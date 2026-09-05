"""Resolve a raw term to a `subject_vectors` row, growing aliases as it goes.

The point is that `air_conditioner` and `air_conditioning` must land on one row
with one vector, while `dogs_allowed` and `last_dogs_entry_time` must stay
apart. Exact-alias lookup handles everything already seen; only genuinely new
surface forms pay for an embedding and two small-LLM calls.

    term -> normalize + force positive phrasing
         -> exact hit on aliases (GIN)                        [no LLM]
         -> 5 nearest by <#>, filtered, adjudicate             [1 LLM call]
            -> match: append the term as an alias, done
         -> insert the TERM as a new subject, storing the probe embedding;
            the classifier is asked only for a category the extractor left out
            (the extractor names subjects -- nothing downstream renames them)

Three filters stand between "near in the vector space" and "same subject":

  distance   too far to be worth asking about
  category   an amenity, a boolean rule and a numeric rule never mix —
             `barbecue_allowed` vs `barbecue`, `late_check_out_allowed` vs
             `late_check_out_end_time`
  opposed    antonyms are opposite facts — `child_min_age` vs `child_max_age`

Whether two names state the same predicate (a permission vs a time vs a price)
is deliberately NOT filtered here: a suffix list used to do it and fragmented
`late_check_out_*` into nine subjects wherever a suffix was missing from the
list. The judge decides that, with both names and their contexts in view. When
the judge rejects every neighbour, those neighbours are handed to the classifier
so the new canonical name is chosen to stay clear of them.

Every resolution emits a one-line trace of what was considered and why it landed
where it did. Over-merges are silent without it (see docs/design.md).

Embeddings are cast through a literal `%s::vector`, so callers do not need
`register_vector` on the connection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from db.models import QualifierUnit, SubjectCategory
from source.scraper.amenity_enrichment.llm import EmbeddingLLMClient, LlmUsage
from source.scraper.subjects.llm import (
    MATCH_MIN_CONFIDENCE,
    Judgement,
    SubjectAdjudicatorLLMClient,
)
from source.scraper.subjects.naming import (
    normalize_alias,
    opposed,
    to_positive_subject,
)

# Five nearest, per the resolution design; `<#>` is negative inner product, so
# more-negative is closer. Mirrors AMENITY_MATCH_MAX_DISTANCE in the planner.
NEAREST_K = 5
MATCH_MAX_DISTANCE = -0.75

# Why a nearest neighbour never reached the adjudicator.
REJECT_FAR = "far"
REJECT_CATEGORY = "category"
REJECT_OPPOSED = "opposed"

_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def ensure_table_name(name: str) -> str:
    """Return `name` if it is a plain table identifier, else raise.

    Table names are injected so experiments can target `test_*` copies, which
    means they get interpolated into SQL rather than passed as parameters.
    """
    if not _IDENT_RE.fullmatch(name or ""):
        raise ValueError(f"not a plain table identifier: {name!r}")
    return name


@dataclass(frozen=True)
class SubjectStore:
    """Which table holds the subjects, and what it can store.

    Injected so an ingestion experiment can point at `test_subject_vectors`
    without the production schema being touched. The table name is interpolated
    into SQL, so it is checked against a plain-identifier pattern.

    `has_context` says whether the table carries the `context` column: the
    sentence a subject was first read from, shown to the sameness judge so it
    can tell a communal toilet block from an in-room bathroom.
    """

    table: str = "subject_vectors"
    has_context: bool = True
    # Where a candidate's existing statements are read from, so the judge can
    # be shown what each side states.
    rules_table: str = "campsite_rules"

    def __post_init__(self) -> None:
        ensure_table_name(self.table)
        ensure_table_name(self.rules_table)


DEFAULT_STORE = SubjectStore()


@dataclass(frozen=True)
class SubjectRef:
    """A resolved subject, plus any polarity implied by the term's phrasing."""

    id: int
    name: str
    category: int
    # False when the incoming term was negatively phrased ("no dogs") and the
    # negation was moved out of the name. None when the term said nothing.
    implied_polarity: bool | None = None


@dataclass(frozen=True)
class Candidate:
    """One nearest neighbour and what became of it."""

    id: int
    name: str
    distance: float
    category: int
    # None means it was offered to the adjudicator.
    rejected_for: str | None = None
    # What this subject was first read from, when the store keeps it.
    context: str | None = None


@dataclass
class ResolutionTrace:
    """What was considered for one term, and how it was decided."""

    term: str
    normalized: str
    category: int | None = None
    # The sentence the term was read from, for reports that show a merge with
    # both sides' original phrasing.
    context: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    outcome: str = ""
    # Which path decided it, for reports that group by decision rather than
    # parse `outcome`: "alias", "merged", "inserted" or "dropped".
    kind: str = ""
    subject_id: int | None = None
    subject_name: str | None = None

    @property
    def offered(self) -> list[Candidate]:
        return [c for c in self.candidates if c.rejected_for is None]


def category_label(category: int | None) -> str:
    if category is None:
        return "?"
    try:
        return SubjectCategory(int(category)).name.lower()
    except ValueError:
        return str(category)


def format_trace(trace: ResolutionTrace) -> str:
    """The one-line story of a resolution, for the ingest log."""
    head = f"{trace.term!r} from extractor ({category_label(trace.category)})."
    if not trace.candidates:
        return f"    {head} {trace.outcome}"

    listed = ", ".join(
        f"{c.name} {c.distance:+.3f}"
        + (f"[{c.rejected_for}]" if c.rejected_for else "")
        for c in trace.candidates
    )
    return (
        f"    {head} no alias match. ran NN, top {len(trace.candidates)}: "
        f"[{listed}]. considered {len(trace.offered)}. {trace.outcome}"
    )


def vector_literal(values: list[float]) -> str:
    """pgvector literal, matching `source.agent.search._query_vec_literal`."""
    return "[" + ",".join(f"{x:.8f}" for x in values) + "]"


def format_states(
    polarity: bool | None, qualifier: object, qualifier_unit: int | None
) -> str | None:
    """What a statement asserts, as the judge is shown it: "polarity=true
    qualifier=30 count". None when it asserts nothing."""
    parts = []
    if polarity is not None:
        parts.append(f"polarity={str(bool(polarity)).lower()}")
    if qualifier is not None:
        try:
            unit = QualifierUnit(int(qualifier_unit or 0)).name.lower()
        except ValueError:
            unit = str(qualifier_unit)
        number = f"{float(qualifier):g}"
        parts.append(f"qualifier={number}" + (f" {unit}" if unit != "none" else ""))
    return " ".join(parts) or None


def resolve_subject(
    conn,
    term: str,
    *,
    embedder: EmbeddingLLMClient,
    adjudicator: SubjectAdjudicatorLLMClient,
    category: int | None = None,
    context: str | None = None,
    states: str | None = None,
    campsite_id: int | None = None,
    store: SubjectStore = DEFAULT_STORE,
    cache: dict[str, SubjectRef] | None = None,
    usage: LlmUsage | None = None,
    trace_sink: list[ResolutionTrace] | None = None,
    verbose: bool = True,
) -> SubjectRef | None:
    """Return the subject this term names, creating it if it is genuinely new.

    `states` is what the statement asserts ("qualifier=30 count"); together
    with each candidate's existing rows it is shown to the judge, which is what
    stopped a 30-person minimum merging into an 80-person one. `campsite_id`
    marks which of a candidate's rows come from the page being read.

    `category` is what the extractor read off the sentence — it has context a
    bare term does not, so it beats the classifier's guess and keeps rules and
    amenities out of each other's candidate lists. None searches every category.

    `context` is the sentence the term was read from. It is stored with a new
    subject and shown to the sameness judge on later comparisons, which is how
    `toilets` (a 30-stall communal block) stays apart from `bathroom` (in a
    room) when the names alone give no clue.

    None is returned when the term was empty, or so negatively phrased it could
    not be rewritten into a positive subject; drop the statement.
    """
    raw_key = normalize_alias(term)
    if not raw_key:
        return None
    if cache is not None and raw_key in cache:
        return cache[raw_key]

    trace = ResolutionTrace(
        term=term, normalized=raw_key, category=category, context=context
    )
    positive, implied = to_positive_subject(raw_key)
    if positive is None:
        trace.outcome = "DROPPED (negative phrasing that cannot be made positive)."
        trace.kind = "dropped"
        _emit(trace, trace_sink, verbose)
        return None
    trace.normalized = positive

    ref = _resolve_positive(
        conn,
        positive,
        embedder=embedder,
        adjudicator=adjudicator,
        category=category,
        context=context,
        states=states,
        campsite_id=campsite_id,
        store=store,
        usage=usage,
        trace=trace,
    )
    _emit(trace, trace_sink, verbose)
    if ref is None:
        return None

    ref = SubjectRef(ref.id, ref.name, ref.category, implied)
    if cache is not None:
        cache[raw_key] = ref
    return ref


def _emit(
    trace: ResolutionTrace,
    trace_sink: list[ResolutionTrace] | None,
    verbose: bool,
) -> None:
    if trace_sink is not None:
        trace_sink.append(trace)
    if verbose:
        print(format_trace(trace))


def _resolve_positive(
    conn,
    key: str,
    *,
    embedder: EmbeddingLLMClient,
    adjudicator: SubjectAdjudicatorLLMClient,
    category: int | None,
    context: str | None,
    states: str | None,
    campsite_id: int | None,
    store: SubjectStore,
    usage: LlmUsage | None,
    trace: ResolutionTrace,
) -> SubjectRef | None:
    hit = _select_by_alias(conn, key, store)
    if hit is not None:
        trace.outcome = f"alias hit -> {hit.name!r}."
        trace.kind = "alias"
        trace.subject_id, trace.subject_name = hit.id, hit.name
        return hit

    probe = embedder.embed([key], usage=usage)[0]
    # The category filter belongs in the SQL: filtering afterwards spends all
    # five slots on the wrong category and can leave nothing to consider.
    trace.candidates = [
        _judge_candidate(key, row, category)
        for row in _nearest(conn, probe, NEAREST_K, category=category, store=store)
    ]
    offered = trace.offered

    matched = None
    judgements: list[Judgement] = []
    if offered:
        matched = adjudicator.pick_match(
            key,
            [c.name for c in offered],
            term_context=context,
            candidate_contexts={c.name: c.context for c in offered if c.context},
            term_states=states,
            candidate_states=_candidate_states(conn, offered, campsite_id, store),
            usage=usage,
            judgement_sink=judgements,
        )
    judgement = judgements[-1] if judgements else None
    if matched is not None:
        winner = next(c for c in offered if c.name == matched)
        _append_alias(conn, winner.id, key, store)
        sure = (
            f" (confidence {judgement.confidence:.2f})"
            if judgement is not None and judgement.confidence is not None
            else ""
        )
        trace.outcome = f"ADJUDICATOR merged into {matched!r}{sure}."
        trace.kind = "merged"
        trace.subject_id, trace.subject_name = winner.id, winner.name
        return SubjectRef(winner.id, winner.name, winner.category)

    # The extractor names subjects; nothing here renames them. Letting the
    # classifier pick a "distinct" name was measured over 40 terms: it reordered
    # words, dropped words, and turned `dogs_entry_time` ("from 16:00") into
    # `last_dogs_entry_time`. The classifier is consulted only for a category
    # the extractor did not supply.
    if category is not None:
        resolved_category = int(category)
    else:
        payload = adjudicator.classify(key, context=context, usage=usage)
        resolved_category = int(payload.category)
    if judgement is not None and judgement.match is not None:
        # Answered a name but below the confidence gate: the merge did not happen.
        verdict = (
            f"ADJUDICATOR said {judgement.match!r} at confidence "
            f"{judgement.confidence:.2f} < {MATCH_MIN_CONFIDENCE}: rejected."
        )
    else:
        verdict = "ADJUDICATOR rejected all." if offered else "nothing near enough to ask."

    # The term is the name, so the probe that ran the neighbour search is the
    # row's vector: no second embedding, and the alias key can never collide
    # with a different subject's alias -- it just missed that lookup.
    inserted = _insert(conn, key, resolved_category, [key], probe, context, store)
    trace.outcome = (
        f"{verdict} INSERTED as {category_label(resolved_category)} {key!r}."
    )
    trace.kind = "inserted"
    trace.subject_id, trace.subject_name = inserted.id, inserted.name
    return inserted


def _judge_candidate(
    key: str,
    row: tuple[int, str, int, float, str | None],
    category: int | None,
) -> Candidate:
    subject_id, name, row_category, distance, context = row
    if distance > MATCH_MAX_DISTANCE:
        reason = REJECT_FAR
    elif category is not None and int(row_category) != int(category):
        # Belt and braces: the SQL already filtered, so this only fires when the
        # caller passed no category to the query.
        reason = REJECT_CATEGORY
    elif opposed(key, name):
        # min vs max, start vs end: opposite facts wearing near-identical names.
        reason = REJECT_OPPOSED
    else:
        # Everything else -- including whether the two names state the same
        # predicate -- is the judge's call, with contexts in view.
        reason = None
    return Candidate(
        id=subject_id,
        name=name,
        distance=distance,
        category=row_category,
        rejected_for=reason,
        context=context,
    )


# A candidate with rows on many campsites is summarised: the first few, then a count.
STATES_SHOWN = 4


def _candidate_states(
    conn, offered: list[Candidate], campsite_id: int | None, store: SubjectStore
) -> dict[str, str | None]:
    """What each offered candidate's existing rows assert, per campsite.

    The page being read is marked "(same page)": two statements from one page
    that state different numbers are two facts, where a different number on
    another campsite is normal.
    """
    if not offered:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT subject_id, campsite_id, polarity, qualifier, qualifier_unit
            FROM {store.rules_table}
            WHERE subject_id = ANY(%s)
            ORDER BY subject_id, campsite_id
            """,
            ([c.id for c in offered],),
        )
        rows = cur.fetchall()
    by_id: dict[int, list[str]] = {}
    # The page being read first: it is the entry that decides "two facts from
    # one page", and must not fall into the "+N more" tail.
    for subject_id, row_campsite, polarity, qualifier, unit in sorted(
        rows, key=lambda r: (r[1] != campsite_id, r[1])
    ):
        states = format_states(polarity, qualifier, unit)
        if states is None:
            continue
        where = "same page" if row_campsite == campsite_id else f"campsite {row_campsite}"
        by_id.setdefault(int(subject_id), []).append(f"{states} ({where})")
    out: dict[str, str | None] = {}
    for c in offered:
        entries = by_id.get(c.id, [])
        if not entries:
            continue
        shown = "; ".join(entries[:STATES_SHOWN])
        if len(entries) > STATES_SHOWN:
            shown += f"; +{len(entries) - STATES_SHOWN} more"
        out[c.name] = shown
    return out


def _select_by_alias(conn, key: str, store: SubjectStore) -> SubjectRef | None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, name, category
            FROM {store.table}
            WHERE aliases @> ARRAY[%s]::text[]
            LIMIT 1
            """,
            (key,),
        )
        row = cur.fetchone()
    return None if row is None else SubjectRef(int(row[0]), row[1], int(row[2]))


def _nearest(
    conn,
    probe: list[float],
    limit: int,
    *,
    category: int | None = None,
    store: SubjectStore = DEFAULT_STORE,
) -> list[tuple[int, str, int, float, str | None]]:
    """The `limit` nearest subjects, restricted to one category when given.

    Restricting in SQL rather than afterwards is the point: the partial HNSW
    indexes from migration 025 serve exactly this, and a post-filter would spend
    all `limit` slots on the wrong category and return nothing to consider.
    """
    literal = vector_literal(probe)
    context_col = "context" if store.has_context else "NULL::text"
    clauses = ["embedding IS NOT NULL"]
    params: list[object] = [literal]
    if category is not None:
        clauses.append("category = %s")
        params.append(int(category))
    params.extend([literal, limit])
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, name, category, embedding <#> %s::vector AS distance,
                   {context_col} AS context
            FROM {store.table}
            WHERE {" AND ".join(clauses)}
            ORDER BY embedding <#> %s::vector
            LIMIT %s
            """,
            params,
        )
        rows = cur.fetchall()
    return [(int(r[0]), r[1], int(r[2]), float(r[3]), r[4]) for r in rows]


def _append_alias(conn, subject_id: int, key: str, store: SubjectStore) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {store.table}
            SET aliases = array_append(aliases, %(alias)s)
            WHERE id = %(id)s
              AND NOT (aliases @> ARRAY[%(alias)s]::text[])
            """,
            {"id": subject_id, "alias": key},
        )


# Past this many aliases a subject is usually swallowing its neighbours: in one
# experiment `check_in_time` took `car_entry_time`, `arrival_time` and
# `earliest_check_in_time` in a single run. The ingest reports offenders after
# every site.
ALIAS_OVERFLOW = 20


def alias_overflow(
    conn, *, threshold: int = ALIAS_OVERFLOW, store: SubjectStore = DEFAULT_STORE
) -> list[dict]:
    """Subjects whose alias list has grown past `threshold`, longest first."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, name, category, aliases
            FROM {store.table}
            WHERE cardinality(aliases) > %s
            ORDER BY cardinality(aliases) DESC, id
            """,
            (threshold,),
        )
        rows = cur.fetchall()
    return [
        {
            "id": int(r[0]),
            "name": r[1],
            "category": category_label(int(r[2])),
            "n_aliases": len(r[3]),
            "aliases": list(r[3]),
        }
        for r in rows
    ]


def _insert(
    conn,
    name: str,
    category: int,
    aliases: list[str],
    vector: list[float],
    context: str | None = None,
    store: SubjectStore = DEFAULT_STORE,
) -> SubjectRef:
    columns = ["name", "category", "aliases", "embedding"]
    values = ["%(name)s", "%(category)s", "%(aliases)s", "%(embedding)s::vector"]
    params: dict[str, object] = {
        "name": name,
        "category": int(category),
        "aliases": aliases,
        "embedding": vector_literal(vector),
    }
    if store.has_context:
        columns.append("context")
        values.append("%(context)s")
        params["context"] = context
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {store.table} ({", ".join(columns)})
            VALUES ({", ".join(values)})
            ON CONFLICT (name) DO UPDATE
            SET embedding = COALESCE({store.table}.embedding, EXCLUDED.embedding)
            RETURNING id, name, category
            """,
            params,
        )
        row = cur.fetchone()
    return SubjectRef(int(row[0]), row[1], int(row[2]))
