"""File every upsert collision for review; undo a wrong merge when the model is sure.

A collision -- two statements from one page landing on one subject -- always
marks something that went wrong upstream. Each one is diagnosed by the 235B
(cause, which side is right, an explanation, as `explain.py` did) and written
to `conflict_cases` for a person to review.

One action is automatic, because it is the one the model chooses and names
well (experiments.md 2026-09-05 §16-§17: 24 of 26 proposals, right whenever
the collision really was two facts):

  rename_new   the new statement is a DIFFERENT fact that was folded into the
               old subject -- a judge over-merge, or an alias hit on a bare name
               missing its qualifying word. It gets its own subject under
               `new_name`, the merged alias is released from the old subject,
               and the statement is written. The old subject and the kept row
               are never touched.

Everything else -- a true duplicate, a hallucination, a rate-specific line, a
kept row that is itself wrong -- is `none`: filed as `open`, nothing changes.
The model's confidence is recorded but not gated on: it said 0.95 on right and
wrong answers alike (§17).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING

from openai import OpenAI
from pydantic import BaseModel, field_validator

from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_MODEL,
    EmbeddingLLMClient,
    LlmUsage,
    _parse_json_payload,
    make_nebius_openai_client,
)
from source.scraper.rules_ingest.db import upsert_campsite_rules
from source.scraper.rules_ingest.explain import (
    PIPELINE_MECHANICS,
    ConflictExplanation,
    _side,
)
from source.scraper.subjects.naming import normalize_alias
from source.scraper.subjects.resolve import (
    DEFAULT_STORE,
    SubjectStore,
    ensure_table_name,
    vector_literal,
)

if TYPE_CHECKING:
    from source.scraper.rules_ingest.db import DroppedRule
    from source.scraper.rules_ingest.ingest import SiteReport
    from source.scraper.subjects.resolve import ResolutionTrace

ACTIONS = ("none", "rename_new")

RESOLVE_SYSTEM_PROMPT = (
    "You review a campsite-rules ingestion pipeline. Two statements from one page "
    "landed on one subject and the second was refused; explain why, and say whether "
    "the one automatic repair applies.\n\n"
    + PIPELINE_MECHANICS
    + """

You are also told about the OLD subject: its name, its aliases (other names that
were merged into it) and how many campsite rows cite it.

Choose exactly one action:
- "rename_new"  the new statement is a DIFFERENT fact about the campsite that was
                wrongly folded into the old subject -- by a judge over-merge, or by
                an alias hit on a bare name that lost its qualifying word. It gets its
                own subject and the merge is undone. Provide `new_name` in the naming
                shape: the extractor's own name when it already carries the
                difference (family_and_friends_group_min_occupancy,
                late_check_out_on_saturday_evening_allowed), or the bare name with the
                missing word restored IN FRONT (accessible_toilets -- never
                toilets_accessible, never a _count or _accessible suffix, never a
                transliterated Hebrew word).
- "none"        anything else. The same fact stated twice (with or without a count).
                A statement not in its sentence. A line about one unit or rate. And
                the case where the KEPT row is the wrong one -- you cannot touch the
                kept row; say so in the explanation and a person will.

When two sentences state two facts about the campsite, choose rename_new.

Output valid JSON only:
{
  "cause": "extractor_wrong_name" | "extractor_wrong_value" | "extractor_hallucination"
           | "judge_over_merge" | "true_duplicate" | "other",
  "which_is_right": "kept" | "dropped" | "both" | "neither",
  "explanation": one or two English sentences citing the Hebrew,
  "action": "rename_new" | "none",
  "new_name": snake_case or null,
  "rationale": one sentence on why this action,
  "confidence": number 0..1
}
"""
)


@dataclass(frozen=True)
class SubjectFacts:
    """What the resolver is told about the old subject."""

    name: str
    category: int
    aliases: tuple[str, ...] = ()
    rule_count: int = 0

    def describe(self) -> str:
        others = [a for a in self.aliases if a != self.name]
        alias_text = ", ".join(others) if others else "none"
        return (
            f"OLD SUBJECT: {self.name!r} (category {self.category}); "
            f"aliases merged into it: {alias_text}; "
            f"rows citing it across campsites: {self.rule_count}"
        )


class ResolutionPayload(BaseModel):
    cause: str = "other"
    which_is_right: str = "neither"
    explanation: str = ""
    action: str = "none"
    new_name: str | None = None
    rationale: str = ""
    confidence: float | None = None

    @field_validator("action", mode="before")
    @classmethod
    def _known_action(cls, value: object) -> str:
        text = str(value or "").strip().casefold()
        return text if text in ACTIONS else "none"

    @field_validator("new_name", mode="before")
    @classmethod
    def _snake(cls, value: object) -> str | None:
        if value is None:
            return None
        return normalize_alias(str(value)) or None

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, value: object) -> float | None:
        try:
            return None if value is None else min(1.0, max(0.0, float(value)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None


@dataclass
class ConflictResolution:
    """The diagnosis and the proposal, after validation; then what was done."""

    cause: str
    which_is_right: str
    explanation: str
    action: str
    new_name: str | None = None
    rationale: str = ""
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)
    # Filled by `apply_and_file`.
    case_id: int | None = None
    applied: bool = False
    applied_subject_id: int | None = None

    def as_explanation(self) -> ConflictExplanation:
        return ConflictExplanation(self.cause, self.which_is_right, self.explanation, self.rationale)

    def one_line(self) -> str:
        if self.action == "rename_new":
            what = f"rename_new -> {self.new_name!r}"
            what += (
                f", applied as subject #{self.applied_subject_id}" if self.applied else ", NOT applied"
            )
        else:
            what = "none: filed for review"
        if self.case_id is not None:
            what += f" (case #{self.case_id})"
        if self.confidence is not None:
            what += f" [confidence {self.confidence:.2f}]"
        tail = f" [{'; '.join(self.notes)}]" if self.notes else ""
        return f"{what}. {self.rationale}{tail}"


def validate(proposed: ResolutionPayload, subject: SubjectFacts, new_term: str) -> ConflictResolution:
    """`rename_new` needs a name other than the old subject's canonical one.

    An alias of the old subject is fine -- the new term usually IS one, added
    by the very merge being undone. Missing or unusable, the extractor's own
    term is used; if that is the old name itself there is nothing to rename to
    and the case is filed as `none`.
    """
    out = ConflictResolution(
        proposed.cause, proposed.which_is_right, proposed.explanation, proposed.action,
        proposed.new_name, proposed.rationale, proposed.confidence,
    )
    if out.action == "rename_new":
        old = normalize_alias(subject.name)
        if not out.new_name or out.new_name == old:
            fallback = normalize_alias(new_term)
            if fallback and fallback != old:
                out.notes.append(f"new_name missing or the old name; using the extractor's term {fallback!r}")
                out.new_name = fallback
            else:
                out.notes.append("no distinct name to rename to; filed as none")
                out.action, out.new_name = "none", None
    if out.action != "rename_new":
        out.new_name = None
    return out


class ConflictResolverLLMClient:
    """Diagnose a collision and say whether `rename_new` applies."""

    MODEL = QWEN_INSTRUCT_MODEL
    TEMPERATURE = 0
    ROLE = "conflict_resolver"

    def __init__(self, client: OpenAI | None = None, *, model: str | None = None) -> None:
        self._client = client
        self.model = model or self.MODEL

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = make_nebius_openai_client()
        return self._client

    def resolve(
        self,
        drop: DroppedRule,
        subject: SubjectFacts,
        *,
        kept_how: str | None = None,
        dropped_how: str | None = None,
        usage: LlmUsage | None = None,
    ) -> ConflictResolution:
        body = "\n".join(
            [
                f"Collision ({drop.label}) on campsite {drop.campsite_id}:",
                "",
                *_side("kept", drop.kept, kept_how),
                *_side("dropped (the new statement)", drop.dropped, dropped_how),
                subject.describe(),
            ]
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RESOLVE_SYSTEM_PROMPT},
                {"role": "user", "content": body},
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(response.usage, role=self.ROLE, model=self.model)
        try:
            data = _parse_json_payload(response.choices[0].message.content or "")
        except ValueError:
            data = {"explanation": (response.choices[0].message.content or "").strip()}
        return validate(ResolutionPayload.model_validate(data), subject, drop.dropped.term or "")


# ------------------------------------------------------------------ database
def subject_facts(conn, subject_id: int, store: SubjectStore = DEFAULT_STORE) -> SubjectFacts:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT s.name, s.category, s.aliases,
                   (SELECT count(*) FROM {store.rules_table} r WHERE r.subject_id = s.id)
            FROM {store.table} s WHERE s.id = %s
            """,
            (subject_id,),
        )
        row = cur.fetchone()
    if row is None:
        return SubjectFacts(f"#{subject_id}", 0)
    return SubjectFacts(row[0], int(row[1]), tuple(row[2] or ()), int(row[3]))


def apply_rename_new(
    conn,
    drop: DroppedRule,
    resolution: ConflictResolution,
    *,
    embedder: EmbeddingLLMClient,
    store: SubjectStore = DEFAULT_STORE,
    usage: LlmUsage | None = None,
) -> int:
    """Undo the merge: release the alias, give the new statement its own subject,
    write its row. Returns the subject id the statement now lives under.

    The old subject keeps its name, its canonical alias and its rows. If a
    subject called `new_name` already exists the statement joins it instead of
    creating a duplicate.
    """
    new_name = resolution.new_name
    assert new_name, "apply_rename_new needs a name"
    term = normalize_alias(drop.dropped.term or "") or new_name
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT name, category FROM {store.table} WHERE id = %s", (drop.kept.subject_id,)
        )
        old_row = cur.fetchone()
        old_name = normalize_alias(old_row[0]) if old_row else ""
        category = int(old_row[1]) if old_row else 1
        # The term travels with the statement only when it is a merged alias.
        # A bare name that IS the old subject (`toilets` renamed to
        # `accessible_toilets`) stays where it is: releasing it would orphan the
        # old subject's own name, aliasing it elsewhere would hijack every later
        # `toilets`.
        movable = term != old_name
        if movable:
            # 1. The merge being undone added `term` as an alias of the old
            #    subject. Release it -- never the canonical alias, aliases[1].
            cur.execute(
                f"""
                UPDATE {store.table}
                SET aliases = array_remove(aliases, %(alias)s)
                WHERE id = %(id)s AND aliases[1] <> %(alias)s
                """,
                {"id": drop.kept.subject_id, "alias": term},
            )
        # 2. A home for the new statement.
        cur.execute(f"SELECT id FROM {store.table} WHERE name = %s", (new_name,))
        row = cur.fetchone()
        if row is not None:
            subject_id = int(row[0])
            if movable and term != new_name:
                cur.execute(
                    f"""
                    UPDATE {store.table}
                    SET aliases = array_append(aliases, %(alias)s)
                    WHERE id = %(id)s AND NOT (aliases @> ARRAY[%(alias)s]::text[])
                    """,
                    {"id": subject_id, "alias": term},
                )
        else:
            aliases = [new_name] + ([term] if movable and term != new_name else [])
            vector = embedder.embed([new_name], usage=usage)[0]
            columns = ["name", "category", "aliases", "embedding"]
            values = ["%(name)s", "%(category)s", "%(aliases)s", "%(embedding)s::vector"]
            params: dict[str, object] = {
                "name": new_name,
                "category": category,
                "aliases": aliases,
                "embedding": vector_literal(vector),
            }
            if store.has_context:
                columns.append("context")
                values.append("%(context)s")
                section = drop.dropped.section_title
                span = drop.dropped.evidence_span or ""
                params["context"] = f"{section}: {span}" if section else span
            cur.execute(
                f"INSERT INTO {store.table} ({', '.join(columns)}) VALUES ({', '.join(values)}) RETURNING id",
                params,
            )
            subject_id = int(cur.fetchone()[0])
        # 3. The statement itself, under its own subject now.
        upsert_campsite_rules(
            cur,
            campsite_id=drop.campsite_id,
            rules=[replace(drop.dropped, subject_id=subject_id)],
            table=store.rules_table,
        )
    return subject_id


def file_conflict_case(
    conn,
    drop: DroppedRule,
    resolution: ConflictResolution,
    *,
    run_at: datetime,
    kept_how: str | None,
    dropped_how: str | None,
    table: str = "conflict_cases",
) -> int:
    """Insert the case; returns its id."""
    table = ensure_table_name(table)
    k, n = drop.kept, drop.dropped
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {table} (
                run_at, campsite_id, subject_id, label,
                kept_term, kept_section, kept_polarity, kept_qualifier, kept_qualifier_unit,
                kept_evidence, kept_resolution,
                new_term, new_section, new_polarity, new_qualifier, new_qualifier_unit,
                new_evidence, new_resolution,
                cause, which_is_right, explanation, rationale, confidence,
                action, new_name, applied, applied_subject_id, status
            ) VALUES (
                %(run_at)s, %(campsite_id)s, %(subject_id)s, %(label)s,
                %(kt)s, %(ks)s, %(kp)s, %(kq)s, %(ku)s, %(ke)s, %(kh)s,
                %(nt)s, %(ns)s, %(np)s, %(nq)s, %(nu)s, %(ne)s, %(nh)s,
                %(cause)s, %(right)s, %(explanation)s, %(rationale)s, %(confidence)s,
                %(action)s, %(new_name)s, %(applied)s, %(applied_subject_id)s, %(status)s
            ) RETURNING id
            """,
            {
                "run_at": run_at, "campsite_id": drop.campsite_id,
                "subject_id": k.subject_id, "label": drop.label,
                "kt": k.term, "ks": k.section_title, "kp": k.polarity, "kq": k.qualifier,
                "ku": int(k.qualifier_unit), "ke": k.evidence_span, "kh": kept_how,
                "nt": n.term, "ns": n.section_title, "np": n.polarity, "nq": n.qualifier,
                "nu": int(n.qualifier_unit), "ne": n.evidence_span, "nh": dropped_how,
                "cause": resolution.cause, "right": resolution.which_is_right,
                "explanation": resolution.explanation, "rationale": resolution.rationale,
                "confidence": resolution.confidence,
                "action": resolution.action, "new_name": resolution.new_name,
                "applied": resolution.applied, "applied_subject_id": resolution.applied_subject_id,
                "status": "applied" if resolution.applied else "open",
            },
        )
        return int(cur.fetchone()[0])


def resolve_page_conflicts(
    conn,
    report: SiteReport,
    resolver: ConflictResolverLLMClient,
    *,
    embedder: EmbeddingLLMClient,
    run_at: datetime,
    store: SubjectStore = DEFAULT_STORE,
    cases_table: str = "conflict_cases",
    usage: LlmUsage | None = None,
    verbose: bool = True,
) -> int:
    """For every collision on the page: diagnose, apply `rename_new` when
    proposed, file the case. Fills `report.resolutions` and
    `report.explanations`; returns how many were applied.

    One model call per collision. A failure on one collision files it as
    `none` with the error as its explanation and moves on; the caller commits.
    """
    if not report.drops:
        return 0
    first: dict[str, ResolutionTrace] = {}
    for trace in report.traces:
        first.setdefault(trace.term, trace)
    if verbose:
        print(f"    resolving {len(report.drops)} collision(s) -> {resolver.model} ", end="", flush=True)
    started = time.monotonic()
    applied = 0
    for index, drop in enumerate(report.drops):
        kept = first.get(drop.kept.term or "")
        dropped = first.get(drop.dropped.term or "")
        kept_how = kept.outcome if kept else None
        dropped_how = dropped.outcome if dropped else None
        try:
            resolution = resolver.resolve(
                drop, subject_facts(conn, drop.kept.subject_id, store),
                kept_how=kept_how, dropped_how=dropped_how, usage=usage,
            )
        except Exception as exc:  # noqa: BLE001 -- file it, keep going
            resolution = ConflictResolution("other", "neither", f"resolver failed: {exc}", "none")
        if resolution.action == "rename_new":
            try:
                resolution.applied_subject_id = apply_rename_new(
                    conn, drop, resolution, embedder=embedder, store=store, usage=usage
                )
                resolution.applied = True
                applied += 1
            except Exception as exc:  # noqa: BLE001 -- the case still gets filed
                resolution.notes.append(f"apply failed: {exc}")
        resolution.case_id = file_conflict_case(
            conn, drop, resolution, run_at=run_at, kept_how=kept_how,
            dropped_how=dropped_how, table=cases_table,
        )
        report.resolutions[index] = resolution
        report.explanations[index] = resolution.as_explanation()
        if verbose:
            print(".", end="", flush=True)
    if verbose:
        print(f" {len(report.drops)} filed, {applied} applied in {time.monotonic() - started:.1f}s")
    return applied
