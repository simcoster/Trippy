"""
Ingest site-level rules and amenities from parks.org.il camping info pages.

Reads the static page plus the AJAX `מידע למבקר` tab (`אפשרויות לינה` is
ingested by rooms.py). Splits into sections, extracts statements, resolves
each subject against `subject_vectors`, and upserts into `campsite_rules`.

  uv run python -m source.scraper.rules_ingest.ingest --limit 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from db.connect import connect, database_url
from db.models import QualifierUnit
from source.scraper.amenity_enrichment.llm import (
    EmbeddingLLMClient,
    LlmUsage,
    record_scrape_cost,
)
from source.scraper.cli import add_site_argument, site_ids
from source.scraper.rules_ingest.db import (
    DroppedRule,
    ResolvedRule,
    upsert_campsite_rules,
)
from source.scraper.rules_ingest.explain import ConflictExplanation
from source.scraper.rules_ingest.fetch import fetch_page_html
from source.scraper.rules_ingest.llm import RuleExtractorLLMClient
from source.scraper.rules_ingest.lodging import fetch_panel
from source.scraper.rules_ingest.report import SiteRun, write_run_report
from source.scraper.rules_ingest.resolve_conflicts import (
    ConflictResolution,
    ConflictResolverLLMClient,
    drop_redundant_permissions,
    resolve_page_conflicts,
)
from source.scraper.rules_ingest.schemas import miscategorised_rule
from source.scraper.rules_ingest.sections import (
    VISITOR_INFO_TITLE,
    Section,
    parse_sections,
    parse_visitor_info_panel,
)
from source.scraper.rules_ingest.subcamps import (
    load_subcamps,
    subcamp_prompt,
    subcamp_sections,
)
from source.scraper.subjects.llm import SubjectAdjudicatorLLMClient
from source.scraper.subjects.resolve import (
    DEFAULT_STORE,
    ResolutionTrace,
    SubjectRef,
    SubjectStore,
    alias_overflow,
    format_states,
    resolve_subject,
)

if hasattr(sys.stdout, "reconfigure"):
    # line_buffering so progress lines land as they happen when stdout is a
    # pipe or a file, not in one burst at exit.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
DEFAULT_LIMIT = 2
DEFAULT_PAUSE_SECONDS = 0.5

# Sections the parser returns but the ingest does not extract. The rate-card
# notes are per-rate by construction ("בונגלו עם מזגן סופי שבוע וחגים: מותנה
# במינימום 2 לילות") and the extractor drops the rate label, so their facts
# landed as campsite-wide rules: `child_min_age 5`, `weekend_min_nights 2`,
# `mattresses 4`. Parked until statements carry a referent that can route them
# to an accommodation type (PLAN 2026-09-05 "Referent field").
PARKED_SECTION_TITLES = ("הערות למחירון",)


def sections_to_extract(sections: list[Section]) -> list[Section]:
    """The parsed sections minus the parked ones."""
    return [s for s in sections if s.title not in PARKED_SECTION_TITLES]


@dataclass
class SiteReport:
    """What one page's ingest dropped at the upsert, and why each term landed
    where it did.

    `traces` is every resolver decision made for the page; `drops` is every
    statement the upsert refused because its subject already had a row in the
    scope. Together they say, for each collision, which two extractor terms
    were judged one subject and on what grounds -- what you need to decide
    whether that was an over-merge.
    """

    drops: list[DroppedRule] = field(default_factory=list)
    traces: list[ResolutionTrace] = field(default_factory=list)
    # Statements whose category the extractor contradicted with its own naming
    # rule: (term, section title). The category was cleared, not the statement.
    miscategorised: list[tuple[str, str]] = field(default_factory=list)
    # `X_allowed` statements deleted because this pass's own `X` already said
    # it: (subject name, the scope it was written in).
    redundant: list[tuple[str, str]] = field(default_factory=list)
    # Per collision, keyed by index into `drops`, filled after the page is
    # done: the resolver's diagnosis (what the explainer used to give) and its
    # full resolution -- the action, whether it was applied, the case id.
    explanations: dict[int, ConflictExplanation] = field(default_factory=dict)
    resolutions: dict[int, ConflictResolution] = field(default_factory=dict)

    def render(self) -> str:
        # A term is traced the first time it is resolved; later repeats come
        # from the cache, so the first trace is the one that explains it.
        first_trace: dict[str, ResolutionTrace] = {}
        for trace in self.traces:
            first_trace.setdefault(trace.term, trace)
        names = {
            t.subject_id: t.subject_name
            for t in self.traces
            if t.subject_id is not None and t.subject_name
        }
        lines = self._subjects_section(first_trace)
        lines.extend(self._miscategorised_section())
        lines.extend(self._redundant_section())
        lines.extend(self._collisions_section(first_trace, names))
        return "\n".join(lines)

    def _redundant_section(self) -> list[str]:
        """Permissions deleted because the pass's own amenity already said it.

        Not a loss, but worth reading: it says the extractor split one sentence
        into two statements where the prompt asks for one.
        """
        if not self.redundant:
            return []
        lines = ["", "    Redundant permissions dropped (the amenity says it):"]
        for name, scope in sorted(set(self.redundant)):
            lines.append(f"      {name}" + (f"  ({scope})" if scope else ""))
        return lines

    def _miscategorised_section(self) -> list[str]:
        """Terms whose category the extractor contradicted with its own naming rule.

        Worth seeing even though nothing was lost: a `boolean_rule` that coins a
        predicate is the extractor drifting off the prompt, and the subject it
        lands on was chosen without the category filter that normally protects
        it.
        """
        if not self.miscategorised:
            return []
        lines = ["", "    Category cleared (boolean_rule with a coined predicate):"]
        for term, section in sorted(set(self.miscategorised)):
            lines.append(f"      {term}  ({section})")
        return lines

    def _subjects_section(self, first_trace: dict[str, ResolutionTrace]) -> list[str]:
        """Every subject the page touched, with each term that reached it and how."""
        by_subject: dict[int | None, list[ResolutionTrace]] = {}
        for trace in first_trace.values():
            by_subject.setdefault(trace.subject_id, []).append(trace)
        kinds = Counter(t.kind or "?" for t in first_trace.values())
        collided = {d.kept.subject_id: d.label for d in self.drops}

        found = [sid for sid in by_subject if sid is not None]
        summary = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
        lines = [f"    subjects on this page: {len(found)} ({summary})"]
        for sid in sorted(found):
            traces = by_subject[sid]
            name = traces[0].subject_name or "?"
            terms = "; ".join(f"{t.term!r} [{t.kind or '?'}]" for t in traces)
            flag = f"   !! {collided[sid]}" if sid in collided else ""
            lines.append(f"    #{sid} {name!r} <- {terms}{flag}")
        for trace in by_subject.get(None, []):
            lines.append(f"    (no subject) {trace.term!r}: {trace.outcome}")
        return lines

    def _collisions_section(
        self, first_trace: dict[str, ResolutionTrace], names: dict[int, str]
    ) -> list[str]:
        if not self.drops:
            return ["    collisions on this page: none"]
        lines = [f"    collisions on this page: {len(self.drops)}"]
        for index, drop in enumerate(self.drops):
            sid = drop.kept.subject_id
            lines.append(
                f"    {drop.label} on subject #{sid} {names.get(sid, '?')!r} "
                f"(campsite {drop.campsite_id})"
            )
            for role, rule in (("kept   ", drop.kept), ("dropped", drop.dropped)):
                trace = first_trace.get(rule.term) if rule.term else None
                lines.extend(_describe_side(role, rule, trace))
            explanation = self.explanations.get(index)
            if explanation is not None:
                lines.append(f"      explainer: {explanation.one_line()}")
            resolution = self.resolutions.get(index)
            if resolution is not None:
                lines.append(f"      resolution: {resolution.one_line()}")
        return lines


def _describe_side(
    role: str, rule: ResolvedRule, trace: ResolutionTrace | None
) -> list[str]:
    """Three lines: the extractor's term and values, its evidence, its resolution."""
    if rule.qualifier is None:
        value = f"polarity={rule.polarity}"
    else:
        try:
            unit = QualifierUnit(int(rule.qualifier_unit)).name.lower()
        except ValueError:
            unit = str(rule.qualifier_unit)
        value = f"polarity={rule.polarity} qualifier={rule.qualifier} {unit}"
    where = f" in {rule.section_title!r}" if rule.section_title else ""
    how = (
        trace.outcome
        if trace is not None
        else "(no trace: the term was already in the cache for this page)"
    )
    return [
        f"      {role}: {rule.term!r}{where} -> {value}",
        f"               {rule.evidence_span!r}",
        f"               {how}",
    ]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_campsites(
    config: dict, *, limit: int, sites: list[int] | None = None
) -> list[dict]:
    """Campsites with a page of their own, or just the ones `sites` names.

    A subcamp has no page — it is ingested as part of its parent's page, once
    per subcamp. Skipping children here is the whole cost of the split to this
    loop, and `WHERE url IS NOT NULL` is what does it.

    `limit` still applies, so naming more sites than it allows scrapes the
    first `limit` of them; pass `--limit` alongside to raise it.
    """
    where = "url IS NOT NULL"
    params: list = []
    if sites:
        where += " AND id = ANY(%s)"
        params.append(list(sites))
    params.append(limit)
    with connect(database_url(config)) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT id, name, url FROM campsites WHERE {where} ORDER BY id LIMIT %s",
            params,
        )
        rows = cur.fetchall()
    return [{"id": r[0], "name": r[1], "url": r[2]} for r in rows]


def _statement_context(section: Section, statement) -> str:
    """Where this statement came from: the section it sits in and its sentence."""
    span = (statement.evidence_span or "").strip()
    return f"{section.title}: {span}" if span else section.title


def rules_from_sections(
    conn,
    sections: list[Section],
    *,
    extractor: RuleExtractorLLMClient,
    embedder: EmbeddingLLMClient,
    adjudicator: SubjectAdjudicatorLLMClient,
    store: SubjectStore = DEFAULT_STORE,
    cache: dict[str, SubjectRef] | None = None,
    usage: LlmUsage | None = None,
    trace_sink: list[ResolutionTrace] | None = None,
    campsite_id: int | None = None,
    report: SiteReport | None = None,
) -> list[ResolvedRule]:
    """Extract each section, then resolve every subject to a subject_vectors id.

    `campsite_id` is the row being written, so the judge can tell a candidate's
    statements from this page apart from those on other campsites.
    """
    resolved: list[ResolvedRule] = []
    shared_cache = cache if cache is not None else {}
    for section in sections:
        # One chat call per section, and a dense Hebrew amenity list keeps the
        # model busy for 30-90s. The reply is streamed and rendered as dots on
        # this line, so a long wait looks like work rather than a hang.
        print(
            f"    extract: {section.title} ({len(section.text):,} chars) "
            f"-> {extractor.model} ",
            end="",
            flush=True,
        )
        extract_usage = LlmUsage()
        started = time.monotonic()
        try:
            extract = extractor.extract(
                section.text,
                section_title=section.title,
                usage=extract_usage,
                progress=_stream_dots(),
            )
        except Exception as exc:  # noqa: BLE001 — one bad section must not stop the page
            print()  # close the dots line
            print(
                f"    extract failed for {section.title!r} after "
                f"{time.monotonic() - started:.1f}s: {exc}"
            )
            if usage is not None:
                usage.merge(extract_usage)
            continue
        print()  # close the dots line
        print(
            f"      {len(extract.statements)} statement(s) in "
            f"{time.monotonic() - started:.1f}s, tokens "
            f"in={extract_usage.chat_prompt_tokens} "
            f"out={extract_usage.chat_completion_tokens}"
        )

        # The resolver prints a trace per term it has to look up (cache hits
        # are silent). This tally is what says the section finished, and what
        # the embedding + adjudicator calls behind it cost.
        resolve_usage = LlmUsage()
        started = time.monotonic()
        rules, dropped = _resolve_statements(
            conn,
            section,
            extract.statements,
            embedder=embedder,
            adjudicator=adjudicator,
            store=store,
            cache=shared_cache,
            usage=resolve_usage,
            trace_sink=trace_sink,
            campsite_id=campsite_id,
            report=report,
        )
        resolved.extend(rules)
        print(
            f"      resolved: {len(rules)} kept, {dropped} dropped in "
            f"{time.monotonic() - started:.1f}s "
            f"({resolve_usage.embed_calls} embed, "
            f"{resolve_usage.chat_calls} adjudicator call(s))"
        )
        if usage is not None:
            usage.merge(extract_usage)
            usage.merge(resolve_usage)
    return resolved


def _stream_dots(every: int = 40) -> Callable[[int], None]:
    """Progress renderer for `extract`: one dot per `every` streamed chunks."""

    def tick(chunks: int) -> None:
        if chunks % every == 0:
            print(".", end="", flush=True)

    return tick


def _resolve_statements(
    conn,
    section: Section,
    statements,
    *,
    embedder: EmbeddingLLMClient,
    adjudicator: SubjectAdjudicatorLLMClient,
    store: SubjectStore,
    cache: dict[str, SubjectRef],
    usage: LlmUsage,
    trace_sink: list[ResolutionTrace] | None,
    campsite_id: int | None = None,
    report: SiteReport | None = None,
) -> tuple[list[ResolvedRule], int]:
    """Resolve one section's statements to subject ids.

    Returns the rules that survived and how many statements were dropped,
    whether for asserting nothing or because the resolver could not place them.
    """
    rules: list[ResolvedRule] = []
    dropped = 0
    for statement in statements:
        # A `boolean_rule` has to end in `_allowed` or `_required`; the prompt
        # says so exhaustively. When it does not, the label is the part the
        # model got wrong, so drop the label rather than the fact and let the
        # resolver place it on the evidence.
        if miscategorised_rule(statement.subject, statement.category):
            print(
                f"    {statement.subject!r} is category "
                f"{statement.category} but coins a predicate; "
                f"searching every category instead"
            )
            if report is not None:
                report.miscategorised.append((statement.subject, section.title))
            statement.category = None
        # A statement with neither a polarity nor a number asserts nothing:
        # it would add a permanent subject no query can use, and one that
        # later terms could be merged into.
        if statement.polarity is None and statement.qualifier is None:
            print(
                f"    dropping {statement.subject!r}: no polarity and no "
                f"qualifier, so it states nothing"
            )
            dropped += 1
            continue
        ref = resolve_subject(
            conn,
            statement.subject,
            embedder=embedder,
            adjudicator=adjudicator,
            category=statement.category,
            # The sentence the statement was read from is the context a
            # later sameness judgement needs; "toilets" means one thing in
            # a site amenity list and another inside a room description.
            context=_statement_context(section, statement),
            # What the statement asserts, so the judge can see that "30" and
            # "80" from one page are two facts.
            states=format_states(
                statement.polarity, statement.qualifier, statement.qualifier_unit
            ),
            campsite_id=campsite_id,
            store=store,
            cache=cache,
            usage=usage,
            trace_sink=trace_sink,
        )
        if ref is None:
            dropped += 1
            continue
        # A term the resolver had to de-negate ("no dogs") overrides whatever
        # polarity the extractor paired with the negative phrasing.
        polarity = (
            ref.implied_polarity
            if ref.implied_polarity is not None
            else statement.polarity
        )
        rules.append(
            ResolvedRule(
                subject_id=ref.id,
                polarity=polarity,
                qualifier=statement.qualifier,
                qualifier_unit=statement.qualifier_unit,
                evidence_span=statement.evidence_span,
                source_url=section.source_url,
                confidence=statement.confidence,
                term=statement.subject,
                section_title=section.title,
            )
        )
    return rules, dropped


def _ajax_visitor_info(site_url: str, html: str) -> list[Section]:
    """The `מידע למבקר` accordion tab, or [] if the page has none / fetch fails."""
    try:
        panel = fetch_panel(site_url, html, title=VISITOR_INFO_TITLE)
    except httpx.HTTPError as exc:
        print(f"    visitor-info panel fetch failed: {exc}")
        return []
    if not panel:
        return []
    sections = parse_visitor_info_panel(panel, source_url=site_url)
    if sections:
        print(f"    visitor-info panel: {len(sections[0].text):,} chars")
    return sections


def ingest_site(
    conn,
    site: dict,
    html: str,
    *,
    extractor: RuleExtractorLLMClient,
    embedder: EmbeddingLLMClient,
    adjudicator: SubjectAdjudicatorLLMClient,
    store: SubjectStore = DEFAULT_STORE,
    rules_table: str = "campsite_rules",
    usage: LlmUsage | None = None,
    report: SiteReport | None = None,
) -> int:
    parsed = parse_sections(html, source_url=site["url"])
    parsed.extend(_ajax_visitor_info(site["url"], html))
    sections = sections_to_extract(parsed)
    parked = [s.title for s in parsed if s not in sections]
    print(f"    {len(sections)} section(s): {', '.join(s.title for s in sections)}")
    if parked:
        print(f"    parked, not extracted: {', '.join(parked)}")
    if not sections:
        return 0

    subcamps = load_subcamps(conn, site["id"])
    if not subcamps:
        return _ingest_scope(
            conn,
            campsite_id=site["id"],
            sections=sections,
            extractor=extractor,
            embedder=embedder,
            adjudicator=adjudicator,
            store=store,
            rules_table=rules_table,
            usage=usage,
            report=report,
        )

    # One pass per subcamp, each writing to its own campsites row — which is why
    # campsite_rules needs no subcamp dimension. The subject cache is shared
    # across the passes so they converge on the same subject ids; without it the
    # two halves of one site would build parallel vocabularies.
    print(f"    {len(subcamps)} subcamp(s): {', '.join(s.heading for s in subcamps)}")
    cache: dict[str, SubjectRef] = {}
    written = 0
    for subcamp in subcamps:
        print(f"    -- {subcamp.heading} (campsite {subcamp.campsite_id})")
        written += _ingest_scope(
            conn,
            campsite_id=subcamp.campsite_id,
            sections=subcamp_sections(sections, subcamp, subcamps),
            extractor=RuleExtractorLLMClient(
                system_prompt=subcamp_prompt(subcamp, subcamps)
            ),
            embedder=embedder,
            adjudicator=adjudicator,
            store=store,
            rules_table=rules_table,
            cache=cache,
            usage=usage,
            report=report,
        )
    return written


def _ingest_scope(
    conn,
    *,
    campsite_id: int,
    sections: list[Section],
    extractor: RuleExtractorLLMClient,
    embedder: EmbeddingLLMClient,
    adjudicator: SubjectAdjudicatorLLMClient,
    store: SubjectStore,
    rules_table: str,
    cache: dict[str, SubjectRef] | None = None,
    usage: LlmUsage | None = None,
    report: SiteReport | None = None,
) -> int:
    """Extract and write one campsite row's worth of rules."""
    rules = rules_from_sections(
        conn,
        sections,
        extractor=extractor,
        embedder=embedder,
        adjudicator=adjudicator,
        store=store,
        cache=cache,
        usage=usage,
        trace_sink=report.traces if report is not None else None,
        campsite_id=campsite_id,
        report=report,
    )
    if not rules:
        print("    no statements extracted")
        return 0

    with conn.cursor() as cur:
        written = upsert_campsite_rules(
            cur,
            campsite_id=campsite_id,
            rules=rules,
            table=rules_table,
            dropped_sink=report.drops if report is not None else None,
        )
        print(f"    {written} rule(s) upserted")
    # Deterministic, so it runs before the model is asked anything -- and only
    # over what this pass wrote, like the conflict resolver.
    drop_redundant_permissions(
        conn,
        campsite_id=campsite_id,
        rules=rules,
        table=rules_table,
        sink=report.redundant if report is not None else None,
        scope="site-level",
    )
    return written


def run(
    config: dict,
    *,
    limit: int,
    sites: list[int] | None = None,
    usage: LlmUsage | None = None,
    runs: list[SiteRun] | None = None,
    run_at: datetime | None = None,
) -> int:
    """Ingest up to `limit` campsites (or just `site`). Returns rules upserted.

    `usage` is filled with every LLM call made, per role and model, so the
    caller can report the run's cost; a fresh one is used when none is given.
    `runs` collects one `SiteRun` per page -- its report, rows written, time
    taken and any failure -- for the run report `main()` writes afterwards.
    `run_at` stamps the conflict cases this run files.
    """
    campsites = fetch_campsites(config, limit=limit, sites=sites)
    if not campsites:
        print("No campsites found")
        return 0

    pause_s = float(
        config.get("info_site", {}).get("request_pause_seconds", DEFAULT_PAUSE_SECONDS)
    )
    extractor = RuleExtractorLLMClient()
    embedder = EmbeddingLLMClient()
    adjudicator = SubjectAdjudicatorLLMClient()
    resolver = ConflictResolverLLMClient()
    usage = usage if usage is not None else LlmUsage()
    run_at = run_at or datetime.now().astimezone()
    total = 0
    run_started = time.monotonic()

    print(f"Ingesting rules for {len(campsites)} campsite(s)")
    with connect(database_url(config)) as conn:
        for site in campsites:
            print("=" * 60)
            print(f"{site['id']}. {site['name']}")
            print(f"   {site['url']}")
            site_started = time.monotonic()
            report = SiteReport()
            outcome = SiteRun(site=site, report=report)
            if runs is not None:
                runs.append(outcome)
            try:
                html = fetch_page_html(site["url"])
            except httpx.HTTPError as exc:
                print(f"    HTTP error: {exc}")
                outcome.error = f"HTTP error: {exc}"
                outcome.seconds = time.monotonic() - site_started
                continue
            print(
                f"    fetched {len(html):,} chars in "
                f"{time.monotonic() - site_started:.1f}s"
            )
            try:
                outcome.written = ingest_site(
                    conn,
                    site,
                    html,
                    extractor=extractor,
                    embedder=embedder,
                    adjudicator=adjudicator,
                    usage=usage,
                    report=report,
                )
                total += outcome.written
                conn.commit()
                print(f"    site done in {time.monotonic() - site_started:.1f}s")
            except Exception as exc:  # noqa: BLE001 — keep going to the next site
                conn.rollback()
                outcome.error = str(exc)
                print(
                    f"    failed, rolled back after "
                    f"{time.monotonic() - site_started:.1f}s: {exc}"
                )
            outcome.seconds = time.monotonic() - site_started
            # Every collision is something extracted or merged wrongly. Each is
            # diagnosed and filed in conflict_cases; a wrong merge the model is
            # sure of is undone on the spot (its own transaction).
            try:
                resolve_page_conflicts(
                    conn, report, resolver, embedder=embedder, run_at=run_at, usage=usage
                )
                conn.commit()
            except Exception as exc:  # noqa: BLE001 -- the page itself is already committed
                conn.rollback()
                print(f"    conflict resolution failed, rolled back: {exc}")
            # Printed after a rollback too: the over-merge behind a collision
            # is still worth seeing even when nothing was written.
            print(report.render())
            # One JSON line per subject whose alias list has outgrown
            # ALIAS_OVERFLOW -- greppable, and the list itself shows what it ate.
            for subject in alias_overflow(conn):
                print("    ALIAS OVERFLOW " + json.dumps(subject, ensure_ascii=False))
            if pause_s > 0:
                time.sleep(pause_s)

    print("-" * 60)
    print(
        f"Done. Upserted {total} rule(s) in "
        f"{time.monotonic() - run_started:.1f}s."
    )
    if usage.chat_calls or usage.embed_calls:
        print(usage.summary(prefix="Rules ingest total: "))
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Campsite rules ingester")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="How many campsites to process (default: info_site.limit_campsites)",
    )
    add_site_argument(
        parser,
        help_text="Campsite ids to ingest (a parent id for a split site); "
        "repeat or comma-separate. Default: all",
    )
    args = parser.parse_args(argv)
    config = load_config()
    limit = args.limit
    if limit is None:
        limit = int(config.get("info_site", {}).get("limit_campsites", DEFAULT_LIMIT))
    usage = LlmUsage()
    runs: list[SiteRun] = []
    started_at = datetime.now()
    started = time.monotonic()
    run(
        config, limit=limit, sites=site_ids(args.site), usage=usage, runs=runs,
        run_at=started_at.astimezone(),
    )
    # Recorded here, not in run(): a test driving run() must not write reports/.
    written = record_scrape_cost("scrape-rules", usage)
    if written:
        print(f"cost report appended to {written}")
    if runs:
        path = write_run_report(
            runs, usage, started_at=started_at, seconds=time.monotonic() - started
        )
        print(f"run report written to {path}")


if __name__ == "__main__":
    main()
