"""
Ingest site-level rules and amenities from parks.org.il camping info pages.

Reads the static page (no AJAX tabs — per-unit data comes from the availability
scrape), splits it into sections, extracts statements, resolves each subject
against `subject_vectors`, and upserts into `campsite_rules`.

  uv run python -m source.scraper.rules_ingest.ingest --limit 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
import psycopg
from dotenv import load_dotenv

from db.models import QualifierUnit
from source.scraper.amenity_enrichment.llm import (
    EmbeddingLLMClient,
    LlmUsage,
    record_scrape_cost,
)
from source.scraper.rules_ingest.db import (
    DroppedRule,
    ResolvedRule,
    upsert_campsite_rules,
)
from source.scraper.rules_ingest.fetch import fetch_page_html
from source.scraper.rules_ingest.llm import RuleExtractorLLMClient
from source.scraper.rules_ingest.report import SiteRun, write_run_report
from source.scraper.rules_ingest.sections import Section, parse_sections
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
        lines.extend(self._collisions_section(first_trace, names))
        return "\n".join(lines)

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
        for drop in self.drops:
            sid = drop.kept.subject_id
            lines.append(
                f"    {drop.label} on subject #{sid} {names.get(sid, '?')!r} "
                f"(campsite {drop.campsite_id})"
            )
            for role, rule in (("kept   ", drop.kept), ("dropped", drop.dropped)):
                trace = first_trace.get(rule.term) if rule.term else None
                lines.extend(_describe_side(role, rule, trace))
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


def database_url(config: dict) -> str:
    url = os.environ.get("DATABASE_URL") or config.get("database_url")
    if not url:
        raise RuntimeError("No database_url in config or DATABASE_URL env")
    return url.replace("@db:", "@localhost:")


def fetch_campsites(
    config: dict, *, limit: int, site: int | None = None
) -> list[dict]:
    """Campsites with a page of their own, or just the one named by `site`.

    A subcamp has no page — it is ingested as part of its parent's page, once
    per subcamp. Skipping children here is the whole cost of the split to this
    loop, and `WHERE url IS NOT NULL` is what does it.
    """
    where = "url IS NOT NULL"
    params: list = []
    if site is not None:
        where += " AND id = %s"
        params.append(site)
    params.append(limit)
    with psycopg.connect(database_url(config)) as conn, conn.cursor() as cur:
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
) -> list[ResolvedRule]:
    """Extract each section, then resolve every subject to a subject_vectors id."""
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
) -> tuple[list[ResolvedRule], int]:
    """Resolve one section's statements to subject ids.

    Returns the rules that survived and how many statements were dropped,
    whether for asserting nothing or because the resolver could not place them.
    """
    rules: list[ResolvedRule] = []
    dropped = 0
    for statement in statements:
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
    sections = parse_sections(html, source_url=site["url"])
    print(f"    {len(sections)} section(s): {', '.join(s.title for s in sections)}")
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
    return written


def run(
    config: dict,
    *,
    limit: int,
    site: int | None = None,
    usage: LlmUsage | None = None,
    runs: list[SiteRun] | None = None,
) -> int:
    """Ingest up to `limit` campsites (or just `site`). Returns rules upserted.

    `usage` is filled with every LLM call made, per role and model, so the
    caller can report the run's cost; a fresh one is used when none is given.
    `runs` collects one `SiteRun` per page -- its report, rows written, time
    taken and any failure -- for the run report `main()` writes afterwards.
    """
    campsites = fetch_campsites(config, limit=limit, site=site)
    if not campsites:
        print("No campsites found")
        return 0

    pause_s = float(
        config.get("info_site", {}).get("request_pause_seconds", DEFAULT_PAUSE_SECONDS)
    )
    extractor = RuleExtractorLLMClient()
    embedder = EmbeddingLLMClient()
    adjudicator = SubjectAdjudicatorLLMClient()
    usage = usage if usage is not None else LlmUsage()
    total = 0
    run_started = time.monotonic()

    print(f"Ingesting rules for {len(campsites)} campsite(s)")
    with psycopg.connect(database_url(config)) as conn:
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
    parser.add_argument(
        "--site",
        type=int,
        default=None,
        help="Ingest one campsite by id (a parent id for a split site)",
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
    run(config, limit=limit, site=args.site, usage=usage, runs=runs)
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
