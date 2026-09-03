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
from pathlib import Path

import httpx
import psycopg
from dotenv import load_dotenv

from source.scraper.amenity_enrichment.llm import EmbeddingLLMClient, LlmUsage
from source.scraper.rules_ingest.db import ResolvedRule, upsert_campsite_rules
from source.scraper.rules_ingest.fetch import fetch_page_html
from source.scraper.rules_ingest.llm import RuleExtractorLLMClient
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
    resolve_subject,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
DEFAULT_LIMIT = 2
DEFAULT_PAUSE_SECONDS = 0.5


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
        print(f"    extract: {section.title}")
        try:
            extract = extractor.extract(
                section.text, section_title=section.title, usage=usage
            )
        except Exception as exc:  # noqa: BLE001 — one bad section must not stop the page
            print(f"    extract failed for {section.title!r}: {exc}")
            continue

        for statement in extract.statements:
            # A statement with neither a polarity nor a number asserts nothing:
            # it would add a permanent subject no query can use, and one that
            # later terms could be merged into.
            if statement.polarity is None and statement.qualifier is None:
                print(
                    f"    dropping {statement.subject!r}: no polarity and no "
                    f"qualifier, so it states nothing"
                )
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
                cache=shared_cache,
                usage=usage,
                trace_sink=trace_sink,
            )
            if ref is None:
                continue
            # A term the resolver had to de-negate ("no dogs") overrides whatever
            # polarity the extractor paired with the negative phrasing.
            polarity = (
                ref.implied_polarity
                if ref.implied_polarity is not None
                else statement.polarity
            )
            resolved.append(
                ResolvedRule(
                    subject_id=ref.id,
                    polarity=polarity,
                    qualifier=statement.qualifier,
                    qualifier_unit=statement.qualifier_unit,
                    evidence_span=statement.evidence_span,
                    source_url=section.source_url,
                    confidence=statement.confidence,
                )
            )
    return resolved


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
    )
    if not rules:
        print("    no statements extracted")
        return 0

    with conn.cursor() as cur:
        written = upsert_campsite_rules(
            cur, campsite_id=campsite_id, rules=rules, table=rules_table
        )
        print(f"    {written} rule(s) upserted")
    return written


def run(config: dict, *, limit: int, site: int | None = None) -> int:
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
    usage = LlmUsage()
    total = 0

    print(f"Ingesting rules for {len(campsites)} campsite(s)")
    with psycopg.connect(database_url(config)) as conn:
        for site in campsites:
            print("=" * 60)
            print(f"{site['id']}. {site['name']}")
            print(f"   {site['url']}")
            try:
                html = fetch_page_html(site["url"])
            except httpx.HTTPError as exc:
                print(f"    HTTP error: {exc}")
                continue
            try:
                total += ingest_site(
                    conn,
                    site,
                    html,
                    extractor=extractor,
                    embedder=embedder,
                    adjudicator=adjudicator,
                    usage=usage,
                )
                conn.commit()
            except Exception as exc:  # noqa: BLE001 — keep going to the next site
                conn.rollback()
                print(f"    failed, rolled back: {exc}")
            if pause_s > 0:
                time.sleep(pause_s)

    print("-" * 60)
    print(f"Done. Upserted {total} rule(s).")
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
    run(config, limit=limit, site=args.site)


if __name__ == "__main__":
    main()
