"""
Ingest accommodation types from the info page's `אפשרויות לינה` panel.

The panel is the operator's own statement of what a campsite has. The booking
engine only says what is free on the nights being scanned, so a fully-booked or
seasonally-closed unit is absent there and no inventory counts exist at all.
This is what creates `accommodation_types`; `scrape-availability` matches
booking names onto them afterwards and never creates one.

Per unit, two passes over the same text, neither feeding the other:

  1  `ExtractorLLMClient`  -> the type's own columns: occupancy, beds, room_count
  2  `ingest_unit_rules`   -> amenities and rules in `campsite_rules`, each with
                              the Hebrew sentence it was read from

Nothing is subtracted between them: `unit_prompt` names the columns' facts as
not its own instead, because subtracting made 48% of evidence spans quote text
that was never on the page (experiments.md 2026-09-06 §2).

  uv run python -m source.scraper.rules_ingest.rooms --site 1
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import psycopg
from dotenv import load_dotenv

from source.scraper.amenity_enrichment.db import update_accommodation_type_details
from source.scraper.amenity_enrichment.llm import (
    EmbeddingLLMClient,
    ExtractorLLMClient,
    LlmUsage,
    make_nebius_openai_client,
    record_scrape_cost,
)
from source.scraper.cli import add_site_argument, site_ids
from source.scraper.info_site.db import get_or_create_info_website_name
from source.scraper.rules_ingest.db import upsert_campsite_rules
from source.scraper.rules_ingest.fetch import fetch_page_html
from source.scraper.rules_ingest.ingest import (
    DEFAULT_LIMIT,
    DEFAULT_PAUSE_SECONDS,
    SiteReport,
    database_url,
    fetch_campsites,
    load_config,
    rules_from_sections,
)
from source.scraper.rules_ingest.llm import RuleExtractorLLMClient
from source.scraper.rules_ingest.lodging import (
    PANEL_TITLE,
    LodgingSegmenterLLMClient,
    fetch_panel,
    fold,
    parse_lodging_blocks,
    segment_panel,
)
from source.scraper.rules_ingest.report import SiteRun, write_run_report
from source.scraper.rules_ingest.resolve_conflicts import drop_redundant_permissions
from source.scraper.rules_ingest.sections import Section
from source.scraper.rules_ingest.subcamps import load_subcamps
from source.scraper.rules_ingest.units import ingest_unit_rules, unit_extractor
from source.scraper.subjects.llm import SubjectAdjudicatorLLMClient
from source.scraper.subjects.resolve import DEFAULT_STORE, SubjectRef, SubjectStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

load_dotenv()

UPSERT_TYPE_SQL = """
INSERT INTO accommodation_types (hotel_id, name, unit_count, aliases)
VALUES (%(hotel_id)s, %(name)s, %(unit_count)s, ARRAY[%(name)s])
ON CONFLICT (hotel_id, name) DO UPDATE
SET unit_count = EXCLUDED.unit_count,
    updated_at = now()
RETURNING id, info_website_name_id
"""


def owner_for(site: dict, subcamps: list, scope: str | None) -> int:
    """Which `campsites` row a unit belongs to, from the panel's own `<h3>`.

    Akhziv's panel groups its units under `חניון צפוני` / `חניון דרומי`, which
    are exactly the headings in `config.json`. Matching on the operator's own
    words is what `docs/design.md` wanted instead of `unit_owner`'s substring
    match on booking unit names, which it flags as provisional and silently
    wrong on a rename.
    """
    if not scope or not subcamps:
        return site["id"]
    for sub in subcamps:
        if any(fold(n) == fold(scope) for n in (sub.heading, *sub.aliases) if n):
            return sub.campsite_id
    print(f"      scope {scope!r} matches no subcamp; keeping the parent")
    return site["id"]


def ingest_rooms(
    conn,
    site: dict,
    html: str,
    *,
    segmenter: LodgingSegmenterLLMClient,
    details: ExtractorLLMClient,
    extractor: RuleExtractorLLMClient,
    site_extractor: RuleExtractorLLMClient,
    embedder: EmbeddingLLMClient,
    adjudicator: SubjectAdjudicatorLLMClient,
    store: SubjectStore = DEFAULT_STORE,
    usage: LlmUsage | None = None,
    report: SiteReport | None = None,
) -> int:
    """One site's panel -> accommodation types and their rules. Rows upserted."""
    panel = fetch_panel(site["url"], html)
    if not panel or not panel.strip():
        print("    no lodging panel on this page")
        return 0
    blocks = parse_lodging_blocks(panel)
    seg = segment_panel(blocks, segmenter, usage=usage)
    print(
        f"    {len(blocks)} block(s) -> {len(seg.units)} unit(s), "
        f"{len(seg.rules)} rule paragraph(s)"
    )
    for note in seg.notes:
        print(f"    NOTE {note}")

    subcamps = load_subcamps(conn, site["id"])
    cache: dict[str, SubjectRef] = {}
    written = 0

    for unit in seg.units:
        owner = owner_for(site, subcamps, unit.scope)
        tail = f"  [{unit.scope} -> campsite {owner}]" if unit.scope else ""
        print(f"    -- {unit.name} ({unit.unit_count}){tail}")
        with conn.cursor() as cur:
            cur.execute(
                UPSERT_TYPE_SQL,
                {
                    "hotel_id": owner,
                    "name": unit.name,
                    "unit_count": unit.unit_count,
                },
            )
            type_id = cur.fetchone()[0]
            # The panel is the catalog, so this is where a lodging product
            # comes into existence. `scrape-prices` matches rate-card labels
            # against these rows and creates none of its own.
            listing_id = get_or_create_info_website_name(
                cur, site_id=site["id"], name=unit.name
            )
            cur.execute(
                "UPDATE accommodation_types SET info_website_name_id = %s "
                "WHERE id = %s",
                (listing_id, type_id),
            )
            columns = details.extract(unit.text or unit.name, type_name=unit.name)
            update_accommodation_type_details(
                cur,
                accommodation_type_id=type_id,
                description=unit.text or unit.name,
                details=columns,
            )
        print(
            f"       occ={columns.get('max_people')} "
            f"beds={columns.get('double_bed')}+{columns.get('single_bed')} "
            f"rooms={columns.get('room_count')}"
        )
        conn.commit()
        written += ingest_unit_rules(
            conn,
            campsite_id=owner,
            accommodation_type_id=type_id,
            type_name=unit.name,
            tooltip=unit.text,
            source_url=site["url"],
            embedder=embedder,
            adjudicator=adjudicator,
            store=store,
            cache=cache,
            usage=usage,
            report=report,
            extractor=extractor,
        )
        conn.commit()

    # Paragraphs that own no unit are about the lodging area as a whole; they
    # go in site-level, exactly as a section of the page does.
    if seg.rules:
        rules = rules_from_sections(
            conn,
            [Section(PANEL_TITLE, "\n".join(seg.rules), site["url"])],
            extractor=site_extractor,
            embedder=embedder,
            adjudicator=adjudicator,
            store=store,
            cache=cache,
            usage=usage,
            trace_sink=report.traces if report is not None else None,
            campsite_id=site["id"],
            report=report,
        )
        with conn.cursor() as cur:
            site_written = upsert_campsite_rules(
                cur,
                campsite_id=site["id"],
                rules=rules,
                dropped_sink=report.drops if report is not None else None,
            )
        conn.commit()
        drop_redundant_permissions(
            conn,
            campsite_id=site["id"],
            rules=rules,
            sink=report.redundant if report is not None else None,
            scope=PANEL_TITLE,
        )
        conn.commit()
        print(f"    {site_written} site-level rule(s) upserted")
        written += site_written
    return written


def _load_listings(conn, site_id: int) -> list[tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name FROM info_website_names WHERE site_id = %s ORDER BY id",
            (site_id,),
        )
        return [(int(r[0]), r[1]) for r in cur.fetchall()]


def run(
    config: dict,
    *,
    limit: int,
    sites: list[int] | None = None,
    usage: LlmUsage | None = None,
    runs: list[SiteRun] | None = None,
) -> int:
    campsites = fetch_campsites(config, limit=limit, sites=sites)
    if not campsites:
        print("No campsites found")
        return 0
    pause_s = float(
        config.get("info_site", {}).get("request_pause_seconds", DEFAULT_PAUSE_SECONDS)
    )
    # One transport, and one extractor for every unit. Building a client costs
    # seconds where the OS trust store is loaded per call (source/scraper/tls.py).
    shared = make_nebius_openai_client()
    clients = {
        "segmenter": LodgingSegmenterLLMClient(shared),
        "details": ExtractorLLMClient(shared),
        "extractor": unit_extractor(shared),
        "site_extractor": RuleExtractorLLMClient(shared),
        "embedder": EmbeddingLLMClient(),
        "adjudicator": SubjectAdjudicatorLLMClient(),
    }
    usage = usage if usage is not None else LlmUsage()
    total = 0

    print(f"Ingesting rooms for {len(campsites)} campsite(s)")
    with psycopg.connect(database_url(config)) as conn:
        for entry in campsites:
            print("=" * 60)
            print(f"{entry['id']}. {entry['name']}")
            started = time.monotonic()
            report = SiteReport()
            outcome = SiteRun(site=entry, report=report)
            if runs is not None:
                runs.append(outcome)
            try:
                html = fetch_page_html(entry["url"])
                outcome.written = ingest_rooms(
                    conn,
                    entry,
                    html,
                    segmenter=clients["segmenter"],
                    details=clients["details"],
                    extractor=clients["extractor"],
                    site_extractor=clients["site_extractor"],
                    embedder=clients["embedder"],
                    adjudicator=clients["adjudicator"],
                    usage=usage,
                    report=report,
                )
                total += outcome.written
                conn.commit()
            except Exception as exc:  # noqa: BLE001 -- one site must not stop the run
                conn.rollback()
                outcome.error = str(exc)
                print(f"    failed, rolled back: {exc}")
            outcome.seconds = time.monotonic() - started
            print(report.render())
            if pause_s > 0:
                time.sleep(pause_s)

    print("-" * 60)
    print(f"Done. Upserted {total} row(s) from the lodging panel.")
    if usage.chat_calls or usage.embed_calls:
        print(usage.summary(prefix="Rooms ingest total: "))
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lodging-panel room ingester")
    parser.add_argument("--limit", type=int, default=None)
    add_site_argument(parser)
    args = parser.parse_args(argv)
    config = load_config()
    limit = args.limit
    if limit is None:
        limit = int(config.get("info_site", {}).get("limit_campsites", DEFAULT_LIMIT))
    usage = LlmUsage()
    runs: list[SiteRun] = []
    started_at = datetime.now()
    started = time.monotonic()
    run(config, limit=limit, sites=site_ids(args.site), usage=usage, runs=runs)
    written = record_scrape_cost("scrape-rooms", usage)
    if written:
        print(f"cost report appended to {written}")
    if runs:
        path = write_run_report(
            runs,
            usage,
            started_at=started_at,
            seconds=time.monotonic() - started,
            title="scrape-rooms",
        )
        print(f"run report written to {path}")


if __name__ == "__main__":
    main()
