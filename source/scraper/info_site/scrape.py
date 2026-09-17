"""
Scrape published rate cards from parks.org.il camping info pages.

Creates info_website_names from classified lodging rows and snapshots
list_prices. Does not create accommodation_types or scrape newsflashes.

  uv run python -m source.scraper.info_site.scrape --prices
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError

from db.connect import SCHEMA_ENV, connect, database_url
from source.scraper.amenity_enrichment.llm import LlmUsage, record_scrape_cost
from source.scraper.cli import add_site_argument, site_ids
from source.scraper.info_site.classify import RateCardClassifier, classify_rows
from source.scraper.info_site.compile_price import (
    FIX_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    assess_compiled_source,
    compile_quote_fix,
    compile_quote_source,
    digest_source,
    gather_visitor_info_text,
    gold_cases_for_site,
    match_compile_rows,
)
from source.scraper.info_site.db import (
    UNCERTAIN_BELOW,
    load_info_website_names,
    maybe_fill_booking_hotel_id,
    snapshot_list_prices,
    store_price_function,
)
from source.scraper.info_site.match_listing import InfoWebsiteNameMatcher, MatchCall
from source.scraper.info_site.parse import (
    parse_booking_hotel_id,
    parse_rate_table,
    parse_rate_tables,
    parse_wp_post_id,
)
from source.scraper.info_site.price_report import (
    PriceFunctionRun,
    run_folder,
    write_run_report,
)
from source.scraper.tls import ssl_context

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

_SCRAPER_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = _SCRAPER_DIR / "config.json"
LISTING_URL = (
    "https://www.parks.org.il/"
    "%D7%94%D7%96%D7%9E%D7%A0%D7%95%D7%AA-%D7%9C%D7%97%D7%A0%D7%99%D7%95%D7%A0%D7%99-%D7%9C%D7%99%D7%9C%D7%94/"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_campsites(config: dict, *, sites: list[int] | None = None) -> list[dict]:
    """The pages to scrape: the ones `sites` names, else the first `limit`.

    `--site` exists so `just scrape-info -- --site 5,14` means the same sites at
    every step of the pipeline; without it the rooms step would run on two and
    the prices step on twenty.
    """
    limit = int(config.get("info_site", {}).get("limit_campsites", 2))
    # Subcamps have no page of their own; their parent's rate card covers them,
    # and scraping a NULL url would fail.
    where = "WHERE url IS NOT NULL" + (" AND id = ANY(%(sites)s)" if sites else "")
    sql = f"""
        SELECT id, name, url, booking_hotel_id
        FROM campsites
        {where}
        ORDER BY id
        LIMIT %(limit)s
    """
    with connect(database_url(config)) as conn, conn.cursor() as cur:
        cur.execute(
            sql,
            {"sites": list(sites or ()), "limit": len(sites) if sites else limit},
        )
        rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "name": row[1],
            "url": row[2],
            "booking_hotel_id": row[3],
        }
        for row in rows
    ]


def fetch_page_html(url: str, *, referer: str = LISTING_URL) -> str:
    delays = (2.0, 8.0)
    last: Exception | None = None
    attempts = len(delays) + 1
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(
                timeout=45.0,
                verify=ssl_context(),
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT, "Referer": referer},
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.text
        except httpx.HTTPError as exc:
            last = exc
            if attempt == attempts:
                break
            wait = delays[attempt - 1]
            print(
                f"    page fetch error (attempt {attempt}/{attempts}): {exc}; "
                f"retry in {wait:.0f}s"
            )
            time.sleep(wait)
    assert last is not None
    raise last


def next_fail_stem(directory: Path, site_id: int) -> str:
    """Next unused `{id}_vN` stem (`13_v1`, then `13_v2`). `{id}.py` is latest."""
    n = 1
    while (directory / f"{site_id}_v{n}.py").exists():
        n += 1
    return f"{site_id}_v{n}"


def _write_quote_files(
    stem: str,
    source: str,
    *,
    directory: Path,
    user_prompt: str = "",
    system: str | None = None,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"{stem}.py"
    dest.write_text(source or "", encoding="utf-8")
    prompt_dest = directory / f"{stem}.prompt.txt"
    prompt_dest.write_text(
        "----- system -----\n"
        + (system if system is not None else SYSTEM_PROMPT)
        + "\n\n----- user -----\n"
        + (user_prompt or ""),
        encoding="utf-8",
    )
    return dest, prompt_dest


def _dump_quote(
    site: dict,
    source: str,
    *,
    directory: Path,
    user_prompt: str = "",
    system: str | None = None,
) -> None:
    dest, prompt_dest = _write_quote_files(
        str(site["id"]),
        source,
        directory=directory,
        user_prompt=user_prompt,
        system=system,
    )
    print(f"    wrote {dest}")
    if user_prompt:
        print(f"    wrote {prompt_dest}")


def _dump_failed_quote(
    site: dict,
    source: str,
    *,
    directory: Path,
    user_prompt: str = "",
    system: str | None = None,
) -> str:
    stem = next_fail_stem(directory, site["id"])
    dest, prompt_dest = _write_quote_files(
        stem, source, directory=directory, user_prompt=user_prompt, system=system
    )
    print(f"    wrote failed {dest}")
    if user_prompt:
        print(f"    wrote failed {prompt_dest}")
    return stem


def _print_ast_failure(kind: str, details: list[str]) -> None:
    print(flush=True)
    print("=" * 60, flush=True)
    print("!!! PRICE FUNCTION AST FAILED !!!", flush=True)
    print(f"!!! {kind}", flush=True)
    for line in details:
        print(f"!!!   {line}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


def _print_store_ok(
    status: str, *, n_gold: int, digest: str, gold_ok: bool = True
) -> None:
    if status == "inserted":
        headline = "PRICE FUNCTION ADDED TO DB"
    elif status == "updated":
        headline = "PRICE FUNCTION UPDATED IN DB"
    else:
        headline = "PRICE FUNCTION UNCHANGED IN DB (hash match)"
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"*** {headline} ***", flush=True)
    if not gold_ok:
        print("*** gold still failing; stored anyway ***", flush=True)
    print(f"*** {n_gold} gold tests  sha256={digest[:12]}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


def _print_gold_failure(details: list[str]) -> None:
    print(flush=True)
    print("=" * 60, flush=True)
    print("!!! PRICE FUNCTION GOLD FAILED !!!", flush=True)
    for line in details:
        print(f"!!!   {line}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


def _print_compile_verdict(verdict) -> None:
    if verdict.stage in {"allowlist", "static"}:
        kind = "allowlist" if verdict.stage == "allowlist" else "static checks"
        _print_ast_failure(kind, verdict.log_lines)
        return
    _print_gold_failure(verdict.log_lines)


def _store_compiled(conn, site: dict, source: str, cases, verdict, run: PriceFunctionRun) -> PriceFunctionRun:
    """AST-ok source goes in the DB even when gold still fails."""
    if verdict.stage in {"allowlist", "static"}:
        _print_compile_verdict(verdict)
        run.outcome = "ast_failed"
        if not run.failures:
            run.failures = list(verdict.log_lines)
        return run
    digest = digest_source(source)
    status = store_price_function(
        conn, site_id=site["id"], source=source, digest=digest
    )
    run.store_status = status
    run.digest = digest
    if verdict.ok:
        _print_store_ok(status, n_gold=len(cases), digest=digest)
        run.outcome = "stored"
        return run
    _print_compile_verdict(verdict)
    _print_store_ok(status, n_gold=len(cases), digest=digest, gold_ok=False)
    run.outcome = "gold_failed"
    if not run.failures:
        run.failures = list(verdict.log_lines)
    return run


def compile_price_function_for_site(
    conn,
    site: dict,
    html: str,
    *,
    quote_dir: Path,
    usage: LlmUsage | None = None,
    matcher: InfoWebsiteNameMatcher | None = None,
) -> PriceFunctionRun:
    run = PriceFunctionRun(
        site_id=site["id"], site_name=site["name"], url=site.get("url", "")
    )
    cases = gold_cases_for_site(url=site["url"])
    if not cases:
        print("    no gold tests for this site; skip price function")
        run.outcome = "skipped"
        run.skip_reason = "no gold tests"
        return run
    run.n_gold = len(cases)
    gathered = parse_rate_tables(html)
    if not gathered:
        print("    no rate-card rows; skip price function")
        run.outcome = "skipped"
        run.skip_reason = "no rate-card rows"
        return run
    names = load_info_website_names(conn, site_id=site["id"])
    if not names:
        print("    no info_website_names; skip price function")
        run.outcome = "skipped"
        run.skip_reason = "no info_website_names"
        return run
    compile_rows = match_compile_rows(
        gathered, names, matcher=matcher, usage=usage
    )
    if not compile_rows:
        print("    no matched rate rows; skip price function")
        run.outcome = "skipped"
        run.skip_reason = "no matched rate rows"
        return run
    lodgings = [name for _id, name in names]
    guest_types = list(dict.fromkeys(row.guest_type for row in compile_rows))
    visitor = gather_visitor_info_text(site["url"], html)
    compile_kwargs = dict(
        rows=compile_rows,
        lodgings=lodgings,
        guest_types=guest_types,
        visitor_info=visitor,
        site_name=site["name"],
        usage=usage,
    )
    try:
        draft = compile_quote_source(**compile_kwargs)
    except Exception as exc:
        print(f"    price function compile failed: {exc}")
        run.outcome = "compile_error"
        run.skip_reason = str(exc)
        return run
    current_system = SYSTEM_PROMPT
    _dump_quote(
        site,
        draft.source,
        directory=quote_dir,
        user_prompt=draft.user_prompt,
        system=current_system,
    )
    verdict = assess_compiled_source(draft.source, cases)
    if not verdict.ok:
        run.fail_stems.append(
            _dump_failed_quote(
                site,
                draft.source,
                directory=quote_dir,
                user_prompt=draft.user_prompt,
                system=current_system,
            )
        )
        run.failures = list(verdict.log_lines)
        if verdict.retry == "fix":
            print("    retry: fix the function")
            for line in verdict.log_lines:
                print(f"      {line}")
            run.retry = "fix"
            try:
                draft = compile_quote_fix(
                    draft.source, verdict.retry_errors, usage=usage
                )
            except Exception as exc:
                print(f"    price function compile failed: {exc}")
                run.outcome = "compile_error"
                run.skip_reason = str(exc)
                run.failures = list(verdict.log_lines)
                return run
            current_system = FIX_SYSTEM_PROMPT
        elif verdict.retry == "regen":
            print("    retry: regenerate occupancy from the rate card")
            for line in verdict.log_lines:
                print(f"      {line}")
            run.retry = "regen"
            try:
                draft = compile_quote_source(
                    **compile_kwargs,
                    role="price_function_compile_retry",
                    retry_suffix=verdict.retry_errors[0] if verdict.retry_errors else "",
                )
            except Exception as exc:
                print(f"    price function compile failed: {exc}")
                run.outcome = "compile_error"
                run.skip_reason = str(exc)
                run.failures = list(verdict.log_lines)
                return run
            current_system = SYSTEM_PROMPT
        else:
            return _store_compiled(conn, site, draft.source, cases, verdict, run)
        _dump_quote(
            site,
            draft.source,
            directory=quote_dir,
            user_prompt=draft.user_prompt,
            system=current_system,
        )
        verdict = assess_compiled_source(draft.source, cases)
        if not verdict.ok:
            retry_lines = list(verdict.log_lines)
            if run.failures and run.failures != retry_lines:
                run.failures = [
                    *(f"attempt 1: {line}" for line in run.failures),
                    *(f"retry: {line}" for line in retry_lines),
                ]
            else:
                run.failures = retry_lines
            run.fail_stems.append(
                _dump_failed_quote(
                    site,
                    draft.source,
                    directory=quote_dir,
                    user_prompt=draft.user_prompt,
                    system=current_system,
                )
            )
            return _store_compiled(conn, site, draft.source, cases, verdict, run)
    return _store_compiled(conn, site, draft.source, cases, verdict, run)


def scrape_prices_for_site(
    conn,
    site: dict,
    html: str,
    *,
    classifier: RateCardClassifier,
    quote_dir: Path,
    usage: LlmUsage | None = None,
    matcher: InfoWebsiteNameMatcher | None = None,
    unmatched_sink: list[str] | None = None,
    compile_runs: list[PriceFunctionRun] | None = None,
) -> int:
    raw_rows = parse_rate_table(html)
    classified = classify_rows(raw_rows, classifier=classifier, usage=usage)
    lodging = snapshot_list_prices(
        conn,
        site_id=site["id"],
        rows=classified,
        matcher=matcher,
        usage=usage,
        unmatched_sink=unmatched_sink,
    )
    hotel_id = parse_booking_hotel_id(html)
    with conn.cursor() as cur:
        filled = maybe_fill_booking_hotel_id(
            cur, site_id=site["id"], booking_hotel_id=hotel_id
        )
    if filled:
        print(f"    filled booking_hotel_id={filled}")
    post_id = parse_wp_post_id(html)
    if post_id:
        print(f"    wp post id={post_id}")
    fees = sum(1 for row in classified if row.kind == "fee")
    print(f"    {len(raw_rows)} table rows, {len(lodging)} lodging stored, {fees} fees skipped")
    run = compile_price_function_for_site(
        conn, site, html, quote_dir=quote_dir, usage=usage, matcher=matcher
    )
    if compile_runs is not None:
        compile_runs.append(run)
    status = run.store_status or run.outcome
    if run.outcome == "gold_failed" and run.store_status:
        status = f"gold_failed (stored {run.store_status})"
    print(f"    compile: {status}", flush=True)
    return len(lodging)


def match_verdict(call: MatchCall) -> str:
    """"forced", "collision", "split", "uncertain", or "" -- the same reading
    `snapshot_list_prices` makes of the answer, taken from the record rather
    than passed alongside it.

    A refusal is the model ignoring an instruction the prompt states plainly, so
    the price is forced onto a candidate; a low number is the model doing as it
    was told and saying the match is poor. A split is the rescue pass finding
    that one rate really does price several products -- confident or not, that
    is worth seeing, because it writes more rows than the rate card has lines.
    """
    if call.kind == "collision":
        # Always shown: the clash was established in code, so this is the one
        # answer with no confidence of its own to hide behind.
        return "collision" if call.picked is not None else "collision unresolved"
    if call.picked is None:
        return "forced"
    if len(call.picked_names) > 1:
        return "split"
    if call.confidence is not None and call.confidence < UNCERTAIN_BELOW:
        return "uncertain"
    return ""


def print_flagged_prompts(matcher: InfoWebsiteNameMatcher) -> None:
    """Every uncertain or forced match, with the exact prompt that produced it.

    A wrong pick is either the prompt's fault or the model's, and the summary
    lines above cannot tell you which: they show the answer, not the question.
    Only the user message is printed -- the system prompt is byte-identical on
    every call and lives in `match_listing.SYSTEM_PROMPT`.
    """
    flagged = [(call, match_verdict(call)) for call in matcher.calls]
    flagged = [(call, verdict) for call, verdict in flagged if verdict]
    if not flagged:
        return
    print()
    print("=" * 60)
    print(f"PROMPTS FOR {len(flagged)} FLAGGED MATCH(ES)")
    print("=" * 60)
    for i, (call, verdict) in enumerate(flagged, start=1):
        confidence = "none" if call.confidence is None else f"{call.confidence:.2f}"
        print("-" * 60)
        print(f"{i}. {verdict.upper()} (confidence {confidence}) -- {call.site}")
        if call.kind == "rescue" and len(call.picked_names) > 1:
            print(f"   priced against {len(call.picked_names)} products")
        print()
        print("[user]")
        print(call.user)
        print()
        print("[reply]")
        print(call.reply)
    print("-" * 60)


def run_prices(
    config: dict,
    *,
    usage: LlmUsage | None = None,
    sites: list[int] | None = None,
    compile_runs: list[PriceFunctionRun] | None = None,
    quote_dir: Path | None = None,
) -> tuple[int, list[PriceFunctionRun]]:
    """Scrape rate cards for the configured campsites. Returns rows stored.

    `usage` collects every LLM call so the caller can report the run's cost.
    `compile_runs` is filled in place so an interrupt can still write the
    Markdown report for sites that finished.
    `quote_dir` holds this run's `{id}.py` dumps; defaults to a new
    timestamped folder under `reports/scrape_prices/`.
    """
    if compile_runs is None:
        compile_runs = []
    if quote_dir is None:
        quote_dir = run_folder(datetime.now())
        quote_dir.mkdir(parents=True, exist_ok=True)
        print(f"run folder {quote_dir}")
    campsites = fetch_campsites(config, sites=sites)
    if not campsites:
        print("No campsites found")
        schema = (os.environ.get(SCHEMA_ENV) or "").strip() or "public"
        if schema == "experiments":
            print(
                "    experiments.campsites is empty. "
                "Re-copy with `just setup-experiments copy`."
            )
        return 0, compile_runs

    pause_s = float(config.get("info_site", {}).get("request_pause_seconds", 0.5))
    classifier = RateCardClassifier()
    usage = usage if usage is not None else LlmUsage()
    matcher = InfoWebsiteNameMatcher()
    unmatched: list[str] = []
    total = 0

    print(f"Scraping list prices for {len(campsites)} campsite(s)")
    with connect(database_url(config)) as conn:
        for site in campsites:
            print("=" * 60)
            print(f"{site['id']}. {site['name']}")
            print(f"   {site['url']}")
            try:
                html = fetch_page_html(site["url"])
            except httpx.HTTPError as exc:
                print(f"    HTTP error: {exc}")
                compile_runs.append(
                    PriceFunctionRun(
                        site_id=site["id"],
                        site_name=site["name"],
                        url=site.get("url", ""),
                        outcome="http_error",
                        skip_reason=str(exc),
                    )
                )
                continue
            calls_before = len(matcher.calls)
            try:
                saved = scrape_prices_for_site(
                    conn,
                    site,
                    html,
                    classifier=classifier,
                    quote_dir=quote_dir,
                    usage=usage,
                    matcher=matcher,
                    unmatched_sink=unmatched,
                    compile_runs=compile_runs,
                )
            except (APIConnectionError, APITimeoutError) as exc:
                print(f"    LLM connection error after retries: {exc}")
                compile_runs.append(
                    PriceFunctionRun(
                        site_id=site["id"],
                        site_name=site["name"],
                        url=site.get("url", ""),
                        outcome="compile_error",
                        skip_reason=str(exc),
                    )
                )
                for call in matcher.calls[calls_before:]:
                    call.site = site["name"]
                conn.commit()
                continue
            for call in matcher.calls[calls_before:]:
                call.site = site["name"]
            conn.commit()
            total += saved
            if pause_s > 0:
                time.sleep(pause_s)

    print("-" * 60)
    print(f"Done. Stored {total} lodging list-price row(s).")
    if unmatched:
        print(f"{len(unmatched)} rate-card label(s) matched no lodging product:")
        for name in sorted(set(unmatched)):
            print(f"  {name}")
    print_flagged_prompts(matcher)
    if usage.chat_calls:
        print(usage.summary(prefix="Classify total: "))
    return total, compile_runs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parks.org.il info-site scraper")
    add_site_argument(parser)
    parser.add_argument(
        "--prices",
        action="store_true",
        help="Scrape רגיל rate-card tables into list_prices",
    )
    args = parser.parse_args(argv)
    if not args.prices:
        parser.error("pass --prices (newsflashes are not wired yet)")
    usage = LlmUsage()
    started_at = datetime.now()
    t0 = time.perf_counter()
    compile_runs: list[PriceFunctionRun] = []
    quote_dir = run_folder(started_at)
    quote_dir.mkdir(parents=True, exist_ok=True)
    print(f"run folder {quote_dir}")
    try:
        run_prices(
            load_config(),
            usage=usage,
            sites=site_ids(args.site),
            compile_runs=compile_runs,
            quote_dir=quote_dir,
        )
    except KeyboardInterrupt:
        print()
        print("interrupted — writing report for sites finished so far")
        raise
    finally:
        seconds = time.perf_counter() - t0
        report = write_run_report(
            compile_runs,
            usage,
            started_at=started_at,
            seconds=seconds,
            directory=quote_dir,
            name="report.md",
        )
        failed = sum(
            1
            for run in compile_runs
            if run.outcome in {"gold_failed", "ast_failed", "compile_error"}
        )
        print(f"run report {report}  ({len(compile_runs)} sites, {failed} failed)")
        written = record_scrape_cost("scrape-prices", usage)
        if written:
            print(f"cost report appended to {written}")


if __name__ == "__main__":
    main()
