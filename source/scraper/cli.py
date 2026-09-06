"""Shared command-line pieces for the scrapers.

`--site` names which campsites to work on. Every scraper takes it, and it takes
a LIST: a pipeline step is rarely wanted for exactly one site, and `just
scrape-info -- --site 5,14,17` has to mean the same three at every step of it.

Both forms work and can be mixed, because both get typed in practice:

    --site 5 --site 14        repeated
    --site 5,14,17            comma-separated
"""

from __future__ import annotations

import argparse


def add_site_argument(
    parser: argparse.ArgumentParser, *, help_text: str = ""
) -> None:
    """Add the repeatable, comma-accepting `--site`."""
    parser.add_argument(
        "--site",
        action="append",
        default=None,
        metavar="ID[,ID...]",
        help=help_text
        or "Campsite ids to scrape; repeat or comma-separate. Default: all",
    )


def site_ids(values: list[str] | None) -> list[int]:
    """The ids `--site` named, in the order given, without duplicates.

    An empty list means "no filter", which every caller reads as "all of them"
    -- so a missing flag and `--site ''` behave the same rather than one of
    them quietly scraping nothing.
    """
    if not values:
        return []
    out: list[int] = []
    for value in values:
        for part in str(value).split(","):
            text = part.strip()
            if not text:
                continue
            try:
                site = int(text)
            except ValueError:
                raise SystemExit(f"--site takes campsite ids, got {text!r}") from None
            if site not in out:
                out.append(site)
    return out
