"""`--site` names campsites, plural.

A pipeline step is rarely wanted for exactly one site, and `just scrape-info --
--site 5,14,17` has to mean the same three at every step of it. Both spellings
get typed in practice, so both work and they mix.
"""

from __future__ import annotations

import argparse

import pytest

from source.scraper.cli import add_site_argument, site_ids


def parse(argv: list[str]) -> list[int]:
    parser = argparse.ArgumentParser()
    add_site_argument(parser)
    return site_ids(parser.parse_args(argv).site)


def test_no_flag_means_every_site():
    """Empty is "no filter", which every caller reads as all of them."""
    assert parse([]) == []


def test_one_id():
    assert parse(["--site", "5"]) == [5]


def test_repeated_flags():
    assert parse(["--site", "5", "--site", "14"]) == [5, 14]


def test_comma_separated():
    assert parse(["--site", "5,14,17"]) == [5, 14, 17]


def test_the_two_forms_mix():
    assert parse(["--site", "5,14", "--site", "17"]) == [5, 14, 17]


def test_order_is_kept_and_repeats_collapse():
    """Order is the caller's; a site named twice is still scraped once."""
    assert parse(["--site", "17,5", "--site", "5"]) == [17, 5]


def test_spaces_and_empty_parts_are_tolerated():
    assert parse(["--site", " 5 , ,14 "]) == [5, 14]


def test_an_empty_value_is_not_silently_no_sites():
    """`--site ''` must not mean "scrape nothing" -- that would look like a
    successful run over zero campsites."""
    assert parse(["--site", ""]) == []


def test_a_non_number_stops_the_run():
    with pytest.raises(SystemExit) as excinfo:
        site_ids(["banana"])
    assert "banana" in str(excinfo.value)
