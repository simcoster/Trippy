"""Infix `_without_` is a topic; `cant_` / `cannot_` rewrite to `can_`."""

from source.scraper.subjects.llm import ADJUDICATE_SYSTEM_PROMPT
from source.scraper.subjects.naming import to_positive_subject


def test_entry_without_reservation_is_kept():
    """The visitor-info extractor named this; infix `_without_` used to
    discard the whole statement. Prefix `without_electricity` is still
    rewritten — that is a different shape."""
    assert to_positive_subject("entry_without_reservation_allowed") == (
        "entry_without_reservation_allowed",
        None,
    )


def test_prefix_without_is_still_rewritten():
    assert to_positive_subject("without_electricity") == (
        "electricity_allowed",
        False,
    )


def test_cant_be_without_muzzle_rewrites_to_can_false():
    assert to_positive_subject("cant_be_without_muzzle") == (
        "can_be_without_muzzle",
        False,
    )


def test_cannot_enter_rewrites_to_can_false():
    assert to_positive_subject("cannot_enter_the_pool") == (
        "can_enter_the_pool",
        False,
    )


def test_scant_is_not_a_cant_token():
    assert to_positive_subject("scant_shade") == ("scant_shade", None)


def test_judge_prompt_keeps_late_fee_cutoff_off_check_in():
    assert "late_entry_exit_end_time" in ADJUDICATE_SYSTEM_PROMPT
    assert "check_in_end_time" in ADJUDICATE_SYSTEM_PROMPT
    assert 'term "late_entry_exit_end_time"' in ADJUDICATE_SYSTEM_PROMPT
