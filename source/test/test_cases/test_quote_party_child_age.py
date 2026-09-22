"""Unstated children are quoted as age 10."""

from source.agent.constraints import DEFAULT_CHILD_AGE, quote_party


def test_unstated_children_are_age_10():
    party = quote_party(party_size=4, child_num=2)
    assert party.adults_num == 2
    assert party.child_num == 2
    assert party.child_ages == (DEFAULT_CHILD_AGE, DEFAULT_CHILD_AGE)


def test_a_stated_age_is_kept_and_the_rest_are_10():
    party = quote_party(party_size=4, child_num=2, child_ages=(5,))
    assert party.child_ages == (5, DEFAULT_CHILD_AGE)


def test_stated_ages_are_not_replaced():
    party = quote_party(party_size=4, child_num=2, child_ages=(5, 8))
    assert party.child_ages == (5, 8)
