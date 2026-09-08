"""Rule search is own+parent, not sisters; judge nos do not veto a yes."""

from __future__ import annotations

import inspect

from source.agent.claim_judge import CLAIM_JUDGE_SYSTEM
from source.agent.search import (
    _OWN_OR_PARENT_RULES,
    search_campsite_rules,
    search_site_amenities,
)


def test_own_or_parent_sql_is_this_site_and_parent():
    assert _OWN_OR_PARENT_RULES.format(alias="site") == (
        "(cr.campsite_id = site.id OR cr.campsite_id = site.parent_id)"
    )


def test_rule_and_site_amenity_search_use_own_or_parent():
    rules_src = inspect.getsource(search_campsite_rules)
    amenity_src = inspect.getsource(search_site_amenities)
    assert "_OWN_OR_PARENT_RULES" in rules_src
    assert "_OWN_OR_PARENT_RULES" in amenity_src
    for src in (rules_src, amenity_src):
        assert "parent_id = site.parent_id" not in src
        assert "parent_id = c.parent_id" not in src
        assert "COALESCE(site.parent_id, site.id)" not in src


def test_judge_prompt_nos_do_not_veto_a_granting_yes():
    assert "do not veto" in CLAIM_JUDGE_SYSTEM
    assert "official listing provides fridges; complaint is a caveat" in (
        CLAIM_JUDGE_SYSTEM
    )
    assert "official hookup grants it; the complaint is a caveat" in CLAIM_JUDGE_SYSTEM
