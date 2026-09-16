"""Each gold JSON file is a list of explicit prices the runner can load."""

import pytest

from source.price_sandbox.gold.runner import (
    cases_from_doc,
    load_site,
    site_paths,
)


@pytest.mark.parametrize("path", site_paths(), ids=lambda p: p.stem)
def test_gold_file_has_explicit_prices(path):
    doc = load_site(path)
    assert doc["match"]
    cases = cases_from_doc(doc)
    assert len(cases) >= 5, path.name
    for case in cases:
        assert case.expected_price >= 0, case.note
        assert case.explanation, case.note
        assert case.params.lodging
