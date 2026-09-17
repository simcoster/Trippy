"""filter is allowed in generated price functions."""

from source.price_sandbox.ast_check import compile_quote
from source.price_sandbox.execute import eval_quote_inprocess
from source.price_sandbox.params import QuoteParams

_FILTER = """
def quote(
    lodging,
    adults_num,
    child_num=0,
    child_ages=(),
    guest_type="רגיל",
    is_weekend_or_holiday=False,
    planned_entry_time=None,
    planned_exit_time=None,
):
    paying = list(filter(None, child_ages))
    return float(adults_num + len(paying)) * 10.0, "ok"
"""


def test_compile_allows_filter():
    compile_quote(_FILTER)
    result = eval_quote_inprocess(
        _FILTER,
        QuoteParams(lodging="x", adults_num=2, child_ages=(7, 0, 4)),
    )
    assert result.price == 40.0
