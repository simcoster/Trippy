"""try/except is allowed in generated price functions."""

from source.price_sandbox.ast_check import compile_quote
from source.price_sandbox.execute import eval_quote_inprocess
from source.price_sandbox.params import QuoteParams

_TRY_EXCEPT = """
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
    hour = 0
    try:
        hour = int((planned_exit_time or "0:00").split(":")[0])
    except Exception:
        hour = 0
    return float(adults_num) * 10.0 + hour, "ok"
"""


def test_compile_allows_try_except():
    compile_quote(_TRY_EXCEPT)
    result = eval_quote_inprocess(
        _TRY_EXCEPT,
        QuoteParams(lodging="x", adults_num=2, planned_exit_time="13:00"),
    )
    assert result.price == 33.0
    bad = eval_quote_inprocess(
        _TRY_EXCEPT,
        QuoteParams(lodging="x", adults_num=2, planned_exit_time="late"),
    )
    assert bad.price == 20.0
