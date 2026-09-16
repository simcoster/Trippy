"""Price functions may use map() (late-exit clock parse)."""

from source.price_sandbox.ast_check import compile_quote
from source.price_sandbox.execute import eval_quote_inprocess
from source.price_sandbox.params import QuoteParams

MAP_QUOTE = """
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
    hour, minute = map(int, "13:00".split(":"))
    return float(hour + minute), "ok"
"""


def test_map_builtin_is_allowed():
    compile_quote(MAP_QUOTE)
    got = eval_quote_inprocess(
        MAP_QUOTE,
        QuoteParams(lodging="x", adults_num=1),
    )
    assert got.price == 13.0
