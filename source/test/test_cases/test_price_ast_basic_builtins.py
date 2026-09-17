"""Safe sequence/conversion builtins are allowed in generated price functions."""

from source.price_sandbox.ast_check import compile_quote
from source.price_sandbox.execute import eval_quote_inprocess
from source.price_sandbox.params import QuoteParams

_BUILTINS = """
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
    ages = list(reversed(child_ages))
    first = next(iter(ages), 0)
    flags = dict(zip(("n",), (adults_num,)))
    uniq = set(ages)
    hour, _minute = divmod(130, 60)
    ok = any(ages) and all(age >= 0 for age in ages)
    if isinstance(adults_num, int) and ok:
        return float(adults_num + first + hour + len(uniq)) * 10.0, str(flags)
    return 0.0, "x"
"""


def test_compile_allows_basic_builtins():
    compile_quote(_BUILTINS)
    result = eval_quote_inprocess(
        _BUILTINS,
        QuoteParams(lodging="x", adults_num=2, child_ages=(7, 4)),
    )
    assert result.price == 100.0
