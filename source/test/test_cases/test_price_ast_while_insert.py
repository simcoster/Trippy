"""while and list.insert are allowed in generated price functions."""

from source.price_sandbox.ast_check import compile_quote
from source.price_sandbox.execute import eval_quote_inprocess
from source.price_sandbox.params import QuoteParams

_WHILE_INSERT = """
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
    ages = list(child_ages)
    while len(ages) < child_num:
        ages.append(5)
    bits = []
    bits.insert(0, str(adults_num))
    return float(adults_num) * 10.0, " ".join(bits)
"""


def test_compile_allows_while_and_list_insert():
    compile_quote(_WHILE_INSERT)
    result = eval_quote_inprocess(
        _WHILE_INSERT,
        QuoteParams(lodging="x", adults_num=2, child_num=1),
    )
    assert result.price == 20.0
