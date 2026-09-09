"""run-eval refreshes experiments from public except occupancy."""

from source.eval.run import EVAL_COPY_SKIP


def test_eval_copy_skips_availability_only():
    assert EVAL_COPY_SKIP == ("availability",)
