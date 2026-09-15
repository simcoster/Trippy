"""Run an approved quote() in a short-lived child process."""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from typing import Any

from .ast_check import PriceFunctionError, compile_quote
from .params import QuoteParams, QuoteResult

DEFAULT_TIMEOUT_S = 0.5
DEFAULT_MEMORY_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class _ChildOutcome:
    ok: bool
    result: QuoteResult | None
    error: str


def _apply_rlimits(memory_bytes: int) -> None:
    try:
        import resource
    except ImportError:
        return
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    except (ValueError, OSError):
        pass


def _child_main(
    source: str,
    params_payload: dict[str, Any],
    memory_bytes: int,
    conn: Any,
) -> None:
    _apply_rlimits(memory_bytes)
    try:
        fn = compile_quote(source)
        params = QuoteParams.from_mapping(params_payload)
        raw = fn(**params.as_call_kwargs())
        result = QuoteResult.from_raw(raw)
        conn.send(_ChildOutcome(ok=True, result=result, error=""))
    except Exception as exc:
        conn.send(_ChildOutcome(ok=False, result=None, error=str(exc)))


def eval_quote_inprocess(source: str, params: QuoteParams) -> QuoteResult:
    """Compile and call quote() in this process (used by scrape gold tests)."""
    fn = compile_quote(source)
    raw = fn(**params.as_call_kwargs())
    return QuoteResult.from_raw(raw)


def run_quote(
    source: str,
    params: QuoteParams,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_bytes: int = DEFAULT_MEMORY_BYTES,
) -> QuoteResult:
    """Evaluate *source* in a child process; kill it when the timeout fires."""
    compile_quote(source)
    parent, child = multiprocessing.Pipe(duplex=False)
    proc = multiprocessing.Process(
        target=_child_main,
        args=(source, params.to_json(), memory_bytes, child),
        daemon=True,
    )
    proc.start()
    child.close()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
        if proc.is_alive():
            proc.kill()
            proc.join(1.0)
        raise TimeoutError("price function exceeded time limit")
    if not parent.poll():
        raise PriceFunctionError("price function produced no result")
    outcome = parent.recv()
    if not isinstance(outcome, _ChildOutcome):
        raise PriceFunctionError("price function returned a corrupt result")
    if not outcome.ok or outcome.result is None:
        raise PriceFunctionError(outcome.error or "price function failed")
    return outcome.result
