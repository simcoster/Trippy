"""Run each quote() in a short-lived child, four at a time.

A child sets the memory cap, evaluates one quote, and exits. The next
queued quote takes its place. Spawn, not fork: the server is multi-threaded.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import Any, NamedTuple

from .ast_check import PriceFunctionError, compile_quote
from .params import QuoteParams, QuoteResult

DEFAULT_TIMEOUT_S = 0.5
DEFAULT_MEMORY_BYTES = 64 * 1024 * 1024
WORKER_COUNT = 4
_STOP_TIMEOUT_S = 1.0

_mp = multiprocessing.get_context("spawn")
_gate = threading.Lock()


class QuoteCall(NamedTuple):
    source: str
    params: QuoteParams


@dataclass(frozen=True)
class _ChildOutcome:
    ok: bool
    result: QuoteResult | None
    error: str


@dataclass
class _Running:
    index: int
    proc: multiprocessing.Process
    conn: Any
    deadline: float


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
        result = QuoteResult.from_raw(fn(**params.as_call_kwargs()))
        conn.send(_ChildOutcome(ok=True, result=result, error=""))
    except Exception as exc:
        conn.send(_ChildOutcome(ok=False, result=None, error=str(exc)))


def _start(
    index: int,
    call: QuoteCall,
    *,
    timeout_s: float,
    memory_bytes: int,
) -> _Running:
    parent, child = _mp.Pipe(duplex=False)
    proc = _mp.Process(
        target=_child_main,
        args=(call.source, call.params.to_json(), memory_bytes, child),
        daemon=True,
    )
    proc.start()
    child.close()
    return _Running(
        index=index,
        proc=proc,
        conn=parent,
        deadline=time.monotonic() + timeout_s,
    )


def _stop(running: _Running) -> None:
    if running.proc.is_alive():
        running.proc.terminate()
        running.proc.join(_STOP_TIMEOUT_S)
        if running.proc.is_alive():
            running.proc.kill()
            running.proc.join(_STOP_TIMEOUT_S)
    try:
        running.conn.close()
    except OSError:
        pass


def _result_of(outcome: Any) -> QuoteResult | BaseException:
    if not isinstance(outcome, _ChildOutcome):
        return PriceFunctionError("price function returned a corrupt result")
    if not outcome.ok or outcome.result is None:
        return PriceFunctionError(outcome.error or "price function failed")
    return outcome.result


def _collect(running: _Running) -> QuoteResult | BaseException:
    try:
        outcome = running.conn.recv()
    except (EOFError, OSError) as exc:
        _stop(running)
        return PriceFunctionError(str(exc))
    running.proc.join(_STOP_TIMEOUT_S)
    if running.proc.is_alive():
        _stop(running)
    else:
        try:
            running.conn.close()
        except OSError:
            pass
    return _result_of(outcome)


def _dispatch(
    calls: list[QuoteCall],
    *,
    timeout_s: float,
    memory_bytes: int,
) -> list[QuoteResult | BaseException]:
    outcomes: list[QuoteResult | BaseException | None] = [None] * len(calls)
    pending = list(enumerate(calls))
    inflight: list[_Running] = []
    while pending or inflight:
        while pending and len(inflight) < WORKER_COUNT:
            index, call = pending.pop(0)
            inflight.append(
                _start(
                    index,
                    call,
                    timeout_s=timeout_s,
                    memory_bytes=memory_bytes,
                )
            )
        now = time.monotonic()
        wait_s = max(0.0, min(item.deadline - now for item in inflight))
        ready = set(wait([item.conn for item in inflight], wait_s))
        now = time.monotonic()
        still: list[_Running] = []
        for running in inflight:
            if running.conn in ready:
                outcomes[running.index] = _collect(running)
            elif now >= running.deadline:
                _stop(running)
                outcomes[running.index] = TimeoutError(
                    "price function exceeded time limit"
                )
            else:
                still.append(running)
        inflight = still
    return [
        item if item is not None else PriceFunctionError("price function produced no result")
        for item in outcomes
    ]


def run_quotes(
    calls: list[QuoteCall],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_bytes: int = DEFAULT_MEMORY_BYTES,
) -> list[QuoteResult | BaseException]:
    """Run calls on up to WORKER_COUNT children. Order matches calls."""
    if not calls:
        return []
    with _gate:
        return _dispatch(calls, timeout_s=timeout_s, memory_bytes=memory_bytes)


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
    """Evaluate one quote in a child process; kill it when the timeout fires."""
    got = run_quotes(
        [QuoteCall(source=source, params=params)],
        timeout_s=timeout_s,
        memory_bytes=memory_bytes,
    )[0]
    if isinstance(got, QuoteResult):
        return got
    raise got
