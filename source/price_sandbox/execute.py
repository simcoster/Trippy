"""A pool of quote processes, each already waiting for one batch.

A worker runs that batch, sends the results, and exits. A replacement
starts as soon as it dies, so the next batch does not wait for spawn.

TODO: one short-lived child per quote again. The pool exists because
spawning a process per quote used the 0.5s budget on startup.
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
from dataclasses import dataclass
from typing import Any, NamedTuple

from .ast_check import PriceFunctionError, compile_quote
from .params import QuoteParams, QuoteResult

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MEMORY_BYTES = 64 * 1024 * 1024
WORKER_COUNT = 4
_STOP_TIMEOUT_S = 1.0

_mp = multiprocessing.get_context("spawn")
_pool: queue.Queue[_Worker] | None = None
_pool_lock = threading.Lock()


class QuoteCall(NamedTuple):
    source: str
    params: QuoteParams


class _Worker(NamedTuple):
    proc: multiprocessing.Process
    conn: Any


@dataclass(frozen=True)
class _ChildOutcome:
    ok: bool
    result: QuoteResult | None
    error: str


def _apply_rlimits(memory_bytes: int) -> None:
    """Cap address space for this one batch. The parent kills a slow batch."""
    try:
        import resource
    except ImportError:
        return
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        pass


def _eval_one(source: str, params_payload: dict[str, Any]) -> _ChildOutcome:
    try:
        fn = compile_quote(source)
        params = QuoteParams.from_mapping(params_payload)
        result = QuoteResult.from_raw(fn(**params.as_call_kwargs()))
    except Exception as exc:
        return _ChildOutcome(ok=False, result=None, error=str(exc))
    return _ChildOutcome(ok=True, result=result, error="")


def _worker_main(conn: Any, memory_bytes: int) -> None:
    """Wait for one batch, send its results, and exit."""
    _apply_rlimits(memory_bytes)
    batch = conn.recv()
    if batch is None:
        return
    conn.send([_eval_one(source, payload) for source, payload in batch])


def _start_worker(memory_bytes: int) -> _Worker:
    parent, child = _mp.Pipe(duplex=True)
    proc = _mp.Process(
        target=_worker_main,
        args=(child, memory_bytes),
        daemon=True,
    )
    proc.start()
    child.close()
    return _Worker(proc=proc, conn=parent)


def _workers(memory_bytes: int) -> queue.Queue[_Worker]:
    global _pool
    with _pool_lock:
        if _pool is None:
            started: queue.Queue[_Worker] = queue.Queue()
            for _ in range(WORKER_COUNT):
                started.put(_start_worker(memory_bytes))
            _pool = started
        return _pool


def _stop(worker: _Worker) -> None:
    if worker.proc.is_alive():
        worker.proc.terminate()
        worker.proc.join(_STOP_TIMEOUT_S)
        if worker.proc.is_alive():
            worker.proc.kill()
            worker.proc.join(_STOP_TIMEOUT_S)
    try:
        worker.conn.close()
    except OSError:
        pass


def _result_of(outcome: Any) -> QuoteResult | BaseException:
    if not isinstance(outcome, _ChildOutcome):
        return PriceFunctionError("price function returned a corrupt result")
    if not outcome.ok or outcome.result is None:
        return PriceFunctionError(outcome.error or "price function failed")
    return outcome.result


def _reap(worker: _Worker) -> None:
    """Let a finished worker exit. Kill it if it is still alive."""
    if worker.proc.is_alive():
        worker.proc.join(_STOP_TIMEOUT_S)
    if worker.proc.is_alive():
        _stop(worker)
        return
    try:
        worker.conn.close()
    except OSError:
        pass


def _fill(pool: queue.Queue[_Worker], worker: _Worker, memory_bytes: int) -> None:
    """After this worker dies, a new one is already waiting for work."""
    _reap(worker)
    pool.put(_start_worker(memory_bytes))


def _timeouts(count: int) -> list[TimeoutError]:
    return [TimeoutError("price function exceeded time limit") for _ in range(count)]


def _run_batch(
    calls: list[QuoteCall],
    *,
    timeout_s: float,
    memory_bytes: int,
) -> list[QuoteResult | BaseException]:
    """One idle worker runs every call. The clock is the whole batch."""
    pool = _workers(memory_bytes)
    worker = pool.get()
    payload = [(call.source, call.params.to_json()) for call in calls]
    try:
        worker.conn.send(payload)
        if not worker.conn.poll(timeout_s):
            _stop(worker)
            pool.put(_start_worker(memory_bytes))
            return _timeouts(len(calls))
        raw = worker.conn.recv()
    except (EOFError, OSError) as exc:
        _stop(worker)
        pool.put(_start_worker(memory_bytes))
        return [PriceFunctionError(str(exc)) for _ in calls]
    if not isinstance(raw, list) or len(raw) != len(calls):
        _stop(worker)
        pool.put(_start_worker(memory_bytes))
        return [
            PriceFunctionError("price function returned a corrupt result")
            for _ in calls
        ]
    _fill(pool, worker, memory_bytes)
    return [_result_of(item) for item in raw]


def run_quotes(
    calls: list[QuoteCall],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_bytes: int = DEFAULT_MEMORY_BYTES,
) -> list[QuoteResult | BaseException]:
    """Run one batch on a process that is already waiting. Order matches calls."""
    if not calls:
        return []
    return _run_batch(calls, timeout_s=timeout_s, memory_bytes=memory_bytes)


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
    """Evaluate one quote on a waiting process; kill it when the timeout fires."""
    got = run_quotes(
        [QuoteCall(source=source, params=params)],
        timeout_s=timeout_s,
        memory_bytes=memory_bytes,
    )[0]
    if isinstance(got, QuoteResult):
        return got
    raise got
