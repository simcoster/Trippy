"""Optional wall-time buckets for extractor + planner stages.

A no-op unless `collect_stages()` is active (eval). Nested stages are
independent: retrieve SQL is timed after the embed call returns.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

STAGE_ORDER: tuple[str, ...] = (
    "extract",
    "sql",
    "embed",
    "retrieve",
    "rules",
    "judge",
)


@dataclass
class _Stat:
    seconds: float = 0.0
    calls: int = 0

    def add(self, elapsed: float, calls: int = 1) -> None:
        self.seconds += elapsed
        self.calls += calls


@dataclass
class StageClock:
    _stats: dict[str, _Stat] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, name: str, elapsed: float, *, calls: int = 1) -> None:
        with self._lock:
            self._stats.setdefault(name, _Stat()).add(elapsed, calls)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for name in STAGE_ORDER:
            stat = self._stats.get(name)
            out[name] = {
                "s": round(stat.seconds, 3) if stat else 0.0,
                "n": stat.calls if stat else 0,
            }
        return out


_clock: ContextVar[StageClock | None] = ContextVar("trippy_stage_clock", default=None)


@contextmanager
def collect_stages() -> Iterator[StageClock]:
    clock = StageClock()
    token: Token = _clock.set(clock)
    try:
        yield clock
    finally:
        _clock.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    clock = _clock.get()
    if clock is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        clock.add(name, time.perf_counter() - started)


def record_stage(name: str, elapsed: float, *, calls: int = 1) -> None:
    clock = _clock.get()
    if clock is None:
        return
    clock.add(name, elapsed, calls=calls)


def format_stages(snap: dict[str, dict[str, float | int]] | None) -> str:
    if not snap:
        return ""
    parts: list[str] = []
    for name in STAGE_ORDER:
        item = snap.get(name) or {}
        seconds = float(item.get("s") or 0)
        n = int(item.get("n") or 0)
        if n <= 0 and seconds <= 0:
            continue
        if n > 1:
            parts.append(f"{name}={seconds:.1f}s×{n}")
        else:
            parts.append(f"{name}={seconds:.1f}s")
    return " ".join(parts)


def merge_snapshots(
    snaps: list[dict[str, dict[str, float | int]]],
) -> dict[str, dict[str, float | int]]:
    merged = {name: {"s": 0.0, "n": 0} for name in STAGE_ORDER}
    for snap in snaps:
        for name in STAGE_ORDER:
            item = snap.get(name) or {}
            merged[name]["s"] += float(item.get("s") or 0)
            merged[name]["n"] += int(item.get("n") or 0)
    for name in STAGE_ORDER:
        merged[name]["s"] = round(float(merged[name]["s"]), 3)
    return merged
