"""Keep Nebius replicas warm so the first Streamlit turn is not a cold start."""

from __future__ import annotations

import contextvars
import logging
import os
import threading
import time
from collections.abc import Mapping, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from datetime import time as wall_time
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage

from source.agent.tracing import (
    bind_to_current_trace,
    emit_child_span,
    tracing_configured,
)

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SEC = 600.0
_KEEPALIVE_TZ = ZoneInfo("Asia/Jerusalem")
_KEEPALIVE_START = wall_time(7, 0)
_KEEPALIVE_END = wall_time(23, 0)
_PING = "hi"
_MAX_TOKENS = 5
_ROLES = ("recommender", "light", "extractor", "embed")

_lock = threading.Lock()
_started = False
_SESSION_PINGED_KEY = "model_keepalive_pinged"
_last_ok_lock = threading.Lock()
_last_ok: dict[str, float] = {}


class KeepaliveTarget(NamedTuple):
    role: str
    model: str
    chat: Any
    kind: str = "chat"


def keepalive_hours_open(moment: datetime | None = None) -> bool:
    """True from 07:00 until 23:00 in Asia/Jerusalem."""
    now = moment or datetime.now(_KEEPALIVE_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_KEEPALIVE_TZ)
    else:
        now = now.astimezone(_KEEPALIVE_TZ)
    current = now.time()
    return _KEEPALIVE_START <= current < _KEEPALIVE_END


def keepalive_interval_sec() -> float:
    raw = (os.environ.get("TRIPPY_KEEPALIVE_INTERVAL_SEC") or "").strip()
    if not raw:
        return DEFAULT_INTERVAL_SEC
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "invalid TRIPPY_KEEPALIVE_INTERVAL_SEC=%r; using %s",
            raw,
            DEFAULT_INTERVAL_SEC,
        )
        return DEFAULT_INTERVAL_SEC


def _model_of(client: Any, role: str) -> str:
    return str(
        getattr(client, "model_name", None)
        or getattr(client, "model", None)
        or getattr(client, "MODEL", None)
        or role
    )


def _unique_by_model(targets: list[KeepaliveTarget]) -> tuple[KeepaliveTarget, ...]:
    """One ping per Nebius endpoint. Light and extractor share the 235B."""
    seen: dict[str, KeepaliveTarget] = {}
    order: list[str] = []
    for target in targets:
        if target.model not in seen:
            seen[target.model] = target
            order.append(target.model)
    return tuple(seen[key] for key in order)


def _default_targets() -> tuple[KeepaliveTarget, ...]:
    from source.agent.graph import _extractor_chat, light_model
    from source.agent.recommender.models import _recommender_chat, recommender_model
    from source.agent.search.embed import _claims_embedder
    from source.scraper.amenity_enrichment.llm import instruct_chat_model

    instruct = instruct_chat_model()
    embed_model = _model_of(_claims_embedder, "embed")
    return _unique_by_model(
        [
            KeepaliveTarget("recommender", recommender_model(), _recommender_chat()),
            KeepaliveTarget("light", instruct, light_model),
            KeepaliveTarget("extractor", instruct, _extractor_chat()),
            KeepaliveTarget("embed", embed_model, _claims_embedder, "embed"),
        ]
    )


def _targets(chats: Mapping[str, Any] | None) -> tuple[KeepaliveTarget, ...]:
    if chats is None:
        return _default_targets()
    out: list[KeepaliveTarget] = []
    for role in _ROLES:
        chat = chats.get(role)
        if chat is None:
            continue
        kind = "embed" if role == "embed" else "chat"
        out.append(KeepaliveTarget(role, _model_of(chat, role), chat, kind))
    return _unique_by_model(out)


def note_model_used(model: str) -> None:
    """A real call (retrieve embed, chat) already warmed this endpoint."""
    name = (model or "").strip()
    if not name:
        return
    with _last_ok_lock:
        _last_ok[name] = time.monotonic()


def _warm_age_sec(model: str) -> float | None:
    with _last_ok_lock:
        last = _last_ok.get(model)
    if last is None:
        return None
    return time.monotonic() - last


def _still_warm(model: str, *, within: float) -> bool:
    age = _warm_age_sec(model)
    return age is not None and age < within


def _keepalive_loop_running() -> bool:
    return any(
        t.name == "model-keepalive" and t.is_alive() for t in threading.enumerate()
    )


def _clip_reply(reply: Any) -> str:
    text = getattr(reply, "content", reply)
    if not isinstance(text, str):
        text = str(text)
    text = " ".join(text.split())
    if len(text) > 160:
        text = text[:160] + "…"
    return text


def _invoke_ping(target: KeepaliveTarget) -> None:
    window = keepalive_interval_sec()
    if target.kind == "embed" and _still_warm(target.model, within=window):
        age = _warm_age_sec(target.model) or 0.0
        skipped = (
            f"keepalive skip role={target.role} model={target.model} "
            f"warm {age:.0f}s ago"
        )
        print(skipped, flush=True)
        logger.info(skipped)
        return
    started = time.perf_counter()
    try:
        if target.kind == "embed":
            vectors = target.chat.embed([_PING])
            reply = f"embed dim={len(vectors[0]) if vectors else 0}"
        else:
            client = target.chat
            if hasattr(client, "bind"):
                client = client.bind(max_tokens=_MAX_TOKENS)
            reply = client.invoke(
                [HumanMessage(content=_PING)],
                config={
                    "run_name": f"keepalive-{target.role}",
                    "tags": ["keepalive", target.role],
                    "metadata": {
                        "keepalive": True,
                        "role": target.role,
                        "model": target.model,
                    },
                },
            )
    except Exception:
        print(
            f"keepalive failed role={target.role} model={target.model}",
            flush=True,
        )
        logger.warning(
            "keepalive failed role=%s model=%s",
            target.role,
            target.model,
            exc_info=True,
        )
        return
    note_model_used(target.model)
    done = (
        f"keepalive reply={_clip_reply(reply)!r} "
        f"role={target.role} model={target.model} "
        f"in {time.perf_counter() - started:.1f}s"
    )
    print(done, flush=True)
    logger.info(done)


def _ping_round(targets: tuple[KeepaliveTarget, ...]) -> None:
    if not targets:
        return
    ping = bind_to_current_trace(_invoke_ping)
    workers = min(len(targets), 4) or 1
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="keepalive") as pool:
        list(pool.map(ping, targets))


def ping_models(
    *,
    chats: Mapping[str, Any] | None = None,
    reason: str = "interval",
) -> None:
    """One `hi` round per model endpoint. Traced as `model-keepalive`."""
    targets = _targets(chats)
    for target in targets:
        scheduled = (
            f"keepalive ping={_PING} role={target.role} "
            f"model={target.model} kind={target.kind} reason={reason}"
        )
        print(scheduled, flush=True)
        logger.info(scheduled)

    def _run() -> str | None:
        if reason == "session" and chats is None:
            from source.agent.claim_judge import judge_model, warmup_claim_judge

            scheduled = (
                f"keepalive ping=system role=claim_judge "
                f"model={judge_model()} kind=chat reason={reason}"
            )
            print(scheduled, flush=True)
            logger.info(scheduled)
            ping = bind_to_current_trace(warmup_claim_judge)
            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="keepalive"
            ) as pool:
                # Each worker needs its own copy. One Context cannot be
                # entered on two threads, and without a copy the model
                # calls land as root runs instead of children.
                chats_done = pool.submit(
                    contextvars.copy_context().run, _ping_round, targets
                )
                judge_done = pool.submit(contextvars.copy_context().run, ping)
            chats_done.result()
            reply = judge_done.result()
            return reply if isinstance(reply, str) else None
        _ping_round(targets)
        return None

    if tracing_configured():
        try:
            from langsmith import trace

            with trace(
                name=f"model-keepalive-{reason}",
                run_type="chain",
                inputs={
                    "models": [target.model for target in targets],
                    "roles": [target.role for target in targets],
                    "reason": reason,
                },
                tags=["keepalive", reason],
                metadata={"keepalive": True, "reason": reason},
            ):
                judge_reply = _run()
                if judge_reply is not None:
                    emit_child_span(
                        name="keepalive-claim_judge",
                        inputs={"role": "claim_judge"},
                        outputs={"reply": judge_reply},
                        tags=["keepalive", reason, "claim_judge"],
                    )
            return
        except Exception:
            logger.warning("keepalive langsmith parent failed", exc_info=True)
    _run()


def ping_new_session(
    session: MutableMapping[str, Any],
    *,
    chats: Mapping[str, Any] | None = None,
    blocking: bool = False,
) -> bool:
    """One `hi` round for a new Streamlit session. Returns True if a ping started."""
    if _SESSION_PINGED_KEY in session:
        return False
    session[_SESSION_PINGED_KEY] = True

    def _run() -> None:
        ping_models(chats=chats, reason="session")

    if blocking:
        _run()
        return True
    threading.Thread(
        target=_run, daemon=True, name="model-keepalive-session"
    ).start()
    return True


def start_model_keepalive(
    *,
    chats: Mapping[str, Any] | None = None,
    interval_sec: float | None = None,
    blocking: bool = False,
) -> None:
    """Start the interval loop once per process. First ping waits for the interval.

    Rounds outside 07:00–23:00 Asia/Jerusalem are skipped. The loop keeps
    sleeping so the next morning's window still fires.
    """
    global _started
    with _lock:
        if _started or _keepalive_loop_running():
            _started = True
            return
        _started = True

    interval = (
        keepalive_interval_sec() if interval_sec is None else max(0.0, interval_sec)
    )
    started = (
        f"model keepalive interval={interval:g}s "
        f"ping={_PING} max_tokens={_MAX_TOKENS}"
    )
    print(started, flush=True)
    logger.info(started)

    def _loop() -> None:
        while True:
            if interval > 0:
                time.sleep(interval)
            if not keepalive_hours_open():
                skipped = "keepalive skip outside 07:00-23:00 Asia/Jerusalem"
                print(skipped, flush=True)
                logger.info(skipped)
            else:
                try:
                    ping_models(chats=chats, reason="interval")
                except Exception:
                    print("keepalive round failed", flush=True)
                    logger.warning("keepalive round failed", exc_info=True)
            if interval <= 0:
                return

    if blocking:
        _loop()
        return
    threading.Thread(target=_loop, daemon=True, name="model-keepalive").start()
