"""Keep Nebius replicas warm so the first Streamlit turn is not a cold start."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

from langchain_core.messages import HumanMessage

from source.agent.tracing import bind_to_current_trace, tracing_configured

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SEC = 240.0
_PING = "hi"
_MAX_TOKENS = 5
_ROLES = ("recommender", "light", "extractor")

_lock = threading.Lock()
_started = False


class KeepaliveTarget(NamedTuple):
    role: str
    model: str
    chat: Any


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


def _default_targets() -> tuple[KeepaliveTarget, ...]:
    from source.agent.graph import _extractor_chat, light_model
    from source.agent.recommender import _recommender_chat, recommender_model
    from source.scraper.amenity_enrichment.llm import instruct_chat_model

    instruct = instruct_chat_model()
    return (
        KeepaliveTarget("recommender", recommender_model(), _recommender_chat()),
        KeepaliveTarget("light", instruct, light_model),
        KeepaliveTarget("extractor", instruct, _extractor_chat()),
    )


def _targets(chats: Mapping[str, Any] | None) -> tuple[KeepaliveTarget, ...]:
    if chats is None:
        return _default_targets()
    out: list[KeepaliveTarget] = []
    for role in _ROLES:
        chat = chats.get(role)
        if chat is None:
            continue
        model = (
            getattr(chat, "model_name", None)
            or getattr(chat, "model", None)
            or role
        )
        out.append(KeepaliveTarget(role, str(model), chat))
    return tuple(out)


def _clip_reply(reply: Any) -> str:
    text = getattr(reply, "content", reply)
    if not isinstance(text, str):
        text = str(text)
    text = " ".join(text.split())
    if len(text) > 160:
        text = text[:160] + "…"
    return text


def _invoke_ping(target: KeepaliveTarget) -> None:
    started = time.perf_counter()
    try:
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
        print(f"keepalive failed role={target.role}", flush=True)
        logger.warning("keepalive failed role=%s", target.role, exc_info=True)
        return
    done = (
        f"keepalive reply={_clip_reply(reply)!r} "
        f"role={target.role} in {time.perf_counter() - started:.1f}s"
    )
    print(done, flush=True)
    logger.info(done)


def _ping_round(targets: tuple[KeepaliveTarget, ...]) -> None:
    if not targets:
        return
    ping = bind_to_current_trace(_invoke_ping)
    workers = min(len(targets), len(_ROLES)) or 1
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="keepalive") as pool:
        list(pool.map(ping, targets))


def ping_models(*, chats: Mapping[str, Any] | None = None) -> None:
    """One `hi` round (max 5 tokens) to each chat client. Traced as `model-keepalive`."""
    targets = _targets(chats)
    for target in targets:
        scheduled = (
            f"keepalive ping={_PING} role={target.role} model={target.model}"
        )
        print(scheduled, flush=True)
        logger.info(scheduled)

    def _run() -> None:
        _ping_round(targets)

    if tracing_configured():
        try:
            from langsmith import trace

            with trace(
                name="model-keepalive",
                run_type="chain",
                inputs={"roles": [target.role for target in targets]},
                tags=["keepalive"],
                metadata={"keepalive": True},
            ):
                _run()
            return
        except Exception:
            logger.warning("keepalive langsmith parent failed", exc_info=True)
    _run()


def start_model_keepalive(
    *,
    chats: Mapping[str, Any] | None = None,
    interval_sec: float | None = None,
    blocking: bool = False,
) -> None:
    """Ping recommender / light / extractor now, then every interval, once per process."""
    global _started
    with _lock:
        if _started:
            return
        _started = True

    interval = (
        keepalive_interval_sec() if interval_sec is None else max(0.0, interval_sec)
    )
    started = (
        f"model keepalive interval={interval:g}s "
        f"roles={','.join(_ROLES)} ping={_PING} max_tokens={_MAX_TOKENS}"
    )
    print(started, flush=True)
    logger.info(started)

    def _loop() -> None:
        while True:
            try:
                ping_models(chats=chats)
            except Exception:
                print("keepalive round failed", flush=True)
                logger.warning("keepalive round failed", exc_info=True)
            if interval <= 0:
                return
            time.sleep(interval)

    if blocking:
        _loop()
        return
    threading.Thread(target=_loop, daemon=True, name="model-keepalive").start()
