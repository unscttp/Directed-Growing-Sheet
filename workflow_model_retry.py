"""Automatic retries of model calls that failed for a reason that may pass.

A model adapter raises ``TransientModelError`` (or an exception with
``transient = True``, or a ``TimeoutError``/``ConnectionError``) when a request
timed out, the connection dropped, or the provider was overloaded or rate
limiting. With ``model_retry_attempts`` > 0 in the settings, the framework
then waits and calls the adapter again, up to that many times, with an
exponential delay (``model_retry_delay_seconds``, doubled each time, at most
``model_retry_max_delay_seconds``). Every retry is journaled as
``<kind>.retry``. Other errors, and the last failure, are raised unchanged.

``model_retry_attempts = 0`` (the default) switches retries off. Hosts can
change it per engine with ``engine.setting_overrides``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from workflow_logging import record_event
from workflow_types import JsonDict

MAX_ERROR_CHARS = 300
_sleep = time.sleep  # replaced in tests


class TransientModelError(RuntimeError):
    """A model call failed in a way a retry may fix (timeout, connection, overload)."""


def is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (TransientModelError, TimeoutError, ConnectionError)) or (
        getattr(exc, "transient", False) is True
    )


def call_model(
    call: Callable[[], Any],
    *,
    settings: JsonDict,
    task,
    execution=None,
    kind: str = "llm",
    before_retry: Callable[[], None] | None = None,
) -> Any:
    """Run ``call``, retrying transient failures as the settings allow."""
    retries = settings.get("model_retry_attempts", 0)
    for attempt in range(retries + 1):
        try:
            return call()
        except Exception as exc:
            if attempt >= retries or not is_transient(exc):
                raise
            delay = min(
                settings["model_retry_delay_seconds"] * 2**attempt,
                settings["model_retry_max_delay_seconds"],
            )
            record_event(
                task,
                f"{kind}.retry",
                {
                    "attempt": attempt + 1,
                    "retries": retries,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:MAX_ERROR_CHARS],
                    "delay_seconds": delay,
                },
                execution,
            )
            if before_retry is not None:
                before_retry()  # for example, stop if the node's time is up
            _sleep(delay)
    raise AssertionError("unreachable")
