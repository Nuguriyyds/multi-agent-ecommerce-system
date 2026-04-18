from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI

from app.v3.hooks import HookBus
from app.v3.models import HookEvent, HookPoint, HookResult


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
    turn_number: int | None = None,
    payload: dict[str, Any] | None = None,
    level: int = logging.INFO,
) -> None:
    logger.log(
        level,
        event,
        extra={
            "event": event,
            "trace_id": trace_id,
            "session_id": session_id,
            "turn_number": turn_number,
            "payload": payload or {},
        },
    )


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = _json_safe(getattr(record, "payload", {}))
        if not isinstance(payload, dict):
            payload = {"value": payload}
        if record.exc_info:
            payload = {
                **payload,
                "exception": self.formatException(record.exc_info),
            }

        body = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "trace_id": getattr(record, "trace_id", None),
            "session_id": getattr(record, "session_id", None),
            "turn_number": getattr(record, "turn_number", None),
            "event": getattr(record, "event", record.getMessage()),
            "payload": payload,
        }
        return json.dumps(body, ensure_ascii=False)


class InMemoryJSONHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.setFormatter(JSONFormatter())
        self._v3_capture_handler = True

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def install_observability(
    application: FastAPI,
    *,
    emit_to_stderr: bool = False,
) -> HookBus:
    logger = logging.getLogger("app.v3")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_v3_capture_handler", False):
            logger.removeHandler(handler)

    capture_handler = InMemoryJSONHandler()
    logger.addHandler(capture_handler)

    if emit_to_stderr and not any(getattr(handler, "_v3_stream_handler", False) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(JSONFormatter())
        stream_handler._v3_stream_handler = True  # type: ignore[attr-defined]
        logger.addHandler(stream_handler)

    hook_bus = HookBus()
    _register_hook_logging(hook_bus)

    application.state.v3_log_capture = capture_handler
    application.state.v3_hook_bus = hook_bus
    return hook_bus


def _register_hook_logging(hook_bus: HookBus) -> None:
    logger = logging.getLogger("app.v3.observability.hooks")

    for point in HookPoint:
        hook_bus.register(point, _build_hook_logger(point, logger))


def _build_hook_logger(
    point: HookPoint,
    logger: logging.Logger,
):
    async def hook_event_logger(event: HookEvent) -> HookResult:
        log_event(
            logger,
            "hook.emit",
            trace_id=event.trace_id,
            session_id=event.session_id,
            turn_number=event.turn_number,
            payload={
                "hook_point": point.value,
                "hook_payload": event.payload,
            },
        )
        return HookResult(
            handler_name=f"hook_event_logger_{point.value}",
            metadata={"hook_point": point.value},
        )

    hook_event_logger.__name__ = f"hook_event_logger_{point.value}"
    return hook_event_logger


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (datetime, Path, UUID)):
        return str(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump(mode="json"))
        except TypeError:
            return _json_safe(model_dump())

    return str(value)


__all__ = [
    "InMemoryJSONHandler",
    "JSONFormatter",
    "install_observability",
    "log_event",
]
