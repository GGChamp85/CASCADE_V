"""
logging_setup.py — Structured logging for CASCADE-V.

Two handlers:
    1. Rich console handler (INFO+) for human-friendly output
    2. JSONL file handler (DEBUG+) at logs/cascade_v.jsonl for the dashboard

Use `event(logger, "stage1.done", **payload)` instead of `print()`. The
event name and payload are emitted as JSON; humans see a one-line summary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rich.logging import RichHandler


_configured = False


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "payload", {})
        out = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "module": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "payload": payload,
        }
        return json.dumps(out)


def configure_logging(json_path: Path | None = None, level: str = "INFO") -> None:
    """Idempotent: safe to call multiple times."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("cascade_v")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(json_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JsonLineFormatter())
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the cascade_v root."""
    if not name.startswith("cascade_v"):
        name = f"cascade_v.{name}"
    return logging.getLogger(name)


def event(logger: logging.Logger, event_name: str, level: int = logging.INFO, **payload: Any) -> None:
    """Emit a structured event. Console gets a one-liner, JSONL gets the payload."""
    parts = [event_name]
    for k, v in payload.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    logger.log(level, " ".join(parts), extra={"event": event_name, "payload": payload})
