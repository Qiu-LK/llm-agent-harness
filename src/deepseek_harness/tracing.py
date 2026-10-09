from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class JsonlTracer:
    def __init__(self, path: Path, *, run_id: str | None = None) -> None:
        self.path = path
        self.run_id = run_id or path.stem
        self._started = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **data: Any) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "elapsed_ms": round((time.monotonic() - self._started) * 1000, 3),
            "event": event,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


class NullTracer:
    def record(self, event: str, **data: Any) -> None:
        del event, data


class CallbackTracer:
    def __init__(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.callback = callback
        self._started = time.monotonic()

    def record(self, event: str, **data: Any) -> None:
        self.callback({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.monotonic() - self._started) * 1000, 3),
            "event": event,
            **data,
        })


class MultiTracer:
    def __init__(self, *tracers: Any) -> None:
        self.tracers = tracers

    def record(self, event: str, **data: Any) -> None:
        for tracer in self.tracers:
            tracer.record(event, **data)
