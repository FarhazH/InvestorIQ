"""
In-memory job store — tracks pipeline runs by run_id.
For production, replace with Cloudant/Redis-backed store.
"""
from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_jobs: dict[str, Any] = {}  # run_id -> PipelineState


def store_job(run_id: str, state: Any) -> None:
    with _lock:
        _jobs[run_id] = state


def get_job(run_id: str) -> Any | None:
    with _lock:
        return _jobs.get(run_id)


def list_jobs() -> list[str]:
    with _lock:
        return list(_jobs.keys())


def delete_job(run_id: str) -> bool:
    with _lock:
        if run_id in _jobs:
            del _jobs[run_id]
            return True
        return False
