"""Iteration results CRUD (JSONL files).

Concurrency: Uses a threading.Lock per file path to serialize appends within a process.
Each line is written with a single write() + flush() call.
"""

import json
import threading
from pathlib import Path
from typing import Any, Iterator

from . import layout

_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_lock(path: Path) -> threading.Lock:
    """Get or create a lock for a specific file path."""
    key = str(path)
    with _locks_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def append_iteration(data_dir: Path, experiment_id: str, record: dict[str, Any]) -> None:
    """Append one iteration record to the JSONL file (thread-safe)."""
    path = layout.iterations_path(data_dir, experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"

    lock = _get_lock(path)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def read_iterations(data_dir: Path, experiment_id: str) -> Iterator[dict[str, Any]]:
    """Read all iteration records from the JSONL file. Yields dicts."""
    path = layout.iterations_path(data_dir, experiment_id)
    if not path.is_file():
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
