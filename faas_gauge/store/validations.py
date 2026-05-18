"""Validation batch and run CRUD (JSON + JSONL files)."""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from . import layout

_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _locks_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def create_validation_batch(data_dir: Path, batch: dict[str, Any]) -> None:
    """Create a validation batch directory with metadata."""
    validation_id = batch["id"]
    val_dir = layout.validation_dir(data_dir, validation_id)
    val_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().isoformat()
    batch.setdefault("started_at", now)
    batch.setdefault("summary", None)

    batch_path = layout.validation_batch_path(data_dir, validation_id)
    batch_path.write_text(
        json.dumps(batch, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )


def load_validation_batch(data_dir: Path, validation_id: str) -> dict[str, Any]:
    """Load validation batch metadata."""
    batch_path = layout.validation_batch_path(data_dir, validation_id)
    if not batch_path.is_file():
        raise FileNotFoundError(f"Validation batch not found: {validation_id}")
    data: dict[str, Any] = json.loads(batch_path.read_text(encoding="utf-8"))
    return data


def update_validation_batch(data_dir: Path, validation_id: str, updates: dict[str, Any]) -> None:
    """Update fields in a validation batch."""
    batch = load_validation_batch(data_dir, validation_id)
    batch.update(updates)
    batch_path = layout.validation_batch_path(data_dir, validation_id)
    batch_path.write_text(
        json.dumps(batch, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )


def append_validation_run(data_dir: Path, validation_id: str, run: dict[str, Any]) -> None:
    """Append one validation run to the JSONL file (thread-safe)."""
    path = layout.validation_runs_path(data_dir, validation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(run, ensure_ascii=False, default=str) + "\n"

    lock = _get_lock(path)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def read_validation_runs(data_dir: Path, validation_id: str) -> Iterator[dict[str, Any]]:
    """Read all validation runs from the JSONL file."""
    path = layout.validation_runs_path(data_dir, validation_id)
    if not path.is_file():
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def list_validations(data_dir: Path, experiment_id: str | None = None) -> list[dict[str, Any]]:
    """List validation batches, optionally filtered by experiment_id."""
    validations_dir = data_dir / "validations"
    if not validations_dir.is_dir():
        return []

    results = []
    for val_dir in sorted(validations_dir.iterdir()):
        if not val_dir.is_dir():
            continue
        batch_file = val_dir / "batch.json"
        if not batch_file.is_file():
            continue

        batch = json.loads(batch_file.read_text(encoding="utf-8"))

        if experiment_id is not None:
            # Check if any run in this batch references the experiment.
            runs_file = val_dir / "runs.jsonl"
            if runs_file.is_file():
                found = False
                for line in runs_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        run = json.loads(line)
                        if run.get("experiment_id") == experiment_id:
                            found = True
                            break
                if not found:
                    continue
            else:
                continue

        results.append(batch)

    return results
