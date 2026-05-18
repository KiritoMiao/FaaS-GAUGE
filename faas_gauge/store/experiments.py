"""Experiment metadata CRUD (JSON files)."""

import fnmatch
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import layout


def create_experiment(data_dir: Path, meta: dict[str, Any]) -> None:
    """Create a new experiment directory with metadata."""
    experiment_id = meta["id"]
    exp_dir = layout.experiment_dir(data_dir, experiment_id)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamps if not present.
    now = datetime.now().isoformat()
    meta.setdefault("created_at", now)
    meta.setdefault("updated_at", now)
    meta.setdefault("summary", None)

    meta_path = layout.experiment_meta_path(data_dir, experiment_id)
    meta_path.write_text(
        json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )


def load_experiment(data_dir: Path, experiment_id: str) -> dict[str, Any]:
    """Load experiment metadata. Raises FileNotFoundError if missing."""
    meta_path = layout.experiment_meta_path(data_dir, experiment_id)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Experiment not found: {experiment_id}")
    data: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    return data


def update_experiment_summary(data_dir: Path, experiment_id: str, summary: dict[str, Any]) -> None:
    """Update the summary field of an existing experiment."""
    meta = load_experiment(data_dir, experiment_id)
    meta["summary"] = summary
    meta["updated_at"] = datetime.now().isoformat()
    meta_path = layout.experiment_meta_path(data_dir, experiment_id)
    meta_path.write_text(
        json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )


def list_experiments(
    data_dir: Path,
    pattern: str = "*",
    test_group: str | None = None,
) -> list[dict[str, Any]]:
    """List experiments matching pattern and optional test_group filter."""
    experiments_dir = data_dir / "experiments"
    if not experiments_dir.is_dir():
        return []

    results = []
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if not fnmatch.fnmatch(exp_dir.name, pattern):
            continue

        meta_file = exp_dir / "experiment.json"
        if not meta_file.is_file():
            continue

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if test_group is not None and meta.get("test_group") != test_group:
            continue

        results.append(meta)

    return results
