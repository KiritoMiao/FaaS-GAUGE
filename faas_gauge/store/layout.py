"""
Directory layout conventions and path helpers for the data store.

Layout:
    data/
    ├── experiments/{experiment_id}/
    │   ├── experiment.json
    │   └── iterations.jsonl
    └── validations/{validation_id}/
        ├── batch.json
        └── runs.jsonl
"""

from pathlib import Path


def experiment_dir(data_dir: Path, experiment_id: str) -> Path:
    """Return path to an experiment's directory."""
    return data_dir / "experiments" / experiment_id


def experiment_meta_path(data_dir: Path, experiment_id: str) -> Path:
    """Return path to experiment.json."""
    return experiment_dir(data_dir, experiment_id) / "experiment.json"


def iterations_path(data_dir: Path, experiment_id: str) -> Path:
    """Return path to iterations.jsonl."""
    return experiment_dir(data_dir, experiment_id) / "iterations.jsonl"


def validation_dir(data_dir: Path, validation_id: str) -> Path:
    """Return path to a validation batch's directory."""
    return data_dir / "validations" / validation_id


def validation_batch_path(data_dir: Path, validation_id: str) -> Path:
    """Return path to batch.json."""
    return validation_dir(data_dir, validation_id) / "batch.json"


def validation_runs_path(data_dir: Path, validation_id: str) -> Path:
    """Return path to runs.jsonl."""
    return validation_dir(data_dir, validation_id) / "runs.jsonl"
