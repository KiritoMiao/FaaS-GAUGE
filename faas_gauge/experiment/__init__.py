"""Experiment orchestration: generate, validate, retry."""

from typing import Any, Iterator

from faas_gauge.store import ExperimentStore

from .generate import generate
from .retry import retry_failed
from .validate import validate_batch


class Experiment:
    """Convenience wrapper around experiment metadata with iteration access."""

    def __init__(self, meta: dict[str, Any], store: ExperimentStore):
        self._meta = meta
        self._store = store
        self.id = meta["id"]
        self.summary = meta.get("summary")

    def iterations(self) -> Iterator[dict[str, Any]]:
        """Lazily read all iterations from JSONL."""
        return self._store.read_iterations(self.id)

    def __getitem__(self, key: str) -> Any:
        return self._meta[key]

    def __repr__(self) -> str:
        return f"Experiment({self.id})"


def load_experiment(experiment_id: str, data_dir: str = "data") -> Experiment:
    """Load an existing experiment by ID."""
    store = ExperimentStore(data_dir=data_dir)
    meta = store.load_experiment(experiment_id)
    return Experiment(meta, store)


__all__ = [
    "generate",
    "validate_batch",
    "retry_failed",
    "load_experiment",
    "Experiment",
]
