"""
JSON/JSONL data store for experiments and validations.

ExperimentStore provides a single facade for all data operations.
"""

from pathlib import Path
from typing import Any, Iterator

from . import experiments as _exp
from . import iterations as _iter
from . import validations as _val


class ExperimentStore:
    """
    Facade for reading and writing experiment/validation data.

    All data is stored as JSON (metadata) and JSONL (records) files
    under the data directory.
    """

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)

    # --- Experiments ---

    def create_experiment(self, meta: dict[str, Any]) -> None:
        """Create a new experiment with metadata."""
        _exp.create_experiment(self.data_dir, meta)

    def load_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Load experiment metadata. Raises FileNotFoundError if missing."""
        return _exp.load_experiment(self.data_dir, experiment_id)

    def update_experiment_summary(self, experiment_id: str, summary: dict[str, Any]) -> None:
        """Update experiment summary statistics."""
        _exp.update_experiment_summary(self.data_dir, experiment_id, summary)

    def list_experiments(
        self,
        pattern: str = "*",
        test_group: str | None = None,
    ) -> list[dict[str, Any]]:
        """List experiments matching pattern and optional test_group."""
        return _exp.list_experiments(self.data_dir, pattern=pattern, test_group=test_group)

    # --- Iterations ---

    def append_iteration(self, experiment_id: str, record: dict[str, Any]) -> None:
        """Append an iteration record (thread-safe JSONL append)."""
        _iter.append_iteration(self.data_dir, experiment_id, record)

    def read_iterations(self, experiment_id: str) -> Iterator[dict[str, Any]]:
        """Read all iterations for an experiment."""
        return _iter.read_iterations(self.data_dir, experiment_id)

    # --- Validations ---

    def create_validation_batch(self, batch: dict[str, Any]) -> None:
        """Create a new validation batch."""
        _val.create_validation_batch(self.data_dir, batch)

    def load_validation_batch(self, validation_id: str) -> dict[str, Any]:
        """Load a validation batch."""
        return _val.load_validation_batch(self.data_dir, validation_id)

    def update_validation_batch(self, validation_id: str, updates: dict[str, Any]) -> None:
        """Update a validation batch."""
        _val.update_validation_batch(self.data_dir, validation_id, updates)

    def append_validation_run(self, validation_id: str, run: dict[str, Any]) -> None:
        """Append a validation run record."""
        _val.append_validation_run(self.data_dir, validation_id, run)

    def read_validation_runs(self, validation_id: str) -> Iterator[dict[str, Any]]:
        """Read all validation runs for a batch."""
        return _val.read_validation_runs(self.data_dir, validation_id)

    def list_validations(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        """List validation batches."""
        return _val.list_validations(self.data_dir, experiment_id=experiment_id)


__all__ = ["ExperimentStore"]
