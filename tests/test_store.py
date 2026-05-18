"""Tests for faas_gauge.store module."""

import json
from pathlib import Path

import pytest

from faas_gauge.store import ExperimentStore


@pytest.fixture
def store(tmp_path):
    """Create an ExperimentStore with a temporary data directory."""
    data_dir = tmp_path / "data"
    for sub in [
        "config",
        "experiments",
        "validations",
        "questions",
        "test_data",
        "prompts",
    ]:
        (data_dir / sub).mkdir(parents=True)
    return ExperimentStore(data_dir=data_dir)


class TestExperimentCRUD:
    def test_create_experiment(self, store):
        meta = {
            "id": "20260314_1530+prime+openai+gpt-4o-mini",
            "question_name": "prime",
            "question_content": "Write primes",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "iterations": 10,
            "test_group": "week1",
        }
        store.create_experiment(meta)
        loaded = store.load_experiment(meta["id"])
        assert loaded["question_name"] == "prime"
        assert loaded["provider"] == "openai"

    def test_list_experiments(self, store):
        for i in range(3):
            store.create_experiment(
                {
                    "id": f"exp_{i}",
                    "question_name": "q",
                    "question_content": "c",
                    "provider": "openai",
                    "model": "m",
                    "iterations": 1,
                }
            )
        exps = store.list_experiments()
        assert len(exps) == 3

    def test_list_experiments_pattern(self, store):
        store.create_experiment(
            {
                "id": "a+openai+gpt",
                "question_name": "q",
                "question_content": "c",
                "provider": "openai",
                "model": "gpt",
                "iterations": 1,
            }
        )
        store.create_experiment(
            {
                "id": "b+deepseek+ds",
                "question_name": "q",
                "question_content": "c",
                "provider": "deepseek",
                "model": "ds",
                "iterations": 1,
            }
        )
        assert len(store.list_experiments(pattern="*openai*")) == 1

    def test_list_experiments_test_group(self, store):
        store.create_experiment(
            {
                "id": "e1",
                "question_name": "q",
                "question_content": "c",
                "provider": "p",
                "model": "m",
                "iterations": 1,
                "test_group": "week1",
            }
        )
        store.create_experiment(
            {
                "id": "e2",
                "question_name": "q",
                "question_content": "c",
                "provider": "p",
                "model": "m",
                "iterations": 1,
                "test_group": "week2",
            }
        )
        assert len(store.list_experiments(test_group="week1")) == 1

    def test_update_experiment_summary(self, store):
        store.create_experiment(
            {
                "id": "e1",
                "question_name": "q",
                "question_content": "c",
                "provider": "p",
                "model": "m",
                "iterations": 10,
            }
        )
        store.update_experiment_summary("e1", {"successful": 9, "failed": 1})
        loaded = store.load_experiment("e1")
        assert loaded["summary"]["successful"] == 9

    def test_load_nonexistent_experiment(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_experiment("nonexistent")


class TestIterationCRUD:
    def test_append_and_read_iterations(self, store):
        store.create_experiment(
            {
                "id": "e1",
                "question_name": "q",
                "question_content": "c",
                "provider": "p",
                "model": "m",
                "iterations": 3,
            }
        )
        store.append_iteration("e1", {"iteration": 1, "output_tokens": 100, "error": None})
        store.append_iteration("e1", {"iteration": 2, "output_tokens": 200, "error": None})
        store.append_iteration("e1", {"iteration": 3, "output_tokens": 0, "error": "timeout"})

        iters = list(store.read_iterations("e1"))
        assert len(iters) == 3
        assert iters[0]["iteration"] == 1
        assert iters[2]["error"] == "timeout"

    def test_read_iterations_empty(self, store):
        store.create_experiment(
            {
                "id": "e1",
                "question_name": "q",
                "question_content": "c",
                "provider": "p",
                "model": "m",
                "iterations": 1,
            }
        )
        iters = list(store.read_iterations("e1"))
        assert iters == []


class TestValidationCRUD:
    def test_create_and_read_validation_batch(self, store):
        batch = {"id": "val_001", "test_group": "week1", "runtime": "local"}
        store.create_validation_batch(batch)
        loaded = store.load_validation_batch("val_001")
        assert loaded["runtime"] == "local"

    def test_append_and_read_validation_runs(self, store):
        store.create_validation_batch({"id": "val_001", "test_group": "w1", "runtime": "local"})
        store.append_validation_run(
            "val_001",
            {"experiment_id": "e1", "iteration": 1, "function_status": "success"},
        )
        store.append_validation_run(
            "val_001",
            {"experiment_id": "e1", "iteration": 2, "function_status": "syntax-error"},
        )

        runs = list(store.read_validation_runs("val_001"))
        assert len(runs) == 2
        assert runs[0]["function_status"] == "success"
