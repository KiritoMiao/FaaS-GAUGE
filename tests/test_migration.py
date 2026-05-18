"""Tests for SQLite to JSON/JSONL migration helpers."""

import json
import sqlite3
from pathlib import Path

import pytest

from faas_gauge.store.migration import (
    migrate_all,
    migrate_credentials,
    migrate_from_sqlite,
    migrate_questions,
    migrate_test_data,
    migrate_validations,
)


def _setup_legacy_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL UNIQUE,
            question_name TEXT NOT NULL,
            question_content TEXT NOT NULL,
            provider TEXT NOT NULL,
            model_name TEXT NOT NULL,
            iterations INTEGER NOT NULL,
            test_group TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            successful_iterations INTEGER NOT NULL DEFAULT 0,
            failed_iterations INTEGER NOT NULL DEFAULT 0,
            total_input_tokens INTEGER NOT NULL DEFAULT 0,
            total_output_tokens INTEGER NOT NULL DEFAULT 0,
            total_time REAL NOT NULL DEFAULT 0,
            total_tokens_per_second REAL NOT NULL DEFAULT 0,
            average_time REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE iteration_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            iteration INTEGER NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            rtt_time REAL NOT NULL DEFAULT 0,
            tokens_per_second REAL,
            error TEXT,
            raw_response TEXT,
            python_code TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE validation_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_group TEXT,
            runtime TEXT,
            iterations_per_test INTEGER,
            max_runtime INTEGER,
            question_filter TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            total_files INTEGER NOT NULL DEFAULT 0,
            completed_files INTEGER NOT NULL DEFAULT 0,
            total_tests INTEGER NOT NULL DEFAULT 0,
            passed INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0,
            errors INTEGER NOT NULL DEFAULT 0,
            notes TEXT
        );

        CREATE TABLE validation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            experiment_id INTEGER NOT NULL,
            experiment_iteration INTEGER NOT NULL,
            code_path TEXT NOT NULL,
            question TEXT,
            runtime TEXT NOT NULL,
            iterations_per_test INTEGER NOT NULL,
            max_runtime INTEGER NOT NULL,
            validation_timestamp TEXT NOT NULL,
            created_at TEXT NOT NULL,
            total_tests INTEGER NOT NULL DEFAULT 0,
            passed INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0,
            errors INTEGER NOT NULL DEFAULT 0,
            total_executions INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL
        );

        CREATE TABLE validation_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            validation_run_id INTEGER NOT NULL,
            test_name TEXT NOT NULL,
            status TEXT,
            message TEXT,
            input_json TEXT,
            actual_result_json TEXT,
            iterations INTEGER,
            successful_runs INTEGER,
            failed_runs INTEGER
        );

        CREATE TABLE validation_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            validation_test_id INTEGER NOT NULL,
            exec_iteration INTEGER,
            success INTEGER NOT NULL,
            execution_time_ms REAL,
            exec_timestamp TEXT,
            error TEXT,
            result_json TEXT,
            output_json TEXT,
            saaf_runtime REAL,
            cpu_user_delta REAL,
            cpu_kernel_delta REAL,
            cpu_idle_delta REAL,
            max_memory_used_mb REAL
        );
        """)

    cur.execute(
        """
        INSERT INTO experiments (
            experiment_name, question_name, question_content, provider, model_name,
            iterations, test_group, created_at, updated_at,
            successful_iterations, failed_iterations, total_input_tokens,
            total_output_tokens, total_time, total_tokens_per_second, average_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "exp_alpha",
            "prime",
            "Return primes",
            "openai",
            "gpt-5-mini",
            2,
            "week1",
            "2026-01-01T00:00:00",
            "2026-01-01T01:00:00",
            1,
            1,
            10,
            20,
            3.5,
            8.2,
            1.75,
        ),
    )

    cur.execute(
        """
        INSERT INTO iteration_results (
            experiment_id, iteration, input_tokens, output_tokens, rtt_time,
            tokens_per_second, error, raw_response, python_code, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            1,
            5,
            12,
            1.0,
            12.0,
            None,
            "raw",
            "print('ok')",
            "2026-01-01T00:10:00",
            "2026-01-01T00:10:01",
        ),
    )

    cur.execute(
        """
        INSERT INTO validation_batches (
            test_group, runtime, iterations_per_test, max_runtime, question_filter,
            started_at, completed_at, total_files, completed_files,
            total_tests, passed, failed, errors, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "week1",
            "local",
            2,
            30,
            "prime",
            "2026-01-01T00:00:00",
            "2026-01-01T00:20:00",
            1,
            1,
            1,
            1,
            0,
            0,
            "ok",
        ),
    )
    cur.execute(
        """
        INSERT INTO validation_runs (
            batch_id, experiment_id, experiment_iteration, code_path, question,
            runtime, iterations_per_test, max_runtime, validation_timestamp,
            created_at, total_tests, passed, failed, errors, total_executions, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            1,
            1,
            "generated.py",
            "prime",
            "local",
            2,
            30,
            "2026-01-01T00:11:00",
            "2026-01-01T00:11:00",
            1,
            1,
            0,
            0,
            1,
            "passed",
        ),
    )
    cur.execute(
        """
        INSERT INTO validation_tests (
            validation_run_id, test_name, status, message, input_json,
            actual_result_json, iterations, successful_runs, failed_runs
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            "test_prime",
            "passed",
            "ok",
            '{"n": 10}',
            '{"result": [2,3,5,7]}',
            1,
            1,
            0,
        ),
    )
    cur.execute(
        """
        INSERT INTO validation_executions (
            validation_test_id, exec_iteration, success, execution_time_ms,
            exec_timestamp, error, result_json, output_json,
            saaf_runtime, cpu_user_delta, cpu_kernel_delta, cpu_idle_delta, max_memory_used_mb
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            1,
            1,
            10.5,
            "2026-01-01T00:11:01",
            None,
            '{"result": [2,3,5,7]}',
            '{"stdout": "ok"}',
            9.0,
            0.2,
            0.1,
            0.7,
            25.0,
        ),
    )

    conn.commit()
    conn.close()


@pytest.fixture
def legacy_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "legacy.db"
    _setup_legacy_db(db_path)
    return db_path


def test_migrate_from_sqlite_writes_experiment_and_iterations(
    legacy_db: Path, tmp_path: Path
) -> None:
    target_data_dir = tmp_path / "data"

    migrate_from_sqlite(legacy_db, target_data_dir)

    experiment_path = target_data_dir / "experiments" / "exp_alpha" / "experiment.json"
    iterations_path = target_data_dir / "experiments" / "exp_alpha" / "iterations.jsonl"

    assert experiment_path.is_file()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    assert experiment["id"] == "exp_alpha"
    assert experiment["summary"]["successful_iterations"] == 1

    lines = iterations_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["iteration"] == 1
    assert record["python_code"] == "print('ok')"


def test_migrate_validations_embeds_tests_and_executions(legacy_db: Path, tmp_path: Path) -> None:
    target_data_dir = tmp_path / "data"

    migrate_validations(legacy_db, target_data_dir)

    batch_path = target_data_dir / "validations" / "batch_1" / "batch.json"
    runs_path = target_data_dir / "validations" / "batch_1" / "runs.jsonl"

    assert batch_path.is_file()
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    assert batch["id"] == "batch_1"
    assert batch["runtime"] == "local"

    run = json.loads(runs_path.read_text(encoding="utf-8").strip())
    assert run["id"] == 1
    assert len(run["tests"]) == 1
    assert run["tests"][0]["test_name"] == "test_prime"
    assert run["tests"][0]["executions"][0]["execution_time_ms"] == 10.5


def test_migrate_credentials_converts_yaml_to_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret_yaml = tmp_path / ".secret.yaml"
    target_data_dir = tmp_path / "data"
    secret_yaml.write_text(
        "openai:\n  api_key: sk-test\n  endpoint: https://api.openai.com/v1\n",
        encoding="utf-8",
    )

    class _YamlStub:
        @staticmethod
        def safe_load(text: str) -> dict[str, dict[str, str]]:
            return {
                "openai": {
                    "api_key": "sk-test",
                    "endpoint": "https://api.openai.com/v1",
                }
            }

    import faas_gauge.store.migration as migration

    monkeypatch.setattr(migration, "yaml", _YamlStub())
    migrate_credentials(secret_yaml, target_data_dir)

    credentials_path = target_data_dir / "config" / "credentials.json"
    creds = json.loads(credentials_path.read_text(encoding="utf-8"))
    assert creds["openai"]["api_key"] == "sk-test"


def test_migrate_credentials_raises_when_yaml_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret_yaml = tmp_path / ".secret.yaml"
    target_data_dir = tmp_path / "data"
    secret_yaml.write_text("openai: {}\n", encoding="utf-8")

    import faas_gauge.store.migration as migration

    monkeypatch.setattr(migration, "yaml", None)
    with pytest.raises(ImportError):
        migrate_credentials(secret_yaml, target_data_dir)


def test_migrate_static_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "legacy"
    target_data_dir = tmp_path / "data"

    (source_dir / "test_data").mkdir(parents=True)
    (source_dir / "question").mkdir(parents=True)

    (source_dir / "test_data" / "case.json").write_text('{"x": 1}', encoding="utf-8")
    (source_dir / "question" / "prime.txt").write_text("Generate prime numbers", encoding="utf-8")

    migrate_test_data(source_dir, target_data_dir)
    migrate_questions(source_dir, target_data_dir)

    assert (target_data_dir / "test_data" / "case.json").read_text(encoding="utf-8") == '{"x": 1}'
    assert (target_data_dir / "questions" / "prime.txt").read_text(
        encoding="utf-8"
    ) == "Generate prime numbers"


def test_migrate_all_runs_every_step(legacy_db: Path, tmp_path: Path) -> None:
    source_dir = tmp_path / "legacy"
    target_data_dir = tmp_path / "data"
    secret_yaml = source_dir / ".secret.yaml"

    (source_dir / "test_data").mkdir(parents=True)
    (source_dir / "question").mkdir(parents=True)
    (source_dir / "test_data" / "case.json").write_text('{"k": "v"}', encoding="utf-8")
    (source_dir / "question" / "q.txt").write_text("Q", encoding="utf-8")
    secret_yaml.write_text("openai: {}\n", encoding="utf-8")

    class _YamlStub:
        @staticmethod
        def safe_load(text: str) -> dict[str, dict[str, str]]:
            return {"openai": {}}

    import faas_gauge.store.migration as migration

    migration.yaml = _YamlStub()

    migrate_all(
        source_db=legacy_db,
        source_dirs=[source_dir],
        target_data_dir=target_data_dir,
        secret_yaml=secret_yaml,
    )

    assert (target_data_dir / "experiments" / "exp_alpha" / "experiment.json").is_file()
    assert (target_data_dir / "validations" / "batch_1" / "batch.json").is_file()
    assert (target_data_dir / "test_data" / "case.json").is_file()
    assert (target_data_dir / "questions" / "q.txt").is_file()
