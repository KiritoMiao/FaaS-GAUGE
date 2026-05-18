"""Utilities for migrating legacy SQLite/YAML data into the JSON/JSONL store."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from . import ExperimentStore

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - tested via monkeypatch
    yaml = None


def _rows(
    conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    cursor = conn.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def migrate_from_sqlite(
    db_path: str | Path,
    target_data_dir: str | Path,
    test_group: str | None = None,
) -> None:
    """Migrate experiments and iteration_results from SQLite into JSON/JSONL.

    If *test_group* is given, only experiments whose ``test_group`` matches are
    migrated.
    """
    store = ExperimentStore(target_data_dir)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        if test_group is not None:
            experiments = _rows(
                conn,
                "SELECT * FROM experiments WHERE test_group = ? ORDER BY id",
                (test_group,),
            )
        else:
            experiments = _rows(conn, "SELECT * FROM experiments ORDER BY id")
        for experiment in experiments:
            experiment_db_id = experiment["id"]
            experiment_id = experiment["experiment_name"]

            meta = {
                "id": experiment_id,
                "question_name": experiment["question_name"],
                "question_content": experiment["question_content"],
                "provider": experiment["provider"],
                "model_name": experiment["model_name"],
                "iterations": experiment["iterations"],
                "test_group": experiment["test_group"],
                "created_at": experiment["created_at"],
                "updated_at": experiment["updated_at"],
            }
            store.create_experiment(meta)
            store.update_experiment_summary(
                experiment_id,
                {
                    "successful_iterations": experiment["successful_iterations"],
                    "failed_iterations": experiment["failed_iterations"],
                    "total_input_tokens": experiment["total_input_tokens"],
                    "total_output_tokens": experiment["total_output_tokens"],
                    "total_time": experiment["total_time"],
                    "total_tokens_per_second": experiment["total_tokens_per_second"],
                    "average_time": experiment["average_time"],
                },
            )

            iterations = _rows(
                conn,
                "SELECT * FROM iteration_results WHERE experiment_id = ? ORDER BY iteration",
                (experiment_db_id,),
            )
            for row in iterations:
                store.append_iteration(
                    experiment_id,
                    {
                        "iteration": row["iteration"],
                        "input_tokens": row["input_tokens"],
                        "output_tokens": row["output_tokens"],
                        "rtt_time": row["rtt_time"],
                        "tokens_per_second": row["tokens_per_second"],
                        "error": row["error"],
                        "raw_response": row["raw_response"],
                        "python_code": row["python_code"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    },
                )
    finally:
        conn.close()


def migrate_validations(
    db_path: str | Path,
    target_data_dir: str | Path,
    test_group: str | None = None,
) -> None:
    """Migrate validation batches/runs/tests/executions into JSON/JSONL.

    If *test_group* is given, only batches whose ``test_group`` matches are
    migrated.
    """
    store = ExperimentStore(target_data_dir)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        if test_group is not None:
            batches = _rows(
                conn,
                "SELECT * FROM validation_batches WHERE test_group = ? ORDER BY id",
                (test_group,),
            )
        else:
            batches = _rows(conn, "SELECT * FROM validation_batches ORDER BY id")
        for batch in batches:
            validation_id = f"batch_{batch['id']}"
            batch_payload = {"id": validation_id}
            for key, value in batch.items():
                if key != "id":
                    batch_payload[key] = value
            store.create_validation_batch(batch_payload)

            runs = _rows(
                conn,
                "SELECT * FROM validation_runs WHERE batch_id = ? ORDER BY id",
                (batch["id"],),
            )
            for run in runs:
                tests = _rows(
                    conn,
                    "SELECT * FROM validation_tests WHERE validation_run_id = ? ORDER BY id",
                    (run["id"],),
                )
                test_payloads: list[dict[str, Any]] = []
                for test in tests:
                    executions = _rows(
                        conn,
                        "SELECT * FROM validation_executions WHERE validation_test_id = ? ORDER BY id",
                        (test["id"],),
                    )
                    test_payload = dict(test)
                    test_payload["executions"] = executions
                    test_payloads.append(test_payload)

                run_payload = dict(run)
                run_payload["tests"] = test_payloads
                store.append_validation_run(validation_id, run_payload)
    finally:
        conn.close()


def migrate_static_analysis(
    source_dir: str | Path,
    target_data_dir: str | Path,
    db_path: str | Path | None = None,
) -> int:
    """Migrate legacy static_analysis CSV files into JSONL per-experiment.

    Looks for ``static_analysis_result/*.csv`` in *source_dir*.  Each CSV
    filename is ``<question>+<provider>+<model>.csv`` (without the timestamp
    prefix).  The function resolves the full experiment name via the SQLite DB
    (when provided) or by scanning existing migrated experiment directories.

    The data is written to
    ``data/experiments/{experiment_id}/static_analysis.jsonl``.

    Returns the number of files migrated.
    """
    import csv as _csv

    csv_dir = Path(source_dir) / "static_analysis_result"
    if not csv_dir.is_dir():
        return 0

    # Build a lookup: suffix → experiment_id from already-migrated experiments
    exp_dir = Path(target_data_dir) / "experiments"
    suffix_to_id: dict[str, str] = {}
    if exp_dir.is_dir():
        for d in exp_dir.iterdir():
            if d.is_dir():
                # Strip the timestamp prefix: "20260121_1430+rest" → "rest"
                parts = d.name.split("+", 1)
                if len(parts) == 2:
                    suffix_to_id[parts[1]] = d.name

    # Also try DB lookup for more reliable matching
    db_lookup: dict[str, str] = {}
    if db_path is not None:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            for row in _rows(conn, "SELECT experiment_name FROM experiments"):
                name = row["experiment_name"]
                parts = name.split("+", 1)
                if len(parts) == 2:
                    db_lookup[parts[1]] = name
        finally:
            conn.close()

    migrated = 0
    for csv_path in sorted(csv_dir.glob("*.csv")):
        suffix = csv_path.stem

        experiment_id = db_lookup.get(suffix) or suffix_to_id.get(suffix)
        if experiment_id is None:
            continue

        out_path = (
            Path(target_data_dir)
            / "experiments"
            / experiment_id
            / "static_analysis.jsonl"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            lines: list[str] = []
            for row in reader:
                record: dict[str, Any] = {}
                for key, value in row.items():
                    if value == "" or value is None:
                        record[key] = None
                    else:
                        try:
                            record[key] = int(value)
                        except ValueError:
                            try:
                                record[key] = float(value)
                            except ValueError:
                                record[key] = value
                lines.append(json.dumps(record, ensure_ascii=False))

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        migrated += 1

    return migrated


def migrate_credentials(
    secret_yaml_path: str | Path, target_data_dir: str | Path
) -> None:
    """Convert .secret.yaml into data/config/credentials.json."""
    if yaml is None:
        raise ImportError("pyyaml is required for credentials migration")

    secret_path = Path(secret_yaml_path)
    target_path = Path(target_data_dir) / "config" / "credentials.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    loaded = yaml.safe_load(secret_path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}

    target_path.write_text(
        json.dumps(loaded, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def migrate_test_data(source_dir: str | Path, target_data_dir: str | Path) -> None:
    """Copy legacy test_data/*.json files into the new store."""
    src_dir = Path(source_dir) / "test_data"
    dst_dir = Path(target_data_dir) / "test_data"
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.is_dir():
        return

    for src in src_dir.glob("*.json"):
        shutil.copy2(src, dst_dir / src.name)


def migrate_questions(source_dir: str | Path, target_data_dir: str | Path) -> None:
    """Copy legacy question/*.txt files into the new store."""
    src_dir = Path(source_dir) / "question"
    dst_dir = Path(target_data_dir) / "questions"
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.is_dir():
        return

    for src in src_dir.glob("*.txt"):
        shutil.copy2(src, dst_dir / src.name)


def migrate_all(
    source_db: str | Path,
    source_dirs: list[str | Path],
    target_data_dir: str | Path,
    secret_yaml: str | Path | None = None,
    test_group: str | None = None,
) -> None:
    """Run the full migration for DB data, static files, and optional credentials."""
    migrate_from_sqlite(source_db, target_data_dir, test_group=test_group)
    migrate_validations(source_db, target_data_dir, test_group=test_group)

    for source_dir in source_dirs:
        migrate_test_data(source_dir, target_data_dir)
        migrate_questions(source_dir, target_data_dir)
        migrate_static_analysis(source_dir, target_data_dir, db_path=source_db)

    if secret_yaml is not None and Path(secret_yaml).is_file():
        migrate_credentials(secret_yaml, target_data_dir)
