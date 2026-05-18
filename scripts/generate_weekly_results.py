from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SIZE_ORDER = [
    (1_000_000, "1m"),
    (10_000_000, "10m"),
    (50_000_000, "50m"),
    (75_000_000, "75m"),
    (100_000_000, "100m"),
    (125_000_000, "125m"),
    (150_000_000, "150m"),
    (175_000_000, "175m"),
    (200_000_000, "200m"),
]
SIZE_LABELS = [label for _, label in SIZE_ORDER]

PRIME_COLUMNS = [
    "model_provider",
    "seq",
    "maximum_successful_prime",
    "function_status_200m",
    *SIZE_LABELS,
    *[f"{s}_max_ram" for s in SIZE_LABELS],
]

TARGET30_COLUMNS = [
    "question",
    "model_provider",
    "seq",
    "function_status",
    "performance_status",
    "run_1_status",
    "run_1_total_ms",
    "run_1_max_ram_mb",
    "run_2_status",
    "run_2_total_ms",
    "run_2_max_ram_mb",
    "avg_of_runs_total_ms",
    "avg_of_runs_total_seconds",
    "max_ram_mb",
    "input_url",
    "validation_message",
    "first_execution_error",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_jsonl_slim(path: Path) -> list[dict[str, Any]]:
    """Read JSONL keeping only timing/status fields in execution_results (saves RAM)."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    _KEEP = {"iteration", "success", "execution_time_ms", "max_memory_used_mb", "error"}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            run = json.loads(line)
            for t in run.get("tests", []):
                execs = t.get("execution_results")
                if isinstance(execs, list):
                    t["execution_results"] = [
                        {k: v for k, v in e.items() if k in _KEEP} for e in execs
                    ]
            rows.append(run)
    return rows


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_ts(run: dict[str, Any]) -> datetime:
    for key in ("validation_timestamp", "created_at"):
        v = run.get(key)
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except ValueError:
                continue
    return datetime.min


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _execution_avg_time(test: dict[str, Any]) -> float | None:
    vals = [
        _to_float(e.get("execution_time_ms")) for e in test.get("execution_results", [])
    ]
    clean = [v for v in vals if v is not None]
    return _avg(clean)


def _execution_max_ram(test: dict[str, Any]) -> float | None:
    vals = [
        _to_float(e.get("max_memory_used_mb"))
        for e in test.get("execution_results", [])
    ]
    clean = [v for v in vals if v is not None]
    return max(clean) if clean else None


def _format_200m_status(run: dict[str, Any], test_200m: dict[str, Any] | None) -> str:
    if not test_200m:
        return "error"
    if test_200m.get("status") == "passed":
        return "success"
    text = " ".join(
        [
            str(run.get("function_status", "")),
            str(run.get("function_status_reason", "")),
            str(test_200m.get("message", "")),
            " ".join(
                str(e.get("error", "")) for e in test_200m.get("execution_results", [])
            ),
        ]
    ).lower()
    if "memory" in text or "out of memory" in text or "oom" in text:
        return "out-of-memory"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "scaling" in text:
        return "scaling-bug"
    return "error"


def _build_prime_row(run: dict[str, Any], model_provider: str) -> dict[str, Any]:
    tests = run.get("tests", [])
    test_by_name = {t.get("name"): t for t in tests}
    non_perf_passed = all(
        t.get("status") == "passed"
        for t in tests
        if not str(t.get("name", "")).startswith("performance[")
    )

    row: dict[str, Any] = {
        "model_provider": model_provider,
        "seq": run.get("experiment_iteration") or run.get("iteration"),
        "maximum_successful_prime": "",
        "function_status_200m": "error",
    }

    highest_success_m: int | None = None
    for n, label in SIZE_ORDER:
        test_name = f"performance[n={n}]"
        t = test_by_name.get(test_name)
        timing = _execution_avg_time(t) if t else None
        ram = _execution_max_ram(t) if t else None
        row[label] = timing if t and t.get("status") == "passed" else ""
        row[f"{label}_max_ram"] = ram if ram is not None else ""

        if t and t.get("status") == "passed" and non_perf_passed:
            highest_success_m = n // 1_000_000

    row["maximum_successful_prime"] = (
        highest_success_m if highest_success_m is not None else ""
    )
    row["function_status_200m"] = _format_200m_status(
        run, test_by_name.get("performance[n=200000000]")
    )
    return row


def _perf_test_summary(run: dict[str, Any]) -> tuple[float, float | None, str, str]:
    """Extract total_ms, max_ram, status, first_error from performance test executions."""
    total_ms = 0.0
    max_ram: float | None = None
    first_error = ""
    status = "passed"
    for t in run.get("tests", []):
        name = str(t.get("name", ""))
        if not (name == "performance" or name.startswith("performance[")):
            continue
        exs = t.get("execution_results") or []
        has_success = False
        for ex in exs:
            if ex.get("error"):
                if not first_error:
                    first_error = str(ex["error"])
            else:
                has_success = True
            tm = _to_float(ex.get("execution_time_ms"))
            if tm is not None:
                total_ms += tm
            ram = _to_float(ex.get("max_memory_used_mb"))
            if ram is not None:
                max_ram = ram if max_ram is None else max(max_ram, ram)
        if t.get("status") == "error":
            status = "error"
        elif t.get("status") == "failed" and status != "error":
            status = "failed"
    return total_ms, max_ram, status, first_error


def _target30_row(
    run1: dict[str, Any],
    run2: dict[str, Any] | None,
    model_provider: str,
    question: str,
) -> dict[str, Any]:
    r1_total, r1_ram, r1_status, r1_error = _perf_test_summary(run1)
    if run2 is not None:
        r2_total, r2_ram, r2_status, r2_error = _perf_test_summary(run2)
    else:
        r2_total, r2_ram, r2_status, r2_error = 0.0, None, "", ""

    first_error = r1_error or r2_error
    avg_ms = (r1_total + r2_total) / 2 if run2 is not None else r1_total

    function_status = str(run1.get("function_status", ""))
    if run2 is not None and run2.get("function_status") != run1.get("function_status"):
        if function_status == "success":
            function_status = "mixed"

    if function_status == "mixed":
        perf_status = "mixed"
    elif "error" in (r1_status, r2_status) or function_status in {
        "error",
        "timeout",
    }:
        perf_status = "error"
    elif "failed" in (r1_status, r2_status) or function_status == "functional-error":
        perf_status = "failed"
    else:
        perf_status = "passed" if avg_ms < 30_000 else "failed"

    max_ram = max([v for v in (r1_ram, r2_ram) if v is not None], default=None)

    validation_message = "OK"
    if function_status != "success":
        for t in run1.get("tests", []):
            msg = str(t.get("message", "")).strip()
            if msg and msg not in ("Execution successful", ""):
                validation_message = msg
                break
        if validation_message == "OK":
            validation_message = str(
                run1.get("function_status_reason")
                or run1.get("status")
                or function_status
            )

    return {
        "question": question,
        "model_provider": model_provider,
        "seq": run1.get("experiment_iteration") or run1.get("iteration"),
        "function_status": function_status,
        "performance_status": perf_status,
        "run_1_status": r1_status,
        "run_1_total_ms": r1_total,
        "run_1_max_ram_mb": r1_ram if r1_ram is not None else "",
        "run_2_status": r2_status if run2 is not None else "",
        "run_2_total_ms": r2_total if run2 is not None else "",
        "run_2_max_ram_mb": r2_ram if r2_ram is not None else "",
        "avg_of_runs_total_ms": avg_ms,
        "avg_of_runs_total_seconds": avg_ms / 1000,
        "max_ram_mb": max_ram if max_ram is not None else "",
        "input_url": "",
        "validation_message": validation_message,
        "first_execution_error": first_error,
    }


def _numeric_columns(columns: list[str], rows: list[dict[str, Any]]) -> list[str]:
    numerics: list[str] = []
    for c in columns:
        if c in {
            "model_provider",
            "question",
            "function_status",
            "performance_status",
            "run_1_status",
            "run_2_status",
            "input_url",
            "validation_message",
            "first_execution_error",
            "function_status_200m",
        }:
            continue
        if any(_to_float(r.get(c)) is not None for r in rows):
            numerics.append(c)
    return numerics


def _stats_rows(columns: list[str], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    nums = _numeric_columns(columns, rows)

    def vals(col: str) -> list[float]:
        out = []
        for r in rows:
            v = _to_float(r.get(col))
            if v is not None:
                out.append(v)
        return out

    labels = ["AVG", "STDEV", "MEDIAN", "CV", "MAX", "MIN", "RANGE", "x Slower"]
    out_rows = [{c: "" for c in columns} for _ in labels]
    label_col = "model_provider" if "model_provider" in columns else columns[0]
    for i, lb in enumerate(labels):
        out_rows[i][label_col] = lb

    for c in nums:
        v = vals(c)
        if not v:
            continue
        mean = statistics.fmean(v)
        std = statistics.stdev(v) if len(v) > 1 else 0.0
        med = statistics.median(v)
        mx = max(v)
        mn = min(v)
        rng = mx - mn
        slower = mx / mn if mn != 0 else ""

        out_rows[0][c] = mean
        out_rows[1][c] = std
        out_rows[2][c] = med
        out_rows[3][c] = f"{(std / mean * 100):.2f}%" if mean != 0 else ""
        out_rows[4][c] = mx
        out_rows[5][c] = mn
        out_rows[6][c] = rng
        out_rows[7][c] = slower

    return out_rows


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows + _stats_rows(columns, rows):
            w.writerow({c: row.get(c, "") for c in columns})


def _collect_runs(
    data_dir: Path, test_group: str
) -> dict[tuple[str, int], dict[str, Any]]:
    """Collect latest run per (exp_id, seq) for low-iteration batches."""
    validations_dir = data_dir / "validations"
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for batch_dir in sorted(validations_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        batch_json = batch_dir / "batch.json"
        runs_jsonl = batch_dir / "runs.jsonl"
        if not batch_json.exists() or not runs_jsonl.exists():
            continue
        batch = _read_json(batch_json)
        if batch.get("test_group") != test_group or batch.get("runtime") != "aws":
            continue
        # Skip high-iteration batches (those are for target30 pair reporting)
        ipt = batch.get("iterations_per_test", 1)
        if isinstance(ipt, int) and ipt >= _TARGET30_MIN_ITERATIONS:
            continue
        for run in _read_jsonl(runs_jsonl):
            exp_id = run.get("experiment_id")
            seq = run.get("experiment_iteration") or run.get("iteration")
            if not isinstance(exp_id, str) or seq is None:
                continue
            key = (exp_id, int(seq))
            current = selected.get(key)
            if current is None or _parse_ts(run) >= _parse_ts(current):
                selected[key] = run
    return selected


# Minimum iterations_per_test to consider a batch as a "target30" run.
_TARGET30_MIN_ITERATIONS = 10


def _collect_target30_run_pairs(
    data_dir: Path, test_group: str, question: str
) -> dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any] | None]]:
    """Collect paired runs (run1, run2) from high-iteration batches for target30.

    For each (exp_id, seq), picks the two most recent batches as run1 / run2.
    If only one batch exists, run2 is None.
    """
    validations_dir = data_dir / "validations"

    # Find batches matching criteria, sorted by creation time
    matching_batches: list[tuple[str, Path]] = []
    for batch_dir in sorted(validations_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        batch_json = batch_dir / "batch.json"
        runs_jsonl = batch_dir / "runs.jsonl"
        if not batch_json.exists() or not runs_jsonl.exists():
            continue
        batch = _read_json(batch_json)
        if batch.get("test_group") != test_group or batch.get("runtime") != "aws":
            continue
        if batch.get("question_filter") != question:
            continue
        ipt = batch.get("iterations_per_test", 1)
        if not isinstance(ipt, int) or ipt < _TARGET30_MIN_ITERATIONS:
            continue
        batch_id = batch.get("id", batch_dir.name)
        matching_batches.append((batch_id, runs_jsonl))

    # Collect runs keyed by (exp_id, seq) -> list of runs across batches
    all_runs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for batch_id, runs_path in matching_batches:
        for run in _read_jsonl_slim(runs_path):
            exp_id = run.get("experiment_id")
            seq = run.get("experiment_iteration") or run.get("iteration")
            if not isinstance(exp_id, str) or seq is None:
                continue
            key = (exp_id, int(seq))
            all_runs.setdefault(key, []).append(run)

    # Pick 2 most recent runs per key
    pairs: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any] | None]] = {}
    for key, runs in all_runs.items():
        runs.sort(key=_parse_ts)
        if len(runs) >= 2:
            pairs[key] = (runs[-2], runs[-1])
        else:
            pairs[key] = (runs[0], None)
    return pairs


def _model_provider_for(data_dir: Path, exp_id: str, cache: dict[str, str]) -> str:
    if exp_id in cache:
        return cache[exp_id]
    exp_file = data_dir / "experiments" / exp_id / "experiment.json"
    if exp_file.exists():
        exp = _read_json(exp_file)
        cache[exp_id] = f"{exp.get('provider', '')}+{exp.get('model', '')}"
    else:
        parts = exp_id.split("+")
        cache[exp_id] = "+".join(parts[-2:]) if len(parts) >= 2 else exp_id
    return cache[exp_id]


def generate_weekly(group: str, output_dir: Path, data_dir: Path) -> list[Path]:
    latest_runs = _collect_runs(data_dir, group)
    model_cache: dict[str, str] = {}

    # Collect questions present in the runs
    questions: set[str] = set()
    for (exp_id, _), run in latest_runs.items():
        questions.add(str(run.get("question") or exp_id.split("+")[1]))

    # Also discover questions from high-iteration (target30) batches
    validations_dir = data_dir / "validations"
    for batch_dir in validations_dir.iterdir():
        if not batch_dir.is_dir():
            continue
        batch_json = batch_dir / "batch.json"
        if not batch_json.exists():
            continue
        batch = _read_json(batch_json)
        if batch.get("test_group") != group or batch.get("runtime") != "aws":
            continue
        ipt = batch.get("iterations_per_test", 1)
        if isinstance(ipt, int) and ipt >= _TARGET30_MIN_ITERATIONS:
            qf = batch.get("question_filter")
            if qf:
                questions.add(qf)

    outputs: list[Path] = []

    # --- Prime: use latest single-run data as before ---
    if "prime_number_generator" in questions:
        rows: list[dict[str, Any]] = []
        for (exp_id, _), run in latest_runs.items():
            q = str(run.get("question") or exp_id.split("+")[1])
            if q != "prime_number_generator":
                continue
            mp = _model_provider_for(data_dir, exp_id, model_cache)
            rows.append(_build_prime_row(run, mp))
        rows.sort(
            key=lambda r: (str(r.get("model_provider", "")), int(r.get("seq", 0)))
        )
        out = output_dir / f"{group}_prime_number_generator_lambda_performance.csv"
        _write_csv(out, PRIME_COLUMNS, rows)
        outputs.append(out)

    # --- Target-30 questions: use paired runs from high-iteration batches ---
    for question in sorted(questions - {"prime_number_generator"}):
        pairs = _collect_target30_run_pairs(data_dir, group, question)
        if not pairs:
            # Fallback: use single-run data with old approach
            pairs_fallback: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
            for (exp_id, _), run in latest_runs.items():
                q = str(run.get("question") or exp_id.split("+")[1])
                if q != question:
                    continue
                pairs_fallback.append((run, None))
            pairs = {
                (
                    run.get("experiment_id", ""),
                    int(run.get("experiment_iteration") or run.get("iteration", 0)),
                ): (run, None)
                for run, _ in pairs_fallback
            }

        rows = []
        for (exp_id, _), (run1, run2) in pairs.items():
            mp = _model_provider_for(data_dir, exp_id, model_cache)
            rows.append(_target30_row(run1, run2, mp, question))
        rows.sort(
            key=lambda r: (str(r.get("model_provider", "")), int(r.get("seq", 0)))
        )
        out = output_dir / f"{group}_{question}_aws_target30_avg2_detailed.csv"
        _write_csv(out, TARGET30_COLUMNS, rows)
        outputs.append(out)

    return outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate weekly result CSVs from validations."
    )
    p.add_argument("-g", "--group", required=True, help="test_group filter, e.g. week3")
    p.add_argument(
        "-o", "--output-dir", default="data/reports/weekly", help="output dir"
    )
    p.add_argument("-d", "--data-dir", default="data", help="data dir")
    return p


def main() -> int:
    args = build_parser().parse_args()
    outputs = generate_weekly(args.group, Path(args.output_dir), Path(args.data_dir))
    if not outputs:
        print("No matching runs found.")
        return 1
    print("Generated files:")
    for p in sorted(outputs):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
