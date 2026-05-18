"""Export experiment data as CSV reports.

Generates CSV files in several report formats:

  static-analysis  — pylint + radon metrics per iteration
  performance      — function status, execution times, RAM per iteration
  merged           — static-analysis + performance joined per iteration
  composite        — single CSV per test_group with ALL models combined

Examples:

  # Export static analysis for all week1 experiments
  python scripts/export_results.py --report static-analysis --test-group week1

  # Export performance for a specific experiment
  python scripts/export_results.py --report performance --experiment "*prime*deepseek*"

  # Export merged (static + runtime) per experiment
  python scripts/export_results.py --report merged --test-group week2

  # Export composite CSVs (all models in one file, grouped by test_group)
  python scripts/export_results.py --report composite --test-group week1

  # Export everything (per-experiment + composite)
  python scripts/export_results.py --report all --test-group week1 -o results/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from faas_gauge.store import ExperimentStore

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

STATIC_ANALYSIS_COLUMNS = [
    "iteration",
    "generate_timestamp",
    "input_token",
    "output_token",
    "rtt_time",
    "token_per_second",
    "error",
    "pylint_score",
    "pylint_fatal",
    "pylint_errors",
    "pylint_warnings",
    "pylint_conventions",
    "pylint_refactors",
    "cc_avg",
    "mi_score",
    "mi_rank",
    "loc",
    "lloc",
    "sloc",
    "comments",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "halstead_bugs",
]

PERFORMANCE_COLUMNS = [
    "model_provider",
    "seq",
    "generate_timestamp",
    "input_url",
    "function_status",
    "performance_status",
    "execution_count",
    "performance_avg_ms",
    "performance_total_ms",
    "performance_max_ram_mb",
    "validation_message",
    "first_execution_error",
]

# Merged columns — static analysis + performance per iteration
MERGED_COLUMNS = [
    "model_provider",
    "seq",
    "generate_timestamp",
    # -- LLM generation metrics --
    "input_token",
    "output_token",
    "rtt_time",
    "token_per_second",
    "error",
    # -- static analysis --
    "pylint_score",
    "pylint_fatal",
    "pylint_errors",
    "pylint_warnings",
    "pylint_conventions",
    "pylint_refactors",
    "cc_avg",
    "mi_score",
    "mi_rank",
    "loc",
    "lloc",
    "sloc",
    "comments",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "halstead_bugs",
    # -- runtime / performance --
    "function_status",
    "performance_status",
    "execution_count",
    "performance_avg_ms",
    "performance_total_ms",
    "performance_max_ram_mb",
    "validation_message",
    "first_execution_error",
]

# Composite columns — experiment identity + all metric fields in one row
_EXPERIMENT_ID_COLUMNS = [
    "experiment_id",
    "question_name",
    "provider",
    "model_name",
    "test_group",
]

COMPOSITE_STATIC_ANALYSIS_COLUMNS = [
    *_EXPERIMENT_ID_COLUMNS,
    *STATIC_ANALYSIS_COLUMNS,
]

COMPOSITE_PERFORMANCE_COLUMNS = [
    *_EXPERIMENT_ID_COLUMNS,
    *PERFORMANCE_COLUMNS,
]

COMPOSITE_MERGED_COLUMNS = [
    *_EXPERIMENT_ID_COLUMNS,
    *MERGED_COLUMNS,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _experiment_suffix(experiment_id: str) -> str:
    """Strip timestamp prefix: '20260121_1430+rest' -> 'rest'."""
    parts = experiment_id.split("+", 1)
    return parts[1] if len(parts) == 2 else experiment_id


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV with the given column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _model_provider(exp: dict[str, Any]) -> str:
    """Build 'provider+model' string from experiment metadata."""
    return f"{exp.get('provider', '')}+{exp.get('model_name') or exp.get('model', '')}"


def _experiment_identity(exp: dict[str, Any]) -> dict[str, Any]:
    """Return the common identity columns for composite rows."""
    return {
        "experiment_id": exp["id"],
        "question_name": exp.get("question_name", ""),
        "provider": exp.get("provider", ""),
        "model_name": exp.get("model_name") or exp.get("model", ""),
        "test_group": exp.get("test_group") or "ungrouped",
    }


def _extract_generate_timestamp(iteration_record: dict[str, Any]) -> str:
    """Extract LLM generation timestamp from an iteration record.

    Handles both week2 format (``timestamp``) and week1 format
    (``created_at``).
    """
    return iteration_record.get("timestamp") or iteration_record.get("created_at") or ""


def _extract_llm_metrics(it: dict[str, Any]) -> dict[str, Any]:
    """Pull token / timing / timestamp fields from an iteration record.

    Normalises across week1 (``rtt_time``, ``created_at``) and
    week2 (``rtt_sec``, ``timestamp``) field names.
    """
    input_token = it.get("input_tokens") or it.get("input_token")
    output_token = it.get("output_tokens") or it.get("output_token")
    rtt_time = it.get("rtt_time") or it.get("rtt_sec") or it.get("duration_s")
    tps = it.get("tokens_per_second") or it.get("token_per_second")
    if tps is None and output_token and rtt_time and rtt_time > 0:
        tps = round(output_token / rtt_time, 2)
    return {
        "generate_timestamp": _extract_generate_timestamp(it),
        "input_token": input_token,
        "output_token": output_token,
        "rtt_time": rtt_time,
        "token_per_second": tps,
        "error": it.get("error"),
    }


# ---------------------------------------------------------------------------
# Validation index
# ---------------------------------------------------------------------------


def _build_validation_index(
    store: ExperimentStore,
) -> dict[tuple[Any, Any], list[dict[str, Any]]]:
    """Build an index: (experiment_id, iteration) -> [runs]."""
    index: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    validations_dir = store.data_dir / "validations"
    if not validations_dir.is_dir():
        return index

    for batch_dir in sorted(validations_dir.iterdir()):
        runs_path = batch_dir / "runs.jsonl"
        if not runs_path.is_file():
            continue
        for run in _read_jsonl(runs_path):
            exp_id = run.get("experiment_id")
            exp_iter = run.get("experiment_iteration") or run.get("iteration")
            key = (exp_id, exp_iter)
            index.setdefault(key, []).append(run)

    return index


def _find_validation_runs(
    val_index: dict[tuple[Any, Any], list[dict[str, Any]]],
    experiment: dict[str, Any],
    iteration: int,
) -> list[dict[str, Any]]:
    """Find validation runs matching an experiment and iteration."""
    exp_name = experiment.get("id", "")
    exp_question = experiment.get("question_name", "")

    # Direct lookup by experiment ID + iteration
    direct = val_index.get((exp_name, iteration))
    if direct:
        return direct

    # Fallback: scan for matching question
    for (exp_id, exp_iter), runs in val_index.items():
        if exp_iter != iteration:
            continue
        for run in runs:
            code_path = run.get("code_path", "")
            question = run.get("question", "")
            if exp_question and (exp_question in code_path or exp_question == question):
                return runs
    return []


def _summarise_run(run: dict[str, Any]) -> dict[str, Any]:
    """Extract performance summary from a single validation run."""
    tests = run.get("tests", [])

    function_status = run.get("function_status")
    if not function_status:
        if run.get("status") == "error":
            function_status = "error"
        elif all(t.get("status") == "passed" for t in tests):
            function_status = "success"
        elif any(t.get("status") == "error" for t in tests):
            function_status = "functional-error"
        else:
            function_status = "failed"

    exec_times: list[float] = []
    max_ram: float | None = None
    first_error: str | None = None
    validation_message: str | None = None

    for test in tests:
        if test.get("message") and not validation_message:
            validation_message = test["message"]
        exec_list = test.get("executions") or test.get("execution_results") or []
        for ex in exec_list:
            t = ex.get("execution_time_ms")
            if t is not None:
                exec_times.append(t)
            ram = ex.get("max_memory_used_mb")
            if ram is not None:
                max_ram = max(max_ram or 0.0, ram)
            if ex.get("error") and first_error is None:
                first_error = ex["error"]

    run_summary = run.get("summary", {})
    passed = run_summary.get("passed", 0)
    failed = run_summary.get("failed", 0)
    errors = run_summary.get("errors", 0)
    performance_status = (
        "passed" if passed > 0 and failed == 0 and errors == 0 else "failed"
    )

    execution_count = len(exec_times)
    avg_ms = sum(exec_times) / execution_count if execution_count else None
    total_ms = sum(exec_times) if execution_count else None

    return {
        "performance_status": performance_status,
        "function_status": function_status,
        "execution_count": execution_count,
        "performance_avg_ms": avg_ms,
        "performance_total_ms": total_ms,
        "performance_max_ram_mb": max_ram,
        "validation_message": validation_message or "",
        "first_execution_error": first_error or "",
    }


def _perf_summary_for_iteration(
    val_index: dict[tuple[Any, Any], list[dict[str, Any]]],
    exp: dict[str, Any],
    it: dict[str, Any],
) -> dict[str, Any]:
    """Return a performance summary dict for one iteration."""
    seq = it.get("iteration", 0)
    runs = _find_validation_runs(val_index, exp, seq)
    if runs:
        return _summarise_run(runs[-1])
    return {
        "performance_status": "",
        "function_status": "success" if not it.get("error") else "error",
        "execution_count": 0,
        "performance_avg_ms": "",
        "performance_total_ms": "",
        "performance_max_ram_mb": "",
        "validation_message": "",
        "first_execution_error": it.get("error") or "",
    }


# ---------------------------------------------------------------------------
# Static-analysis export
# ---------------------------------------------------------------------------


def export_static_analysis(
    store: ExperimentStore,
    experiments: list[dict[str, Any]],
    output_dir: Path,
) -> int:
    """Export static analysis CSVs. Returns number of files written."""
    count = 0
    for exp in experiments:
        exp_id = exp["id"]
        sa_path = store.data_dir / "experiments" / exp_id / "static_analysis.jsonl"
        records = _read_jsonl(sa_path)
        if not records:
            continue

        suffix = _experiment_suffix(exp_id)
        out_path = output_dir / "static_analysis" / f"{suffix}.csv"
        _write_csv(out_path, STATIC_ANALYSIS_COLUMNS, records)
        count += 1

    return count


# ---------------------------------------------------------------------------
# Performance export
# ---------------------------------------------------------------------------


def export_performance(
    store: ExperimentStore,
    experiments: list[dict[str, Any]],
    output_dir: Path,
) -> int:
    """Export performance CSVs. Returns number of files written."""
    val_index = _build_validation_index(store)
    count = 0

    for exp in experiments:
        exp_id = exp["id"]
        iterations = _read_jsonl(
            store.data_dir / "experiments" / exp_id / "iterations.jsonl"
        )
        if not iterations:
            continue

        mp = _model_provider(exp)
        rows: list[dict[str, Any]] = []

        for it in iterations:
            seq = it.get("iteration", 0)
            summary = _perf_summary_for_iteration(val_index, exp, it)
            row = {
                "model_provider": mp,
                "seq": seq,
                "generate_timestamp": _extract_generate_timestamp(it),
                "input_url": "",
                **summary,
            }
            rows.append(row)

        if rows:
            suffix = _experiment_suffix(exp_id)
            out_path = output_dir / "performance" / f"{suffix}.csv"
            _write_csv(out_path, PERFORMANCE_COLUMNS, rows)
            count += 1

    return count


# ---------------------------------------------------------------------------
# Merged export — static analysis + performance joined per iteration
# ---------------------------------------------------------------------------


def export_merged(
    store: ExperimentStore,
    experiments: list[dict[str, Any]],
    output_dir: Path,
) -> int:
    """Export merged (static + performance) CSVs per experiment."""
    val_index = _build_validation_index(store)
    count = 0

    for exp in experiments:
        exp_id = exp["id"]
        iterations = _read_jsonl(
            store.data_dir / "experiments" / exp_id / "iterations.jsonl"
        )
        if not iterations:
            continue

        # Build static-analysis lookup: iteration -> record
        sa_path = store.data_dir / "experiments" / exp_id / "static_analysis.jsonl"
        sa_records = _read_jsonl(sa_path)
        sa_by_iter: dict[int, dict[str, Any]] = {
            r["iteration"]: r for r in sa_records if "iteration" in r
        }

        mp = _model_provider(exp)
        rows: list[dict[str, Any]] = []

        for it in iterations:
            seq = it.get("iteration", 0)
            llm = _extract_llm_metrics(it)
            perf = _perf_summary_for_iteration(val_index, exp, it)
            sa = sa_by_iter.get(seq, {})

            row: dict[str, Any] = {
                "model_provider": mp,
                "seq": seq,
                # LLM generation metrics (prefer static_analysis.jsonl which
                # already merged these; fall back to iterations.jsonl)
                "generate_timestamp": sa.get("generate_timestamp")
                or llm["generate_timestamp"],
                "input_token": sa.get("input_token") or llm["input_token"],
                "output_token": sa.get("output_token") or llm["output_token"],
                "rtt_time": sa.get("rtt_time") or llm["rtt_time"],
                "token_per_second": sa.get("token_per_second")
                or llm["token_per_second"],
                "error": sa.get("error")
                if sa.get("error") is not None
                else llm["error"],
            }

            # Static analysis metrics
            for col in STATIC_ANALYSIS_COLUMNS:
                if col not in row:
                    row[col] = sa.get(col, "")

            # Performance metrics
            row.update(perf)

            rows.append(row)

        if rows:
            suffix = _experiment_suffix(exp_id)
            out_path = output_dir / "merged" / f"{suffix}.csv"
            _write_csv(out_path, MERGED_COLUMNS, rows)
            count += 1

    return count


# ---------------------------------------------------------------------------
# Composite export — one CSV per test_group with all models
# ---------------------------------------------------------------------------


def _group_by_test_group(
    experiments: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group experiments by test_group.  NULL/missing -> 'ungrouped'."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for exp in experiments:
        key = exp.get("test_group") or "ungrouped"
        groups.setdefault(key, []).append(exp)
    return groups


def export_composite_static_analysis(
    store: ExperimentStore,
    experiments: list[dict[str, Any]],
    output_dir: Path,
) -> int:
    """Export composite static-analysis CSVs (one per test_group)."""
    groups = _group_by_test_group(experiments)
    count = 0

    for group_name, group_exps in sorted(groups.items()):
        all_rows: list[dict[str, Any]] = []
        for exp in group_exps:
            sa_path = (
                store.data_dir / "experiments" / exp["id"] / "static_analysis.jsonl"
            )
            records = _read_jsonl(sa_path)
            if not records:
                continue
            identity = _experiment_identity(exp)
            for rec in records:
                all_rows.append({**identity, **rec})

        if all_rows:
            out_path = output_dir / "composite" / f"static_analysis_{group_name}.csv"
            _write_csv(out_path, COMPOSITE_STATIC_ANALYSIS_COLUMNS, all_rows)
            count += 1

    return count


def export_composite_performance(
    store: ExperimentStore,
    experiments: list[dict[str, Any]],
    output_dir: Path,
) -> int:
    """Export composite performance CSVs (one per test_group)."""
    val_index = _build_validation_index(store)
    groups = _group_by_test_group(experiments)
    count = 0

    for group_name, group_exps in sorted(groups.items()):
        all_rows: list[dict[str, Any]] = []
        for exp in group_exps:
            exp_id = exp["id"]
            iterations = _read_jsonl(
                store.data_dir / "experiments" / exp_id / "iterations.jsonl"
            )
            if not iterations:
                continue

            identity = _experiment_identity(exp)
            mp = _model_provider(exp)

            for it in iterations:
                seq = it.get("iteration", 0)
                summary = _perf_summary_for_iteration(val_index, exp, it)
                all_rows.append(
                    {
                        **identity,
                        "model_provider": mp,
                        "seq": seq,
                        "generate_timestamp": _extract_generate_timestamp(it),
                        "input_url": "",
                        **summary,
                    }
                )

        if all_rows:
            out_path = output_dir / "composite" / f"performance_{group_name}.csv"
            _write_csv(out_path, COMPOSITE_PERFORMANCE_COLUMNS, all_rows)
            count += 1

    return count


def export_composite_merged(
    store: ExperimentStore,
    experiments: list[dict[str, Any]],
    output_dir: Path,
) -> int:
    """Export composite merged CSVs (one per test_group)."""
    val_index = _build_validation_index(store)
    groups = _group_by_test_group(experiments)
    count = 0

    for group_name, group_exps in sorted(groups.items()):
        all_rows: list[dict[str, Any]] = []
        for exp in group_exps:
            exp_id = exp["id"]
            iterations = _read_jsonl(
                store.data_dir / "experiments" / exp_id / "iterations.jsonl"
            )
            if not iterations:
                continue

            sa_path = store.data_dir / "experiments" / exp_id / "static_analysis.jsonl"
            sa_records = _read_jsonl(sa_path)
            sa_by_iter: dict[int, dict[str, Any]] = {
                r["iteration"]: r for r in sa_records if "iteration" in r
            }

            identity = _experiment_identity(exp)
            mp = _model_provider(exp)

            for it in iterations:
                seq = it.get("iteration", 0)
                llm = _extract_llm_metrics(it)
                perf = _perf_summary_for_iteration(val_index, exp, it)
                sa = sa_by_iter.get(seq, {})

                row: dict[str, Any] = {
                    **identity,
                    "model_provider": mp,
                    "seq": seq,
                    "generate_timestamp": sa.get("generate_timestamp")
                    or llm["generate_timestamp"],
                    "input_token": sa.get("input_token") or llm["input_token"],
                    "output_token": sa.get("output_token") or llm["output_token"],
                    "rtt_time": sa.get("rtt_time") or llm["rtt_time"],
                    "token_per_second": sa.get("token_per_second")
                    or llm["token_per_second"],
                    "error": sa.get("error")
                    if sa.get("error") is not None
                    else llm["error"],
                }

                for col in STATIC_ANALYSIS_COLUMNS:
                    if col not in row:
                        row[col] = sa.get(col, "")

                row.update(perf)
                all_rows.append(row)

        if all_rows:
            out_path = output_dir / "composite" / f"merged_{group_name}.csv"
            _write_csv(out_path, COMPOSITE_MERGED_COLUMNS, all_rows)
            count += 1

    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export experiment data as CSV reports.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--report",
        choices=["static-analysis", "performance", "merged", "composite", "all"],
        default="all",
        help="Which report(s) to generate (default: all).",
    )
    parser.add_argument(
        "--experiment",
        default="*",
        help="Glob pattern for experiment IDs (default: '*').",
    )
    parser.add_argument(
        "--test-group",
        default=None,
        help="Filter experiments by test_group (e.g. week1).",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        default="data",
        help="Path to the data store (default: data).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory for CSV files (default: data/reports/).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    store = ExperimentStore(args.data_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else store.data_dir / "reports"
    )

    experiments = store.list_experiments(
        pattern=args.experiment,
        test_group=args.test_group,
    )

    if not experiments:
        print("No experiments matched the filters.", file=sys.stderr)
        return 1

    print(f"Found {len(experiments)} experiment(s)")

    report = args.report
    total = 0

    if report in ("static-analysis", "all"):
        n = export_static_analysis(store, experiments, output_dir)
        print(
            f"  Static analysis: {n} CSV(s) written to {output_dir / 'static_analysis'}"
        )
        total += n

    if report in ("performance", "all"):
        n = export_performance(store, experiments, output_dir)
        print(f"  Performance:     {n} CSV(s) written to {output_dir / 'performance'}")
        total += n

    if report in ("merged", "all"):
        n = export_merged(store, experiments, output_dir)
        print(f"  Merged:          {n} CSV(s) written to {output_dir / 'merged'}")
        total += n

    if report in ("composite", "all"):
        n = export_composite_static_analysis(store, experiments, output_dir)
        print(f"  Composite SA:    {n} CSV(s) written to {output_dir / 'composite'}")
        total += n
        n = export_composite_performance(store, experiments, output_dir)
        print(f"  Composite Perf:  {n} CSV(s) written to {output_dir / 'composite'}")
        total += n
        n = export_composite_merged(store, experiments, output_dir)
        print(f"  Composite Merge: {n} CSV(s) written to {output_dir / 'composite'}")
        total += n

    if total == 0:
        print(
            "No data to export (experiments matched but had no analysis/validation data)."
        )
        return 1

    print(f"Done — {total} file(s) total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
