"""T-test comparison: week4_1 vs week4_2 on static analysis + performance metrics.

Loads composite merged CSVs for week4_1 and week4_2, runs Welch's t-test
per (question, model, metric) and outputs results in the same format as
t_test_analysis.py.

Usage:
    python3 scripts/t_test_week4.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from faas_gauge.store import ExperimentStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QUESTIONS = [
    "car_position",
    "minimal_cost_split",
    "prime_number_generator",
]

WEEKS = ["week4_1", "week4_2"]
WEEK_PAIRS = [("week4_1", "week4_2")]

# Static analysis + performance metrics to compare
STATIC_METRICS = [
    "pylint_score",
    "cc_avg",
    "mi_score",
    "lloc",
    "sloc",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "halstead_bugs",
]

PERFORMANCE_METRICS = [
    "performance_avg_ms",
    "performance_total_ms",
    "performance_max_ram_mb",
]

LLM_METRICS = [
    "input_token",
    "output_token",
    "rtt_time",
    "token_per_second",
]

TEST_METRICS = STATIC_METRICS + PERFORMANCE_METRICS + LLM_METRICS

MODEL_SHORT_NAMES = {
    "openai+gpt-5.2": "GPT-5.2",
    "openai+gpt-5-mini": "GPT-5-mini",
    "router/openrouter+anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "router/openrouter+anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "router/openrouter+google/gemini-3-flash-preview": "Gemini 3 Flash",
    "xai+grok-4-1-fast-reasoning": "Grok 4.1 Fast",
}

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Welch's t-test (pure Python, from t_test_analysis.py)
# ---------------------------------------------------------------------------


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _var(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return sum((x - m) ** 2 for x in vals) / (len(vals) - 1)


def _stdev(vals: list[float]) -> float:
    return math.sqrt(_var(vals))


def _log_gamma(x: float) -> float:
    if x <= 0:
        return float("inf")
    coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]
    y = x
    tmp = x + 5.5
    tmp -= (x - 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in coeffs:
        y += 1
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _beta_cf(a: float, b: float, x: float, max_iter: int = 200) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            return h
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x < 0 or x > 1:
        return 0.0
    if x == 0 or x == 1:
        return x
    log_bt = (
        _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_cf(a, b, x) / a
    else:
        return 1.0 - bt * _beta_cf(b, a, 1.0 - x) / b


def _t_distribution_two_tailed_p(t_abs: float, df: float) -> float:
    if df <= 0:
        return 1.0
    x = df / (df + t_abs * t_abs)
    return _betai(df / 2.0, 0.5, x)


def _welch_t_test(a: list[float], b: list[float]) -> dict[str, float | None]:
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return {
            "t_stat": None, "df": None, "p_value": None,
            "mean_a": _mean(a) if a else None,
            "mean_b": _mean(b) if b else None,
            "stdev_a": _stdev(a) if len(a) >= 2 else None,
            "stdev_b": _stdev(b) if len(b) >= 2 else None,
            "n_a": n_a, "n_b": n_b, "mean_diff": None,
        }
    m_a, m_b = _mean(a), _mean(b)
    v_a, v_b = _var(a), _var(b)
    se = math.sqrt(v_a / n_a + v_b / n_b) if (v_a / n_a + v_b / n_b) > 0 else 0.0
    if se == 0:
        return {
            "t_stat": 0.0, "df": n_a + n_b - 2, "p_value": 1.0,
            "mean_a": m_a, "mean_b": m_b,
            "stdev_a": _stdev(a), "stdev_b": _stdev(b),
            "n_a": n_a, "n_b": n_b, "mean_diff": m_a - m_b,
        }
    t_stat = (m_a - m_b) / se
    num = (v_a / n_a + v_b / n_b) ** 2
    denom = (v_a / n_a) ** 2 / (n_a - 1) + (v_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else n_a + n_b - 2
    p_value = _t_distribution_two_tailed_p(abs(t_stat), df)
    return {
        "t_stat": round(t_stat, 6), "df": round(df, 2),
        "p_value": round(p_value, 8) if p_value is not None else None,
        "mean_a": round(m_a, 8), "mean_b": round(m_b, 8),
        "stdev_a": round(_stdev(a), 8), "stdev_b": round(_stdev(b), 8),
        "n_a": n_a, "n_b": n_b, "mean_diff": round(m_a - m_b, 8),
    }


# ---------------------------------------------------------------------------
# Data loading — builds merged rows directly from store (no huge index)
# ---------------------------------------------------------------------------


def safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_merged_data(data_dir: Path, weeks: list[str]) -> list[dict[str, Any]]:
    """Load merged (static + performance) data directly from the store.

    Only loads validation data for the specified weeks' experiments,
    avoiding the memory-heavy full validation index build.
    """
    store = ExperimentStore(str(data_dir))
    all_rows: list[dict[str, Any]] = []

    for week in weeks:
        experiments = store.list_experiments(test_group=week)
        print(f"  {week}: {len(experiments)} experiments")

        # Build a targeted validation index: only load batches that contain
        # experiments from this week
        exp_ids = {e["id"] for e in experiments}
        val_index: dict[tuple[str, int], dict[str, Any]] = {}
        validations_dir = data_dir / "validations"
        if validations_dir.is_dir():
            for batch_dir in sorted(validations_dir.iterdir()):
                runs_path = batch_dir / "runs.jsonl"
                if not runs_path.is_file():
                    continue
                # Quick check: does this batch mention any of our experiments?
                for run in _read_jsonl(runs_path):
                    exp_id = run.get("experiment_id")
                    if exp_id in exp_ids:
                        exp_iter = run.get("experiment_iteration") or run.get("iteration")
                        val_index[(exp_id, exp_iter)] = run

        for exp in experiments:
            exp_id = exp["id"]
            question = exp.get("question_name", "")
            provider = exp.get("provider", "")
            model = exp.get("model_name") or exp.get("model", "")
            mp = f"{provider}+{model}"

            iterations = _read_jsonl(data_dir / "experiments" / exp_id / "iterations.jsonl")
            sa_records = _read_jsonl(data_dir / "experiments" / exp_id / "static_analysis.jsonl")
            sa_by_iter = {r["iteration"]: r for r in sa_records if "iteration" in r}

            for it in iterations:
                seq = it.get("iteration", 0)
                sa = sa_by_iter.get(seq, {})
                val_run = val_index.get((exp_id, seq), {})

                # Extract performance from validation run
                perf_avg = None
                perf_total = None
                perf_ram = None
                if val_run:
                    tests = val_run.get("tests", [])
                    exec_times: list[float] = []
                    max_ram: float | None = None
                    for t in tests:
                        for ex in t.get("executions") or t.get("execution_results") or []:
                            ms = ex.get("execution_time_ms")
                            if ms is not None:
                                exec_times.append(ms)
                            ram = ex.get("max_memory_used_mb")
                            if ram is not None:
                                max_ram = max(max_ram or 0.0, ram)
                    if exec_times:
                        perf_avg = sum(exec_times) / len(exec_times)
                        perf_total = sum(exec_times)
                    perf_ram = max_ram

                # LLM metrics
                input_token = it.get("input_tokens") or it.get("input_token")
                output_token = it.get("output_tokens") or it.get("output_token")
                rtt = it.get("rtt_time") or it.get("rtt_sec") or it.get("duration_s")
                tps = it.get("tokens_per_second") or it.get("token_per_second")
                if tps is None and output_token and rtt and rtt > 0:
                    tps = round(output_token / rtt, 2)

                row = {
                    "_week": week,
                    "question_name": question,
                    "model_provider": mp,
                    "iteration": seq,
                    # LLM
                    "input_token": input_token,
                    "output_token": output_token,
                    "rtt_time": rtt,
                    "token_per_second": tps,
                    # Static analysis
                    "pylint_score": sa.get("pylint_score"),
                    "cc_avg": sa.get("cc_avg"),
                    "mi_score": sa.get("mi_score"),
                    "lloc": sa.get("lloc"),
                    "sloc": sa.get("sloc"),
                    "halstead_volume": sa.get("halstead_volume"),
                    "halstead_difficulty": sa.get("halstead_difficulty"),
                    "halstead_effort": sa.get("halstead_effort"),
                    "halstead_bugs": sa.get("halstead_bugs"),
                    # Performance
                    "performance_avg_ms": perf_avg,
                    "performance_total_ms": perf_total,
                    "performance_max_ram_mb": perf_ram,
                }
                all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# CSV / text helpers
# ---------------------------------------------------------------------------


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    print(f"  Wrote {path}  ({len(rows)} rows)")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Wrote {path}")


def short(mp: str) -> str:
    return MODEL_SHORT_NAMES.get(mp, mp)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_t_tests(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Run t-tests for all question × model × metric for week4_1 vs week4_2."""
    # Index: (question, model, week) -> {metric: [values]}
    index: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for r in rows:
        q = r.get("question_name", "")
        if q not in QUESTIONS:
            continue
        mp = r.get("model_provider", "")
        w = r.get("_week", "")
        key = (q, mp, w)
        if key not in index:
            index[key] = {m: [] for m in TEST_METRICS}
        for metric in TEST_METRICS:
            val = safe_float(r.get(metric))
            if val is not None:
                index[key][metric].append(val)

    models = sorted({mp for _, mp, _ in index})

    # --- Full detail CSV ---
    detail_rows: list[dict[str, Any]] = []
    for func in QUESTIONS:
        for mp in models:
            for metric in TEST_METRICS:
                for w_a, w_b in WEEK_PAIRS:
                    vals_a = index.get((func, mp, w_a), {}).get(metric, [])
                    vals_b = index.get((func, mp, w_b), {}).get(metric, [])
                    result = _welch_t_test(vals_a, vals_b)
                    sig = ""
                    if result["p_value"] is not None:
                        sig = "YES" if result["p_value"] < ALPHA else "no"
                    detail_rows.append({
                        "function": func,
                        "model": short(mp),
                        "model_provider": mp,
                        "metric": metric,
                        "comparison": f"{w_a} vs {w_b}",
                        "week_a": w_a, "week_b": w_b,
                        "n_a": result["n_a"], "n_b": result["n_b"],
                        "mean_a": result["mean_a"], "mean_b": result["mean_b"],
                        "stdev_a": result["stdev_a"], "stdev_b": result["stdev_b"],
                        "mean_diff": result["mean_diff"],
                        "t_stat": result["t_stat"], "df": result["df"],
                        "p_value": result["p_value"],
                        f"significant_at_{ALPHA}": sig,
                    })

    detail_cols = [
        "function", "model", "model_provider", "metric",
        "comparison", "week_a", "week_b",
        "n_a", "n_b", "mean_a", "mean_b", "stdev_a", "stdev_b",
        "mean_diff", "t_stat", "df", "p_value", f"significant_at_{ALPHA}",
    ]
    write_csv(output_dir / "t_test_week4_full_results.csv", detail_cols, detail_rows)

    sig_rows = [r for r in detail_rows if r[f"significant_at_{ALPHA}"] == "YES"]
    write_csv(output_dir / "t_test_week4_significant_only.csv", detail_cols, sig_rows)

    for func in QUESTIONS:
        func_rows = [r for r in detail_rows if r["function"] == func]
        write_csv(output_dir / f"t_test_week4_{func}.csv", detail_cols, func_rows)

    # --- Text report ---
    lines = [
        "T-Test: week4_1 vs week4_2 (Static Analysis + Performance)",
        "=" * 70,
        "",
        f"Questions analyzed: {', '.join(QUESTIONS)}",
        f"Comparison: week4_1 vs week4_2",
        f"Models: {', '.join(short(m) for m in models)}",
        f"Metrics tested: {len(TEST_METRICS)}",
        f"  Static:      {', '.join(STATIC_METRICS)}",
        f"  Performance: {', '.join(PERFORMANCE_METRICS)}",
        f"  LLM:         {', '.join(LLM_METRICS)}",
        f"Significance level: α = {ALPHA}",
        "",
        f"Total tests run: {len(detail_rows)}",
        f"Significant results: {len(sig_rows)} "
        f"({len(sig_rows) / len(detail_rows) * 100:.1f}%)" if detail_rows else "",
        "",
    ]

    # By function
    lines.append("-" * 70)
    lines.append("SIGNIFICANCE COUNTS BY FUNCTION")
    lines.append("-" * 70)
    for func in QUESTIONS:
        func_all = [r for r in detail_rows if r["function"] == func]
        func_sig = [r for r in sig_rows if r["function"] == func]
        lines.append(
            f"\n  {func}: {len(func_sig)}/{len(func_all)} tests significant "
            f"({len(func_sig) / len(func_all) * 100:.1f}%)" if func_all else ""
        )
        for metric in TEST_METRICS:
            m_sig = [r for r in func_sig if r["metric"] == metric]
            if m_sig:
                lines.append(f"    {metric}:")
                for r in m_sig:
                    direction = "↑" if (r["mean_diff"] or 0) > 0 else "↓"
                    lines.append(
                        f"      {r['model']:25s} "
                        f"p={r['p_value']:.6f}  "
                        f"mean: {r['mean_a']:.4f} → {r['mean_b']:.4f} {direction}"
                    )

    # By model
    lines.append("")
    lines.append("-" * 70)
    lines.append("SIGNIFICANCE COUNTS BY MODEL")
    lines.append("-" * 70)
    for mp in models:
        mp_all = [r for r in detail_rows if r["model_provider"] == mp]
        mp_sig = [r for r in sig_rows if r["model_provider"] == mp]
        lines.append(
            f"  {short(mp):30s}  {len(mp_sig)}/{len(mp_all)} significant "
            f"({len(mp_sig) / len(mp_all) * 100:.1f}%)" if mp_all else ""
        )

    # By metric category
    lines.append("")
    lines.append("-" * 70)
    lines.append("SIGNIFICANCE COUNTS BY METRIC")
    lines.append("-" * 70)
    for metric in TEST_METRICS:
        m_all = [r for r in detail_rows if r["metric"] == metric]
        m_sig = [r for r in sig_rows if r["metric"] == metric]
        cat = (
            "STATIC" if metric in STATIC_METRICS
            else "PERF" if metric in PERFORMANCE_METRICS
            else "LLM"
        )
        lines.append(
            f"  [{cat:6s}] {metric:25s}  {len(m_sig)}/{len(m_all)} significant "
            f"({len(m_sig) / len(m_all) * 100:.1f}%)" if m_all else ""
        )

    # Top findings
    lines.append("")
    lines.append("-" * 70)
    lines.append("TOP SIGNIFICANT RESULTS (lowest p-values)")
    lines.append("-" * 70)
    computable = [r for r in detail_rows if r["p_value"] is not None]
    top_sig = sorted(computable, key=lambda r: r["p_value"])[:20]
    for i, r in enumerate(top_sig, 1):
        sig_marker = " ***" if r["p_value"] < 0.001 else " **" if r["p_value"] < 0.01 else " *" if r["p_value"] < 0.05 else ""
        lines.append(
            f"  {i:2d}. {r['function']:28s} {r['model']:25s} "
            f"{r['metric']:25s} p={r['p_value']:.8f}{sig_marker}"
        )

    # Conclusion
    lines.append("")
    lines.append("-" * 70)
    lines.append("CONCLUSION")
    lines.append("-" * 70)
    pct = len(sig_rows) / len(detail_rows) * 100 if detail_rows else 0
    if pct > 60:
        lines.append(f"  {pct:.1f}% of tests show significant differences between week4_1 and week4_2.")
        lines.append("  The two batches show substantial variation — results differ meaningfully.")
    elif pct > 30:
        lines.append(f"  {pct:.1f}% of tests show significant differences between week4_1 and week4_2.")
        lines.append("  Moderate variation — some metrics are stable, others shift between batches.")
    else:
        lines.append(f"  Only {pct:.1f}% of tests show significant differences between week4_1 and week4_2.")
        lines.append("  Results are largely stable — high reproducibility between the two batches.")

    write_text(output_dir / "t_test_week4_report.txt", "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="T-test: week4_1 vs week4_2 (static + performance)"
    )
    parser.add_argument("-d", "--data-dir", default="data", help="Path to data directory")
    parser.add_argument("-o", "--output-dir", default=None, help="Output dir")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else data_dir / "reports" / "t_test_week4"
    )

    print("Loading merged data for week4_1 and week4_2...")
    rows = load_merged_data(data_dir, WEEKS)
    if not rows:
        print("ERROR: No data found.", file=sys.stderr)
        return 1
    print(f"  Loaded {len(rows)} rows total")

    print("\nRunning t-tests (week4_1 vs week4_2)...")
    run_t_tests(rows, output_dir)

    print(f"\nDone. Outputs in {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
