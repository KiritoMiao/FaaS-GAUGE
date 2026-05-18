"""Statistical significance analysis (t-tests) for the 4 repeated functions.

For each function × model × metric, runs independent two-sample t-tests
across week pairs (week1↔week2, week2↔week3, week1↔week3).

Reports p-values, t-statistics, means, and significance at α=0.05.

Usage:
    python3 scripts/t_test_analysis.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPEATED_FUNCTIONS = [
    "car_position",
    "distinct_integer_counter",
    "minimal_cost_split",
    "prime_number_generator",
]

WEEKS = ["week1", "week2", "week3"]
WEEK_PAIRS = [("week1", "week2"), ("week2", "week3"), ("week1", "week3")]

# Metrics to test
TEST_METRICS = [
    "total_cost",
    "input_cost",
    "output_cost",
    "performance_avg_ms",
    "pylint_score",
    "cc_avg",
    "mi_score",
    "lloc",
    "halstead_volume",
    "halstead_effort",
]

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
# Pure-Python t-test (Welch's, no scipy needed)
# ---------------------------------------------------------------------------


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _var(vals: list[float]) -> float:
    """Sample variance (ddof=1)."""
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return sum((x - m) ** 2 for x in vals) / (len(vals) - 1)


def _stdev(vals: list[float]) -> float:
    return math.sqrt(_var(vals))


def _welch_t_test(a: list[float], b: list[float]) -> dict[str, float | None]:
    """Welch's two-sample t-test (unequal variances).

    Returns dict with t_stat, df, p_value (two-tailed), mean_a, mean_b,
    stdev_a, stdev_b, n_a, n_b, mean_diff.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return {
            "t_stat": None,
            "df": None,
            "p_value": None,
            "mean_a": _mean(a) if a else None,
            "mean_b": _mean(b) if b else None,
            "stdev_a": _stdev(a) if len(a) >= 2 else None,
            "stdev_b": _stdev(b) if len(b) >= 2 else None,
            "n_a": n_a,
            "n_b": n_b,
            "mean_diff": None,
        }

    m_a, m_b = _mean(a), _mean(b)
    v_a, v_b = _var(a), _var(b)
    se = math.sqrt(v_a / n_a + v_b / n_b) if (v_a / n_a + v_b / n_b) > 0 else 0.0

    if se == 0:
        return {
            "t_stat": 0.0,
            "df": n_a + n_b - 2,
            "p_value": 1.0,
            "mean_a": m_a,
            "mean_b": m_b,
            "stdev_a": _stdev(a),
            "stdev_b": _stdev(b),
            "n_a": n_a,
            "n_b": n_b,
            "mean_diff": m_a - m_b,
        }

    t_stat = (m_a - m_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (v_a / n_a + v_b / n_b) ** 2
    denom = (v_a / n_a) ** 2 / (n_a - 1) + (v_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else n_a + n_b - 2

    # Two-tailed p-value via regularized incomplete beta function
    p_value = _t_distribution_two_tailed_p(abs(t_stat), df)

    return {
        "t_stat": round(t_stat, 6),
        "df": round(df, 2),
        "p_value": round(p_value, 8) if p_value is not None else None,
        "mean_a": round(m_a, 8),
        "mean_b": round(m_b, 8),
        "stdev_a": round(_stdev(a), 8),
        "stdev_b": round(_stdev(b), 8),
        "n_a": n_a,
        "n_b": n_b,
        "mean_diff": round(m_a - m_b, 8),
    }


# ---------------------------------------------------------------------------
# P-value computation from t-distribution (pure Python)
# ---------------------------------------------------------------------------


def _log_gamma(x: float) -> float:
    """Lanczos approximation for log(Gamma(x))."""
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
    """Continued fraction for incomplete beta (Lentz's method)."""
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
        # even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # odd step
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
    """Regularized incomplete beta function I_x(a, b)."""
    if x < 0 or x > 1:
        return 0.0
    if x == 0 or x == 1:
        return x
    log_bt = (
        _log_gamma(a + b)
        - _log_gamma(a)
        - _log_gamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_cf(a, b, x) / a
    else:
        return 1.0 - bt * _beta_cf(b, a, 1.0 - x) / b


def _t_distribution_two_tailed_p(t_abs: float, df: float) -> float:
    """Two-tailed p-value from Student's t-distribution."""
    if df <= 0:
        return 1.0
    x = df / (df + t_abs * t_abs)
    p = _betai(df / 2.0, 0.5, x)
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_merged_with_cost(data_dir: Path) -> list[dict[str, Any]]:
    """Load merged_with_cost CSVs from rq4_analysis output."""
    rows: list[dict[str, Any]] = []
    rq4_dir = data_dir / "reports" / "rq4_analysis"
    for week in WEEKS:
        path = rq4_dir / f"merged_with_cost_{week}.csv"
        if not path.is_file():
            print(f"  Warning: {path} not found")
            continue
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_week"] = week
                rows.append(row)
    return rows


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
    """Run t-tests for all function × model × metric × week-pair combos."""
    # Index: (question, model, week) -> {metric: [values]}
    index: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for r in rows:
        q = r.get("question_name", "")
        if q not in REPEATED_FUNCTIONS:
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

    for func in REPEATED_FUNCTIONS:
        for mp in models:
            for metric in TEST_METRICS:
                for w_a, w_b in WEEK_PAIRS:
                    vals_a = index.get((func, mp, w_a), {}).get(metric, [])
                    vals_b = index.get((func, mp, w_b), {}).get(metric, [])
                    result = _welch_t_test(vals_a, vals_b)
                    sig = ""
                    if result["p_value"] is not None:
                        sig = "YES" if result["p_value"] < ALPHA else "no"
                    detail_rows.append(
                        {
                            "function": func,
                            "model": short(mp),
                            "model_provider": mp,
                            "metric": metric,
                            "comparison": f"{w_a} vs {w_b}",
                            "week_a": w_a,
                            "week_b": w_b,
                            "n_a": result["n_a"],
                            "n_b": result["n_b"],
                            "mean_a": result["mean_a"],
                            "mean_b": result["mean_b"],
                            "stdev_a": result["stdev_a"],
                            "stdev_b": result["stdev_b"],
                            "mean_diff": result["mean_diff"],
                            "t_stat": result["t_stat"],
                            "df": result["df"],
                            "p_value": result["p_value"],
                            f"significant_at_{ALPHA}": sig,
                        }
                    )

    detail_cols = [
        "function",
        "model",
        "model_provider",
        "metric",
        "comparison",
        "week_a",
        "week_b",
        "n_a",
        "n_b",
        "mean_a",
        "mean_b",
        "stdev_a",
        "stdev_b",
        "mean_diff",
        "t_stat",
        "df",
        "p_value",
        f"significant_at_{ALPHA}",
    ]
    write_csv(output_dir / "t_test_full_results.csv", detail_cols, detail_rows)

    # --- Summary: only significant results ---
    sig_rows = [r for r in detail_rows if r[f"significant_at_{ALPHA}"] == "YES"]
    write_csv(output_dir / "t_test_significant_only.csv", detail_cols, sig_rows)

    # --- Per-function summary CSV ---
    for func in REPEATED_FUNCTIONS:
        func_rows = [r for r in detail_rows if r["function"] == func]
        write_csv(output_dir / f"t_test_{func}.csv", detail_cols, func_rows)

    # --- Text report ---
    lines = [
        "T-Test Significance Report",
        "=" * 70,
        "",
        f"Functions analyzed: {', '.join(REPEATED_FUNCTIONS)}",
        f"Weeks compared: {', '.join(f'{a} vs {b}' for a, b in WEEK_PAIRS)}",
        f"Models: {', '.join(short(m) for m in models)}",
        f"Metrics tested: {', '.join(TEST_METRICS)}",
        f"Significance level: α = {ALPHA}",
        "",
        f"Total tests run: {len(detail_rows)}",
        f"Significant results: {len(sig_rows)} ({len(sig_rows) / len(detail_rows) * 100:.1f}%)",
        "",
    ]

    # Summary table: count significant per function
    lines.append("-" * 70)
    lines.append("SIGNIFICANCE COUNTS BY FUNCTION")
    lines.append("-" * 70)
    for func in REPEATED_FUNCTIONS:
        func_all = [r for r in detail_rows if r["function"] == func]
        func_sig = [r for r in sig_rows if r["function"] == func]
        lines.append(
            f"\n  {func}: {len(func_sig)}/{len(func_all)} tests significant "
            f"({len(func_sig) / len(func_all) * 100:.1f}%)"
        )
        # Break down by metric
        for metric in TEST_METRICS:
            m_all = [r for r in func_all if r["metric"] == metric]
            m_sig = [r for r in func_sig if r["metric"] == metric]
            if m_sig:
                lines.append(f"    {metric}: {len(m_sig)}/{len(m_all)} significant")
                for r in m_sig:
                    direction = "↑" if (r["mean_diff"] or 0) > 0 else "↓"
                    lines.append(
                        f"      {r['model']:25s} {r['comparison']:15s}  "
                        f"p={r['p_value']:.6f}  "
                        f"mean: {r['mean_a']:.6f} → {r['mean_b']:.6f} {direction}"
                    )

    # Summary by model
    lines.append("")
    lines.append("-" * 70)
    lines.append("SIGNIFICANCE COUNTS BY MODEL")
    lines.append("-" * 70)
    for mp in models:
        mp_all = [r for r in detail_rows if r["model_provider"] == mp]
        mp_sig = [r for r in sig_rows if r["model_provider"] == mp]
        lines.append(
            f"  {short(mp):30s}  {len(mp_sig)}/{len(mp_all)} significant "
            f"({len(mp_sig) / len(mp_all) * 100:.1f}%)"
        )

    # Summary by week pair
    lines.append("")
    lines.append("-" * 70)
    lines.append("SIGNIFICANCE COUNTS BY WEEK PAIR")
    lines.append("-" * 70)
    for w_a, w_b in WEEK_PAIRS:
        comp = f"{w_a} vs {w_b}"
        wp_all = [r for r in detail_rows if r["comparison"] == comp]
        wp_sig = [r for r in sig_rows if r["comparison"] == comp]
        lines.append(
            f"  {comp:20s}  {len(wp_sig)}/{len(wp_all)} significant "
            f"({len(wp_sig) / len(wp_all) * 100:.1f}%)"
        )

    # Summary by metric
    lines.append("")
    lines.append("-" * 70)
    lines.append("SIGNIFICANCE COUNTS BY METRIC")
    lines.append("-" * 70)
    for metric in TEST_METRICS:
        m_all = [r for r in detail_rows if r["metric"] == metric]
        m_sig = [r for r in sig_rows if r["metric"] == metric]
        lines.append(
            f"  {metric:25s}  {len(m_sig)}/{len(m_all)} significant "
            f"({len(m_sig) / len(m_all) * 100:.1f}%)"
        )

    # Key findings
    lines.append("")
    lines.append("-" * 70)
    lines.append("KEY FINDINGS")
    lines.append("-" * 70)

    # Most significant results (lowest p-values)
    computable = [r for r in detail_rows if r["p_value"] is not None]
    top_sig = sorted(computable, key=lambda r: r["p_value"])[:15]
    lines.append("\nTop 15 most significant results (lowest p-values):")
    for i, r in enumerate(top_sig, 1):
        lines.append(
            f"  {i:2d}. {r['function']:28s} {r['model']:25s} "
            f"{r['metric']:22s} {r['comparison']:15s}  p={r['p_value']:.8f}"
        )

    # Non-significant cost comparisons (interesting: cost didn't change)
    cost_nonsig = [
        r
        for r in detail_rows
        if r["metric"] == "total_cost" and r[f"significant_at_{ALPHA}"] == "no"
    ]
    if cost_nonsig:
        lines.append(
            f"\nCost comparisons that are NOT significant ({len(cost_nonsig)}):"
        )
        for r in cost_nonsig:
            lines.append(
                f"  {r['function']:28s} {r['model']:25s} {r['comparison']:15s}  "
                f"p={r['p_value']:.6f}  "
                f"(${r['mean_a']:.6f} vs ${r['mean_b']:.6f})"
            )

    # Overall conclusion
    lines.append("")
    lines.append("-" * 70)
    lines.append("CONCLUSION")
    lines.append("-" * 70)

    pct = len(sig_rows) / len(detail_rows) * 100 if detail_rows else 0
    if pct > 60:
        lines.append(
            f"  {pct:.1f}% of tests show significant differences across weeks."
        )
        lines.append(
            "  Results vary substantially across weeks — week-to-week variation is real."
        )
    elif pct > 30:
        lines.append(
            f"  {pct:.1f}% of tests show significant differences across weeks."
        )
        lines.append(
            "  Moderate week-to-week variation — some metrics are stable, others shift."
        )
    else:
        lines.append(
            f"  Only {pct:.1f}% of tests show significant differences across weeks."
        )
        lines.append(
            "  Results are largely stable across weeks — high reproducibility."
        )

    write_text(output_dir / "t_test_report.txt", "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="T-test significance analysis")
    parser.add_argument(
        "-d", "--data-dir", default="data", help="Path to data directory"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output dir (default: data/reports/t_test_analysis/)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else data_dir / "reports" / "t_test_analysis"
    )

    print("Loading merged-with-cost data...")
    rows = load_merged_with_cost(data_dir)
    if not rows:
        print("ERROR: No data. Run rq4_analysis.py first.", file=sys.stderr)
        return 1
    print(f"  Loaded {len(rows)} rows")

    print("\nRunning t-tests...")
    run_t_tests(rows, output_dir)

    print(f"\nDone. All outputs in {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
