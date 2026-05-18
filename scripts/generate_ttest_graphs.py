#!/usr/bin/env python3
"""Generate visualization graphs for t-test significance analysis."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FUNCTIONS = [
    "car_position",
    "distinct_integer_counter",
    "minimal_cost_split",
    "prime_number_generator",
]

MODELS = [
    "GPT-5-mini",
    "GPT-5.2",
    "Claude Opus 4.5",
    "Claude Sonnet 4.5",
    "Gemini 3 Flash",
    "Grok 4.1 Fast",
]

METRICS = [
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

WEEK_PAIRS = [("week1", "week2"), ("week2", "week3"), ("week1", "week3")]
WEEK_PAIR_LABELS = [f"{a} vs {b}" for a, b in WEEK_PAIRS]

MODEL_COLORS = {
    "GPT-5-mini": "#1f77b4",
    "GPT-5.2": "#ff7f0e",
    "Claude Opus 4.5": "#2ca02c",
    "Claude Sonnet 4.5": "#d62728",
    "Gemini 3 Flash": "#9467bd",
    "Grok 4.1 Fast": "#8c564b",
}


def parse_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_bool(value):
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f", ""}:
        return False
    return False


def read_csv_rows(path):
    rows = []
    if not path.exists():
        print(f"Skipping missing CSV: {path}")
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["p_value_num"] = parse_float(row.get("p_value"))
            row["mean_a_num"] = parse_float(row.get("mean_a"))
            row["mean_b_num"] = parse_float(row.get("mean_b"))
            row["mean_diff_num"] = parse_float(row.get("mean_diff"))
            row["significant"] = parse_bool(row.get("significant_at_0.05"))
            rows.append(row)
    return rows


def save_fig(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(str(out_path))


def is_core_row(row):
    return (
        row.get("function") in FUNCTIONS
        and row.get("model") in MODELS
        and row.get("metric") in METRICS
        and (row.get("week_a"), row.get("week_b")) in WEEK_PAIRS
    )


def heatmap(data, row_labels, col_labels, ann, title, out_path):
    fig, ax = plt.subplots(figsize=(1.8 * len(col_labels), 0.6 * len(row_labels) + 2))
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% significant")
    save_fig(fig, out_path)


def plot_significance_heatmap_by_function_metric(rows, out_dir):
    counts = np.zeros((len(METRICS), len(FUNCTIONS)), dtype=int)
    for row in rows:
        if not is_core_row(row):
            continue
        if row["significant"]:
            mi = METRICS.index(row["metric"])
            fi = FUNCTIONS.index(row["function"])
            counts[mi, fi] += 1

    denom = 18
    pct = counts / denom * 100.0
    ann = [
        [f"{counts[i, j]}/{denom}" for j in range(len(FUNCTIONS))]
        for i in range(len(METRICS))
    ]
    heatmap(
        pct,
        METRICS,
        FUNCTIONS,
        ann,
        "Significance % by Function and Metric",
        out_dir / "significance_heatmap_by_function_metric.png",
    )


def plot_significance_heatmap_by_model_metric(rows, out_dir):
    counts = np.zeros((len(METRICS), len(MODELS)), dtype=int)
    for row in rows:
        if not is_core_row(row):
            continue
        if row["significant"]:
            mi = METRICS.index(row["metric"])
            mdi = MODELS.index(row["model"])
            counts[mi, mdi] += 1

    denom = 12
    pct = counts / denom * 100.0
    ann = [
        [f"{counts[i, j]}/{denom}" for j in range(len(MODELS))]
        for i in range(len(METRICS))
    ]
    heatmap(
        pct,
        METRICS,
        MODELS,
        ann,
        "Significance % by Model and Metric",
        out_dir / "significance_heatmap_by_model_metric.png",
    )


def plot_significance_heatmap_by_function_model(rows, out_dir):
    counts = np.zeros((len(MODELS), len(FUNCTIONS)), dtype=int)
    for row in rows:
        if not is_core_row(row):
            continue
        if row["significant"]:
            mdi = MODELS.index(row["model"])
            fi = FUNCTIONS.index(row["function"])
            counts[mdi, fi] += 1

    denom = 30
    pct = counts / denom * 100.0
    ann = [
        [f"{counts[i, j]}/{denom}" for j in range(len(FUNCTIONS))]
        for i in range(len(MODELS))
    ]
    heatmap(
        pct,
        MODELS,
        FUNCTIONS,
        ann,
        "Significance % by Function and Model",
        out_dir / "significance_heatmap_by_function_model.png",
    )


def plot_pvalue_distribution(rows, out_dir):
    pvals = [
        r["p_value_num"]
        for r in rows
        if r["p_value_num"] is not None and is_core_row(r)
    ]
    if not pvals:
        print("Skipping pvalue_distribution.png: no valid p-values")
        return

    sig = sum(1 for p in pvals if p < 0.05)
    nonsig = len(pvals) - sig
    bins = np.arange(0.0, 1.0001, 0.05)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pvals, bins=bins, color="#4c78a8", edgecolor="black", alpha=0.85)
    ax.axvline(0.05, color="red", linestyle="--", linewidth=2, label="α = 0.05")
    ax.set_title("Distribution of p-values across all 720 t-tests")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.text(
        0.60,
        0.95,
        f"Significant (p<0.05): {sig}\nNon-significant: {nonsig}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="upper right")
    save_fig(fig, out_dir / "pvalue_distribution.png")


def plot_significance_by_week_pair(rows, out_dir):
    counts = np.zeros((len(WEEK_PAIRS), len(METRICS)), dtype=int)
    for row in rows:
        if not is_core_row(row):
            continue
        if not row["significant"]:
            continue
        w = (row.get("week_a"), row.get("week_b"))
        wi = WEEK_PAIRS.index(w)
        mi = METRICS.index(row["metric"])
        counts[wi, mi] += 1

    x = np.arange(len(WEEK_PAIRS))
    width = 0.08
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, metric in enumerate(METRICS):
        offset = (i - (len(METRICS) - 1) / 2) * width
        ax.bar(x + offset, counts[:, i], width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(WEEK_PAIR_LABELS)
    ax.set_ylabel("Count significant")
    ax.set_title("Significance by Week Pair and Metric")
    ax.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, out_dir / "significance_by_week_pair.png")


def plot_mean_diff_volcano(rows, out_dir):
    x_vals = []
    y_vals = []
    sig_mask = []

    for row in rows:
        if not is_core_row(row):
            continue
        p = row["p_value_num"]
        mean_a = row["mean_a_num"]
        mean_diff = row["mean_diff_num"]
        if p is None or p <= 0:
            continue
        if mean_a in (None, 0) or mean_diff is None:
            continue
        pct_change = mean_diff / mean_a
        x_vals.append(pct_change)
        y_vals.append(-math.log10(p))
        sig_mask.append(p < 0.05)

    if not x_vals:
        print("Skipping mean_diff_volcano.png: insufficient numeric rows")
        return

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    sig_mask = np.array(sig_mask)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        x_vals[~sig_mask],
        y_vals[~sig_mask],
        color="gray",
        alpha=0.6,
        s=20,
        label="p ≥ 0.05",
    )
    ax.scatter(
        x_vals[sig_mask],
        y_vals[sig_mask],
        color="red",
        alpha=0.7,
        s=22,
        label="p < 0.05",
    )
    ax.axhline(
        -math.log10(0.05),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="-log10(0.05)",
    )
    ax.set_xlabel("Relative mean difference (mean_diff / mean_a)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Volcano Plot: Effect Size vs Significance")
    ax.legend(loc="upper right")
    save_fig(fig, out_dir / "mean_diff_volcano.png")


def _build_trend_values(function_rows, metric):
    values = defaultdict(lambda: defaultdict(list))
    signif = defaultdict(list)

    for row in function_rows:
        if row.get("metric") != metric or row.get("model") not in MODELS:
            continue
        wa, wb = row.get("week_a"), row.get("week_b")
        if (wa, wb) not in WEEK_PAIRS:
            continue
        ma = row["mean_a_num"]
        mb = row["mean_b_num"]
        if ma is not None:
            values[row["model"]][wa].append(ma)
        if mb is not None:
            values[row["model"]][wb].append(mb)
        p = row["p_value_num"]
        if p is not None and p < 0.05:
            signif[(wa, wb)].append(row["model"])

    week_values = defaultdict(dict)
    for model in MODELS:
        for week in ["week1", "week2", "week3"]:
            vals = values[model].get(week, [])
            if vals:
                week_values[model][week] = float(np.mean(vals))

    return week_values, signif


def plot_trends_by_function(rows, out_dir):
    key_metrics = ["total_cost", "performance_avg_ms", "pylint_score", "lloc"]
    x_weeks = ["week1", "week2", "week3"]
    x = np.arange(len(x_weeks))
    pair_to_x = {
        ("week1", "week2"): (0, 1),
        ("week2", "week3"): (1, 2),
        ("week1", "week3"): (0, 2),
    }

    for func in FUNCTIONS:
        function_rows = [
            r for r in rows if r.get("function") == func and is_core_row(r)
        ]
        if not function_rows:
            print(f"Skipping trends_{func}.png: no rows")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()

        for ax, metric in zip(axes, key_metrics):
            week_values, signif = _build_trend_values(function_rows, metric)
            all_y = []

            for model in MODELS:
                ys = [week_values.get(model, {}).get(w, np.nan) for w in x_weeks]
                arr = np.array(ys, dtype=float)
                if np.all(np.isnan(arr)):
                    continue
                ax.plot(
                    x,
                    arr,
                    marker="o",
                    label=model,
                    color=MODEL_COLORS[model],
                    linewidth=1.8,
                )
                all_y.extend([v for v in arr if not np.isnan(v)])

            ax.set_title(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(x_weeks)
            ax.set_ylabel("value")
            ax.grid(alpha=0.25)

            if all_y:
                y_min, y_max = min(all_y), max(all_y)
                span = y_max - y_min if y_max != y_min else max(1.0, abs(y_max) * 0.1)
                ax.set_ylim(y_min - 0.1 * span, y_max + 0.25 * span)
                for pair, models_sig in signif.items():
                    if not models_sig:
                        continue
                    x0, x1 = pair_to_x[pair]
                    y = y_max + 0.08 * span + (x1 - x0) * 0.01 * span
                    ax.plot([x0, x1], [y, y], color="black", linewidth=1)
                    ax.text(
                        (x0 + x1) / 2,
                        y + 0.01 * span,
                        "*",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.suptitle(f"Metric trends across weeks: {func}", fontsize=14)
        save_fig(fig, out_dir / f"trends_{func}.png")


def plot_overall_significance_summary(rows, out_dir):
    core = [r for r in rows if is_core_row(r)]
    if not core:
        print("Skipping overall_significance_summary.png: no core rows")
        return

    def pct_by(category_values, key):
        vals = []
        for c in category_values:
            subset = [r for r in core if key(r) == c]
            if not subset:
                vals.append(0.0)
                continue
            sig = sum(1 for r in subset if r["significant"])
            vals.append(100.0 * sig / len(subset))
        return np.array(vals)

    func_pct = pct_by(FUNCTIONS, lambda r: r.get("function"))
    model_pct = pct_by(MODELS, lambda r: r.get("model"))
    metric_pct = pct_by(METRICS, lambda r: r.get("metric"))
    week_pct = pct_by(
        WEEK_PAIR_LABELS, lambda r: f"{r.get('week_a')} vs {r.get('week_b')}"
    )

    cmap = plt.cm.get_cmap("RdYlGn_r")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        (axes[0, 0], FUNCTIONS, func_pct, "% significant by function"),
        (axes[0, 1], MODELS, model_pct, "% significant by model"),
        (axes[1, 0], METRICS, metric_pct, "% significant by metric"),
        (axes[1, 1], WEEK_PAIR_LABELS, week_pct, "% significant by week pair"),
    ]

    for ax, labels, vals, title in panels:
        y = np.arange(len(labels))
        colors = cmap(vals / 100.0)
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(title)
        ax.set_xlabel("% significant")
        for yi, v in zip(y, vals):
            ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=8)

    save_fig(fig, out_dir / "overall_significance_summary.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate t-test significance visualizations"
    )
    parser.add_argument(
        "--data-dir", default="data", help="Base data directory (default: data)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    analysis_dir = data_dir / "reports" / "t_test_analysis"
    out_dir = analysis_dir / "graphs"

    full_results_path = analysis_dir / "t_test_full_results.csv"
    rows = read_csv_rows(full_results_path)

    for func in FUNCTIONS:
        _ = read_csv_rows(analysis_dir / f"t_test_{func}.csv")

    _ = read_csv_rows(analysis_dir / "t_test_significant_only.csv")

    if not rows:
        print(
            "No rows available in t_test_full_results.csv; skipping graph generation."
        )
        return

    plot_significance_heatmap_by_function_metric(rows, out_dir)
    plot_significance_heatmap_by_model_metric(rows, out_dir)
    plot_significance_heatmap_by_function_model(rows, out_dir)
    plot_pvalue_distribution(rows, out_dir)
    plot_significance_by_week_pair(rows, out_dir)
    plot_mean_diff_volcano(rows, out_dir)
    plot_trends_by_function(rows, out_dir)
    plot_overall_significance_summary(rows, out_dir)


if __name__ == "__main__":
    main()
