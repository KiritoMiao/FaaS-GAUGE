#!/usr/bin/env python3
"""Generate visualization graphs for week4_1 vs week4_2 t-test analysis."""

from __future__ import annotations

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

PERF_METRICS = [
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

ALL_METRICS = STATIC_METRICS + PERF_METRICS + LLM_METRICS

WEEK_PAIRS = [("week4_1", "week4_2")]
WEEK_PAIR_LABELS = ["week4_1 vs week4_2"]

MODEL_COLORS = {
    "GPT-5-mini": "#1f77b4",
    "GPT-5.2": "#ff7f0e",
    "Claude Opus 4.5": "#2ca02c",
    "Claude Sonnet 4.5": "#d62728",
    "Gemini 3 Flash": "#9467bd",
    "Grok 4.1 Fast": "#8c564b",
}

METRIC_CATEGORIES = {}
for m in STATIC_METRICS:
    METRIC_CATEGORIES[m] = "Static"
for m in PERF_METRICS:
    METRIC_CATEGORIES[m] = "Performance"
for m in LLM_METRICS:
    METRIC_CATEGORIES[m] = "LLM"


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
            row["significant"] = str(row.get("significant_at_0.05", "")).strip().upper() == "YES"
            rows.append(row)
    return rows


def save_fig(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def is_core_row(row):
    return (
        row.get("function") in FUNCTIONS
        and row.get("model") in MODELS
        and row.get("metric") in ALL_METRICS
    )


def heatmap(data, row_labels, col_labels, ann, title, out_path, figw_mult=1.8):
    fig, ax = plt.subplots(figsize=(figw_mult * len(col_labels) + 1, 0.5 * len(row_labels) + 2))
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=12)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% significant")
    save_fig(fig, out_path)


# --- Graph 1: Heatmap by Function x Metric ---
def plot_heatmap_function_metric(rows, out_dir):
    counts = np.zeros((len(ALL_METRICS), len(FUNCTIONS)), dtype=int)
    denom = np.zeros((len(ALL_METRICS), len(FUNCTIONS)), dtype=int)
    for row in rows:
        if not is_core_row(row):
            continue
        mi = ALL_METRICS.index(row["metric"])
        fi = FUNCTIONS.index(row["function"])
        denom[mi, fi] += 1
        if row["significant"]:
            counts[mi, fi] += 1

    pct = np.where(denom > 0, counts / denom * 100.0, 0)
    ann = [
        [f"{counts[i, j]}/{denom[i, j]}" for j in range(len(FUNCTIONS))]
        for i in range(len(ALL_METRICS))
    ]
    # Add category prefix to metric labels
    metric_labels = [f"[{METRIC_CATEGORIES[m][:4]}] {m}" for m in ALL_METRICS]
    heatmap(
        pct, metric_labels, FUNCTIONS, ann,
        "week4_1 vs week4_2: Significance % by Function & Metric",
        out_dir / "significance_heatmap_by_function_metric.png",
        figw_mult=2.5,
    )


# --- Graph 2: Heatmap by Model x Metric ---
def plot_heatmap_model_metric(rows, out_dir):
    counts = np.zeros((len(ALL_METRICS), len(MODELS)), dtype=int)
    denom = np.zeros((len(ALL_METRICS), len(MODELS)), dtype=int)
    for row in rows:
        if not is_core_row(row):
            continue
        mi = ALL_METRICS.index(row["metric"])
        mdi = MODELS.index(row["model"])
        denom[mi, mdi] += 1
        if row["significant"]:
            counts[mi, mdi] += 1

    pct = np.where(denom > 0, counts / denom * 100.0, 0)
    ann = [
        [f"{counts[i, j]}/{denom[i, j]}" for j in range(len(MODELS))]
        for i in range(len(ALL_METRICS))
    ]
    metric_labels = [f"[{METRIC_CATEGORIES[m][:4]}] {m}" for m in ALL_METRICS]
    heatmap(
        pct, metric_labels, MODELS, ann,
        "week4_1 vs week4_2: Significance % by Model & Metric",
        out_dir / "significance_heatmap_by_model_metric.png",
    )


# --- Graph 3: Heatmap by Function x Model ---
def plot_heatmap_function_model(rows, out_dir):
    counts = np.zeros((len(MODELS), len(FUNCTIONS)), dtype=int)
    denom = np.zeros((len(MODELS), len(FUNCTIONS)), dtype=int)
    for row in rows:
        if not is_core_row(row):
            continue
        mdi = MODELS.index(row["model"])
        fi = FUNCTIONS.index(row["function"])
        denom[mdi, fi] += 1
        if row["significant"]:
            counts[mdi, fi] += 1

    pct = np.where(denom > 0, counts / denom * 100.0, 0)
    ann = [
        [f"{counts[i, j]}/{denom[i, j]}" for j in range(len(FUNCTIONS))]
        for i in range(len(MODELS))
    ]
    heatmap(
        pct, MODELS, FUNCTIONS, ann,
        "week4_1 vs week4_2: Significance % by Function & Model",
        out_dir / "significance_heatmap_by_function_model.png",
        figw_mult=2.5,
    )


# --- Graph 4: P-value distribution ---
def plot_pvalue_distribution(rows, out_dir):
    pvals = [r["p_value_num"] for r in rows if r["p_value_num"] is not None and is_core_row(r)]
    if not pvals:
        return

    sig = sum(1 for p in pvals if p < 0.05)
    nonsig = len(pvals) - sig
    bins = np.arange(0.0, 1.0001, 0.05)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pvals, bins=bins, color="#4c78a8", edgecolor="black", alpha=0.85)
    ax.axvline(0.05, color="red", linestyle="--", linewidth=2, label="α = 0.05")
    ax.set_title(f"Distribution of p-values: week4_1 vs week4_2 ({len(pvals)} tests)")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.text(
        0.60, 0.95,
        f"Significant (p<0.05): {sig}\nNon-significant: {nonsig}\n({sig/len(pvals)*100:.1f}% significant)",
        transform=ax.transAxes, va="top", ha="left",
        bbox={"facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="upper right")
    save_fig(fig, out_dir / "pvalue_distribution.png")


# --- Graph 5: Volcano plot ---
def plot_volcano(rows, out_dir):
    x_vals, y_vals, sig_mask, labels = [], [], [], []
    for row in rows:
        if not is_core_row(row):
            continue
        p = row["p_value_num"]
        mean_a = row["mean_a_num"]
        mean_diff = row["mean_diff_num"]
        if p is None or p <= 0 or mean_a in (None, 0) or mean_diff is None:
            continue
        pct_change = mean_diff / mean_a
        x_vals.append(pct_change)
        y_vals.append(-math.log10(p))
        sig_mask.append(p < 0.05)
        labels.append(f"{row['function'][:8]}|{row['model'][:10]}|{row['metric'][:12]}")

    if not x_vals:
        return

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    sig_mask = np.array(sig_mask)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(x_vals[~sig_mask], y_vals[~sig_mask], color="gray", alpha=0.5, s=20, label="p >= 0.05")
    ax.scatter(x_vals[sig_mask], y_vals[sig_mask], color="red", alpha=0.7, s=25, label="p < 0.05")
    ax.axhline(-math.log10(0.05), color="black", linestyle="--", linewidth=1.5, label="-log10(0.05)")
    ax.set_xlabel("Relative mean difference (week4_1 - week4_2) / week4_1")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Volcano Plot: Effect Size vs Significance (week4_1 vs week4_2)")
    ax.legend(loc="upper right")
    save_fig(fig, out_dir / "mean_diff_volcano.png")


# --- Graph 6: Overall summary bar charts ---
def plot_overall_summary(rows, out_dir):
    core = [r for r in rows if is_core_row(r)]
    if not core:
        return

    def pct_by(categories, key):
        vals = []
        for c in categories:
            subset = [r for r in core if key(r) == c]
            if not subset:
                vals.append(0.0)
                continue
            sig = sum(1 for r in subset if r["significant"])
            vals.append(100.0 * sig / len(subset))
        return np.array(vals)

    func_pct = pct_by(FUNCTIONS, lambda r: r.get("function"))
    model_pct = pct_by(MODELS, lambda r: r.get("model"))
    # Group metrics by category
    cat_names = ["Static", "Performance", "LLM"]
    cat_pct = pct_by(cat_names, lambda r: METRIC_CATEGORIES.get(r.get("metric", ""), ""))

    cmap = plt.cm.get_cmap("RdYlGn_r")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    panels = [
        (axes[0], FUNCTIONS, func_pct, "% significant by function"),
        (axes[1], MODELS, model_pct, "% significant by model"),
        (axes[2], cat_names, cat_pct, "% significant by metric category"),
    ]

    for ax, labels, vals, title in panels:
        y = np.arange(len(labels))
        colors = cmap(vals / 100.0)
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("% significant")
        for yi, v in zip(y, vals):
            ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=9)

    fig.suptitle("week4_1 vs week4_2: Overall Significance Summary", fontsize=14)
    save_fig(fig, out_dir / "overall_significance_summary.png")


# --- Graph 7: Mean comparison bar charts per function ---
def plot_mean_comparison(rows, out_dir):
    key_metrics = ["pylint_score", "performance_avg_ms", "halstead_effort", "rtt_time"]

    for func in FUNCTIONS:
        func_rows = [r for r in rows if r.get("function") == func and is_core_row(r)]
        if not func_rows:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        for ax, metric in zip(axes, key_metrics):
            metric_rows = [r for r in func_rows if r["metric"] == metric]
            if not metric_rows:
                ax.set_visible(False)
                continue

            models_present = [m for m in MODELS if any(r["model"] == m for r in metric_rows)]
            x = np.arange(len(models_present))
            width = 0.35

            mean_a_vals = []
            mean_b_vals = []
            sig_markers = []
            for model in models_present:
                mr = [r for r in metric_rows if r["model"] == model]
                if mr:
                    mean_a_vals.append(mr[0]["mean_a_num"] or 0)
                    mean_b_vals.append(mr[0]["mean_b_num"] or 0)
                    sig_markers.append(mr[0]["significant"])
                else:
                    mean_a_vals.append(0)
                    mean_b_vals.append(0)
                    sig_markers.append(False)

            bars1 = ax.bar(x - width / 2, mean_a_vals, width, label="week4_1", color="#4c78a8", alpha=0.85)
            bars2 = ax.bar(x + width / 2, mean_b_vals, width, label="week4_2", color="#f58518", alpha=0.85)

            # Mark significant differences
            for i, sig in enumerate(sig_markers):
                if sig:
                    y_max = max(mean_a_vals[i], mean_b_vals[i])
                    ax.text(i, y_max * 1.02, "*", ha="center", va="bottom", fontsize=14, color="red", fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels([m[:15] for m in models_present], rotation=15, ha="right", fontsize=9)
            ax.set_title(f"{metric} [{METRIC_CATEGORIES[metric]}]", fontsize=11)
            ax.set_ylabel("Mean value")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.2)

        fig.suptitle(f"Mean Comparison: {func} (week4_1 vs week4_2)\n* = significant at α=0.05", fontsize=13)
        save_fig(fig, out_dir / f"mean_comparison_{func}.png")


# --- Graph 8: Metric-level detail per function ---
def plot_metric_significance_bars(rows, out_dir):
    core = [r for r in rows if is_core_row(r)]

    fig, ax = plt.subplots(figsize=(14, 8))
    metric_labels = [f"[{METRIC_CATEGORIES[m][:4]}] {m}" for m in ALL_METRICS]
    x = np.arange(len(ALL_METRICS))
    width = 0.25

    for i, func in enumerate(FUNCTIONS):
        sig_counts = []
        for metric in ALL_METRICS:
            subset = [r for r in core if r["function"] == func and r["metric"] == metric]
            sig = sum(1 for r in subset if r["significant"])
            total = len(subset) if subset else 1
            sig_counts.append(100 * sig / total)
        offset = (i - 1) * width
        ax.bar(x + offset, sig_counts, width, label=func, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% significant (across 6 models)")
    ax.set_title("Significance Rate by Metric and Function (week4_1 vs week4_2)")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.grid(alpha=0.15, axis="y")
    save_fig(fig, out_dir / "significance_by_metric_function.png")


def main():
    parser = argparse.ArgumentParser(description="Generate week4 t-test graphs")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    analysis_dir = data_dir / "reports" / "t_test_week4"
    out_dir = analysis_dir / "graphs"

    full_results_path = analysis_dir / "t_test_week4_full_results.csv"
    rows = read_csv_rows(full_results_path)

    if not rows:
        print("No rows in t_test_week4_full_results.csv; aborting.")
        return

    print(f"Loaded {len(rows)} t-test results. Generating graphs...")

    plot_heatmap_function_metric(rows, out_dir)
    plot_heatmap_model_metric(rows, out_dir)
    plot_heatmap_function_model(rows, out_dir)
    plot_pvalue_distribution(rows, out_dir)
    plot_volcano(rows, out_dir)
    plot_overall_summary(rows, out_dir)
    plot_mean_comparison(rows, out_dir)
    plot_metric_significance_bars(rows, out_dir)

    print(f"\nDone. All graphs in {out_dir}/")


if __name__ == "__main__":
    main()
