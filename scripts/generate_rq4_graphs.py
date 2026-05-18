#!/usr/bin/env python3
"""Generate RQ4 publication-quality graphs from CSV summaries.

Usage:
    python3 scripts/generate_rq4_graphs.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


plt.style.use("seaborn-v0_8-whitegrid")

MODEL_ORDER = [
    "Grok 4.1 Fast",
    "Gemini 3 Flash",
    "GPT-5-mini",
    "GPT-5.2",
    "Claude Sonnet 4.5",
    "Claude Opus 4.5",
]

MODEL_COLORS = {
    "Grok 4.1 Fast": "#4E79A7",
    "Gemini 3 Flash": "#59A14F",
    "GPT-5-mini": "#9C755F",
    "GPT-5.2": "#F28E2B",
    "Claude Sonnet 4.5": "#B07AA1",
    "Claude Opus 4.5": "#E15759",
}

MODEL_MARKERS = {
    "Grok 4.1 Fast": "o",
    "Gemini 3 Flash": "s",
    "GPT-5-mini": "^",
    "GPT-5.2": "D",
    "Claude Sonnet 4.5": "P",
    "Claude Opus 4.5": "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RQ4 graphs from CSV files")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (default: data)",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> Optional[List[Dict[str, str]]]:
    if not path.exists():
        print(f"[WARN] Missing CSV, skipping dependent graph(s): {path}")
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def order_models(models: Sequence[str]) -> List[str]:
    known = [m for m in MODEL_ORDER if m in models]
    extra = sorted([m for m in models if m not in MODEL_ORDER])
    return known + extra


def color_for_model(model: str) -> str:
    return MODEL_COLORS.get(model, "#7F7F7F")


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(str(output_path))


def graph_cost_by_model(base_dir: Path, out_dir: Path) -> None:
    rows = read_csv_rows(base_dir / "cost_summary_by_model.csv")
    if not rows:
        return

    rows_sorted = sorted(rows, key=lambda r: to_float(r.get("mean_total_cost", "0")))
    models = [r.get("model", "") for r in rows_sorted]
    means = np.array([to_float(r.get("mean_total_cost", "0")) for r in rows_sorted])
    stdevs = np.array([to_float(r.get("stdev_total_cost", "0")) for r in rows_sorted])

    y = np.arange(len(models))
    colors = [color_for_model(m) for m in models]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(y, means, xerr=stdevs, color=colors, edgecolor="black", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Mean Total Cost (USD)")
    ax.set_title("RQ4A: Mean Cost by Model")
    ax.invert_yaxis()

    x_margin = max(means) * 0.04 if len(means) else 0.01
    for bar, val in zip(bars, means):
        ax.text(
            val + x_margin,
            bar.get_y() + bar.get_height() / 2,
            f"${val:.4f}",
            va="center",
            fontsize=9,
        )

    save_figure(fig, out_dir / "rq4a_cost_by_model_bar.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    positive_means = np.where(means <= 0, 1e-9, means)
    bars = ax.barh(
        y, positive_means, xerr=stdevs, color=colors, edgecolor="black", alpha=0.9
    )
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Mean Total Cost (USD, log scale)")
    ax.set_title("RQ4A: Mean Cost by Model (Log Scale)")
    ax.invert_yaxis()

    for bar, val in zip(bars, means):
        ax.text(
            (val if val > 0 else 1e-9) * 1.12,
            bar.get_y() + bar.get_height() / 2,
            f"${val:.4f}",
            va="center",
            fontsize=9,
        )

    save_figure(fig, out_dir / "rq4a_cost_by_model_log.png")


def graph_cost_prompt_model(base_dir: Path, out_dir: Path) -> None:
    rows = read_csv_rows(base_dir / "cost_by_prompt_and_model.csv")
    if not rows:
        return

    prompts = sorted({r.get("question", "") for r in rows})
    models = order_models({r.get("model", "") for r in rows})
    matrix = np.full((len(prompts), len(models)), np.nan)

    p_idx = {p: i for i, p in enumerate(prompts)}
    m_idx = {m: i for i, m in enumerate(models)}
    for r in rows:
        p = r.get("question", "")
        m = r.get("model", "")
        if p in p_idx and m in m_idx:
            matrix[p_idx[p], m_idx[m]] = to_float(
                r.get("mean_cost", "nan"), float("nan")
            )

    fig, ax = plt.subplots(figsize=(12, max(4.5, len(prompts) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(prompts)))
    ax.set_yticklabels(prompts)
    ax.set_title("RQ4B: Mean Cost Heatmap (Prompt × Model)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Cost (USD)")

    for i in range(len(prompts)):
        for j in range(len(models)):
            if not np.isnan(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"${matrix[i, j]:.4f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

    save_figure(fig, out_dir / "rq4b_cost_heatmap.png")

    x = np.arange(len(prompts))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(13, max(5, len(prompts) * 0.7)))
    for i, model in enumerate(models):
        vals = [
            matrix[p_idx[p], m_idx[model]] if model in m_idx else np.nan
            for p in prompts
        ]
        offsets = x - 0.4 + (i + 0.5) * width
        ax.bar(
            offsets,
            vals,
            width=width,
            label=model,
            color=color_for_model(model),
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=25, ha="right")
    ax.set_ylabel("Mean Cost (USD)")
    ax.set_title("RQ4B: Cost Comparison Across Prompts and Models")
    ax.legend(fontsize=8, ncol=2)

    save_figure(fig, out_dir / "rq4b_cost_grouped_bar.png")


def graph_cost_runtime(base_dir: Path, out_dir: Path) -> None:
    detail_rows = read_csv_rows(base_dir / "cost_vs_runtime_detail.csv")
    ranked_rows = read_csv_rows(base_dir / "cost_vs_runtime_ranked.csv")

    if detail_rows:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for i, r in enumerate(detail_rows):
            model = r.get("model", "")
            x = to_float(r.get("mean_cost_usd", "0"))
            y = to_float(r.get("mean_runtime_ms", "0"))
            q = r.get("question", "")
            offset_x = (i % 3 - 1) * 0.0003
            offset_y = ((i // 3) % 3 - 1) * 15
            ax.scatter(
                x,
                y,
                s=70,
                color=color_for_model(model),
                marker=MODEL_MARKERS.get(model, "o"),
                edgecolor="black",
                linewidth=0.4,
                alpha=0.9,
                label=model,
            )
            ax.text(x + offset_x, y + offset_y, q, fontsize=7, alpha=0.9)

        handles, labels = ax.get_legend_handles_labels()
        dedup = {}
        for h, l in zip(handles, labels):
            if l not in dedup:
                dedup[l] = h
        ax.legend(dedup.values(), dedup.keys(), fontsize=8, loc="best")
        ax.set_xlabel("Mean Cost (USD)")
        ax.set_ylabel("Mean Runtime (ms)")
        ax.set_title("RQ4C: Cost vs Runtime by Prompt and Model")
        ax.grid(True, alpha=0.4)
        save_figure(fig, out_dir / "rq4c_scatter.png")

    if ranked_rows:
        ranked_sorted = sorted(
            ranked_rows, key=lambda r: int(to_float(r.get("rank", "999")))
        )
        models = [r.get("model", "") for r in ranked_sorted]
        scores = [to_float(r.get("mean_composite_score", "0")) for r in ranked_sorted]

        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        bars = ax.bar(
            np.arange(len(models)),
            scores,
            color=[color_for_model(m) for m in models],
            edgecolor="black",
        )
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylabel("Mean Composite Score")
        ax.set_title("RQ4C: Composite Cost-Runtime Ranking")

        max_score = max(scores) if scores else 1.0
        for bar, r in zip(bars, ranked_sorted):
            cost = to_float(r.get("mean_cost_usd", "0"))
            runtime = to_float(r.get("mean_runtime_ms", "0"))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_score * 0.02,
                f"${cost:.4f} | {runtime:.0f}ms",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

        save_figure(fig, out_dir / "rq4c_composite_bar.png")


def graph_cost_breakdown_and_quality(base_dir: Path, out_dir: Path) -> None:
    rows = read_csv_rows(base_dir / "cost_detail_by_prompt_model.csv")
    if not rows:
        return

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault(r.get("model", ""), []).append(r)

    models = order_models(grouped.keys())
    mean_in = []
    mean_out = []

    for model in models:
        g = grouped.get(model, [])
        if not g:
            mean_in.append(0.0)
            mean_out.append(0.0)
            continue
        mean_in.append(
            float(np.mean([to_float(x.get("mean_input_cost", "0")) for x in g]))
        )
        mean_out.append(
            float(np.mean([to_float(x.get("mean_output_cost", "0")) for x in g]))
        )

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(x, mean_in, color="#76B7B2", edgecolor="black", label="Input cost")
    ax.bar(
        x,
        mean_out,
        bottom=mean_in,
        color="#F28E2B",
        edgecolor="black",
        label="Output cost",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Average Cost (USD)")
    ax.set_title("RQ4: Input vs Output Cost by Model")
    ax.legend()
    save_figure(fig, out_dir / "rq4_cost_breakdown_stacked.png")

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for model in models:
        g = grouped.get(model, [])
        xvals = [to_float(r.get("mean_total_cost", "0")) for r in g]
        yvals = [to_float(r.get("mean_pylint_score", "0")) for r in g]
        ax.scatter(
            xvals,
            yvals,
            color=color_for_model(model),
            marker=MODEL_MARKERS.get(model, "o"),
            s=60,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            label=model,
        )

    ax.set_xlabel("Mean Total Cost (USD)")
    ax.set_ylabel("Mean Pylint Score")
    ax.set_title("RQ4: Cost vs Quality (Pylint)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, out_dir / "rq4_cost_vs_quality_scatter.png")


def get_metric_matrix(base_dir: Path) -> Tuple[List[str], List[str], np.ndarray]:
    metrics = [
        "pylint_score",
        "cc_avg",
        "mi_score",
        "lloc",
        "halstead_volume",
        "halstead_difficulty",
        "halstead_effort",
        "halstead_bugs",
    ]

    values: Dict[str, Dict[str, float]] = {}
    present_metrics: List[str] = []

    for metric in metrics:
        file_path = base_dir / f"{metric}_summary_by_model.csv"
        rows = read_csv_rows(file_path)
        if not rows:
            continue

        col_candidates = [c for c in rows[0].keys() if c.startswith("mean_normalized_")]
        if not col_candidates:
            print(
                f"[WARN] No normalized mean column in {file_path}, skipping metric '{metric}'."
            )
            continue
        val_col = col_candidates[0]

        present_metrics.append(metric)
        for r in rows:
            model = r.get("model", "")
            values.setdefault(model, {})[metric] = to_float(
                r.get(val_col, "nan"), float("nan")
            )

    models = order_models(values.keys())
    matrix = np.full((len(models), len(present_metrics)), np.nan)
    for i, model in enumerate(models):
        for j, metric in enumerate(present_metrics):
            if metric in values.get(model, {}):
                matrix[i, j] = values[model][metric]

    return models, present_metrics, matrix


def graph_metrics(base_dir: Path, out_dir: Path) -> None:
    models, metrics, matrix = get_metric_matrix(base_dir)
    if len(models) == 0 or len(metrics) == 0:
        print(
            "[WARN] Missing normalized metric files; skipping metric comparison graphs."
        )
        return

    # Radar chart (fallback requested if too complex; here we provide radar).
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, polar=True)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for i, model in enumerate(models):
        vals = np.nan_to_num(matrix[i, :], nan=0.0).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.8, label=model, color=color_for_model(model))
        ax.fill(angles, vals, alpha=0.08, color=color_for_model(model))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticklabels([])
    ax.set_title("TODO5: Normalized Metric Radar by Model", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    save_figure(fig, out_dir / "todo5_normalized_metrics_radar.png")

    fig, ax = plt.subplots(figsize=(12, max(4.8, len(models) * 0.6)))
    im = ax.imshow(
        matrix,
        cmap="RdYlGn",
        aspect="auto",
        vmin=np.nanmin(matrix),
        vmax=np.nanmax(matrix),
    )
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("TODO5: Normalized Metric Comparison Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized Score (higher is better)")

    for i in range(len(models)):
        for j in range(len(metrics)):
            if not np.isnan(matrix[i, j]):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8
                )

    save_figure(fig, out_dir / "todo5_metric_comparison_heatmap.png")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    base_dir = data_dir / "reports" / "rq4_analysis"
    out_dir = base_dir / "graphs"

    graph_cost_by_model(base_dir, out_dir)
    graph_cost_prompt_model(base_dir, out_dir)
    graph_cost_runtime(base_dir, out_dir)
    graph_cost_breakdown_and_quality(base_dir, out_dir)
    graph_metrics(base_dir, out_dir)


if __name__ == "__main__":
    main()
