#!/usr/bin/env python3
"""Generate three Week 1 graphs: Correctness, Runtime vs Baseline, Cost of 1M Runs.

Usage:
    python3 scripts/generate_week1_graphs.py [--data-dir data] [--out-dir data/reports/graphs]
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

# --- Constants (consistent with generate_rq4_graphs.py) ---

MODEL_NAME_MAP: dict[str, str] = {
    "openai+gpt-5-mini": "GPT-5-mini",
    "openai+gpt-5.2": "GPT-5.2",
    "router/openrouter+anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "router/openrouter+anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "router/openrouter+google/gemini-3-flash-preview": "Gemini 3 Flash",
    "xai+grok-4-1-fast-reasoning": "Grok 4.1 Fast",
}

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

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "openai+gpt-5.2": (1.75, 14.00),
    "openai+gpt-5-mini": (0.25, 2.00),
    "router/openrouter+anthropic/claude-opus-4.5": (5.00, 25.00),
    "router/openrouter+anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "router/openrouter+google/gemini-3-flash-preview": (0.25, 1.50),
    "xai+grok-4-1-fast-reasoning": (0.20, 0.50),
}

STAT_ROWS = {"", "AVG", "STDEV", "MEDIAN", "CV", "MAX", "MIN", "RANGE", "x Slower"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Week 1 graphs")
    p.add_argument("--data-dir", default="data", help="Base data directory")
    p.add_argument("--out-dir", default="data/reports/graphs", help="Output directory")
    return p.parse_args()


def display_name(raw: str) -> str:
    return MODEL_NAME_MAP.get(raw, raw)


def order_models(models: list[str]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in models]
    unknown = sorted(set(models) - set(MODEL_ORDER))
    return known + unknown


# --- Data loaders ---


def load_aws_detailed(data_dir: Path) -> list[dict[str, Any]]:
    """Load all week1 AWS target-30 detailed CSVs, filtering out summary rows."""
    pattern = str(data_dir / "reports" / "weekly" / "week1_*_aws_target30_avg2_detailed.csv")
    rows: list[dict[str, Any]] = []
    for fp in sorted(glob.glob(pattern)):
        with open(fp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                q = r.get("question", "")
                mp = r.get("model_provider", "")
                if q in STAT_ROWS or mp in STAT_ROWS or not mp:
                    continue
                rows.append(r)
    return rows


def load_merged_tokens(data_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """Load deduplicated (input_token, output_token) per model from merged CSV."""
    path = data_dir / "reports" / "composite" / "merged_week1.csv"
    seen: set[tuple[str, str, str]] = set()
    model_tokens: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["model_provider"], r["question_name"], r["seq"])
            if key in seen:
                continue
            seen.add(key)
            try:
                inp = int(float(r.get("input_token") or "0"))
                out = int(float(r.get("output_token") or "0"))
                if inp > 0 and out > 0:
                    model_tokens[r["model_provider"]].append((inp, out))
            except (ValueError, TypeError):
                pass
    return dict(model_tokens)


# --- Graph 1: Correctness by Model ---


def graph_correctness(aws_rows: list[dict[str, Any]], out_dir: Path) -> None:
    model_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
    for r in aws_rows:
        mp = display_name(r["model_provider"])
        model_stats[mp]["total"] += 1
        if r["function_status"] == "success":
            model_stats[mp]["success"] += 1

    models = order_models(list(model_stats.keys()))
    pcts = [model_stats[m]["success"] / model_stats[m]["total"] * 100 for m in models]
    colors = [MODEL_COLORS.get(m, "#999999") for m in models]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(models, pcts, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    for bar, pct, m in zip(bars, pcts, models):
        s = model_stats[m]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{pct:.1f}%\n({s['success']}/{s['total']})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Correctness Rate (%)", fontsize=11)
    ax.set_title("Correctness by Model — Week 1 (AWS Lambda, 1769 MB)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "week1_correctness_by_model.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# --- Graph 2: Runtime vs Baseline (mean) ---


def graph_runtime_vs_baseline(aws_rows: list[dict[str, Any]], out_dir: Path) -> None:
    # Collect mean runtime per model (successful runs only)
    model_runtimes: dict[str, list[float]] = defaultdict(list)
    for r in aws_rows:
        if r["function_status"] == "success":
            try:
                ms = float(r["avg_of_runs_total_ms"])
                model_runtimes[display_name(r["model_provider"])].append(ms)
            except (ValueError, TypeError):
                pass

    models = order_models(list(model_runtimes.keys()))
    means = [np.mean(model_runtimes[m]) for m in models]

    # Baseline = mean of all models' means
    baseline = float(np.mean(means))
    ratios = [m / baseline for m in means]
    colors = [MODEL_COLORS.get(m, "#999999") for m in models]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(models, ratios, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    # Baseline reference line
    ax.axhline(y=1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.5)

    for bar, ratio, mean_ms in zip(bars, ratios, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{ratio:.2f}x\n({mean_ms / 1000:.1f}s)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Runtime Ratio (vs Baseline)", fontsize=11)
    ax.set_title(
        "Runtime vs Baseline — Week 1 (AWS Lambda, 1769 MB)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, max(ratios) * 1.2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "week1_runtime_vs_baseline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# --- Graph 3: Cost of 1M Runs ---


def graph_cost_1m(model_tokens: dict[str, list[tuple[int, int]]], out_dir: Path) -> None:
    model_costs: dict[str, float] = {}
    for raw_mp, tokens in model_tokens.items():
        pricing = MODEL_PRICING.get(raw_mp)
        if pricing is None:
            continue
        input_price, output_price = pricing
        costs = [
            inp * input_price / 1e6 + out * output_price / 1e6
            for inp, out in tokens
        ]
        avg_cost = np.mean(costs)
        model_costs[display_name(raw_mp)] = avg_cost * 1_000_000  # scale to 1M runs

    models = order_models(list(model_costs.keys()))
    costs = [model_costs[m] for m in models]
    colors = [MODEL_COLORS.get(m, "#999999") for m in models]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(models, costs, color=colors, edgecolor="white", linewidth=0.5, height=0.55)

    for bar, cost in zip(bars, costs):
        ax.text(
            bar.get_width() + max(costs) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"${cost:,.0f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Cost (USD) for 1M Code-Generation Runs", fontsize=11)
    ax.set_title(
        "Cost of 1M Runs — Week 1 (AWS Lambda, 1769 MB)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, max(costs) * 1.25)
    ax.xaxis.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    path = out_dir / "week1_cost_1m_runs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# --- Main ---


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Week 1 data...")
    aws_rows = load_aws_detailed(data_dir)
    model_tokens = load_merged_tokens(data_dir)
    print(f"  AWS rows: {len(aws_rows)}, Token models: {len(model_tokens)}")

    print("\nGenerating graphs...")
    graph_correctness(aws_rows, out_dir)
    graph_runtime_vs_baseline(aws_rows, out_dir)
    graph_cost_1m(model_tokens, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
