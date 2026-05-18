from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

MODEL_DISPLAY = {
    "gpt-5-mini": "GPT-5-mini",
    "gpt-5.2": "GPT-5.2",
    "anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "grok-4-1-fast-reasoning": "Grok 4.1 Fast",
}

MODEL_COLORS = {
    "GPT-5-mini": "#9C755F",
    "GPT-5.2": "#F28E2B",
    "Claude Opus 4.5": "#E15759",
    "Claude Sonnet 4.5": "#B07AA1",
    "Gemini 3 Flash": "#59A14F",
    "Grok 4.1 Fast": "#4E79A7",
}

MODEL_ORDER = [
    "GPT-5-mini",
    "GPT-5.2",
    "Claude Opus 4.5",
    "Claude Sonnet 4.5",
    "Gemini 3 Flash",
    "Grok 4.1 Fast",
]

WEEKS = ["week1", "week2", "week3"]
WEEK_LABELS = ["Week 1", "Week 2", "Week 3"]


def read_csv_rows(path: Path) -> Optional[List[Dict[str, str]]]:
    if not path.exists():
        print(f"[WARN] Missing CSV, skipping: {path}")
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(str(output_path))


def parse_float(val: str) -> Optional[float]:
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def display_name(raw_model: str) -> str:
    return MODEL_DISPLAY.get(raw_model, raw_model)


def prettify_question(name: str) -> str:
    return name.replace("_", " ").title()


def compute_mean_memory(
    rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, float]]:
    """Return {question: {display_model: mean_mb}}."""
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        mem = parse_float(row.get("performance_max_ram_mb", ""))
        if mem is None:
            continue
        q = row["question_name"]
        m = display_name(row["model_name"])
        acc[q][m].append(mem)
    return {
        q: {m: np.mean(vals) for m, vals in models.items()}
        for q, models in acc.items()
    }


def generate_week1_bar(base_dir: Path, output_dir: Path) -> None:
    rows = read_csv_rows(base_dir / "merged_week1.csv")
    if rows is None:
        return

    mem = compute_mean_memory(rows)
    questions = sorted(mem.keys())
    models = [m for m in MODEL_ORDER if any(m in mem[q] for q in questions)]

    x = np.arange(len(questions))
    n = len(models)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(14, len(questions) * 1.3), 6))

    for i, model in enumerate(models):
        vals = [mem[q].get(model, 0) for q in questions]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=model,
            color=MODEL_COLORS.get(model, "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

    ax.set_xlabel("Question")
    ax.set_ylabel("Mean Memory Utilization (MB)")
    ax.set_title(
        "Memory Utilization by Model — Week 1 (lower is better)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [prettify_question(q) for q in questions], rotation=35, ha="right", fontsize=9
    )
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.set_ylim(bottom=0)

    save_figure(fig, output_dir / "memory_utilization_week1.png")


MULTIWEEK_QUESTIONS = ["car_position", "minimal_cost_split"]


def generate_multiweek_line(
    base_dir: Path, output_dir: Path, question: str,
) -> None:
    model_week_mem: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for week in WEEKS:
        rows = read_csv_rows(base_dir / f"merged_{week}.csv")
        if rows is None:
            continue
        for row in rows:
            if row["question_name"] != question:
                continue
            mem = parse_float(row.get("performance_max_ram_mb", ""))
            if mem is None:
                continue
            m = display_name(row["model_name"])
            model_week_mem[m][week].append(mem)

    models = [m for m in MODEL_ORDER if m in model_week_mem]
    if not models:
        print(f"[WARN] No memory data for {question} across weeks")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in models:
        means = []
        valid_weeks = []
        for wi, week in enumerate(WEEKS):
            vals = model_week_mem[model].get(week, [])
            if vals:
                means.append(np.mean(vals))
                valid_weeks.append(wi)
        if means:
            ax.plot(
                valid_weeks,
                means,
                marker="o",
                label=model,
                color=MODEL_COLORS.get(model, "#999999"),
                linewidth=2,
                markersize=7,
            )
            for xp, yp in zip(valid_weeks, means):
                ax.annotate(
                    f"{yp:.1f}",
                    (xp, yp),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=7,
                )

    ax.set_xlabel("Week")
    ax.set_ylabel("Mean Memory Utilization (MB)")
    ax.set_title(
        f"Memory Utilization Over Time — {question} (lower is better)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(range(len(WEEKS)))
    ax.set_xticklabels(WEEK_LABELS)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    save_figure(fig, output_dir / f"memory_utilization_multiweek_{question}.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate memory utilization graphs"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/reports/composite"),
        help="Directory containing merged CSV files (default: data/reports/composite)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reports"),
        help="Directory to save output PNGs (default: data/reports)",
    )
    args = parser.parse_args()

    generate_week1_bar(args.data_dir, args.output_dir)
    for question in MULTIWEEK_QUESTIONS:
        generate_multiweek_line(args.data_dir, args.output_dir, question)


if __name__ == "__main__":
    main()
