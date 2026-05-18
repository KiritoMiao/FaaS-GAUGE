#!/usr/bin/env python3
"""Generate stacked bar charts showing normalized code quality metrics.

For each week, produces a figure with 3 question groups x 6 model bars.
Each bar stacks three min-max-normalized (0-1) metrics:
  - Function Memory Utilization (max RAM MB)
  - Function Runtime (ms)
  - Logical Lines of Code (LLOC)

Y-axis range: 0-3 (sum of three normalized metrics).

Usage:
    python3 scripts/generate_stacked_quality_bars.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEEKS = ["week1", "week2", "week3", "week4_1", "week4_2"]
QUESTIONS = ["prime_number_generator", "car_position", "minimal_cost_split"]
QUESTION_SHORT = {
    "prime_number_generator": "Prime Number",
    "car_position": "Car Position",
    "minimal_cost_split": "Minimal Cost Split",
}

MODELS_ORDER = [
    "openai+gpt-5-mini",
    "openai+gpt-5.2",
    "router/openrouter+anthropic/claude-opus-4.5",
    "router/openrouter+anthropic/claude-sonnet-4.5",
    "router/openrouter+google/gemini-3-flash-preview",
    "xai+grok-4-1-fast-reasoning",
]

MODEL_SHORT = {
    "openai+gpt-5-mini": "GPT-5-mini",
    "openai+gpt-5.2": "GPT-5.2",
    "router/openrouter+anthropic/claude-opus-4.5": "Opus 4.5",
    "router/openrouter+anthropic/claude-sonnet-4.5": "Sonnet 4.5",
    "router/openrouter+google/gemini-3-flash-preview": "Gemini 3",
    "xai+grok-4-1-fast-reasoning": "Grok 4.1",
}

MODEL_COLORS = {
    "openai+gpt-5-mini": "#1f77b4",
    "openai+gpt-5.2": "#ff7f0e",
    "router/openrouter+anthropic/claude-opus-4.5": "#2ca02c",
    "router/openrouter+anthropic/claude-sonnet-4.5": "#d62728",
    "router/openrouter+google/gemini-3-flash-preview": "#9467bd",
    "xai+grok-4-1-fast-reasoning": "#8c564b",
}

# Segment colors for the 3 stacked metrics
SEG_COLORS = {
    "memory": "#3498db",   # blue
    "runtime": "#e67e22",  # orange
    "lloc": "#27ae60",     # green
}

STATIC_HEADER_MAP = {
    "openai+gpt-5-mini": "openai+gpt-5-mini",
    "openai+gpt-5.2": "openai+gpt-5.2",
    "router-openrouter+anthropic-claude-opus-4.5": "router/openrouter+anthropic/claude-opus-4.5",
    "router-openrouter+anthropic-claude-sonnet-4.5": "router/openrouter+anthropic/claude-sonnet-4.5",
    "router-openrouter+google-gemini-3-flash-preview": "router/openrouter+google/gemini-3-flash-preview",
    "xai+grok-4-1-fast-reasoning": "xai+grok-4-1-fast-reasoning",
}

STATS_ROW_LABELS = {"AVG", "STDEV", "MEDIAN", "CV", "MAX", "MIN", "RANGE",
                     "x Slower", "AVERAGE", "x slower"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(val: str) -> float | None:
    if val is None:
        return None
    val = val.strip()
    if not val or val.startswith("="):
        return None
    try:
        f = float(val)
        return f if not math.isnan(f) else None
    except (ValueError, TypeError):
        return None


def min_max_norm(values: list[float]) -> list[float]:
    """Min-max normalize a list to [0, 1]. Returns zeros if all values equal."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    rng = hi - lo
    if rng < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / rng for v in values]


# ---------------------------------------------------------------------------
# Data loaders (self-contained copies for standalone use)
# ---------------------------------------------------------------------------

def _parse_static_header(line: str, question: str) -> str | None:
    line = line.rstrip(",").strip()
    if not line.endswith(".csv"):
        return None
    body = line[:-4]
    prefix = question + "+"
    if not body.startswith(prefix):
        return None
    model_part = body[len(prefix):]
    model_part = re.sub(r"-mock-copy-\d+$", "", model_part)
    return STATIC_HEADER_MAP.get(model_part)


def load_static_lloc(path: Path, question: str) -> dict[str, list[float]]:
    """Parse static CSV → {canonical_provider: [lloc_values]}."""
    result: dict[str, list[float]] = {}
    current: str | None = None
    lloc_idx: int | None = None

    with open(path, newline="", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n\r")
            cells = raw.split(",")
            prov = _parse_static_header(raw, question)
            if prov is not None:
                current = prov
                result.setdefault(prov, [])
                lloc_idx = None
                continue
            if cells and cells[0].strip() == "iteration" and current:
                for i, h in enumerate(cells):
                    if h.strip() == "lloc":
                        lloc_idx = i
                        break
                continue
            first = cells[0].strip() if cells else ""
            if first in STATS_ROW_LABELS or first.startswith("="):
                continue
            if current and lloc_idx is not None and first.isdigit():
                v = safe_float(cells[lloc_idx]) if lloc_idx < len(cells) else None
                if v is not None:
                    result[current].append(v)
    return result


def load_lambda_perf(path: Path) -> dict[str, dict[str, list[float]]]:
    """Parse lambda perf CSV → {provider: {metric: [values]}}."""
    result: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {h.strip(): i for i, h in enumerate(header)}
        mp_idx, seq_idx = col.get("model_provider", 0), col.get("seq", 1)
        t_idx, r_idx = col.get("200m"), col.get("200m_max_ram")
        for row in reader:
            if not row or len(row) <= mp_idx:
                continue
            mp = row[mp_idx].strip()
            if not mp or mp in STATS_ROW_LABELS:
                continue
            sv = row[seq_idx].strip() if seq_idx < len(row) else ""
            if not sv or not sv.replace(".", "").isdigit():
                continue
            if t_idx is not None:
                v = safe_float(row[t_idx]) if t_idx < len(row) else None
                if v is not None:
                    result[mp]["runtime"].append(v)
            if r_idx is not None:
                v = safe_float(row[r_idx]) if r_idx < len(row) else None
                if v is not None:
                    result[mp]["memory"].append(v)
    return dict(result)


def load_aws_target30(path: Path) -> dict[str, dict[str, list[float]]]:
    """Parse AWS target30 CSV → {provider: {metric: [values]}}."""
    result: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {h.strip(): i for i, h in enumerate(header)}
        mp_idx, seq_idx = col.get("model_provider", 1), col.get("seq", 2)
        t_idx = col.get("avg_of_runs_total_ms")
        r_idx = col.get("max_ram_mb")
        for row in reader:
            if not row or len(row) <= mp_idx:
                continue
            mp = row[mp_idx].strip()
            if not mp or mp in STATS_ROW_LABELS:
                continue
            sv = row[seq_idx].strip() if seq_idx < len(row) else ""
            if not sv or not sv.replace(".", "").isdigit():
                continue
            if t_idx is not None:
                v = safe_float(row[t_idx]) if t_idx < len(row) else None
                if v is not None:
                    result[mp]["runtime"].append(v)
            if r_idx is not None:
                v = safe_float(row[r_idx]) if r_idx < len(row) else None
                if v is not None:
                    result[mp]["memory"].append(v)
    return dict(result)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_static(dd: Path, week: str, q: str) -> Path | None:
    p = dd / "reports" / "weekly" / f"{week}_{q}_static.csv"
    return p if p.exists() else None


def find_perf(dd: Path, week: str, q: str) -> Path | None:
    if q == "prime_number_generator":
        for name in [
            f"{week}_prime_number_generator_lambda_performance.csv",
            f"{week}_prime_number_generator_lambda_performance_prime_number_generator_lambda_performance.csv",
        ]:
            p = dd / "reports" / "weekly" / name
            if p.exists():
                return p
        return None
    p = dd / "reports" / "weekly" / f"{week}_{q}_aws_target30_avg2_detailed.csv"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_means(data_dir: Path):
    """Build per-(week, question, model) mean values for the 3 metrics.

    Returns:
        means[week][question][provider] = {"memory": float, "runtime": float, "lloc": float}
    """
    means: dict[str, dict[str, dict[str, dict[str, float | None]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    for week in WEEKS:
        for question in QUESTIONS:
            # LLOC from static CSV
            sp = find_static(data_dir, week, question)
            lloc_data: dict[str, list[float]] = {}
            if sp:
                lloc_data = load_static_lloc(sp, question)

            # Runtime + Memory from performance CSVs
            pp = find_perf(data_dir, week, question)
            perf_data: dict[str, dict[str, list[float]]] = {}
            if pp:
                if question == "prime_number_generator":
                    perf_data = load_lambda_perf(pp)
                else:
                    perf_data = load_aws_target30(pp)

            for provider in MODELS_ORDER:
                lloc_vals = lloc_data.get(provider, [])
                pdata = perf_data.get(provider, {})

                m_lloc = statistics.mean(lloc_vals) if lloc_vals else None
                rt_vals = pdata.get("runtime", [])
                m_runtime = statistics.mean(rt_vals) if rt_vals else None
                mem_vals = pdata.get("memory", [])
                m_memory = statistics.mean(mem_vals) if mem_vals else None

                means[week][question][provider] = {
                    "memory": m_memory,
                    "runtime": m_runtime,
                    "lloc": m_lloc,
                }

    return means


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def plot_week_stacked_bars(
    week: str,
    week_data: dict[str, dict[str, dict[str, float | None]]],
    out_dir: Path,
) -> None:
    """One figure per week: 3 question groups x 6 model stacked bars."""
    n_models = len(MODELS_ORDER)
    n_questions = len(QUESTIONS)
    bar_width = 0.12
    group_gap = 0.4

    fig, ax = plt.subplots(figsize=(16, 7))

    x_ticks = []
    x_labels = []
    group_centers = []

    for qi, question in enumerate(QUESTIONS):
        group_start = qi * (n_models * bar_width + group_gap)

        # Collect raw means for normalization across 6 models within this question
        raw_mem = []
        raw_rt = []
        raw_lloc = []
        for provider in MODELS_ORDER:
            d = week_data.get(question, {}).get(provider, {})
            raw_mem.append(d.get("memory") or 0.0)
            raw_rt.append(d.get("runtime") or 0.0)
            raw_lloc.append(d.get("lloc") or 0.0)

        norm_mem = min_max_norm(raw_mem)
        norm_rt = min_max_norm(raw_rt)
        norm_lloc = min_max_norm(raw_lloc)

        positions = []
        for mi, provider in enumerate(MODELS_ORDER):
            x = group_start + mi * bar_width
            positions.append(x)

            m_val = norm_mem[mi]
            r_val = norm_rt[mi]
            l_val = norm_lloc[mi]

            # Stack: memory (bottom), runtime (middle), lloc (top)
            ax.bar(x, m_val, bar_width * 0.9, bottom=0,
                   color=SEG_COLORS["memory"],
                   edgecolor=MODEL_COLORS[provider], linewidth=1.2)
            ax.bar(x, r_val, bar_width * 0.9, bottom=m_val,
                   color=SEG_COLORS["runtime"],
                   edgecolor=MODEL_COLORS[provider], linewidth=1.2)
            ax.bar(x, l_val, bar_width * 0.9, bottom=m_val + r_val,
                   color=SEG_COLORS["lloc"],
                   edgecolor=MODEL_COLORS[provider], linewidth=1.2)

            x_ticks.append(x)
            x_labels.append(MODEL_SHORT[provider])

        group_centers.append(statistics.mean(positions))

    # Axes
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylim(0, 3.2)
    ax.set_ylabel("Normalized Score (0-3)", fontsize=11)
    ax.set_title(f"Code Quality Profile \u2014 {week}",
                 fontsize=14, fontweight="bold")

    # Question group labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(group_centers)
    ax2.set_xticklabels([QUESTION_SHORT[q] for q in QUESTIONS],
                         fontsize=11, fontweight="bold")
    ax2.tick_params(length=0, pad=8)

    # Group separators
    for qi in range(1, n_questions):
        sep_x = qi * (n_models * bar_width + group_gap) - group_gap / 2
        ax.axvline(x=sep_x, color="gray", linestyle="--", alpha=0.4)

    # Horizontal guide lines
    for y in [1.0, 2.0]:
        ax.axhline(y=y, color="gray", linestyle=":", alpha=0.3)

    # Legend for metric segments
    legend_patches = [
        mpatches.Patch(color=SEG_COLORS["memory"], label="Memory Utilization"),
        mpatches.Patch(color=SEG_COLORS["runtime"], label="Function Runtime"),
        mpatches.Patch(color=SEG_COLORS["lloc"], label="Logical LOC"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
              framealpha=0.9)

    ax.grid(axis="y", alpha=0.2)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"quality_stack_{week}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / f'quality_stack_{week}.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stacked bar charts for code quality metrics")
    parser.add_argument("--data-dir", default="data",
                        help="Path to data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "reports" / "cv_analysis"

    print("Loading weekly data...")
    means = build_means(data_dir)

    print("Generating stacked bar charts...")
    for week in WEEKS:
        plot_week_stacked_bars(week, means[week], out_dir)

    print(f"\nDone \u2014 {len(WEEKS)} chart(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
