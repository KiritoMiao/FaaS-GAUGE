#!/usr/bin/env python3
"""Generate RQ3 CV (Coefficient of Variation) analysis graphs.

Computes CV = (stdev / |mean|) * 100 from raw 10-iteration data per
(model, metric, question, week) and produces:
  - 6 per-model CV profile heatmaps
  - 1 LLOC variability line chart across weeks (aggregated)
  - 3 per-question LLOC variability line charts (raw)
  - 3 per-question normalized LLOC variability charts (0-1, 1 = best)

Usage:
    python3 scripts/generate_rq3_cv_graphs.py [--data-dir data]
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

MODELS_ORDER = [
    "openai+gpt-5-mini",
    "openai+gpt-5.2",
    "router/openrouter+anthropic/claude-opus-4.5",
    "router/openrouter+anthropic/claude-sonnet-4.5",
    "router/openrouter+google/gemini-3-flash-preview",
    "xai+grok-4-1-fast-reasoning",
]

MODEL_SHORT_NAMES = {
    "openai+gpt-5-mini": "GPT-5-mini",
    "openai+gpt-5.2": "GPT-5.2",
    "router/openrouter+anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "router/openrouter+anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "router/openrouter+google/gemini-3-flash-preview": "Gemini 3 Flash",
    "xai+grok-4-1-fast-reasoning": "Grok 4.1 Fast",
}

MODEL_COLORS = {
    "GPT-5-mini": "#1f77b4",
    "GPT-5.2": "#ff7f0e",
    "Claude Opus 4.5": "#2ca02c",
    "Claude Sonnet 4.5": "#d62728",
    "Gemini 3 Flash": "#9467bd",
    "Grok 4.1 Fast": "#8c564b",
}

MODEL_MARKERS = {
    "GPT-5-mini": "o",
    "GPT-5.2": "s",
    "Claude Opus 4.5": "^",
    "Claude Sonnet 4.5": "D",
    "Gemini 3 Flash": "v",
    "Grok 4.1 Fast": "P",
}

# Mapping from static CSV header model fragments to canonical provider keys.
# Static headers use - instead of / and strip mock-copy suffixes.
STATIC_HEADER_MAP = {
    "openai+gpt-5-mini": "openai+gpt-5-mini",
    "openai+gpt-5.2": "openai+gpt-5.2",
    "router-openrouter+anthropic-claude-opus-4.5": "router/openrouter+anthropic/claude-opus-4.5",
    "router-openrouter+anthropic-claude-sonnet-4.5": "router/openrouter+anthropic/claude-sonnet-4.5",
    "router-openrouter+google-gemini-3-flash-preview": "router/openrouter+google/gemini-3-flash-preview",
    "xai+grok-4-1-fast-reasoning": "xai+grok-4-1-fast-reasoning",
}

STATIC_METRICS = [
    "pylint_score", "cc_avg", "mi_score", "lloc", "sloc",
    "halstead_volume", "halstead_difficulty", "halstead_effort", "halstead_bugs",
]
LLM_METRICS = ["input_token", "output_token", "rtt_time", "token_per_second"]
ALL_STATIC_LLM_METRICS = STATIC_METRICS + LLM_METRICS

STATS_ROW_LABELS = {"AVG", "STDEV", "MEDIAN", "CV", "MAX", "MIN", "RANGE",
                     "x Slower", "AVERAGE", "x slower"}

# Display names for heatmap rows
METRIC_DISPLAY = {
    "pylint_score": "Pylint Score",
    "cc_avg": "Cyclomatic Complexity",
    "mi_score": "Maintainability Index",
    "lloc": "Logical LOC",
    "sloc": "Source LOC",
    "halstead_volume": "Halstead Volume",
    "halstead_difficulty": "Halstead Difficulty",
    "halstead_effort": "Halstead Effort",
    "halstead_bugs": "Halstead Bugs",
    "input_token": "Input Tokens",
    "output_token": "Output Tokens",
    "rtt_time": "RTT Time (s)",
    "token_per_second": "Tokens/sec",
    "prime_200m_time": "Prime 200m Time (ms)",
    "prime_200m_ram": "Prime 200m RAM (MB)",
    "car_avg_time_ms": "Car Avg Time (ms)",
    "car_max_ram_mb": "Car Max RAM (MB)",
    "minimal_avg_time_ms": "Minimal Avg Time (ms)",
    "minimal_max_ram_mb": "Minimal Max RAM (MB)",
}

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


def compute_cv(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if len(clean) < 2:
        return None
    m = statistics.mean(clean)
    if abs(m) < 1e-12:
        return None
    return (statistics.stdev(clean) / abs(m)) * 100.0


def model_slug(display_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", display_name.lower()).strip("_")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _parse_static_header(line: str, question: str) -> str | None:
    """Extract canonical model provider from a static CSV model-header line."""
    line = line.rstrip(",").strip()
    if not line.endswith(".csv"):
        return None
    # Strip .csv
    body = line[:-4]
    # Strip question prefix (e.g., "prime_number_generator+")
    prefix = question + "+"
    if not body.startswith(prefix):
        return None
    model_part = body[len(prefix):]
    # Strip optional -mock-copy-N suffix
    model_part = re.sub(r"-mock-copy-\d+$", "", model_part)
    return STATIC_HEADER_MAP.get(model_part)


def load_static_csv(path: Path, question: str) -> dict[str, dict[str, list[float]]]:
    """Parse a multi-section static CSV → {canonical_provider: {metric: [values]}}."""
    result: dict[str, dict[str, list[float]]] = {}
    current_provider: str | None = None
    col_indices: dict[str, int] = {}

    with open(path, newline="", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.rstrip("\n\r")
            cells = raw_line.split(",")

            # Try to detect a model-header line
            provider = _parse_static_header(raw_line, question)
            if provider is not None:
                current_provider = provider
                if provider not in result:
                    result[provider] = {m: [] for m in ALL_STATIC_LLM_METRICS}
                col_indices = {}
                continue

            # Column header line
            if cells and cells[0].strip() == "iteration" and current_provider:
                for i, h in enumerate(cells):
                    h = h.strip()
                    if h in ALL_STATIC_LLM_METRICS:
                        col_indices[h] = i
                continue

            # Skip formula/stats rows
            first = cells[0].strip() if cells else ""
            if first in STATS_ROW_LABELS or first.startswith("="):
                continue

            # Data row: iteration number in first cell
            if current_provider and col_indices and first.isdigit():
                for metric, idx in col_indices.items():
                    val = safe_float(cells[idx]) if idx < len(cells) else None
                    if val is not None:
                        result[current_provider][metric].append(val)

    return result


def load_lambda_perf_csv(path: Path) -> dict[str, dict[str, list[float]]]:
    """Parse a lambda performance CSV → {canonical_provider: {metric: [values]}}."""
    result: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_map: dict[str, int] = {}
        for i, h in enumerate(header):
            h = h.strip()
            col_map[h] = i

        mp_idx = col_map.get("model_provider", 0)
        # We want 200m timing and 200m_max_ram
        time_idx = col_map.get("200m")
        ram_idx = col_map.get("200m_max_ram")

        for row in reader:
            if not row or len(row) <= mp_idx:
                continue
            mp = row[mp_idx].strip()
            if not mp or mp in STATS_ROW_LABELS:
                continue
            # Check seq is numeric (data row, not stats)
            seq_idx = col_map.get("seq", 1)
            seq_val = row[seq_idx].strip() if seq_idx < len(row) else ""
            if not seq_val or not seq_val.replace(".", "").isdigit():
                continue

            if time_idx is not None:
                v = safe_float(row[time_idx]) if time_idx < len(row) else None
                if v is not None:
                    result[mp]["prime_200m_time"].append(v)
            if ram_idx is not None:
                v = safe_float(row[ram_idx]) if ram_idx < len(row) else None
                if v is not None:
                    result[mp]["prime_200m_ram"].append(v)

    return dict(result)


def load_aws_target30_csv(path: Path, question: str) -> dict[str, dict[str, list[float]]]:
    """Parse an AWS target30 CSV → {canonical_provider: {metric: [values]}}."""
    prefix = "car" if question == "car_position" else "minimal"
    time_key = f"{prefix}_avg_time_ms"
    ram_key = f"{prefix}_max_ram_mb"

    result: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_map = {h.strip(): i for i, h in enumerate(header)}

        mp_idx = col_map.get("model_provider", 1)
        seq_idx = col_map.get("seq", 2)
        time_idx = col_map.get("avg_of_runs_total_ms")
        ram_idx = col_map.get("max_ram_mb")

        for row in reader:
            if not row or len(row) <= mp_idx:
                continue
            mp = row[mp_idx].strip()
            if not mp or mp in STATS_ROW_LABELS:
                continue
            seq_val = row[seq_idx].strip() if seq_idx < len(row) else ""
            if not seq_val or not seq_val.replace(".", "").isdigit():
                continue

            if time_idx is not None:
                v = safe_float(row[time_idx]) if time_idx < len(row) else None
                if v is not None:
                    result[mp][time_key].append(v)
            if ram_idx is not None:
                v = safe_float(row[ram_idx]) if ram_idx < len(row) else None
                if v is not None:
                    result[mp][ram_key].append(v)

    return dict(result)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_static_csv(data_dir: Path, week: str, question: str) -> Path | None:
    p = data_dir / "reports" / "weekly" / f"{week}_{question}_static.csv"
    return p if p.exists() else None


def find_lambda_perf_csv(data_dir: Path, week: str) -> Path | None:
    for name in [
        f"{week}_prime_number_generator_lambda_performance.csv",
        f"{week}_prime_number_generator_lambda_performance_prime_number_generator_lambda_performance.csv",
    ]:
        p = data_dir / "reports" / "weekly" / name
        if p.exists():
            return p
    return None


def find_aws_target30_csv(data_dir: Path, week: str, question: str) -> Path | None:
    p = data_dir / "reports" / "weekly" / f"{week}_{question}_aws_target30_avg2_detailed.csv"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_cv_table(data_dir: Path):
    """Build CV and raw-data tables from all weekly CSVs.

    Returns:
        cv_data:  {display_model: {metric: {week: cv_float|None}}}
        raw_data: {display_model: {metric: {week: [values]}}}
    """
    # Nested defaultdicts for accumulation
    # per_question stores per-(provider, metric, week, question) raw values
    per_q: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for week in WEEKS:
        # --- Static + LLM metrics (all 3 questions) ---
        for question in QUESTIONS:
            path = find_static_csv(data_dir, week, question)
            if path is None:
                continue
            data = load_static_csv(path, question)
            for provider, metrics in data.items():
                for metric, values in metrics.items():
                    per_q[provider][metric][week][question].extend(values)

        # --- Lambda performance (prime only) ---
        path = find_lambda_perf_csv(data_dir, week)
        if path:
            data = load_lambda_perf_csv(path)
            for provider, metrics in data.items():
                for metric, values in metrics.items():
                    per_q[provider][metric][week]["prime_number_generator"].extend(values)

        # --- AWS target30 (car + minimal) ---
        for question in ["car_position", "minimal_cost_split"]:
            path = find_aws_target30_csv(data_dir, week, question)
            if path is None:
                continue
            data = load_aws_target30_csv(path, question)
            for provider, metrics in data.items():
                for metric, values in metrics.items():
                    per_q[provider][metric][week][question].extend(values)

    # Build CV table: for static/LLM metrics, average CV across questions;
    # for performance metrics, use question-specific CV directly.
    cv_data: dict[str, dict[str, dict[str, float | None]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    raw_data: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    perf_metrics = {
        "prime_200m_time", "prime_200m_ram",
        "car_avg_time_ms", "car_max_ram_mb",
        "minimal_avg_time_ms", "minimal_max_ram_mb",
    }

    for provider in MODELS_ORDER:
        display = MODEL_SHORT_NAMES[provider]

        for metric in ALL_STATIC_LLM_METRICS:
            for week in WEEKS:
                # Collect values across all questions
                all_vals: list[float] = []
                cvs: list[float] = []
                for question in QUESTIONS:
                    vals = per_q[provider][metric][week].get(question, [])
                    all_vals.extend(vals)
                    cv = compute_cv(vals)
                    if cv is not None:
                        cvs.append(cv)
                raw_data[display][metric][week] = all_vals
                # Average CV across questions
                cv_data[display][metric][week] = (
                    statistics.mean(cvs) if cvs else None
                )

        for metric in perf_metrics:
            for week in WEEKS:
                # Performance metrics are question-specific; determine question
                if metric.startswith("prime"):
                    q = "prime_number_generator"
                elif metric.startswith("car"):
                    q = "car_position"
                else:
                    q = "minimal_cost_split"
                vals = per_q[provider][metric][week].get(q, [])
                raw_data[display][metric][week] = vals
                cv_data[display][metric][week] = compute_cv(vals)

    # Per-question raw LLOC for per-question variability charts
    per_q_lloc: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for provider in MODELS_ORDER:
        display = MODEL_SHORT_NAMES[provider]
        for question in QUESTIONS:
            for week in WEEKS:
                vals = per_q[provider]["lloc"][week].get(question, [])
                per_q_lloc[display][question][week] = vals

    return cv_data, raw_data, per_q_lloc


# ---------------------------------------------------------------------------
# Heatmap rows definition
# ---------------------------------------------------------------------------

HEATMAP_ROWS = [
    # (metric_key, display_name, category)
    ("pylint_score", "Pylint Score", "Static Analysis"),
    ("cc_avg", "Cyclomatic Complexity", "Static Analysis"),
    ("mi_score", "Maintainability Index", "Static Analysis"),
    ("lloc", "Logical LOC", "Static Analysis"),
    ("sloc", "Source LOC", "Static Analysis"),
    ("halstead_volume", "Halstead Volume", "Static Analysis"),
    ("halstead_difficulty", "Halstead Difficulty", "Static Analysis"),
    ("halstead_effort", "Halstead Effort", "Static Analysis"),
    ("halstead_bugs", "Halstead Bugs", "Static Analysis"),
    ("input_token", "Input Tokens", "LLM"),
    ("output_token", "Output Tokens", "LLM"),
    ("rtt_time", "RTT Time (s)", "LLM"),
    ("token_per_second", "Tokens/sec", "LLM"),
    ("prime_200m_time", "Prime 200m Time (ms)", "Performance"),
    ("prime_200m_ram", "Prime 200m RAM (MB)", "Performance"),
    ("car_avg_time_ms", "Car Avg Time (ms)", "Performance"),
    ("car_max_ram_mb", "Car Max RAM (MB)", "Performance"),
    ("minimal_avg_time_ms", "Minimal Avg Time (ms)", "Performance"),
    ("minimal_max_ram_mb", "Minimal Max RAM (MB)", "Performance"),
]


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def plot_model_cv_heatmap(
    model_display: str,
    cv_model: dict[str, dict[str, float | None]],
    out_dir: Path,
) -> None:
    """Generate a per-model heatmap: rows = metrics, columns = weeks."""
    n_rows = len(HEATMAP_ROWS)
    n_cols = len(WEEKS)

    matrix = np.full((n_rows, n_cols), np.nan)
    for ri, (metric_key, _, _) in enumerate(HEATMAP_ROWS):
        for ci, week in enumerate(WEEKS):
            val = cv_model.get(metric_key, {}).get(week)
            if val is not None:
                matrix[ri, ci] = val

    fig, ax = plt.subplots(figsize=(10, 12))

    # Use masked array so NaN cells render as gray
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.colormaps["RdYlGn_r"]
    cmap.set_bad(color="#e0e0e0")

    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    # Axes labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(WEEKS, fontsize=10)
    ax.set_yticks(range(n_rows))
    row_labels = [row[1] for row in HEATMAP_ROWS]
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate cells
    for ri in range(n_rows):
        for ci in range(n_cols):
            val = matrix[ri, ci]
            if np.isnan(val):
                ax.text(ci, ri, "N/A", ha="center", va="center",
                        fontsize=8, color="#999999")
            else:
                color = "white" if val > 60 else "black"
                ax.text(ci, ri, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)

    # Category separators
    prev_cat = HEATMAP_ROWS[0][2]
    for ri, (_, _, cat) in enumerate(HEATMAP_ROWS):
        if cat != prev_cat:
            ax.axhline(y=ri - 0.5, color="black", linewidth=1.5)
            prev_cat = cat

    # Category labels on the left (outside the graph)
    categories = []
    start = 0
    prev_cat = HEATMAP_ROWS[0][2]
    for ri, (_, _, cat) in enumerate(HEATMAP_ROWS):
        if cat != prev_cat:
            categories.append((prev_cat, start, ri - 1))
            start = ri
            prev_cat = cat
    categories.append((prev_cat, start, n_rows - 1))

    for cat_name, s, e in categories:
        mid_y = (s + e) / 2
        ax.text(-0.7, mid_y, cat_name, ha="right", va="center",
                fontsize=9, fontweight="bold", color="#555555",
                transform=ax.get_yaxis_transform())

    # RQ3 region highlight
    ax.axvspan(3 - 0.5, 4 + 0.5, alpha=0.08, color="blue")
    ax.text(3.5, -0.8, "RQ3 region", ha="center", va="center",
            fontsize=9, color="blue", fontstyle="italic")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.12)
    cbar.set_label("CV (%)", fontsize=10)

    ax.set_title(f"RQ3: CV Profile \u2014 {model_display}",
                 fontsize=14, fontweight="bold", pad=15)

    save_figure(fig, out_dir / f"cv_profile_{model_slug(model_display)}.png")


def plot_lloc_variability(
    raw_data: dict[str, dict[str, dict[str, list[float]]]],
    out_dir: Path,
) -> None:
    """Line chart: mean LLOC per model across weeks, with error bands."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x_pos = np.arange(len(WEEKS))

    for display in [MODEL_SHORT_NAMES[p] for p in MODELS_ORDER]:
        means = []
        stds = []
        for week in WEEKS:
            vals = raw_data.get(display, {}).get("lloc", {}).get(week, [])
            if vals:
                means.append(statistics.mean(vals))
                stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
            else:
                means.append(np.nan)
                stds.append(0)

        means_arr = np.array(means, dtype=float)
        stds_arr = np.array(stds, dtype=float)

        color = MODEL_COLORS[display]
        marker = MODEL_MARKERS[display]

        ax.plot(x_pos, means_arr, marker=marker, label=display,
                color=color, linewidth=2, markersize=7, zorder=3)
        ax.fill_between(x_pos,
                        means_arr - stds_arr,
                        means_arr + stds_arr,
                        alpha=0.15, color=color, zorder=2)

    # RQ3 region
    ax.axvspan(2.5, 4.5, alpha=0.08, color="blue", zorder=1)
    ax.axvline(x=2.5, color="blue", linestyle="--", alpha=0.4, zorder=1)
    ylim = ax.get_ylim()
    ax.text(3.5, ylim[1] * 0.97, "RQ3 region", ha="center", va="top",
            fontsize=10, color="blue", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="blue", alpha=0.7))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(WEEKS, fontsize=11)
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Mean LLOC (across 3 questions)", fontsize=12)
    ax.set_title("Average Logical LOC by Model Across Weeks",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    save_figure(fig, out_dir / "lloc_variability_across_weeks.png")


def plot_lloc_variability_per_question(
    per_q_lloc: dict[str, dict[str, dict[str, list[float]]]],
    out_dir: Path,
) -> None:
    """Per-question line charts: mean LLOC per model across weeks, with error bands."""
    for question in QUESTIONS:
        fig, ax = plt.subplots(figsize=(12, 7))
        x_pos = np.arange(len(WEEKS))

        for display in [MODEL_SHORT_NAMES[p] for p in MODELS_ORDER]:
            means = []
            stds = []
            for week in WEEKS:
                vals = per_q_lloc.get(display, {}).get(question, {}).get(week, [])
                if vals:
                    means.append(statistics.mean(vals))
                    stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
                else:
                    means.append(np.nan)
                    stds.append(0)

            means_arr = np.array(means, dtype=float)
            stds_arr = np.array(stds, dtype=float)

            color = MODEL_COLORS[display]
            marker = MODEL_MARKERS[display]

            ax.plot(x_pos, means_arr, marker=marker, label=display,
                    color=color, linewidth=2, markersize=7, zorder=3)
            ax.fill_between(x_pos,
                            means_arr - stds_arr,
                            means_arr + stds_arr,
                            alpha=0.15, color=color, zorder=2)

        # RQ3 region
        ax.axvspan(2.5, 4.5, alpha=0.08, color="blue", zorder=1)
        ax.axvline(x=2.5, color="blue", linestyle="--", alpha=0.4, zorder=1)
        ylim = ax.get_ylim()
        ax.text(3.5, ylim[1] * 0.97, "RQ3 region", ha="center", va="top",
                fontsize=10, color="blue", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="blue", alpha=0.7))

        q_display = question.replace("_", " ").title()
        ax.set_xticks(x_pos)
        ax.set_xticklabels(WEEKS, fontsize=11)
        ax.set_xlabel("Week", fontsize=12)
        ax.set_ylabel("Mean LLOC", fontsize=12)
        ax.set_title(f"Logical LOC Variability \u2014 {q_display}",
                     fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.25)

        save_figure(fig, out_dir / f"lloc_variability_{question}.png")


def plot_lloc_variability_per_question_normalized(
    per_q_lloc: dict[str, dict[str, dict[str, list[float]]]],
    out_dir: Path,
) -> None:
    """Per-question normalized LLOC (0-1, direction-aware: 1 = fewest lines = best)."""
    for question in QUESTIONS:
        # Collect all mean LLOC values across all models and weeks for global min/max
        all_means: list[float] = []
        model_means: dict[str, list[float]] = {}
        model_stds: dict[str, list[float]] = {}

        for display in [MODEL_SHORT_NAMES[p] for p in MODELS_ORDER]:
            means = []
            stds = []
            for week in WEEKS:
                vals = per_q_lloc.get(display, {}).get(question, {}).get(week, [])
                if vals:
                    m = statistics.mean(vals)
                    means.append(m)
                    stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
                    all_means.append(m)
                else:
                    means.append(np.nan)
                    stds.append(0)
            model_means[display] = means
            model_stds[display] = stds

        clean_means = [v for v in all_means if not math.isnan(v)]
        if len(clean_means) < 2:
            continue

        lo = min(clean_means)
        hi = max(clean_means)
        rng = hi - lo

        fig, ax = plt.subplots(figsize=(12, 7))
        x_pos = np.arange(len(WEEKS))

        for display in [MODEL_SHORT_NAMES[p] for p in MODELS_ORDER]:
            raw_m = model_means[display]
            raw_s = model_stds[display]

            if rng < 1e-12:
                norm_m = [0.5 if not math.isnan(v) else np.nan for v in raw_m]
                norm_s = [0.0] * len(raw_s)
            else:
                norm_m = []
                norm_s = []
                for m, s in zip(raw_m, raw_s):
                    if math.isnan(m):
                        norm_m.append(np.nan)
                        norm_s.append(0)
                    else:
                        norm_m.append(1.0 - (m - lo) / rng)
                        norm_s.append(s / rng)

            means_arr = np.array(norm_m, dtype=float)
            stds_arr = np.array(norm_s, dtype=float)

            color = MODEL_COLORS[display]
            marker = MODEL_MARKERS[display]

            ax.plot(x_pos, means_arr, marker=marker, label=display,
                    color=color, linewidth=2, markersize=7, zorder=3)
            ax.fill_between(x_pos,
                            means_arr - stds_arr,
                            means_arr + stds_arr,
                            alpha=0.15, color=color, zorder=2)

        # RQ3 region
        ax.axvspan(2.5, 4.5, alpha=0.08, color="blue", zorder=1)
        ax.axvline(x=2.5, color="blue", linestyle="--", alpha=0.4, zorder=1)
        ax.text(3.5, 0.97, "RQ3 region", ha="center", va="top",
                fontsize=10, color="blue", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="blue", alpha=0.7))

        q_display = question.replace("_", " ").title()
        ax.set_xticks(x_pos)
        ax.set_xticklabels(WEEKS, fontsize=11)
        ax.set_xlabel("Week", fontsize=12)
        ax.set_ylabel("Normalized LLOC (1 = fewest lines = best)", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Normalized Logical LOC \u2014 {q_display}",
                     fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.25)

        save_figure(fig, out_dir / f"lloc_variability_{question}_normalized.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 CV analysis graphs")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "reports" / "cv_analysis"

    print("Loading weekly data and computing CV...")
    cv_data, raw_data, per_q_lloc = build_cv_table(data_dir)

    print(f"\nGenerating per-model CV heatmaps...")
    for provider in MODELS_ORDER:
        display = MODEL_SHORT_NAMES[provider]
        cv_model = cv_data.get(display, {})
        plot_model_cv_heatmap(display, cv_model, out_dir)

    print(f"\nGenerating LLOC variability plot (aggregated)...")
    plot_lloc_variability(raw_data, out_dir)

    print(f"\nGenerating per-question LLOC variability plots...")
    plot_lloc_variability_per_question(per_q_lloc, out_dir)

    print(f"\nGenerating per-question normalized LLOC plots...")
    plot_lloc_variability_per_question_normalized(per_q_lloc, out_dir)

    n_graphs = len(MODELS_ORDER) + 1 + len(QUESTIONS) * 2
    print(f"\nDone \u2014 {n_graphs} graph(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
