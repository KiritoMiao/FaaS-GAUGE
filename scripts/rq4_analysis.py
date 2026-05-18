"""RQ4 Cost & Quality Analysis.

Generates tables, charts, and text summaries for:
  - Todo-2:  Merged data with USD cost columns
  - RQ 4a:   Aggregate generation cost per model
  - RQ 4b:   Per-prompt cost analysis — do expensive models ever produce cheap output?
  - RQ 4c:   Cost vs runtime composite metric (0.5*norm_cost + 0.5*norm_runtime)
  - New RQ:  Input/output/total cost per prompt per model → cost vs quality trade-off
  - Todo-5:  Normalized (0–1) static metrics per prompt, aggregated per model

Usage:
    python scripts/rq4_analysis.py [--data-dir data] [--output-dir data/reports/rq4_analysis]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pricing: USD per 1 million tokens (input, output)
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Key format: "provider+model" matching model_provider in composite CSVs
    # OpenAI
    "openai+gpt-5.2": (1.75, 14.00),
    "openai+gpt-5-mini": (0.25, 2.00),
    # Anthropic (via OpenRouter)
    "router/openrouter+anthropic/claude-opus-4.5": (5.00, 25.00),
    "router/openrouter+anthropic/claude-sonnet-4.5": (3.00, 15.00),
    # Google (via OpenRouter)
    "router/openrouter+google/gemini-3-flash-preview": (0.25, 1.50),
    # xAI
    "xai+grok-4-1-fast-reasoning": (0.20, 0.50),
}

# Short display names for tables/charts
MODEL_SHORT_NAMES: dict[str, str] = {
    "openai+gpt-5.2": "GPT-5.2",
    "openai+gpt-5-mini": "GPT-5-mini",
    "router/openrouter+anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "router/openrouter+anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "router/openrouter+google/gemini-3-flash-preview": "Gemini 3 Flash",
    "xai+grok-4-1-fast-reasoning": "Grok 4.1 Fast",
}

# Static analysis metrics to normalize for Todo-5
STATIC_METRICS = [
    "pylint_score",
    "cc_avg",
    "mi_score",
    "loc",
    "lloc",
    "sloc",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "halstead_bugs",
]

# Metrics where HIGHER = BETTER (don't invert after normalization)
HIGHER_IS_BETTER = {"pylint_score", "mi_score"}
# All others: lower = better → we invert so 0=best, 1=worst... but actually
# we keep consistent: 0 = best after inversion for "lower is better" metrics.
# Actually let's be clear: after normalization, 1 = best for ALL metrics.
# For "higher is better" metrics: normalized = (val - min) / (max - min) → 1 = best
# For "lower is better" metrics: normalized = 1 - (val - min) / (max - min) → 1 = best


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

WEEKS = ["week1", "week2", "week3"]


def load_composite_merged(data_dir: Path) -> list[dict[str, Any]]:
    """Load all composite merged CSVs into a flat list of row dicts."""
    rows: list[dict[str, Any]] = []
    for week in WEEKS:
        csv_path = data_dir / "reports" / "composite" / f"merged_{week}.csv"
        if not csv_path.is_file():
            print(f"  Warning: {csv_path} not found, skipping")
            continue
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_week"] = week
                rows.append(row)
    return rows


def safe_float(val: Any) -> float | None:
    """Parse a value to float, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val: Any) -> int | None:
    """Parse a value to int, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


def compute_cost(row: dict[str, Any]) -> dict[str, float | None]:
    """Compute input_cost, output_cost, total_cost for a row."""
    mp = row.get("model_provider", "")
    pricing = MODEL_PRICING.get(mp)
    input_tokens = safe_int(row.get("input_token"))
    output_tokens = safe_int(row.get("output_token"))

    if pricing is None or input_tokens is None or output_tokens is None:
        return {"input_cost": None, "output_cost": None, "total_cost": None}

    input_price, output_price = pricing
    input_cost = input_tokens * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000
    total_cost = input_cost + output_cost
    return {
        "input_cost": round(input_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": round(total_cost, 8),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    print(f"  Wrote {path}  ({len(rows)} rows)")


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def stdev(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def short_name(model_provider: str) -> str:
    return MODEL_SHORT_NAMES.get(model_provider, model_provider)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Todo-2: Merged with cost columns
# ---------------------------------------------------------------------------

MERGED_COST_EXTRA_COLS = ["input_cost", "output_cost", "total_cost"]


def todo2_merged_with_cost(
    rows: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    """Add cost columns and write per-week CSVs. Returns enriched rows."""
    # Determine original column order from first row
    if not rows:
        return rows

    orig_cols = [k for k in rows[0].keys() if not k.startswith("_")]
    out_cols = orig_cols + MERGED_COST_EXTRA_COLS

    enriched: list[dict[str, Any]] = []
    for row in rows:
        costs = compute_cost(row)
        enriched_row = {**row, **costs}
        enriched.append(enriched_row)

    # Write per-week
    by_week: dict[str, list[dict[str, Any]]] = {}
    for r in enriched:
        by_week.setdefault(r["_week"], []).append(r)

    for week, week_rows in sorted(by_week.items()):
        write_csv(output_dir / f"merged_with_cost_{week}.csv", out_cols, week_rows)

    return enriched


# ---------------------------------------------------------------------------
# RQ 4a: Aggregate cost per model
# ---------------------------------------------------------------------------


def rq4a_cost_summary_by_model(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Aggregate cost per model across all prompts and weeks."""
    # Group by model
    by_model: dict[str, list[float]] = {}
    for r in rows:
        tc = r.get("total_cost")
        if tc is None:
            continue
        mp = r.get("model_provider", "")
        by_model.setdefault(mp, []).append(tc)

    summary_rows: list[dict[str, Any]] = []
    for mp in sorted(by_model, key=lambda m: mean(by_model[m])):
        vals = by_model[mp]
        summary_rows.append(
            {
                "model": short_name(mp),
                "model_provider": mp,
                "n_iterations": len(vals),
                "mean_total_cost": round(mean(vals), 8),
                "stdev_total_cost": round(stdev(vals), 8),
                "median_total_cost": round(median(vals), 8),
                "min_total_cost": round(min(vals), 8),
                "max_total_cost": round(max(vals), 8),
                "sum_total_cost": round(sum(vals), 6),
            }
        )

    cols = [
        "model",
        "model_provider",
        "n_iterations",
        "mean_total_cost",
        "stdev_total_cost",
        "median_total_cost",
        "min_total_cost",
        "max_total_cost",
        "sum_total_cost",
    ]
    write_csv(output_dir / "cost_summary_by_model.csv", cols, summary_rows)

    # Bar chart
    _try_plot_bar(
        output_dir / "rq4a_cost_by_model.png",
        [r["model"] for r in summary_rows],
        [r["mean_total_cost"] for r in summary_rows],
        [r["stdev_total_cost"] for r in summary_rows],
        title="RQ 4a: Mean Generation Cost per Iteration by Model",
        ylabel="Cost (USD)",
    )


# ---------------------------------------------------------------------------
# RQ 4b: Per-prompt cost analysis
# ---------------------------------------------------------------------------


def rq4b_cost_by_prompt_and_model(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Per-prompt per-model cost breakdown with text analysis."""
    # Group by (question, model)
    by_qm: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        tc = r.get("total_cost")
        if tc is None:
            continue
        q = r.get("question_name", "")
        mp = r.get("model_provider", "")
        by_qm.setdefault((q, mp), []).append(tc)

    # Build table
    questions = sorted({q for q, _ in by_qm})
    models = sorted({mp for _, mp in by_qm})

    detail_rows: list[dict[str, Any]] = []
    for q in questions:
        for mp in models:
            vals = by_qm.get((q, mp), [])
            if not vals:
                continue
            detail_rows.append(
                {
                    "question": q,
                    "model": short_name(mp),
                    "model_provider": mp,
                    "n": len(vals),
                    "mean_cost": round(mean(vals), 8),
                    "stdev_cost": round(stdev(vals), 8),
                    "median_cost": round(median(vals), 8),
                    "min_cost": round(min(vals), 8),
                    "max_cost": round(max(vals), 8),
                }
            )

    cols = [
        "question",
        "model",
        "model_provider",
        "n",
        "mean_cost",
        "stdev_cost",
        "median_cost",
        "min_cost",
        "max_cost",
    ]
    write_csv(output_dir / "cost_by_prompt_and_model.csv", cols, detail_rows)

    # Heatmap
    _try_plot_heatmap(
        output_dir / "rq4b_cost_heatmap.png",
        questions,
        models,
        by_qm,
        title="RQ 4b: Mean Cost per Iteration by Prompt × Model",
    )

    # Text analysis
    lines: list[str] = ["RQ 4b: Cost Analysis per Prompt", "=" * 50, ""]
    lines.append(
        "Question: Do more expensive models occasionally produce low-cost output?"
    )
    lines.append(
        "Do the least expensive models always outperform cost-wise on every prompt?"
    )
    lines.append("")

    # For each prompt, rank models by cost
    cheapest_wins: dict[str, int] = {mp: 0 for mp in models}
    for q in questions:
        q_costs = {mp: mean(by_qm[(q, mp)]) for mp in models if (q, mp) in by_qm}
        if not q_costs:
            continue
        ranked = sorted(q_costs.items(), key=lambda x: x[1])
        winner = ranked[0]
        cheapest_wins[winner[0]] = cheapest_wins.get(winner[0], 0) + 1

        lines.append(f"Prompt: {q}")
        for i, (mp, cost) in enumerate(ranked, 1):
            lines.append(f"  {i}. {short_name(mp):30s}  ${cost:.6f}")
        lines.append(f"  → Cheapest: {short_name(winner[0])}")
        lines.append("")

    lines.append("-" * 50)
    lines.append("Summary: Number of prompts each model is cheapest on:")
    for mp in sorted(cheapest_wins, key=lambda m: cheapest_wins[m], reverse=True):
        if cheapest_wins[mp] > 0:
            lines.append(f"  {short_name(mp):30s}  {cheapest_wins[mp]} prompt(s)")

    # Overall cheapest model
    overall_model_costs = {}
    for mp in models:
        all_vals = []
        for q in questions:
            all_vals.extend(by_qm.get((q, mp), []))
        if all_vals:
            overall_model_costs[mp] = mean(all_vals)

    overall_ranked = sorted(overall_model_costs.items(), key=lambda x: x[1])

    lines.append("")
    lines.append("Overall mean cost per iteration (across all prompts):")
    for mp, cost in overall_ranked:
        lines.append(f"  {short_name(mp):30s}  ${cost:.6f}")

    # Conclusion
    cheapest_overall = overall_ranked[0][0] if overall_ranked else ""
    prompt_winners = {mp for mp, c in cheapest_wins.items() if c > 0}

    lines.append("")
    lines.append("Conclusion:")
    if len(prompt_winners) == 1 and cheapest_overall in prompt_winners:
        lines.append(
            f"  {short_name(cheapest_overall)} is the cheapest model on every single prompt."
        )
        lines.append(
            "  More expensive models never produce cheaper output per iteration."
        )
    else:
        lines.append(
            f"  The cheapest model overall ({short_name(cheapest_overall)}) is NOT always cheapest per prompt."
        )
        other_winners = prompt_winners - {cheapest_overall}
        if other_winners:
            lines.append(
                f"  Other models that win on at least one prompt: "
                f"{', '.join(short_name(m) for m in other_winners)}"
            )
        lines.append(
            "  More expensive models CAN occasionally produce lower-cost output on specific prompts."
        )

    write_text(output_dir / "rq4b_analysis.txt", "\n".join(lines))


# ---------------------------------------------------------------------------
# RQ 4c: Composite metric — 0.5*norm_cost + 0.5*norm_runtime
# ---------------------------------------------------------------------------


def rq4c_cost_vs_runtime(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Composite cost-performance metric."""
    # Collect (question, model) → (costs, runtimes)
    by_qm_cost: dict[tuple[str, str], list[float]] = {}
    by_qm_runtime: dict[tuple[str, str], list[float]] = {}

    for r in rows:
        tc = r.get("total_cost")
        rt = safe_float(r.get("performance_avg_ms"))
        q = r.get("question_name", "")
        mp = r.get("model_provider", "")
        if tc is not None:
            by_qm_cost.setdefault((q, mp), []).append(tc)
        if rt is not None and rt > 0:
            by_qm_runtime.setdefault((q, mp), []).append(rt)

    questions = sorted({q for q, _ in by_qm_cost})
    models = sorted({mp for _, mp in by_qm_cost})

    # Compute mean cost and mean runtime per (question, model)
    mean_cost: dict[tuple[str, str], float] = {}
    mean_runtime: dict[tuple[str, str], float] = {}
    for q in questions:
        for mp in models:
            c_vals = by_qm_cost.get((q, mp), [])
            r_vals = by_qm_runtime.get((q, mp), [])
            if c_vals:
                mean_cost[(q, mp)] = mean(c_vals)
            if r_vals:
                mean_runtime[(q, mp)] = mean(r_vals)

    # Normalize per prompt (min-max, lower = better → 0 = best)
    norm_cost: dict[tuple[str, str], float] = {}
    norm_runtime: dict[tuple[str, str], float] = {}

    for q in questions:
        # Cost normalization
        q_costs = [mean_cost[(q, mp)] for mp in models if (q, mp) in mean_cost]
        if len(q_costs) >= 2:
            c_min, c_max = min(q_costs), max(q_costs)
            c_range = c_max - c_min if c_max > c_min else 1.0
            for mp in models:
                if (q, mp) in mean_cost:
                    norm_cost[(q, mp)] = (mean_cost[(q, mp)] - c_min) / c_range
        elif len(q_costs) == 1:
            for mp in models:
                if (q, mp) in mean_cost:
                    norm_cost[(q, mp)] = 0.0

        # Runtime normalization
        q_rts = [mean_runtime[(q, mp)] for mp in models if (q, mp) in mean_runtime]
        if len(q_rts) >= 2:
            r_min, r_max = min(q_rts), max(q_rts)
            r_range = r_max - r_min if r_max > r_min else 1.0
            for mp in models:
                if (q, mp) in mean_runtime:
                    norm_runtime[(q, mp)] = (mean_runtime[(q, mp)] - r_min) / r_range
        elif len(q_rts) == 1:
            for mp in models:
                if (q, mp) in mean_runtime:
                    norm_runtime[(q, mp)] = 0.0

    # Composite score per (question, model): 0.5*norm_cost + 0.5*norm_runtime
    # Lower = better
    composite: dict[tuple[str, str], float] = {}
    for q in questions:
        for mp in models:
            nc = norm_cost.get((q, mp))
            nr = norm_runtime.get((q, mp))
            if nc is not None and nr is not None:
                composite[(q, mp)] = 0.5 * nc + 0.5 * nr

    # Aggregate composite score per model (mean across prompts)
    model_composites: dict[str, list[float]] = {}
    for (q, mp), score in composite.items():
        model_composites.setdefault(mp, []).append(score)

    # Detail table
    detail_rows: list[dict[str, Any]] = []
    for q in questions:
        for mp in models:
            if (q, mp) not in composite:
                continue
            detail_rows.append(
                {
                    "question": q,
                    "model": short_name(mp),
                    "model_provider": mp,
                    "mean_cost_usd": round(mean_cost.get((q, mp), 0), 8),
                    "mean_runtime_ms": round(mean_runtime.get((q, mp), 0), 4),
                    "norm_cost": round(norm_cost.get((q, mp), 0), 6),
                    "norm_runtime": round(norm_runtime.get((q, mp), 0), 6),
                    "composite_score": round(composite[(q, mp)], 6),
                }
            )

    cols = [
        "question",
        "model",
        "model_provider",
        "mean_cost_usd",
        "mean_runtime_ms",
        "norm_cost",
        "norm_runtime",
        "composite_score",
    ]
    write_csv(output_dir / "cost_vs_runtime_detail.csv", cols, detail_rows)

    # Summary table ranked
    summary_rows: list[dict[str, Any]] = []
    for mp in sorted(model_composites, key=lambda m: mean(model_composites[m])):
        vals = model_composites[mp]
        # Also get overall mean cost & runtime for context
        all_costs = []
        all_rts = []
        for q in questions:
            all_costs.extend(by_qm_cost.get((q, mp), []))
            all_rts.extend(by_qm_runtime.get((q, mp), []))
        summary_rows.append(
            {
                "rank": len(summary_rows) + 1,
                "model": short_name(mp),
                "model_provider": mp,
                "mean_composite_score": round(mean(vals), 6),
                "n_prompts": len(vals),
                "mean_cost_usd": round(mean(all_costs), 8) if all_costs else "",
                "mean_runtime_ms": round(mean(all_rts), 4) if all_rts else "",
            }
        )

    summary_cols = [
        "rank",
        "model",
        "model_provider",
        "mean_composite_score",
        "n_prompts",
        "mean_cost_usd",
        "mean_runtime_ms",
    ]
    write_csv(output_dir / "cost_vs_runtime_ranked.csv", summary_cols, summary_rows)

    # Scatter plot
    _try_plot_scatter(
        output_dir / "rq4c_cost_vs_runtime.png",
        detail_rows,
        title="RQ 4c: Cost vs Runtime by Model",
    )

    # Text analysis
    lines = ["RQ 4c: Cost vs Runtime Composite Metric", "=" * 50, ""]
    lines.append("Composite = 0.5 × normalized_cost + 0.5 × normalized_runtime")
    lines.append("Both normalized 0–1 per prompt (min-max). Lower composite = better.")
    lines.append("")
    lines.append("Ranked models (best to worst):")
    for r in summary_rows:
        lines.append(
            f"  #{r['rank']} {r['model']:30s}  "
            f"composite={r['mean_composite_score']:.4f}  "
            f"cost=${r['mean_cost_usd']:.6f}  "
            f"runtime={r['mean_runtime_ms']}ms"
            if r["mean_runtime_ms"]
            else f"  #{r['rank']} {r['model']:30s}  "
            f"composite={r['mean_composite_score']:.4f}  "
            f"cost=${r['mean_cost_usd']:.6f}"
        )

    lines.append("")
    if summary_rows:
        best = summary_rows[0]
        lines.append(
            f"Best overall: {best['model']} with composite score "
            f"{best['mean_composite_score']:.4f}"
        )
        lines.append(
            f"  → Mean cost: ${best['mean_cost_usd']:.6f}/iteration, "
            f"Mean runtime: {best['mean_runtime_ms']}ms"
        )

    write_text(output_dir / "rq4c_analysis.txt", "\n".join(lines))


# ---------------------------------------------------------------------------
# New RQ: Cost vs quality trade-off (input/output/total per prompt per model)
# ---------------------------------------------------------------------------


def new_rq_cost_quality(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Input/output/total cost detail + cost vs quality analysis."""
    by_qm: dict[tuple[str, str], list[dict[str, float | None]]] = {}
    for r in rows:
        q = r.get("question_name", "")
        mp = r.get("model_provider", "")
        by_qm.setdefault((q, mp), []).append(
            {
                "input_cost": r.get("input_cost"),
                "output_cost": r.get("output_cost"),
                "total_cost": r.get("total_cost"),
                "pylint_score": safe_float(r.get("pylint_score")),
                "cc_avg": safe_float(r.get("cc_avg")),
                "mi_score": safe_float(r.get("mi_score")),
            }
        )

    questions = sorted({q for q, _ in by_qm})
    models = sorted({mp for _, mp in by_qm})

    detail_rows: list[dict[str, Any]] = []
    for q in questions:
        for mp in models:
            items = by_qm.get((q, mp), [])
            if not items:
                continue
            ic = [x["input_cost"] for x in items if x["input_cost"] is not None]
            oc = [x["output_cost"] for x in items if x["output_cost"] is not None]
            tc = [x["total_cost"] for x in items if x["total_cost"] is not None]
            ps = [x["pylint_score"] for x in items if x["pylint_score"] is not None]
            cc = [x["cc_avg"] for x in items if x["cc_avg"] is not None]
            mi = [x["mi_score"] for x in items if x["mi_score"] is not None]

            detail_rows.append(
                {
                    "question": q,
                    "model": short_name(mp),
                    "model_provider": mp,
                    "n": len(tc),
                    "mean_input_cost": round(mean(ic), 8) if ic else "",
                    "mean_output_cost": round(mean(oc), 8) if oc else "",
                    "mean_total_cost": round(mean(tc), 8) if tc else "",
                    "mean_pylint_score": round(mean(ps), 4) if ps else "",
                    "mean_cc_avg": round(mean(cc), 4) if cc else "",
                    "mean_mi_score": round(mean(mi), 4) if mi else "",
                }
            )

    cols = [
        "question",
        "model",
        "model_provider",
        "n",
        "mean_input_cost",
        "mean_output_cost",
        "mean_total_cost",
        "mean_pylint_score",
        "mean_cc_avg",
        "mean_mi_score",
    ]
    write_csv(output_dir / "cost_detail_by_prompt_model.csv", cols, detail_rows)

    # Text analysis
    lines = [
        "New RQ: Cost ($) vs Code Quality Trade-off",
        "=" * 50,
        "",
        "This analysis examines whether paying more for code generation",
        "translates to higher code quality (pylint, cyclomatic complexity, MI).",
        "",
    ]

    # Per model: correlation between cost and quality
    for mp in models:
        mp_rows = [r for r in detail_rows if r["model_provider"] == mp]
        costs = [
            r["mean_total_cost"]
            for r in mp_rows
            if isinstance(r["mean_total_cost"], float)
        ]
        pylints = [
            r["mean_pylint_score"]
            for r in mp_rows
            if isinstance(r["mean_pylint_score"], float)
        ]

        lines.append(f"Model: {short_name(mp)}")
        if costs and pylints:
            lines.append(f"  Prompts analyzed: {len(costs)}")
            lines.append(f"  Cost range: ${min(costs):.6f} – ${max(costs):.6f}")
            lines.append(f"  Pylint range: {min(pylints):.2f} – {max(pylints):.2f}")
        lines.append("")

    # Cross-model analysis
    lines.append("-" * 50)
    lines.append("Cross-model comparison:")
    lines.append("")

    model_summary: list[tuple[str, float, float]] = []
    for mp in models:
        mp_items = by_qm_flat = []
        for q in questions:
            mp_items = by_qm.get((q, mp), [])
            by_qm_flat.extend(mp_items)
        tc_vals = [x["total_cost"] for x in by_qm_flat if x["total_cost"] is not None]
        ps_vals = [
            x["pylint_score"] for x in by_qm_flat if x["pylint_score"] is not None
        ]
        if tc_vals and ps_vals:
            model_summary.append((mp, mean(tc_vals), mean(ps_vals)))

    model_summary.sort(key=lambda x: x[1])  # sort by cost
    for mp, cost, pylint in model_summary:
        lines.append(f"  {short_name(mp):30s}  cost=${cost:.6f}  pylint={pylint:.2f}")

    lines.append("")
    if len(model_summary) >= 2:
        cheapest = model_summary[0]
        priciest = model_summary[-1]
        lines.append(
            f"Cheapest model: {short_name(cheapest[0])} "
            f"(${cheapest[1]:.6f}/iter, pylint={cheapest[2]:.2f})"
        )
        lines.append(
            f"Most expensive: {short_name(priciest[0])} "
            f"(${priciest[1]:.6f}/iter, pylint={priciest[2]:.2f})"
        )

        if priciest[2] > cheapest[2]:
            diff_pct = ((priciest[2] - cheapest[2]) / cheapest[2]) * 100
            cost_mult = priciest[1] / cheapest[1] if cheapest[1] > 0 else float("inf")
            lines.append(
                f"\nThe most expensive model costs {cost_mult:.1f}x more but only "
                f"scores {diff_pct:.1f}% higher on pylint."
            )
        elif priciest[2] <= cheapest[2]:
            lines.append(
                "\nThe most expensive model does NOT produce higher pylint scores."
            )
            lines.append(
                "Higher generation cost does not guarantee better code quality."
            )

    write_text(output_dir / "cost_quality_analysis.txt", "\n".join(lines))


# ---------------------------------------------------------------------------
# Todo-5: Normalize static metrics 0–1 per prompt, aggregate per model
# ---------------------------------------------------------------------------


def todo5_normalized_static_metrics(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    """Normalize each static metric 0-1 per prompt, then aggregate per model."""
    # Collect per (question, model) → list of metric values
    by_qm: dict[tuple[str, str], list[dict[str, float | None]]] = {}
    for r in rows:
        q = r.get("question_name", "")
        mp = r.get("model_provider", "")
        metric_vals = {m: safe_float(r.get(m)) for m in STATIC_METRICS}
        by_qm.setdefault((q, mp), []).append(metric_vals)

    questions = sorted({q for q, _ in by_qm})
    models = sorted({mp for _, mp in by_qm})

    # For each metric, produce normalized values and summary
    for metric in STATIC_METRICS:
        higher_better = metric in HIGHER_IS_BETTER

        # Step 1: compute mean per (question, model)
        qm_means: dict[tuple[str, str], float] = {}
        for q in questions:
            for mp in models:
                vals = [
                    x[metric] for x in by_qm.get((q, mp), []) if x[metric] is not None
                ]
                if vals:
                    qm_means[(q, mp)] = mean(vals)

        # Step 2: normalize per prompt (min-max)
        qm_norm: dict[tuple[str, str], float] = {}
        for q in questions:
            q_vals = [qm_means[(q, mp)] for mp in models if (q, mp) in qm_means]
            if len(q_vals) < 2:
                for mp in models:
                    if (q, mp) in qm_means:
                        qm_norm[(q, mp)] = 0.5  # can't normalize with single value
                continue
            v_min, v_max = min(q_vals), max(q_vals)
            v_range = v_max - v_min if v_max > v_min else 1.0
            for mp in models:
                if (q, mp) in qm_means:
                    raw_norm = (qm_means[(q, mp)] - v_min) / v_range
                    if higher_better:
                        qm_norm[(q, mp)] = raw_norm  # 1 = best
                    else:
                        qm_norm[(q, mp)] = 1.0 - raw_norm  # invert: 1 = best

        # Step 3: write normalized detail
        detail_rows: list[dict[str, Any]] = []
        for q in questions:
            for mp in models:
                if (q, mp) not in qm_norm:
                    continue
                detail_rows.append(
                    {
                        "question": q,
                        "model": short_name(mp),
                        "model_provider": mp,
                        f"mean_{metric}": round(qm_means.get((q, mp), 0), 6),
                        f"normalized_{metric}": round(qm_norm[(q, mp)], 6),
                    }
                )

        detail_cols = [
            "question",
            "model",
            "model_provider",
            f"mean_{metric}",
            f"normalized_{metric}",
        ]
        write_csv(
            output_dir / f"{metric}_normalized_by_model.csv", detail_cols, detail_rows
        )

        # Step 4: aggregate per model
        model_norms: dict[str, list[float]] = {}
        for (q, mp), nv in qm_norm.items():
            model_norms.setdefault(mp, []).append(nv)

        summary_rows: list[dict[str, Any]] = []
        for mp in sorted(model_norms, key=lambda m: mean(model_norms[m]), reverse=True):
            vals = model_norms[mp]
            summary_rows.append(
                {
                    "model": short_name(mp),
                    "model_provider": mp,
                    "n_prompts": len(vals),
                    f"mean_normalized_{metric}": round(mean(vals), 6),
                    f"stdev_normalized_{metric}": round(stdev(vals), 6),
                    f"min_normalized_{metric}": round(min(vals), 6),
                    f"max_normalized_{metric}": round(max(vals), 6),
                }
            )

        summary_cols = [
            "model",
            "model_provider",
            "n_prompts",
            f"mean_normalized_{metric}",
            f"stdev_normalized_{metric}",
            f"min_normalized_{metric}",
            f"max_normalized_{metric}",
        ]
        write_csv(
            output_dir / f"{metric}_summary_by_model.csv", summary_cols, summary_rows
        )


# ---------------------------------------------------------------------------
# Plotting helpers (graceful no-op if matplotlib unavailable)
# ---------------------------------------------------------------------------

_HAS_MPL = False
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    _HAS_MPL = True
except ImportError:
    pass


def _try_plot_bar(
    path: Path,
    labels: list[str],
    values: list[float],
    errors: list[float],
    title: str,
    ylabel: str,
) -> None:
    if not _HAS_MPL:
        print(f"  (skipping plot {path.name} — matplotlib not installed)")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2([i / max(len(labels) - 1, 1) for i in range(len(labels))])
    bars = ax.bar(
        labels,
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${val:.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


def _try_plot_heatmap(
    path: Path,
    questions: list[str],
    models: list[str],
    by_qm: dict[tuple[str, str], list[float]],
    title: str,
) -> None:
    if not _HAS_MPL:
        print(f"  (skipping plot {path.name} — matplotlib not installed)")
        return
    import numpy as np

    short_models = [short_name(m) for m in models]
    data = np.zeros((len(questions), len(models)))
    for i, q in enumerate(questions):
        for j, mp in enumerate(models):
            vals = by_qm.get((q, mp), [])
            data[i, j] = mean(vals) if vals else float("nan")

    fig, ax = plt.subplots(figsize=(12, max(6, len(questions) * 0.6)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_models, rotation=40, ha="right")
    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels(questions)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(len(questions)):
        for j in range(len(models)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"${val:.5f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black"
                    if val < data[~np.isnan(data)].max() * 0.7
                    else "white",
                )

    plt.colorbar(im, label="Mean Cost (USD)")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


def _try_plot_scatter(
    path: Path,
    detail_rows: list[dict[str, Any]],
    title: str,
) -> None:
    if not _HAS_MPL:
        print(f"  (skipping plot {path.name} — matplotlib not installed)")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    models_in_data = list({r["model_provider"] for r in detail_rows})
    color_map = {
        mp: plt.cm.Set1(i / max(len(models_in_data) - 1, 1))
        for i, mp in enumerate(models_in_data)
    }
    marker_map = {
        mp: ["o", "s", "^", "D", "v", "P"][i % 6] for i, mp in enumerate(models_in_data)
    }

    for mp in models_in_data:
        mp_rows = [r for r in detail_rows if r["model_provider"] == mp]
        xs = [r["mean_cost_usd"] for r in mp_rows]
        ys = [r["mean_runtime_ms"] for r in mp_rows]
        ax.scatter(
            xs,
            ys,
            c=[color_map[mp]],
            marker=marker_map[mp],
            s=80,
            label=short_name(mp),
            edgecolors="black",
            linewidth=0.5,
            alpha=0.8,
        )

    ax.set_xlabel("Mean Cost per Iteration (USD)")
    ax.set_ylabel("Mean Runtime (ms)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="RQ4 Cost & Quality Analysis")
    parser.add_argument(
        "-d", "--data-dir", default="data", help="Path to the data directory"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: data/reports/rq4_analysis/)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else data_dir / "reports" / "rq4_analysis"
    )

    print("Loading composite merged CSVs...")
    rows = load_composite_merged(data_dir)
    if not rows:
        print("ERROR: No data loaded.", file=sys.stderr)
        return 1
    print(f"  Loaded {len(rows)} rows across {len(WEEKS)} weeks")

    # Todo-2: Add cost columns
    print("\n--- Todo-2: Merged with cost ---")
    enriched = todo2_merged_with_cost(rows, output_dir)

    # RQ 4a
    print("\n--- RQ 4a: Cost summary by model ---")
    rq4a_cost_summary_by_model(enriched, output_dir)

    # RQ 4b
    print("\n--- RQ 4b: Cost by prompt and model ---")
    rq4b_cost_by_prompt_and_model(enriched, output_dir)

    # RQ 4c
    print("\n--- RQ 4c: Cost vs runtime composite ---")
    rq4c_cost_vs_runtime(enriched, output_dir)

    # New RQ
    print("\n--- New RQ: Cost vs quality ---")
    new_rq_cost_quality(enriched, output_dir)

    # Todo-5
    print("\n--- Todo-5: Normalized static metrics ---")
    todo5_normalized_static_metrics(enriched, output_dir)

    print(f"\nDone. All outputs in {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
