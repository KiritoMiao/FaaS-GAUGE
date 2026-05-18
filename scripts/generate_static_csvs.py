from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
OUTPUT_DIR = DATA_DIR / "reports" / "weekly"

WEEKS = ("week2", "week3")
QUESTIONS = (
    "prime_number_generator",
    "car_position",
    "minimal_cost_split",
)

MODEL_ORDER: list[tuple[str, str]] = [
    ("openai", "gpt-5-mini"),
    ("openai", "gpt-5.2"),
    ("router/openrouter", "anthropic-claude-opus-4.5"),
    ("router/openrouter", "anthropic-claude-sonnet-4.5"),
    ("router/openrouter", "google-gemini-3-flash-preview"),
    ("xai", "grok-4-1-fast-reasoning"),
]

CSV_COLUMNS = [
    "iteration",
    "generate_timestamp",
    "input_token",
    "output_token",
    "rtt_time",
    "token_per_second",
    "error",
    "pylint_score",
    "pylint_fatal",
    "pylint_errors",
    "pylint_warnings",
    "pylint_conventions",
    "pylint_refactors",
    "cc_avg",
    "mi_score",
    "mi_rank",
    "loc",
    "lloc",
    "sloc",
    "comments",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "halstead_bugs",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _discover_experiments() -> dict[tuple[str, str], dict[tuple[str, str], Path]]:
    """Map (week, question) -> {(provider, model): latest_experiment_dir}."""
    discovered: dict[tuple[str, str], dict[tuple[str, str], Path]] = {}

    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        experiment_json = exp_dir / "experiment.json"
        if not experiment_json.exists():
            continue

        try:
            exp = _read_json(experiment_json)
        except json.JSONDecodeError:
            continue

        week = str(exp.get("test_group", ""))
        question = str(exp.get("question_name", ""))
        provider = str(exp.get("provider", ""))

        parts = exp_dir.name.split("+")
        if len(parts) < 4:
            continue
        model = parts[-1]

        if week not in WEEKS or question not in QUESTIONS:
            continue

        key = (week, question)
        model_key = (provider, model)
        discovered.setdefault(key, {})

        prev = discovered[key].get(model_key)
        if prev is None or exp_dir.name > prev.name:
            discovered[key][model_key] = exp_dir

    return discovered


def _build_section_header(question_name: str, provider: str, model: str) -> list[str]:
    provider_slug = provider.replace("/", "-")
    filename = f"{question_name}+{provider_slug}+{model}-mock-copy-2.csv"
    return [filename] + [""] * (len(CSV_COLUMNS) - 1)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for col in CSV_COLUMNS:
        val = row.get(col, "")
        if col == "error":
            normalized[col] = "" if val is None else val
        else:
            normalized[col] = "" if val is None else val
    return normalized


def _parse_iteration(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _write_output_csv(
    output_path: Path, sections: list[tuple[list[str], list[dict[str, Any]]]]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    for header_row, data_rows in sections:
        writer.writerow(header_row)
        writer.writerow(CSV_COLUMNS)
        for row in sorted(data_rows, key=lambda r: int(r.get("iteration", 0))):
            normalized = _normalize_row(row)
            writer.writerow([normalized[col] for col in CSV_COLUMNS])

    output_path.write_text(buf.getvalue(), encoding="utf-8")


def main() -> int:
    discovered = _discover_experiments()
    generated: list[Path] = []

    for week in WEEKS:
        for question in QUESTIONS:
            key = (week, question)
            by_model = discovered.get(key, {})
            sections: list[tuple[list[str], list[dict[str, Any]]]] = []

            for provider, model in MODEL_ORDER:
                exp_dir = by_model.get((provider, model))
                if exp_dir is None:
                    continue

                sa_rows = _read_jsonl(exp_dir / "static_analysis.jsonl")
                row_by_iteration: dict[int, dict[str, Any]] = {}
                for row in sa_rows:
                    parsed = _parse_iteration(row.get("iteration"))
                    if parsed is not None:
                        row_by_iteration[parsed] = row
                padded_rows: list[dict[str, Any]] = []
                for iteration in range(1, 11):
                    if iteration in row_by_iteration:
                        padded_rows.append(row_by_iteration[iteration])
                    else:
                        # Missing iterations are represented as blank-metric rows.
                        padded_rows.append({"iteration": iteration})

                header = _build_section_header(question, provider, model)
                sections.append((header, padded_rows))

            output_path = OUTPUT_DIR / f"{week}_{question}_static.csv"
            _write_output_csv(output_path, sections)
            generated.append(output_path)

    print("Generated static CSV files:")
    for path in generated:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
