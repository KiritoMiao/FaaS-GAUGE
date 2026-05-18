"""Add generate_timestamp column to all existing *_static.csv files in data/reports/weekly/.

Reads timestamps from data/experiments/*/static_analysis.jsonl and inserts a
generate_timestamp column after the iteration column in each static CSV file.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "data" / "experiments"
WEEKLY_DIR = ROOT / "data" / "reports" / "weekly"

TIMESTAMP_COL = "generate_timestamp"


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


def _build_timestamp_index() -> dict[tuple[str, str, str, str, int], str]:
    """Build (week, question, provider_slug, model, iteration) -> timestamp."""
    index: dict[tuple[str, str, str, str, int], str] = {}

    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_json = exp_dir / "experiment.json"
        if not exp_json.exists():
            continue
        try:
            exp = json.loads(exp_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        week = str(exp.get("test_group", "") or "")
        question = str(exp.get("question_name", "") or "")
        provider = str(exp.get("provider", "") or "")
        # provider slug: router/openrouter -> router-openrouter
        provider_slug = provider.replace("/", "-")

        parts = exp_dir.name.split("+")
        if len(parts) < 4:
            continue
        model = parts[-1]

        sa_rows = _read_jsonl(exp_dir / "static_analysis.jsonl")
        for row in sa_rows:
            ts = row.get("generate_timestamp", "")
            iteration = row.get("iteration")
            if iteration is None:
                continue
            try:
                iteration = int(iteration)
            except (ValueError, TypeError):
                continue
            if ts:
                key = (week, question, provider_slug, model, iteration)
                # Keep latest experiment (sorted order means last wins)
                index[key] = str(ts)

    return index


def _parse_section_header(header_cell: str) -> tuple[str, str, str] | None:
    """Parse 'question+provider_slug+model[-mock-copy-N].csv' -> (question, provider_slug, model)."""
    name = header_cell.rstrip()
    if not name.endswith(".csv") or "+" not in name:
        return None
    name = name[:-4]  # strip .csv
    # Strip -mock-copy-N suffix
    name = re.sub(r"-mock-copy-\d+$", "", name)
    parts = name.split("+")
    if len(parts) < 3:
        return None
    question = parts[0]
    provider_slug = parts[1]
    model = "+".join(parts[2:])  # model may not contain +, but just in case
    return question, provider_slug, model


def _extract_week(filename: str) -> str:
    """Extract week from filename like 'week1_car_position_static.csv'."""
    m = re.match(r"(week\d+)_", filename)
    return m.group(1) if m else ""


def _is_section_header(cells: list[str]) -> bool:
    """Check if row is a section header (first cell is a filename, rest are empty)."""
    if not cells:
        return False
    first = cells[0].strip()
    return "+" in first and first.endswith(".csv")


def _is_column_header(cells: list[str]) -> bool:
    """Check if row is a column header row."""
    return len(cells) > 0 and cells[0].strip() == "iteration"


def _is_summary_row(cells: list[str]) -> bool:
    """Check if row is STDEV/AVERAGE/CV."""
    return len(cells) > 0 and cells[0].strip() in ("STDEV", "AVERAGE", "CV")


def _is_data_row(cells: list[str]) -> bool:
    """Check if row is a data row (starts with a number)."""
    if not cells:
        return False
    first = cells[0].strip()
    try:
        float(first)
        return True
    except ValueError:
        return False


def process_file(
    filepath: Path,
    ts_index: dict[tuple[str, str, str, str, int], str],
) -> bool:
    """Process one static CSV file, adding generate_timestamp column.

    Returns True if the file was modified.
    """
    week = _extract_week(filepath.name)
    if not week:
        print(f"  SKIP {filepath.name}: cannot extract week")
        return False

    text = filepath.read_text(encoding="utf-8")

    # Check if already has generate_timestamp
    if TIMESTAMP_COL in text.split("\n")[1] if len(text.split("\n")) > 1 else "":
        print(f"  SKIP {filepath.name}: already has {TIMESTAMP_COL}")
        return False

    reader = csv.reader(io.StringIO(text))
    all_rows = list(reader)

    current_question = ""
    current_provider_slug = ""
    current_model = ""
    modified_rows: list[list[str]] = []
    found_any_ts = False

    for row in all_rows:
        if _is_section_header(row):
            parsed = _parse_section_header(row[0])
            if parsed:
                current_question, current_provider_slug, current_model = parsed
            # Add one empty cell to maintain column alignment
            new_row = [row[0]] + [""] + row[1:]
            modified_rows.append(new_row)

        elif _is_column_header(row):
            # Insert generate_timestamp after iteration
            new_row = [row[0], TIMESTAMP_COL] + row[1:]
            modified_rows.append(new_row)

        elif _is_data_row(row):
            # Look up timestamp
            try:
                iteration = int(float(row[0].strip()))
            except (ValueError, IndexError):
                iteration = 0

            ts = ts_index.get(
                (
                    week,
                    current_question,
                    current_provider_slug,
                    current_model,
                    iteration,
                ),
                "",
            )
            if ts:
                found_any_ts = True

            new_row = [row[0], ts] + row[1:]
            modified_rows.append(new_row)

        elif _is_summary_row(row):
            # Add empty cell for timestamp column
            new_row = [row[0], ""] + row[1:]
            modified_rows.append(new_row)

        else:
            # Unknown row type - just add empty cell to keep alignment
            if row:
                new_row = [row[0], ""] + row[1:]
            else:
                new_row = row
            modified_rows.append(new_row)

    # Write back
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    for row in modified_rows:
        writer.writerow(row)

    filepath.write_text(buf.getvalue(), encoding="utf-8")

    status = "OK" if found_any_ts else "OK (no timestamps found in experiments)"
    print(f"  {status}: {filepath.name}")
    return True


def main() -> int:
    print("Building timestamp index from experiments...")
    ts_index = _build_timestamp_index()
    print(
        f"  Indexed {len(ts_index)} (week, question, provider, model, iteration) entries"
    )

    static_files = sorted(WEEKLY_DIR.glob("*_static.csv"))
    print(f"\nProcessing {len(static_files)} static CSV files...")

    modified_count = 0
    for filepath in static_files:
        if process_file(filepath, ts_index):
            modified_count += 1

    print(f"\nDone. Modified {modified_count}/{len(static_files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
