"""Notebook display helpers for the faas_gauge package."""

from __future__ import annotations

from typing import Any


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truncate_text(value: Any, max_len: int = 40) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_experiment_summary(experiment_id: str, summary: dict[str, Any]) -> None:
    """Print a formatted experiment summary."""
    total_iterations = _to_int(summary.get("total_iterations"))
    successful = _to_int(summary.get("successful_iterations"))
    failed = _to_int(summary.get("failed_iterations"))
    input_tokens = _to_int(summary.get("total_input_tokens"))
    output_tokens = _to_int(summary.get("total_output_tokens"))
    total_time = _to_float(summary.get("total_time"))
    avg_time = _to_float(summary.get("average_time"))
    total_tps = _to_float(summary.get("total_tokens_per_second"))

    print(f"Experiment: {experiment_id}")
    print("-" * 72)
    print(f"Iterations  : {total_iterations}")
    print(f"Successful  : {successful}")
    print(f"Failed      : {failed}")
    print(f"Input tokens: {input_tokens:,}")
    print(f"Output tokens: {output_tokens:,}")
    print(f"Total time  : {total_time:.2f}s")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Total TPS   : {total_tps:.2f}")


def format_iteration_table(iterations: list[dict[str, Any]], max_rows: int = 20) -> str:
    """Format iterations as a text table. Show first max_rows with '...' if truncated."""
    headers = ["iteration", "tokens_in", "tokens_out", "tps", "rtt_sec", "error"]
    rows: list[list[str]] = []

    for row in iterations[:max_rows]:
        tokens_in = row.get("input_tokens", row.get("tokens_in", 0))
        tokens_out = row.get("output_tokens", row.get("tokens_out", 0))
        tps = row.get("tokens_per_second", row.get("tps", ""))
        rtt_sec = row.get("rtt_sec", row.get("rtt_time", ""))
        rows.append(
            [
                str(row.get("iteration", "")),
                str(_to_int(tokens_in)),
                str(_to_int(tokens_out)),
                f"{_to_float(tps):.2f}" if tps not in (None, "") else "",
                f"{_to_float(rtt_sec):.2f}" if rtt_sec not in (None, "") else "",
                _truncate_text(row.get("error", ""), max_len=50),
            ]
        )

    widths = [len(h) for h in headers]
    for row_cells in rows:
        for idx, cell in enumerate(row_cells):
            widths[idx] = max(widths[idx], len(cell))

    sep = "-+-".join("-" * w for w in widths)
    header_line = " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
    row_lines = [
        " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) for row in rows
    ]

    lines = [header_line, sep, *row_lines]
    if len(iterations) > max_rows:
        lines.append(f"... ({len(iterations) - max_rows} more rows)")
    return "\n".join(lines)


def print_validation_summary(batch_id: str, summary: dict[str, Any]) -> None:
    """Print a formatted validation summary."""
    total_tests = _to_int(summary.get("total_tests"))
    passed = _to_int(summary.get("passed"))
    failed = _to_int(summary.get("failed"))
    errors = _to_int(summary.get("errors"))

    print(f"Validation Batch: {batch_id}")
    print("-" * 72)
    print(f"Total tests: {total_tests}")
    print(f"Passed     : {passed}")
    print(f"Failed     : {failed}")
    print(f"Errors     : {errors}")
