"""Run static analysis (pylint + radon) on experiment iterations.

Reads generated code from experiments' iterations.jsonl, runs pylint and
radon on each iteration's python_code, and writes results to
static_analysis.jsonl inside each experiment directory.

Examples:

  # All week2 experiments
  python scripts/static_analysis.py -g week2

  # Filter to specific question
  python scripts/static_analysis.py -g week2 -q prime_number_generator

  # Custom workers
  python scripts/static_analysis.py -g week2 -w 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root so we can import the store
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faas_gauge.store import ExperimentStore

# ---------------------------------------------------------------------------
# Resolve tool paths from the same venv as sys.executable
# ---------------------------------------------------------------------------


def _tool_path(name: str) -> str:
    """Return the full path to a CLI tool in the active venv or on PATH."""
    # sys.prefix points to the venv root (not the resolved Homebrew path)
    venv_bin = Path(sys.prefix) / "bin" / name
    if venv_bin.exists():
        return str(venv_bin)
    # Also check next to the executable (without resolve, to stay in venv)
    exe_bin = Path(sys.executable).parent / name
    if exe_bin.exists():
        return str(exe_bin)
    # Fallback to bare name (relies on PATH)
    return name


_PYLINT = _tool_path("pylint")
_RADON = _tool_path("radon")

# ---------------------------------------------------------------------------
# Pylint
# ---------------------------------------------------------------------------


def run_pylint(code_path: Path) -> dict:
    """Run pylint on a Python file and return parsed results."""
    result = {
        "score": None,
        "errors": 0,
        "warnings": 0,
        "conventions": 0,
        "refactors": 0,
        "fatal": 0,
    }

    try:
        proc = subprocess.run(
            [_PYLINT, str(code_path), "--output-format=text"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = proc.stdout + proc.stderr

        score_match = re.search(r"Your code has been rated at ([-\d.]+)/10", output)
        if score_match:
            result["score"] = float(score_match.group(1))

        result["fatal"] = len(
            re.findall(r"^[^:]+:\d+:\d+: F\d+:", output, re.MULTILINE)
        )
        result["errors"] = len(
            re.findall(r"^[^:]+:\d+:\d+: E\d+:", output, re.MULTILINE)
        )
        result["warnings"] = len(
            re.findall(r"^[^:]+:\d+:\d+: W\d+:", output, re.MULTILINE)
        )
        result["conventions"] = len(
            re.findall(r"^[^:]+:\d+:\d+: C\d+:", output, re.MULTILINE)
        )
        result["refactors"] = len(
            re.findall(r"^[^:]+:\d+:\d+: R\d+:", output, re.MULTILINE)
        )

    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        print("  WARNING: pylint not found", file=sys.stderr)
    except Exception as e:
        print(f"  WARNING: pylint error: {e}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Radon
# ---------------------------------------------------------------------------


def run_radon(code_path: Path) -> dict:
    """Run radon on a Python file and return parsed results."""
    result: dict = {
        "cc_avg": None,
        "mi_score": None,
        "mi_rank": None,
        "loc": None,
        "lloc": None,
        "sloc": None,
        "comments": None,
        "halstead_volume": None,
        "halstead_difficulty": None,
        "halstead_effort": None,
        "halstead_bugs": None,
    }

    path_key = str(code_path)

    try:
        # Cyclomatic Complexity
        cc_proc = subprocess.run(
            [_RADON, "cc", str(code_path), "-a", "-j"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        try:
            cc_data = json.loads(cc_proc.stdout)
            if path_key in cc_data:
                blocks = cc_data[path_key]
                if blocks:
                    avg = sum(b.get("complexity", 0) for b in blocks) / len(blocks)
                    result["cc_avg"] = round(avg, 2)
        except json.JSONDecodeError:
            pass

        # Maintainability Index
        mi_proc = subprocess.run(
            [_RADON, "mi", str(code_path), "-j"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        try:
            mi_data = json.loads(mi_proc.stdout)
            if path_key in mi_data:
                mi_info = mi_data[path_key]
                result["mi_score"] = mi_info.get("mi")
                result["mi_rank"] = mi_info.get("rank")
        except json.JSONDecodeError:
            pass

        # Raw Metrics
        raw_proc = subprocess.run(
            [_RADON, "raw", str(code_path), "-j"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        try:
            raw_data = json.loads(raw_proc.stdout)
            if path_key in raw_data:
                raw_info = raw_data[path_key]
                result["loc"] = raw_info.get("loc")
                result["lloc"] = raw_info.get("lloc")
                result["sloc"] = raw_info.get("sloc")
                result["comments"] = raw_info.get("comments")
        except json.JSONDecodeError:
            pass

        # Halstead Metrics
        hal_proc = subprocess.run(
            [_RADON, "hal", str(code_path), "-j"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        try:
            hal_data = json.loads(hal_proc.stdout)
            if path_key in hal_data:
                hal_info = hal_data[path_key].get("total", [{}])
                if hal_info:
                    h = hal_info[0] if isinstance(hal_info, list) else hal_info
                    result["halstead_volume"] = h.get("volume")
                    result["halstead_difficulty"] = h.get("difficulty")
                    result["halstead_effort"] = h.get("effort")
                    result["halstead_bugs"] = h.get("bugs")
        except json.JSONDecodeError:
            pass

    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        print("  WARNING: radon not found", file=sys.stderr)
    except Exception as e:
        print(f"  WARNING: radon error: {e}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Iteration analysis
# ---------------------------------------------------------------------------


def analyze_iteration(iteration_record: dict, tmp_dir: str) -> dict | None:
    """Analyze a single iteration's python_code with pylint + radon.

    Returns a dict suitable for appending to static_analysis.jsonl,
    or None if there's no code.
    """
    code = iteration_record.get("python_code", "")
    if not code or not code.strip():
        return None

    iteration_num = iteration_record.get("iteration", 0)

    # Write code to temp file
    code_path = Path(tmp_dir) / f"iter_{iteration_num}.py"
    code_path.write_text(code, encoding="utf-8")

    pylint = run_pylint(code_path)
    radon = run_radon(code_path)

    # Extract token/timing info from iteration metadata.
    # Handle both week1 format (input_tokens, rtt_time, created_at)
    # and week2 format (input_tokens, rtt_sec, timestamp).
    input_token = iteration_record.get("input_tokens") or iteration_record.get(
        "input_token"
    )
    output_token = iteration_record.get("output_tokens") or iteration_record.get(
        "output_token"
    )
    rtt_time = (
        iteration_record.get("rtt_time")
        or iteration_record.get("rtt_sec")
        or iteration_record.get("duration_s")
    )
    token_per_second = iteration_record.get("tokens_per_second")
    if token_per_second is None and output_token and rtt_time and rtt_time > 0:
        token_per_second = round(output_token / rtt_time, 2)

    # LLM generation timestamp (Todo-1)
    generate_timestamp = iteration_record.get("timestamp") or iteration_record.get(
        "created_at"
    )

    row = {
        "iteration": iteration_num,
        "generate_timestamp": generate_timestamp,
        "input_token": input_token,
        "output_token": output_token,
        "rtt_time": rtt_time,
        "token_per_second": token_per_second,
        "error": iteration_record.get("error"),
        # Pylint
        "pylint_score": pylint["score"],
        "pylint_fatal": pylint["fatal"],
        "pylint_errors": pylint["errors"],
        "pylint_warnings": pylint["warnings"],
        "pylint_conventions": pylint["conventions"],
        "pylint_refactors": pylint["refactors"],
        # Radon
        "cc_avg": radon["cc_avg"],
        "mi_score": radon["mi_score"],
        "mi_rank": radon["mi_rank"],
        "loc": radon["loc"],
        "lloc": radon["lloc"],
        "sloc": radon["sloc"],
        "comments": radon["comments"],
        "halstead_volume": radon["halstead_volume"],
        "halstead_difficulty": radon["halstead_difficulty"],
        "halstead_effort": radon["halstead_effort"],
        "halstead_bugs": radon["halstead_bugs"],
    }

    # Cleanup temp file
    try:
        code_path.unlink()
    except OSError:
        pass

    return row


# ---------------------------------------------------------------------------
# Experiment-level analysis
# ---------------------------------------------------------------------------


def analyze_experiment(
    exp_id: str,
    data_dir: Path,
    max_workers: int | None = None,
) -> list[dict]:
    """Analyze all iterations in an experiment. Returns list of result rows."""
    iterations_path = data_dir / "experiments" / exp_id / "iterations.jsonl"
    if not iterations_path.is_file():
        return []

    records: list[dict] = []
    with open(iterations_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return []

    tmp_dir = tempfile.mkdtemp(prefix="sa_")
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(analyze_iteration, rec, tmp_dir): rec for rec in records
        }
        for future in as_completed(future_map):
            try:
                row = future.result()
                if row is not None:
                    results.append(row)
            except Exception as e:
                rec = future_map[future]
                print(
                    f"  WARNING: iteration {rec.get('iteration')} failed: {e}",
                    file=sys.stderr,
                )

    results.sort(key=lambda x: x["iteration"])

    # Write static_analysis.jsonl
    sa_path = data_dir / "experiments" / exp_id / "static_analysis.jsonl"
    with open(sa_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    # Cleanup temp dir
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run static analysis (pylint + radon) on experiment iterations.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-g", "--test-group", required=True, help="Test group name.")
    parser.add_argument(
        "-q", "--question", default=None, help="Filter by question name."
    )
    parser.add_argument(
        "-w", "--max-workers", type=int, default=None, help="Max parallel workers."
    )
    parser.add_argument("-d", "--data-dir", default="data", help="Data directory path.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    store = ExperimentStore(data_dir=data_dir)

    experiments = store.list_experiments(test_group=args.test_group)
    if args.question:
        experiments = [
            e
            for e in experiments
            if args.question.lower() in e.get("question_name", "").lower()
        ]

    if not experiments:
        print("No experiments matched the filters.", file=sys.stderr)
        return 1

    print(f"Found {len(experiments)} experiment(s) to analyze")
    print("=" * 60)

    total_iterations = 0
    for exp in experiments:
        exp_id = exp["id"]
        question = exp.get("question_name", "")
        model = exp.get("model_name", "")
        print(f"\nAnalyzing: {question} / {model}")
        print(f"  Experiment: {exp_id}")

        results = analyze_experiment(exp_id, data_dir, max_workers=args.max_workers)
        total_iterations += len(results)

        if results:
            pylint_scores = [
                r["pylint_score"] for r in results if r["pylint_score"] is not None
            ]
            mi_scores = [r["mi_score"] for r in results if r["mi_score"] is not None]
            cc_avgs = [r["cc_avg"] for r in results if r["cc_avg"] is not None]

            print(f"  Iterations analyzed: {len(results)}")
            if pylint_scores:
                print(
                    f"  Avg pylint score: {sum(pylint_scores) / len(pylint_scores):.2f}"
                )
            if mi_scores:
                print(f"  Avg MI score: {sum(mi_scores) / len(mi_scores):.2f}")
            if cc_avgs:
                print(f"  Avg CC: {sum(cc_avgs) / len(cc_avgs):.2f}")
        else:
            print("  No iterations found")

    print("\n" + "=" * 60)
    print(
        f"Done - {total_iterations} iteration(s) analyzed across {len(experiments)} experiment(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
