"""CLI wrapper for faas_gauge.experiment.generate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from faas_gauge.experiment import generate


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for generation runs."""
    parser = argparse.ArgumentParser(
        description="Generate code samples for a question with a provider/model.",
        epilog=(
            "Examples:\n"
            "  python scripts/generate.py -f question/prime_number_generator.txt "
            "-p openai -m gpt-5-mini\n"
            "  python scripts/generate.py -f question/sort.txt -p openai -m gpt-5-mini "
            "-n 20 -w 4 -g week1 -d data"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--question-file",
        required=True,
        help="Path to question .txt file.",
    )
    parser.add_argument("-p", "--provider", required=True, help="AI provider name.")
    parser.add_argument("-m", "--model", required=True, help="Model name.")
    parser.add_argument("-n", "--iterations", type=int, default=100, help="Number of runs.")
    parser.add_argument("-w", "--max-workers", type=int, default=None, help="Worker count.")
    parser.add_argument("-g", "--test-group", default=None, help="Optional test group label.")
    parser.add_argument("-d", "--data-dir", default="data", help="Data directory path.")
    parser.add_argument("-c", "--config", default=None, help="Optional config file path.")
    return parser


def main() -> int:
    """Run generation CLI and return process exit code."""
    parser = build_parser()
    args = parser.parse_args()

    question_file = Path(args.question_file)
    question = question_file.stem

    try:
        question_content = question_file.read_text(encoding="utf-8")

        print(
            f"Starting generation: question={question}, provider={args.provider}, "
            f"model={args.model}, iterations={args.iterations}"
        )
        result = generate(
            question=question,
            question_content=question_content,
            provider=args.provider,
            model=args.model,
            iterations=args.iterations,
            max_workers=args.max_workers,
            test_group=args.test_group,
            data_dir=args.data_dir,
            config_path=args.config,
        )

        summary = result.get("summary", {})
        print(f"Experiment ID: {result.get('id')}")
        print(
            "Summary: "
            f"success={summary.get('successful_iterations', 0)}, "
            f"failed={summary.get('failed_iterations', 0)}, "
            f"total={summary.get('total_iterations', args.iterations)}"
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
