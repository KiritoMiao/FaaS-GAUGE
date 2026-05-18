"""CLI wrapper for faas_gauge.experiment.validate_batch."""

from __future__ import annotations

import argparse
import sys

from faas_gauge.experiment import validate_batch


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for validation batches."""
    parser = argparse.ArgumentParser(
        description="Validate generated code for a test group.",
        epilog=(
            "Examples:\n"
            "  python scripts/validate.py -g week1\n"
            "  python scripts/validate.py -g week1 -r aws -n 3 -t 60 "
            "--performance-mode strict --aws-memory 2048 --no-reuse"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-g", "--test-group", required=True, help="Test group name.")
    parser.add_argument(
        "-n", "--iterations", type=int, default=1, help="Iterations per test."
    )
    parser.add_argument(
        "-r", "--runtime", default="local", help="Runtime: local or aws."
    )
    parser.add_argument(
        "-t", "--max-runtime", type=int, default=30, help="Per-test timeout (sec)."
    )
    parser.add_argument(
        "-q", "--question", default=None, help="Optional question filter."
    )
    parser.add_argument(
        "--performance-mode",
        default="loose",
        choices=["loose", "strict"],
        help="Performance validation mode.",
    )
    parser.add_argument(
        "--aws-memory", type=int, default=1769, help="AWS memory size MB."
    )
    parser.add_argument(
        "--no-reuse",
        action="store_false",
        dest="reuse_deployed_function",
        help="Disable deployed function reuse.",
    )
    parser.set_defaults(reuse_deployed_function=True)
    parser.add_argument(
        "--max-total-seconds",
        type=float,
        default=None,
        help="Max total seconds budget for performance test iterations per code version.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Do not skip already-validated iterations from prior batches.",
    )
    parser.add_argument("-d", "--data-dir", default="data", help="Data directory path.")
    parser.add_argument(
        "-c", "--config", default=None, help="Optional config file path."
    )
    return parser


def main() -> int:
    """Run batch validation CLI and return process exit code."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        print(
            f"Starting validation: test_group={args.test_group}, runtime={args.runtime}, "
            f"iterations={args.iterations}"
        )
        result = validate_batch(
            test_group=args.test_group,
            iterations=args.iterations,
            runtime=args.runtime,
            max_runtime=args.max_runtime,
            question=args.question,
            performance_mode=args.performance_mode,
            aws_memory_size=args.aws_memory,
            reuse_deployed_function=args.reuse_deployed_function,
            data_dir=args.data_dir,
            config_path=args.config,
            max_total_seconds=args.max_total_seconds,
            skip_resume=args.no_resume,
        )

        summary = result.get("summary", {})
        print(f"Batch ID: {result.get('batch_id')}")
        print(
            "Summary: "
            f"completed={summary.get('completed', 0)}/{summary.get('total_experiments', 0)}, "
            f"tests={summary.get('total_tests', 0)}, "
            f"passed={summary.get('passed', 0)}, "
            f"failed={summary.get('failed', 0)}, "
            f"errors={summary.get('errors', 0)}"
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
