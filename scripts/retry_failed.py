"""CLI wrapper for faas_gauge.experiment.retry_failed."""

from __future__ import annotations

import argparse
import sys

from faas_gauge.experiment import retry_failed


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for retrying failed iterations."""
    parser = argparse.ArgumentParser(
        description="Retry failed iterations for an existing experiment.",
        epilog=(
            "Examples:\n"
            "  python scripts/retry_failed.py -e 20260125_2100+prime_number_generator+openai+gpt-5-mini\n"
            "  python scripts/retry_failed.py -e EXPERIMENT_ID -w 8 -d data -c .secret.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-e", "--experiment", required=True, help="Experiment ID to retry.")
    parser.add_argument("-w", "--max-workers", type=int, default=None, help="Worker count.")
    parser.add_argument("-d", "--data-dir", default="data", help="Data directory path.")
    parser.add_argument("-c", "--config", default=None, help="Optional config file path.")
    return parser


def main() -> int:
    """Run retry CLI and return process exit code."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        print(f"Retrying failed iterations for experiment={args.experiment}")
        result = retry_failed(
            experiment_id=args.experiment,
            max_workers=args.max_workers,
            data_dir=args.data_dir,
            config_path=args.config,
        )
        print(
            "Summary: "
            f"retried={result.get('retried', 0)}, "
            f"newly_successful={result.get('newly_successful', 0)}, "
            f"still_failed={result.get('still_failed', 0)}"
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
