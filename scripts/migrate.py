"""CLI wrapper for migration utilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from faas_gauge.store.migration import (
    migrate_all,
    migrate_credentials,
    migrate_from_sqlite,
    migrate_validations,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for migration operations."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy SQLite/YAML data into the new data store.",
        epilog=(
            "Examples:\n"
            "  python scripts/migrate.py --db legacy.db --source-dir . -t data\n"
            "  python scripts/migrate.py --db legacy.db --source-dir . -t data --only experiments\n"
            "  python scripts/migrate.py --db legacy.db --source-dir . -t data --only credentials "
            "-s .secret.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", required=True, help="Path to source SQLite DB.")
    parser.add_argument(
        "--source-dir",
        action="append",
        required=True,
        help="Source directory to migrate static files from (repeatable).",
    )
    parser.add_argument("-t", "--target", required=True, help="Target data directory.")
    parser.add_argument(
        "-s", "--secret-yaml", default=None, help="Optional .secret.yaml path."
    )
    parser.add_argument(
        "--only",
        choices=["experiments", "validations", "credentials"],
        default=None,
        help="Run only one migration step.",
    )
    parser.add_argument(
        "--test-group",
        default=None,
        help="Only migrate data belonging to this test_group (e.g. week1).",
    )
    return parser


def main() -> int:
    """Run migration CLI and return process exit code."""
    parser = build_parser()
    args = parser.parse_args()

    source_db = Path(args.db)
    source_dirs = [Path(path) for path in args.source_dir]
    target = Path(args.target)
    secret_yaml = Path(args.secret_yaml) if args.secret_yaml else None

    test_group = args.test_group
    group_msg = f" (test_group={test_group})" if test_group else ""

    try:
        if args.only is None:
            print(
                f"Migrating all data: experiments, validations, static files, credentials{group_msg}"
            )
            migrate_all(
                source_db=source_db,
                source_dirs=source_dirs,
                target_data_dir=target,
                secret_yaml=secret_yaml,
                test_group=test_group,
            )
            print("Migration complete")
            return 0

        if args.only == "experiments":
            print(f"Migrating experiments from SQLite{group_msg}")
            migrate_from_sqlite(source_db, target, test_group=test_group)
            print("Experiments migration complete")
            return 0

        if args.only == "validations":
            print(f"Migrating validations from SQLite{group_msg}")
            migrate_validations(source_db, target, test_group=test_group)
            print("Validations migration complete")
            return 0

        if args.only == "credentials":
            if secret_yaml is None:
                raise ValueError(
                    "--secret-yaml is required when using --only credentials"
                )
            print("Migrating credentials from secret YAML")
            migrate_credentials(secret_yaml, target)
            print("Credentials migration complete")
            return 0

        raise ValueError(f"Unsupported --only option: {args.only}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
