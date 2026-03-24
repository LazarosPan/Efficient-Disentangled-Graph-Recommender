#!/usr/bin/env python
"""Legacy compatibility alias for the unified quick validation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Legacy alias for scripts/quick_validate.py",
        add_help=False,
    )
    parser.add_argument(
        "--keep-db", action="store_true", help="Deprecated legacy flag; ignored."
    )
    parser.add_argument(
        "--db-path", type=str, default=None, help="Deprecated legacy flag; ignored."
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show quick_validate help."
    )
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    from scripts.quick_validate import main as quick_validate_main

    if args.keep_db or args.db_path is not None:
        print(
            "verify-pipeline is now an alias for scripts/quick_validate.py; --keep-db and --db-path are ignored."
        )

    forwarded_argv = [sys.argv[0], *passthrough]
    if args.help:
        forwarded_argv.append("--help")

    original_argv = sys.argv[:]
    try:
        sys.argv = forwarded_argv
        return quick_validate_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
