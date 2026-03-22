#!/usr/bin/env python
"""List the repository's main uv commands and short option terminology."""

from __future__ import annotations


def main() -> int:
    print("U-CaGNN command list")
    print()
    print("Runs")
    print("  uv run experiment --list-recipes")
    print("  uv run benchmark --tier small --dry-run")
    print("  uv run ablation --dataset movielens1m")
    print()
    print("Maintenance")
    print("  uv run preflight --dry-run")
    print("  uv run query-results")
    print("  uv run reset-experiment-db")
    print("  uv run cleanup-experiment-artifacts --all")
    print("  uv run verify-setup")
    print("  uv run verify-pipeline")
    print("  uv run verify-sqlite")
    print("  uv run visualize-results")
    print()
    print("Terms")
    print("  dry-run: preview only, changes nothing")
    print("  --yes: execute a destructive command")
    print("  preflight: short safety check before long runs")
    print("  ablation: run variants with parts removed to measure impact")
    print("  recipe: named experiment setup")
    print("  preset: model configuration family")
    print("  sample-interactions: run on a smaller sampled dataset")
    print("  --keep-db: keep the temporary verification database")
    print("  --no-mlflow: disable MLflow logging for that run")
    print("  --no-auto-resume: start fresh even if a checkpoint exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())