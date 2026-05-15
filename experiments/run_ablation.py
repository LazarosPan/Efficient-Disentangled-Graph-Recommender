#!/usr/bin/env python
"""Ablation study runner: test each thesis-facing U-CaGNN component contribution.

Runs named variants around the UCaGNN mainline on specified datasets.

Usage:
    uv run ablation --dataset movielens1m --epochs 3
    uv run ablation --dataset movielens1m --variants mainline no_independence no_features
    uv run ablation --dataset movielens1m --dry-run
"""

from __future__ import annotations

import logging
import time
import traceback

from scripts._workflow_helpers import (
    configure_cli_logging,
    metric_value,
    print_batch_summary_counts,
    resolve_batch_id,
    thesis_metric_values,
)
from src.training import THESIS_PRIMARY_METRICS
from src.utils.cli_parsers import (
    normalize_benchmark_datasets_arg,
    resolve_benchmark_datasets,
)
from src.utils.config import DEFAULT_SEED
from src.utils.experiment_logger import ExperimentLogger
from src.utils.project_paths import THESIS_DB_PATH

from experiments.ablation_configs import (
    ABLATION_VARIANTS,
    build_ablation_base_kwargs,
    make_ablation_config,
)
from experiments.cli_parsers import build_ablation_parser
from experiments.run_experiment import run_experiment

logger = logging.getLogger("ucagnn.ablation")
UCAGNN_PRESET = "ucagnn"


def _resolve_target_datasets(args) -> list[str]:
    """Resolve the ablation dataset selection from CLI args.

    Args:
        args: Parsed ablation CLI arguments.

    Returns:
        Ordered dataset list with benchmark-tier selectors expanded.

    Raises:
        ValueError: If neither ``--dataset`` nor ``--datasets`` is supplied.

    """
    raw_datasets = args.datasets if args.datasets else ([args.dataset] if args.dataset else None)
    if raw_datasets is None:
        raise ValueError("Pass --dataset or --datasets for the ablation study.")
    return resolve_benchmark_datasets(normalize_benchmark_datasets_arg(raw_datasets))


def main() -> int:
    """Parse arguments and run the ablation sweep."""
    args = build_ablation_parser().parse_args()
    datasets = _resolve_target_datasets(args)

    configure_cli_logging()
    batch_id = resolve_batch_id(args.batch_id, prefix="ablation")

    print("=" * 70)
    print(f"ABLATION STUDY: {', '.join(datasets)}")
    print(f"  Variants: {', '.join(args.variants)}")
    print(f"  Batch ID: {batch_id}")
    print(f"  Resume batch: {args.resume_batch}")
    print("=" * 70)

    if args.dry_run:
        print(f"\n{'#':>4} | {'Dataset':<12} | {'Variant':<20} | Overrides")
        print("-" * 76)
        row_number = 0
        for dataset in datasets:
            for variant in args.variants:
                row_number += 1
                overrides = ABLATION_VARIANTS[variant]
                override_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(baseline)"
                print(f"{row_number:>4} | {dataset:<12} | {variant:<20} | {override_str}")
        print(
            f"\nTotal: {len(datasets) * len(args.variants)} runs (dry run, nothing executed)",
        )
        return 0

    # Run ablation experiments
    tracker = ExperimentLogger(db_path=str(THESIS_DB_PATH))
    completed = 0
    failed = 0
    skipped = 0
    results = []

    total_runs = len(datasets) * len(args.variants)
    run_index = 0
    for dataset in datasets:
        for variant in args.variants:
            run_index += 1
            print(f"\n{'=' * 70}")
            print(f"[{run_index}/{total_runs}] Ablation: {dataset} :: {variant}")
            print("=" * 70)

            existing = tracker.find_latest_batch_experiment(
                batch_id=batch_id,
                dataset=dataset,
                preset=UCAGNN_PRESET,
                intervention=variant,
                seed=DEFAULT_SEED,
                training_mode=None,
            )
            if (
                args.resume_batch
                and existing is not None
                and existing["status"] in ExperimentLogger.TERMINAL_STATUSES
            ):
                skipped += 1
                logger.info(
                    "Skipping existing batch ablation (dataset=%s, exp_id=%s, status=%s)",
                    dataset,
                    existing["id"],
                    existing["status"],
                )
                if existing["status"] == "completed":
                    results.append(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "exp_id": existing["id"],
                            "metrics": tracker.get_metrics_for_split(
                                int(existing["id"]),
                                split="test",
                            ),
                            "elapsed_s": 0.0,
                        },
                    )
                continue

            try:
                config = make_ablation_config(
                    variant,
                    **build_ablation_base_kwargs(
                        dataset=dataset,
                        data_dir=args.data_dir,
                        seed=DEFAULT_SEED,
                        device=args.device,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        sample_interactions=args.sample_interactions,
                        loader_max_rows=args.loader_max_rows,
                    ),
                )

                t0 = time.time()
                result = run_experiment(
                    config,
                    preset=UCAGNN_PRESET,
                    intervention=variant,
                    save_checkpoint=True,
                    enable_mlflow=not args.no_mlflow,
                    mlflow_tracking_uri=args.mlflow_tracking_uri,
                    mlflow_experiment_name=args.mlflow_experiment_name,
                    batch_id=batch_id,
                )
                elapsed = time.time() - t0

                results.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "exp_id": result["exp_id"],
                        "metrics": result["test_metrics"],
                        "elapsed_s": elapsed,
                    },
                )
                completed += 1
                logger.info("Completed %s in %.1fs", dataset, elapsed)

            except Exception as e:
                failed += 1
                logger.error("FAILED (%s): %s", dataset, e)
                traceback.print_exc()
                continue

    # Summary
    print_batch_summary_counts(
        title=f"ABLATION SUMMARY ({', '.join(datasets)})",
        completed=completed,
        failed=failed,
        skipped=skipped,
        total=total_runs,
    )

    if results:
        print("Note: AvgPop@20 and AvgPop@40 are lower-is-better.")
        results_by_dataset: dict[str, list[dict]] = {}
        for result_entry in results:
            results_by_dataset.setdefault(result_entry["dataset"], []).append(result_entry)

        for dataset in datasets:
            dataset_results = results_by_dataset.get(dataset)
            if not dataset_results:
                continue

            baseline_metrics = None
            for result_entry in dataset_results:
                if result_entry["variant"] == "mainline":
                    baseline_metrics = result_entry["metrics"]
                    break

            print(
                ""
                f"\nDATASET: {dataset}\n"
                f"{'Variant':<20} | {'NDCG@20':>8} | {'Recall@20':>10} | {'AvgPop@20':>10} | "
                f"{'NDCG@40':>8} | {'Recall@40':>10} | {'AvgPop@40':>10} | Time",
            )
            print("-" * 117)
            for result_entry in dataset_results:
                metric_values = thesis_metric_values(
                    result_entry["metrics"],
                    THESIS_PRIMARY_METRICS,
                )

                print(
                    f"{result_entry['variant']:<20} | {metric_values['NDCG@20']:>8.4f} | "
                    f"{metric_values['Recall@20']:>10.4f} | "
                    f"{metric_values['AveragePopularity@20']:>10.4f} | "
                    f"{metric_values['NDCG@40']:>8.4f} | "
                    f"{metric_values['Recall@40']:>10.4f} | "
                    f"{metric_values['AveragePopularity@40']:>10.4f} | "
                    f"{result_entry['elapsed_s']:.0f}s",
                )
                if baseline_metrics and result_entry["variant"] != "mainline":
                    deltas = []
                    for metric_name in THESIS_PRIMARY_METRICS:
                        delta = metric_values[metric_name] - metric_value(
                            baseline_metrics,
                            metric_name,
                        )
                        deltas.append(f"{metric_name} {delta:+.4f}")
                    print(f"{'':<20}   deltas: {' | '.join(deltas)}")
                else:
                    print(f"{'':<20}   deltas: baseline")

    tracker.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
