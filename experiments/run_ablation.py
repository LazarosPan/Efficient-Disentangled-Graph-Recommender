#!/usr/bin/env python
"""Ablation study runner: test each U-CaGNN component's contribution.

Runs named variants around the UCaGNN mainline on specified datasets.

Usage:
    uv run ablation --dataset movielens1m --epochs 3
    uv run ablation --dataset movielens1m --variants mainline no_ipw no_dual_branch
    uv run ablation --dataset movielens1m --dry-run
"""

from __future__ import annotations

import logging
import time
import traceback

from experiments.ablation_configs import ABLATION_VARIANTS, make_ablation_config
from experiments.cli_parsers import build_ablation_parser
from experiments.run_experiment import DB_PATH, run_experiment
from scripts._workflow_helpers import metric_value, resolve_batch_id
from src.training import THESIS_PRIMARY_METRICS
from src.utils.config import DEFAULT_SEED
from src.utils.experiment_logger import ExperimentLogger

logger = logging.getLogger("ucagnn.ablation")
UCAGNN_PRESET = "ucagnn"


def main() -> int:
    """Parse arguments and run the ablation sweep."""
    args = build_ablation_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    batch_id = resolve_batch_id(args.batch_id, prefix="ablation")

    print("=" * 70)
    print(f"ABLATION STUDY: {args.dataset}")
    print(f"  Variants: {', '.join(args.variants)}")
    print(f"  Batch ID: {batch_id}")
    print(f"  Resume batch: {args.resume_batch}")
    print("=" * 70)

    if args.dry_run:
        print(f"\n{'#':>4} | {'Variant':<20} | Overrides")
        print("-" * 60)
        for i, variant in enumerate(args.variants, 1):
            overrides = ABLATION_VARIANTS[variant]
            override_str = (
                ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(baseline)"
            )
            print(f"{i:>4} | {variant:<20} | {override_str}")
        print(f"\nTotal: {len(args.variants)} variants (dry run, nothing executed)")
        return 0

    # Run ablation experiments
    tracker = ExperimentLogger(db_path=str(DB_PATH))
    completed = 0
    failed = 0
    skipped = 0
    results = []

    for i, variant in enumerate(args.variants, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(args.variants)}] Ablation: {variant}")
        print("=" * 70)

        existing = tracker.find_latest_batch_experiment(
            batch_id=batch_id,
            dataset=args.dataset,
            preset=UCAGNN_PRESET,
            intervention=variant,
            seed=DEFAULT_SEED,
            training_mode=None,
            graph_method=None,
        )
        if (
            args.resume_batch
            and existing is not None
            and existing["status"] in ExperimentLogger.TERMINAL_STATUSES
        ):
            skipped += 1
            logger.info(
                "Skipping existing batch ablation (exp_id=%s, status=%s)",
                existing["id"],
                existing["status"],
            )
            if existing["status"] == "completed":
                results.append(
                    {
                        "variant": variant,
                        "exp_id": existing["id"],
                        "metrics": tracker.get_metrics_for_split(
                            int(existing["id"]),
                            split="test",
                        ),
                        "elapsed_s": 0.0,
                    }
                )
            continue

        try:
            base_kwargs = {
                "dataset": args.dataset,
                "data_dir": args.data_dir,
                "seed": DEFAULT_SEED,
                "device": args.device,
            }
            if args.epochs is not None:
                base_kwargs["epochs"] = args.epochs
            if args.batch_size is not None:
                base_kwargs["batch_size"] = args.batch_size
            if args.sample_interactions is not None:
                base_kwargs["sample_interactions"] = args.sample_interactions
            if args.loader_max_rows is not None:
                base_kwargs["loader_max_rows"] = args.loader_max_rows

            config = make_ablation_config(variant, **base_kwargs)

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
                    "variant": variant,
                    "exp_id": result["exp_id"],
                    "metrics": result["test_metrics"],
                    "elapsed_s": elapsed,
                }
            )
            completed += 1
            logger.info(f"Completed in {elapsed:.1f}s")

        except Exception as e:
            failed += 1
            logger.error(f"FAILED: {e}")
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 70)
    print(f"ABLATION SUMMARY ({args.dataset})")
    print("=" * 70)
    print(f"Completed: {completed}/{len(args.variants)}")
    print(f"Failed: {failed}/{len(args.variants)}")
    print(f"Skipped via resume: {skipped}")

    if results:
        # Find baseline (mainline) metrics for delta computation.
        baseline_metrics = None
        for r in results:
            if r["variant"] == "mainline":
                baseline_metrics = r["metrics"]
                break

        print("Note: AvgPop@20 and AvgPop@40 are lower-is-better.")
        print(
            f"\n{'Variant':<20} | {'NDCG@20':>8} | {'Recall@20':>10} | {'AvgPop@20':>10} | {'NDCG@40':>8} | {'Recall@40':>10} | {'AvgPop@40':>10} | Time"
        )
        print("-" * 117)
        for r in results:
            metric_values = {
                metric_name: metric_value(r["metrics"], metric_name)
                for metric_name in THESIS_PRIMARY_METRICS
            }

            print(
                f"{r['variant']:<20} | {metric_values['NDCG@20']:>8.4f} | "
                f"{metric_values['Recall@20']:>10.4f} | {metric_values['AveragePopularity@20']:>10.4f} | "
                f"{metric_values['NDCG@40']:>8.4f} | {metric_values['Recall@40']:>10.4f} | "
                f"{metric_values['AveragePopularity@40']:>10.4f} | {r['elapsed_s']:.0f}s"
            )
            if baseline_metrics and r["variant"] != "mainline":
                deltas = []
                for metric_name in THESIS_PRIMARY_METRICS:
                    delta = metric_values[metric_name] - metric_value(
                        baseline_metrics, metric_name
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
