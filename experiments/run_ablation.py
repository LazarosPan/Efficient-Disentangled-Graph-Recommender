#!/usr/bin/env python
"""Ablation study runner: test each U-CaGNN component's contribution.

Runs 8 variants (full + 7 component removals) on specified datasets.

Usage:
    python experiments/run_ablation.py --dataset movielens1m --epochs 3
    python experiments/run_ablation.py --dataset movielens1m --variants full no_ipw no_dual_branch
    python experiments/run_ablation.py --dataset movielens1m --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.ablation_configs import ABLATION_VARIANTS, make_ablation_config
from experiments.run_experiment import run_experiment
from src.training import THESIS_PRIMARY_METRICS

logger = logging.getLogger("ucagnn.ablation")


def _metric_value(metrics: dict[str, float], metric_name: str) -> float:
    """Return a metric value, falling back from @50 to @20 when needed."""
    if metric_name in metrics:
        return metrics[metric_name]
    if metric_name.endswith("@50"):
        fallback = metric_name.replace("@50", "@20")
        return metrics.get(fallback, 0.0)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Run U-CaGNN ablation study")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(ABLATION_VARIANTS.keys()),
        help="Ablation variants to run",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--sample-interactions",
        type=int,
        default=None,
        help="Optional interaction budget for sampled ablation smoke runs.",
    )
    parser.add_argument(
        "--loader-max-rows",
        type=int,
        default=None,
        help="Optional early row cap for dataset loading during fast ablation smoke runs.",
    )
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking for all ablation runs",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Override MLflow tracking URI for all ablation runs",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="ucagnn-ablation",
        help="MLflow experiment name for ablation runs",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without running"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate variants
    for v in args.variants:
        if v not in ABLATION_VARIANTS:
            print(f"Unknown variant: {v}. Available: {list(ABLATION_VARIANTS.keys())}")
            return 1

    print("=" * 70)
    print(f"ABLATION STUDY: {args.dataset}")
    print(f"  Variants: {', '.join(args.variants)}")
    print(f"  Seed: {args.seed}")
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
    completed = 0
    failed = 0
    results = []

    for i, variant in enumerate(args.variants, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(args.variants)}] Ablation: {variant}")
        print("=" * 70)

        try:
            base_kwargs = {
                "dataset": args.dataset,
                "data_dir": args.data_dir,
                "seed": args.seed,
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
                preset="full",
                intervention=variant,
                save_checkpoint=True,
                enable_mlflow=not args.no_mlflow,
                mlflow_tracking_uri=args.mlflow_tracking_uri,
                mlflow_experiment_name=args.mlflow_experiment_name,
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

    if results:
        # Find baseline (full) metrics for delta computation
        baseline_metrics = None
        for r in results:
            if r["variant"] == "full":
                baseline_metrics = r["metrics"]
                break

        print("Note: AvgPop@20 and AvgPop@50 are lower-is-better.")
        print(
            f"\n{'Variant':<20} | {'NDCG@20':>8} | {'Recall@20':>10} | {'AvgPop@20':>10} | {'NDCG@50':>8} | {'Recall@50':>10} | {'AvgPop@50':>10} | Time"
        )
        print("-" * 117)
        for r in results:
            metric_values = {
                metric_name: _metric_value(r["metrics"], metric_name)
                for metric_name in THESIS_PRIMARY_METRICS
            }

            print(
                f"{r['variant']:<20} | {metric_values['NDCG@20']:>8.4f} | "
                f"{metric_values['Recall@20']:>10.4f} | {metric_values['AveragePopularity@20']:>10.4f} | "
                f"{metric_values['NDCG@50']:>8.4f} | {metric_values['Recall@50']:>10.4f} | "
                f"{metric_values['AveragePopularity@50']:>10.4f} | {r['elapsed_s']:.0f}s"
            )
            if baseline_metrics and r["variant"] != "full":
                deltas = []
                for metric_name in THESIS_PRIMARY_METRICS:
                    delta = metric_values[metric_name] - _metric_value(
                        baseline_metrics, metric_name
                    )
                    deltas.append(f"{metric_name} {delta:+.4f}")
                print(f"{'':<20}   deltas: {' | '.join(deltas)}")
            else:
                print(f"{'':<20}   deltas: baseline")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
