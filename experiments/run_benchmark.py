#!/usr/bin/env python
"""Formal matrix benchmark runner for dataset x preset x training_mode x graph_method.

Usage:
    python experiments/run_benchmark.py --tier small --dry-run
    python experiments/run_benchmark.py --tier small --seeds 13 298 132
    python experiments/run_benchmark.py --tier all
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from experiments.run_experiment import DB_PATH, PRESETS, build_config, run_experiment
from src.training import THESIS_PRIMARY_METRICS
from src.utils.experiment_logger import ExperimentLogger

logger = logging.getLogger("ucagnn.benchmark")

TIERS = {
    "small": ["amazonbook", "movielens1m"],
    "medium": ["movielens20m", "kuairec_v2"],
    "large": ["taobao", "kuairand1k"],
}
TIERS["all"] = TIERS["small"] + TIERS["medium"] + TIERS["large"]

DEFAULT_SEEDS = [13, 298, 132]
DEFAULT_TRAINING_MODES = ["full_graph", "cached_propagation", "mini_batch"]
DEFAULT_GRAPH_METHODS = ["dense", "knn", "cagra"]


def _resolve_batch_id(provided: str | None, prefix: str) -> str:
    """Return an explicit or generated batch id for grouped runs."""
    if provided:
        return provided
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return whether an exception represents a CUDA OOM failure."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _metric_value(metrics: dict[str, float], metric_name: str) -> float:
    """Return a metric value, falling back from @50 to @20 when needed."""
    if metric_name in metrics:
        return metrics[metric_name]
    if metric_name.endswith("@50"):
        fallback = metric_name.replace("@50", "@20")
        return metrics.get(fallback, 0.0)
    return 0.0


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""
    parser = argparse.ArgumentParser(description="Run U-CaGNN benchmark matrix")
    parser.add_argument(
        "--tier", choices=list(TIERS.keys()), default="small", help="Dataset tier"
    )
    parser.add_argument(
        "--presets", nargs="*", default=list(PRESETS.keys()), help="Presets to run"
    )
    parser.add_argument(
        "--training-modes",
        nargs="*",
        default=DEFAULT_TRAINING_MODES,
        choices=DEFAULT_TRAINING_MODES,
        help="Training modes to run across the matrix",
    )
    parser.add_argument(
        "--graph-methods",
        nargs="*",
        default=DEFAULT_GRAPH_METHODS,
        choices=DEFAULT_GRAPH_METHODS,
        help="Graph construction methods to run across the matrix",
    )
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=DEFAULT_SEEDS, help="Random seeds"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs for all"
    )
    parser.add_argument(
        "--sample-interactions",
        type=int,
        default=None,
        help="Optional interaction budget for sampled benchmark passes.",
    )
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking for all benchmark runs",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Override MLflow tracking URI for all benchmark runs",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="ucagnn-benchmark",
        help="MLflow experiment name for benchmark runs",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional batch identifier for grouping and resuming benchmark runs.",
    )
    parser.add_argument(
        "--resume-batch",
        action="store_true",
        help="Skip benchmark items already recorded with a terminal status for this batch id.",
    )
    parser.add_argument(
        "--fallback-on-oom",
        choices=["none", "cached_propagation", "mini_batch"],
        default="none",
        help="Optional fallback training mode to retry after a CUDA OOM. The fallback is logged as a separate explicit run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without running"
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> int:
    """Execute the formal benchmark matrix from parsed arguments."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    datasets = TIERS[args.tier]
    batch_id = _resolve_batch_id(args.batch_id, prefix="benchmark")
    experiments = []
    for dataset in datasets:
        for preset in args.presets:
            for training_mode in args.training_modes:
                for graph_method in args.graph_methods:
                    for seed in args.seeds:
                        experiments.append(
                            (dataset, preset, training_mode, graph_method, seed)
                        )

    print("=" * 70)
    print(f"BENCHMARK PLAN: {len(experiments)} experiments")
    print(f"  Tier: {args.tier} ({len(datasets)} datasets: {', '.join(datasets)})")
    print(f"  Presets: {', '.join(args.presets)}")
    print(f"  Training modes: {', '.join(args.training_modes)}")
    print(f"  Graph methods: {', '.join(args.graph_methods)}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Batch ID: {batch_id}")
    print(f"  Resume batch: {args.resume_batch}")
    print(f"  Fallback on OOM: {args.fallback_on_oom}")
    print("=" * 70)

    if args.dry_run:
        print(
            f"\n{'#':>4} | {'Dataset':<15} | {'Preset':<12} | {'Mode':<18} | {'Graph':<5} | Seed"
        )
        print("-" * 86)
        for i, (ds, pr, tm, gm, sd) in enumerate(experiments, 1):
            print(f"{i:>4} | {ds:<15} | {pr:<12} | {tm:<18} | {gm:<5} | {sd}")
        print(f"\nTotal: {len(experiments)} experiments (dry run, nothing executed)")
        return 0

    # Run experiments
    tracker = ExperimentLogger(db_path=str(DB_PATH))
    completed = 0
    failed = 0
    skipped = 0
    fallback_completed = 0
    results = []

    for i, (dataset, preset, training_mode, graph_method, seed) in enumerate(
        experiments, 1
    ):
        print(f"\n{'=' * 70}")
        print(
            f"[{i}/{len(experiments)}] {dataset} / {preset} / {training_mode} / {graph_method} / seed={seed}"
        )
        print("=" * 70)

        existing = tracker.find_latest_batch_experiment(
            batch_id=batch_id,
            dataset=dataset,
            preset=preset,
            intervention=None,
            seed=seed,
            training_mode=training_mode,
            graph_method=graph_method,
        )
        if (
            args.resume_batch
            and existing is not None
            and existing["status"] in ExperimentLogger.TERMINAL_STATUSES
        ):
            skipped += 1
            logger.info(
                "Skipping existing batch item (exp_id=%s, status=%s)",
                existing["id"],
                existing["status"],
            )
            if existing["status"] == "completed":
                results.append(
                    {
                        "dataset": dataset,
                        "preset": preset,
                        "training_mode": training_mode,
                        "graph_method": graph_method,
                        "seed": seed,
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
            # Build a namespace matching run_experiment.build_config expectations
            exp_args = argparse.Namespace(
                dataset=dataset,
                recipe=None,
                preset=preset,
                seed=seed,
                epochs=args.epochs,
                batch_size=None,
                embed_dim=None,
                lr=None,
                graph_method=graph_method,
                training_mode=training_mode,
                num_neighbors=None,
                sample_interactions=args.sample_interactions,
                device=args.device,
                data_dir=args.data_dir,
                intervention=None,
            )

            config = build_config(exp_args)
            t0 = time.time()
            result = run_experiment(
                config,
                preset=preset,
                enable_mlflow=not args.no_mlflow,
                mlflow_tracking_uri=args.mlflow_tracking_uri,
                mlflow_experiment_name=args.mlflow_experiment_name,
                batch_id=batch_id,
            )
            elapsed = time.time() - t0

            results.append(
                {
                    "dataset": dataset,
                    "preset": preset,
                    "training_mode": training_mode,
                    "graph_method": graph_method,
                    "seed": seed,
                    "exp_id": result["exp_id"],
                    "metrics": result["test_metrics"],
                    "elapsed_s": elapsed,
                }
            )
            completed += 1
            logger.info(f"Completed in {elapsed:.1f}s (exp_id={result['exp_id']})")

        except Exception as e:
            failed += 1
            logger.error(f"FAILED: {e}")
            traceback.print_exc()

            if (
                _is_cuda_oom(e)
                and args.fallback_on_oom != "none"
                and training_mode != args.fallback_on_oom
            ):
                logger.info(
                    "Retrying batch item with fallback training_mode=%s",
                    args.fallback_on_oom,
                )
                fallback_args = argparse.Namespace(
                    dataset=dataset,
                    recipe=None,
                    preset=preset,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=None,
                    embed_dim=None,
                    n_gnn_layers=None,
                    interest_gnn_layers=None,
                    conformity_gnn_layers=None,
                    lr=None,
                    eval_scoring_mode=None,
                    scoring_weight_mode=None,
                    use_features=None,
                    feature_policy=None,
                    graph_method=graph_method,
                    training_mode=args.fallback_on_oom,
                    num_neighbors=None,
                    sample_interactions=args.sample_interactions,
                    loader_max_rows=None,
                    device=args.device,
                    data_dir=args.data_dir,
                    intervention=None,
                )

                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    fallback_config = build_config(fallback_args)
                    t0 = time.time()
                    fallback_result = run_experiment(
                        fallback_config,
                        preset=preset,
                        enable_mlflow=not args.no_mlflow,
                        mlflow_tracking_uri=args.mlflow_tracking_uri,
                        mlflow_experiment_name=args.mlflow_experiment_name,
                        batch_id=batch_id,
                    )
                    elapsed = time.time() - t0
                    results.append(
                        {
                            "dataset": dataset,
                            "preset": preset,
                            "training_mode": args.fallback_on_oom,
                            "graph_method": graph_method,
                            "seed": seed,
                            "exp_id": fallback_result["exp_id"],
                            "metrics": fallback_result["test_metrics"],
                            "elapsed_s": elapsed,
                        }
                    )
                    fallback_completed += 1
                    logger.info(
                        "Fallback completed in %.1fs (exp_id=%s)",
                        elapsed,
                        fallback_result["exp_id"],
                    )
                except Exception as fallback_exc:
                    logger.error("Fallback FAILED: %s", fallback_exc)
                    traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Completed: {completed}/{len(experiments)}")
    print(f"Fallback completed: {fallback_completed}")
    print(f"Failed: {failed}/{len(experiments)}")
    print(f"Skipped via resume: {skipped}")

    if results:
        print("Note: AvgPop@20 and AvgPop@50 are lower-is-better.")
        print(
            f"\n{'Dataset':<15} | {'Preset':<12} | {'Mode':<18} | {'Graph':<5} | Seed | {'NDCG@20':>8} | {'Recall@20':>10} | {'AvgPop@20':>10} | {'NDCG@50':>8} | {'Recall@50':>10} | {'AvgPop@50':>10} | Time"
        )
        print("-" * 169)
        for r in results:
            ndcg_20 = _metric_value(r["metrics"], "NDCG@20")
            recall_20 = _metric_value(r["metrics"], "Recall@20")
            avg_pop_20 = _metric_value(r["metrics"], "AveragePopularity@20")
            ndcg_50 = _metric_value(r["metrics"], "NDCG@50")
            recall_50 = _metric_value(r["metrics"], "Recall@50")
            avg_pop_50 = _metric_value(r["metrics"], "AveragePopularity@50")
            print(
                f"{r['dataset']:<15} | {r['preset']:<12} | {r['training_mode']:<18} | {r['graph_method']:<5} | {r['seed']:>4} | "
                f"{ndcg_20:>8.4f} | {recall_20:>10.4f} | {avg_pop_20:>10.4f} | {ndcg_50:>8.4f} | {recall_50:>10.4f} | {avg_pop_50:>10.4f} | {r['elapsed_s']:.0f}s"
            )

    tracker.close()
    return 0 if failed == 0 else 1


def main() -> int:
    """Parse CLI arguments and run the benchmark matrix."""
    parser = build_parser()
    args = parser.parse_args()
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
