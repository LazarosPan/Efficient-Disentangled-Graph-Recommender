#!/usr/bin/env python
"""Formal matrix benchmark runner for dataset x preset x training_mode x graph_method.

Usage:
    python experiments/run_benchmark.py --tier small --dry-run
    python experiments/run_benchmark.py --tier small --seeds 13 298 132
    python experiments/run_benchmark.py --tier all
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_experiment import build_config, run_experiment, PRESETS

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


def main():
    parser = argparse.ArgumentParser(description="Run U-CaGNN benchmark matrix")
    parser.add_argument("--tier", choices=list(TIERS.keys()), default="small", help="Dataset tier")
    parser.add_argument("--presets", nargs="*", default=list(PRESETS.keys()), help="Presets to run")
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
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS, help="Random seeds")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all")
    parser.add_argument(
        "--sample-interactions",
        type=int,
        default=None,
        help="Optional interaction budget for sampled benchmark passes.",
    )
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    datasets = TIERS[args.tier]
    experiments = []
    for dataset in datasets:
        for preset in args.presets:
            for training_mode in args.training_modes:
                for graph_method in args.graph_methods:
                    for seed in args.seeds:
                        experiments.append((dataset, preset, training_mode, graph_method, seed))

    print("=" * 70)
    print(f"BENCHMARK PLAN: {len(experiments)} experiments")
    print(f"  Tier: {args.tier} ({len(datasets)} datasets: {', '.join(datasets)})")
    print(f"  Presets: {', '.join(args.presets)}")
    print(f"  Training modes: {', '.join(args.training_modes)}")
    print(f"  Graph methods: {', '.join(args.graph_methods)}")
    print(f"  Seeds: {args.seeds}")
    print("=" * 70)

    if args.dry_run:
        print(f"\n{'#':>4} | {'Dataset':<15} | {'Preset':<12} | {'Mode':<18} | {'Graph':<5} | Seed")
        print("-" * 86)
        for i, (ds, pr, tm, gm, sd) in enumerate(experiments, 1):
            print(f"{i:>4} | {ds:<15} | {pr:<12} | {tm:<18} | {gm:<5} | {sd}")
        print(f"\nTotal: {len(experiments)} experiments (dry run, nothing executed)")
        return 0

    # Run experiments
    completed = 0
    failed = 0
    results = []

    for i, (dataset, preset, training_mode, graph_method, seed) in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(
            f"[{i}/{len(experiments)}] {dataset} / {preset} / {training_mode} / {graph_method} / seed={seed}"
        )
        print("=" * 70)

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
            result = run_experiment(config, preset=preset)
            elapsed = time.time() - t0

            results.append({
                "dataset": dataset,
                "preset": preset,
                "training_mode": training_mode,
                "graph_method": graph_method,
                "seed": seed,
                "exp_id": result["exp_id"],
                "metrics": result["test_metrics"],
                "elapsed_s": elapsed,
            })
            completed += 1
            logger.info(f"Completed in {elapsed:.1f}s (exp_id={result['exp_id']})")

        except Exception as e:
            failed += 1
            logger.error(f"FAILED: {e}")
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Completed: {completed}/{len(experiments)}")
    print(f"Failed: {failed}/{len(experiments)}")

    if results:
        print(
            f"\n{'Dataset':<15} | {'Preset':<12} | {'Mode':<18} | {'Graph':<5} | Seed | {'NDCG@50':>8} | {'Recall@50':>10} | Time"
        )
        print("-" * 114)
        for r in results:
            ndcg = r["metrics"].get("NDCG@50", r["metrics"].get("NDCG@20", 0.0))
            recall = r["metrics"].get("Recall@50", r["metrics"].get("Recall@20", 0.0))
            print(
                f"{r['dataset']:<15} | {r['preset']:<12} | {r['training_mode']:<18} | {r['graph_method']:<5} | {r['seed']:>4} | "
                f"{ndcg:>8.4f} | {recall:>10.4f} | {r['elapsed_s']:.0f}s"
            )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
