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

logger = logging.getLogger("ucagnn.ablation")


def main():
    parser = argparse.ArgumentParser(description="Run U-CaGNN ablation study")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--variants", nargs="*", default=list(ABLATION_VARIANTS.keys()),
                        help="Ablation variants to run")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
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
            override_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(baseline)"
            print(f"{i:>4} | {variant:<20} | {override_str}")
        print(f"\nTotal: {len(args.variants)} variants (dry run, nothing executed)")
        return 0

    # Run ablation experiments
    completed = 0
    failed = 0
    results = []

    for i, variant in enumerate(args.variants, 1):
        print(f"\n{'='*70}")
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

            config = make_ablation_config(variant, **base_kwargs)

            t0 = time.time()
            result = run_experiment(
                config,
                preset="full",
                intervention=variant,
                save_checkpoint=True,
            )
            elapsed = time.time() - t0

            results.append({
                "variant": variant,
                "exp_id": result["exp_id"],
                "metrics": result["test_metrics"],
                "elapsed_s": elapsed,
            })
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

        print(f"\n{'Variant':<20} | {'NDCG@50':>8} | {'Delta':>7} | {'Recall@50':>10} | {'Delta':>7} | Time")
        print("-" * 80)
        for r in results:
            ndcg = r["metrics"].get("NDCG@50", r["metrics"].get("NDCG@20", 0.0))
            recall = r["metrics"].get("Recall@50", r["metrics"].get("Recall@20", 0.0))

            if baseline_metrics and r["variant"] != "full":
                base_ndcg = baseline_metrics.get("NDCG@50", baseline_metrics.get("NDCG@20", 0.0))
                base_recall = baseline_metrics.get("Recall@50", baseline_metrics.get("Recall@20", 0.0))
                ndcg_delta = f"{ndcg - base_ndcg:+.4f}"
                recall_delta = f"{recall - base_recall:+.4f}"
            else:
                ndcg_delta = "  base"
                recall_delta = "  base"

            print(
                f"{r['variant']:<20} | {ndcg:>8.4f} | {ndcg_delta:>7} | "
                f"{recall:>10.4f} | {recall_delta:>7} | {r['elapsed_s']:.0f}s"
            )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
