#!/usr/bin/env python
"""Validate all 6 benchmark loaders and print per-dataset statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loaders import LOADERS, load_dataset

BENCHMARK_DATASETS = [
    "amazonbook",
    "movielens1m",
    "movielens20m",
    "kuairec_v2",
    "taobao",
    "kuairand1k",
]


def describe(name: str, data_dir: str = "data") -> dict | None:
    """Load a dataset and return a stats dict, or None on failure."""
    try:
        c = load_dataset(name, data_dir)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None

    print(f"  {repr(c)}")

    stats = {
        "dataset": name,
        "n_users": c.n_users,
        "n_items": c.n_items,
        "n_interactions": len(c),
        "density": len(c) / (c.n_users * c.n_items) * 100
        if c.n_users * c.n_items > 0
        else 0,
        "pos_rate": float(c.label.mean()),
        "sign_min": float(c.sign.min()),
        "sign_q25": float(np.percentile(c.sign, 25)),
        "sign_median": float(np.median(c.sign)),
        "sign_q75": float(np.percentile(c.sign, 75)),
        "sign_max": float(c.sign.max()),
        "user_feat_shape": c.user_features.shape
        if c.user_features is not None
        else None,
        "item_feat_shape": c.item_features.shape
        if c.item_features is not None
        else None,
        "has_predefined_splits": c.train_mask is not None,
        "split_source": "predefined" if c.train_mask is not None else "derived",
        "feedback_type": c.feedback_type,
        "preprocessing_preset": c.preprocessing_preset,
    }

    # Split sizes
    resolved_train, resolved_val, resolved_test = c.get_splits()
    stats["train_size"] = int(resolved_train.sum())
    stats["val_size"] = int(resolved_val.sum())
    stats["test_size"] = int(resolved_test.sum())
    if c.train_mask is not None:
        if c.val_mask is None and c.test_mask is not None:
            stats["split_source"] = "train/test"
        else:
            stats["split_source"] = "predefined"
    else:
        stats["split_source"] = "derived:per_user"
    print(
        f"  Splits ({stats['split_source']}): "
        f"train={stats['train_size']:,} val={stats['val_size']:,} test={stats['test_size']:,}"
    )

    # Popularity distribution
    pop = c.popularity
    print(
        f"  Popularity: min={pop.min():.4f} median={np.median(pop):.4f} max={pop.max():.4f}"
    )
    print(f"  Sign range: [{stats['sign_min']:.2f}, {stats['sign_max']:.2f}]")
    print(f"  Positive rate: {stats['pos_rate']:.2%}")
    if stats["feedback_type"] is not None:
        print(f"  Feedback type: {stats['feedback_type']}")
    if stats["preprocessing_preset"] is not None:
        print(f"  Preprocessing preset: {stats['preprocessing_preset']}")
    if c.raw_target is not None:
        print(
            "  Raw target range: "
            f"[{float(np.min(c.raw_target)):.2f}, {float(np.max(c.raw_target)):.2f}]"
        )
    if c.exposure_flag is not None:
        print(f"  Random exposure rate: {float(np.mean(c.exposure_flag)):.2%}")
    if c.behavior_type is not None:
        preview = np.unique(c.behavior_type).tolist()[:8]
        print(f"  Behavior types: {preview}")
    if c.source_domain is not None:
        preview = np.unique(c.source_domain).tolist()[:8]
        print(f"  Source domains: {preview}")

    if c.metadata:
        for k, v in c.metadata.items():
            if isinstance(v, np.ndarray):
                print(
                    f"  Metadata[{k}]: shape={v.shape}, unique={np.unique(v).tolist()[:10]}"
                )
            else:
                print(f"  Metadata[{k}]: {v}")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Explore all benchmark datasets")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--datasets", nargs="*", default=BENCHMARK_DATASETS, help="Datasets to explore"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("U-CaGNN DATASET EXPLORATION")
    print("=" * 70)

    results = []
    for name in args.datasets:
        print(f"\n--- {name} ---")
        if name not in LOADERS:
            print(f"  SKIPPED: unknown loader '{name}'")
            continue
        stats = describe(name, args.data_dir)
        if stats:
            results.append(stats)

    # Comparative summary table
    if results:
        print("\n" + "=" * 70)
        print("COMPARATIVE SUMMARY")
        print("=" * 70)
        print(
            f"{'Dataset':<15} {'Users':>10} {'Items':>10} {'Interact.':>12} {'Density':>8} {'Pos%':>7} {'Splits':>10}"
        )
        print("-" * 70)
        for s in results:
            print(
                f"{s['dataset']:<15} {s['n_users']:>10,} {s['n_items']:>10,} "
                f"{s['n_interactions']:>12,} {s['density']:>7.4f}% {s['pos_rate']:>6.2%} {s['split_source']:>10}"
            )

    print(f"\nLoaded {len(results)}/{len(args.datasets)} datasets successfully.")
    return 0 if len(results) == len(args.datasets) else 1


if __name__ == "__main__":
    sys.exit(main())
