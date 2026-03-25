#!/usr/bin/env python
"""Generate thesis-quality figures from SQLite experiment results.

Usage:
    python scripts/visualize_results.py                    # All plots
    python scripts/visualize_results.py --plot performance  # Just performance table
    python scripts/visualize_results.py --plot ablation     # Ablation heatmap
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import LOWER_IS_BETTER_METRICS, THESIS_PRIMARY_METRICS  # noqa: E402

DB_PATH = REPO_ROOT / "results" / "thesis_experiments.db"
FIGURE_DIR = REPO_ROOT / "results" / "figures"
THESIS_METRIC_LABELS = {
    "NDCG@20": "NDCG@20",
    "Recall@20": "Recall@20",
    "AveragePopularity@20": "AveragePopularity@20 (lower is better)",
    "NDCG@50": "NDCG@50",
    "Recall@50": "Recall@50",
    "AveragePopularity@50": "AveragePopularity@50 (lower is better)",
}

# Thesis-quality defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def connect():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run an experiment first.")
        sys.exit(1)

    from src.utils.experiment_logger import ExperimentLogger

    migrator = ExperimentLogger(db_path=str(DB_PATH))
    migrator.close()
    return sqlite3.connect(DB_PATH)


def plot_performance_table(conn):
    """Generate thesis-primary performance comparisons across datasets."""
    rows = conn.execute("""
        SELECT e.dataset, e.preset, m.metric_name, AVG(m.metric_value) as avg_val,
               COUNT(DISTINCT e.seed) as n_seeds
        FROM experiments e
        JOIN metrics m ON m.experiment_id = e.id
        WHERE m.split = 'test' AND m.epoch IS NULL
          AND e.intervention IS NULL
        GROUP BY e.dataset, e.preset, m.metric_name
        ORDER BY e.dataset, e.preset
    """).fetchall()

    if not rows:
        print("No test metrics found for performance table.")
        return

    # Organize data
    data = defaultdict(lambda: defaultdict(dict))
    for dataset, preset, metric, avg_val, n_seeds in rows:
        data[dataset][preset][metric] = (avg_val, n_seeds)

    datasets = sorted(data.keys())
    presets = sorted({p for d in data.values() for p in d})
    fig, axes = plt.subplots(2, 3, figsize=(max(12, len(datasets) * 2.8), 8))
    axes = axes.flatten()
    x = np.arange(len(datasets))
    width = 0.8 / max(len(presets), 1)

    for axis, metric_name in zip(axes, THESIS_PRIMARY_METRICS):
        for i, preset in enumerate(presets):
            values = [
                data[ds].get(preset, {}).get(metric_name, (0, 0))[0] for ds in datasets
            ]
            offset = (i - len(presets) / 2 + 0.5) * width
            axis.bar(x + offset, values, width * 0.9, label=preset or "custom")

        axis.set_xlabel("Dataset")
        axis.set_ylabel(THESIS_METRIC_LABELS[metric_name])
        axis.set_title(THESIS_METRIC_LABELS[metric_name])
        axis.set_xticks(x)
        axis.set_xticklabels(datasets, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best")

    fig.suptitle("Thesis-Primary Performance Comparison", fontsize=14)
    fig.tight_layout()

    path = FIGURE_DIR / "performance_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_training_curves(conn):
    """Plot training loss vs epoch for each experiment."""
    experiments = conn.execute("""
        SELECT id, dataset, preset, seed FROM experiments
        ORDER BY dataset, preset
    """).fetchall()

    if not experiments:
        return

    # Group by (dataset, preset)
    groups = defaultdict(list)
    for exp_id, dataset, preset, seed in experiments:
        groups[(dataset, preset or "custom")].append(exp_id)

    fig, axes = plt.subplots(
        1, min(len(groups), 4), figsize=(5 * min(len(groups), 4), 4), squeeze=False
    )
    axes = axes.flatten()

    for idx, ((dataset, preset), exp_ids) in enumerate(sorted(groups.items())):
        if idx >= len(axes):
            break
        ax = axes[idx]
        for exp_id in exp_ids:
            losses = conn.execute(
                """
                SELECT epoch, metric_value FROM metrics
                WHERE experiment_id = ? AND split = 'train' AND metric_name = 'loss'
                ORDER BY epoch
            """,
                (exp_id,),
            ).fetchall()
            if losses:
                epochs, values = zip(*losses)
                ax.plot(epochs, values, alpha=0.7)

        ax.set_title(f"{dataset}\n{preset}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

    fig.suptitle("Training Curves", fontsize=14)
    fig.tight_layout()
    path = FIGURE_DIR / "training_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_profiling_breakdown(conn):
    """Compare profiling stage cost as Avg Total/Epoch and Avg/Call."""
    rows = conn.execute("""
        SELECT e.dataset,
               e.preset,
               p.stage,
               AVG(p.duration_ms) as avg_epoch_total_ms,
               SUM(p.duration_ms) / NULLIF(SUM(p.stage_call_count), 0) as avg_call_ms
        FROM experiments e
        JOIN profiling p ON p.experiment_id = e.id
        GROUP BY e.dataset, e.preset, p.stage
        ORDER BY e.dataset, e.preset
    """).fetchall()

    if not rows:
        print("No profiling data found.")
        return

    # Organize
    epoch_total_data = defaultdict(lambda: defaultdict(float))
    avg_call_data = defaultdict(lambda: defaultdict(float))
    labels = set()
    for dataset, preset, stage, avg_epoch_total_ms, avg_call_ms in rows:
        key = f"{dataset}\n{preset or 'custom'}"
        epoch_total_data[key][stage] = avg_epoch_total_ms
        avg_call_data[key][stage] = avg_call_ms
        labels.add(stage)

    groups = sorted(epoch_total_data.keys())
    stages = sorted(labels)

    fig, axes = plt.subplots(
        1, 2, figsize=(max(12, len(groups) * 2.4), 5), squeeze=False
    )
    ax_epoch, ax_call = axes[0]
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))

    for stage in stages:
        values = [epoch_total_data[g].get(stage, 0) for g in groups]
        ax_epoch.bar(x, values, 0.6, bottom=bottom, label=stage)
        bottom += values

    ax_epoch.set_xlabel("Experiment")
    ax_epoch.set_ylabel("Avg Total/Epoch (ms)")
    ax_epoch.set_title("Stage Cost by Experiment")
    ax_epoch.set_xticks(x)
    ax_epoch.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax_epoch.legend(loc="upper left", fontsize=8)
    ax_epoch.grid(axis="y", alpha=0.3)

    width = 0.8 / max(len(stages), 1)
    for idx, stage in enumerate(stages):
        values = [avg_call_data[g].get(stage, 0) for g in groups]
        offset = (idx - len(stages) / 2 + 0.5) * width
        ax_call.bar(x + offset, values, width * 0.9, label=stage)

    ax_call.set_xlabel("Experiment")
    ax_call.set_ylabel("Avg/Call (ms)")
    ax_call.set_title("Per-Call Stage Latency")
    ax_call.set_xticks(x)
    ax_call.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax_call.grid(axis="y", alpha=0.3)

    handles, labels = ax_call.get_legend_handles_labels()
    if handles:
        ax_call.legend(loc="upper left", fontsize=8)

    fig.suptitle("Profiling Breakdown: Avg Total/Epoch vs Avg/Call", fontsize=14)
    fig.tight_layout()

    path = FIGURE_DIR / "profiling_breakdown.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scaling_analysis(conn):
    """Wall-clock time vs dataset size."""
    rows = conn.execute("""
        SELECT e.dataset, SUM(p.duration_ms) / COUNT(DISTINCT e.id) as avg_total_ms
        FROM experiments e
        JOIN profiling p ON p.experiment_id = e.id
        GROUP BY e.dataset
        ORDER BY avg_total_ms
    """).fetchall()

    if not rows:
        return

    datasets, times = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(datasets)), [t / 1000 for t in times], color="steelblue")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Total Training Time (seconds)")
    ax.set_title("Scaling Analysis: Wall-clock Time per Dataset")
    ax.grid(axis="x", alpha=0.3)

    path = FIGURE_DIR / "scaling_analysis.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ablation_heatmap(conn):
    """Ablation heatmaps for thesis-primary metrics."""
    rows = conn.execute("""
        SELECT e.dataset, e.intervention, m.metric_name, AVG(m.metric_value) as avg_val
        FROM experiments e
        JOIN metrics m ON m.experiment_id = e.id
        WHERE m.split = 'test' AND m.epoch IS NULL
          AND e.intervention IS NOT NULL
        GROUP BY e.dataset, e.intervention, m.metric_name
        ORDER BY e.dataset, e.intervention
    """).fetchall()

    if not rows:
        print("No ablation results found.")
        return

    data = defaultdict(lambda: defaultdict(dict))
    for dataset, intervention, metric, avg_val in rows:
        if metric in THESIS_PRIMARY_METRICS:
            data[dataset][intervention][metric] = avg_val

    if not data:
        print("No thesis-primary ablation results found.")
        return

    datasets = sorted(data.keys())
    all_variants = set()
    for d in data.values():
        all_variants.update(d.keys())
    variants = sorted(all_variants)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(max(12, len(variants) * 2.2), max(7, len(datasets) * 1.5)),
    )
    axes = axes.flatten()

    for axis, metric_name in zip(axes, THESIS_PRIMARY_METRICS):
        matrix = np.full((len(datasets), len(variants)), np.nan)
        for i, ds in enumerate(datasets):
            baseline = data[ds].get("full", {}).get(metric_name)
            for j, var in enumerate(variants):
                metric_value = data[ds].get(var, {}).get(metric_name)
                if baseline is not None and metric_value is not None:
                    matrix[i, j] = metric_value - baseline

        masked = np.ma.masked_invalid(matrix)
        finite = matrix[np.isfinite(matrix)]
        vmax = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 0.001)
        cmap = "RdYlGn_r" if metric_name in LOWER_IS_BETTER_METRICS else "RdYlGn"
        im = axis.imshow(masked, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

        axis.set_xticks(range(len(variants)))
        axis.set_xticklabels(variants, rotation=45, ha="right", fontsize=9)
        axis.set_yticks(range(len(datasets)))
        axis.set_yticklabels(datasets)
        axis.set_title(f"{THESIS_METRIC_LABELS[metric_name]} Delta")

        for i in range(len(datasets)):
            for j in range(len(variants)):
                if not np.isnan(matrix[i, j]):
                    axis.text(
                        j,
                        i,
                        f"{matrix[i, j]:+.4f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("Ablation Deltas for Thesis-Primary Metrics", fontsize=14)
    fig.tight_layout()
    path = FIGURE_DIR / "ablation_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


PLOT_FUNCTIONS = {
    "performance": plot_performance_table,
    "training": plot_training_curves,
    "profiling": plot_profiling_breakdown,
    "scaling": plot_scaling_analysis,
    "ablation": plot_ablation_heatmap,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis figures from experiment results"
    )
    parser.add_argument(
        "--plot", choices=list(PLOT_FUNCTIONS.keys()), help="Generate specific plot"
    )
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    conn = connect()

    if args.plot:
        PLOT_FUNCTIONS[args.plot](conn)
    else:
        for name, func in PLOT_FUNCTIONS.items():
            print(f"\nGenerating: {name}...")
            try:
                func(conn)
            except Exception as e:
                print(f"  Failed: {e}")

    conn.close()
    print(f"\nFigures saved to: {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
