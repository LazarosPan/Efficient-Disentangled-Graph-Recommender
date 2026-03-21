#!/usr/bin/env python
"""Query experiment results from SQLite database.

Usage:
    python scripts/query_results.py                    # Show all experiments
    python scripts/query_results.py --exp 1            # Show experiment 1 details
    python scripts/query_results.py --metrics 1        # Show metrics for experiment 1
    python scripts/query_results.py --profiling 1      # Show profiling for experiment 1
    python scripts/query_results.py --alpha 1          # Show alpha drift for experiment 1
    python scripts/query_results.py --bottleneck 1     # Show bottleneck breakdown for experiment 1
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DB_PATH = REPO_ROOT / "results" / "thesis_experiments.db"


def connect():
    """Connect to experiment database."""
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH.resolve()}")
        print("Run a real experiment first to create the persistent database in results/.")
        print("The verify scripts use temporary .db files and remove them when they finish.")
        sys.exit(1)

    from src.utils.experiment_logger import ExperimentLogger

    migrator = ExperimentLogger(db_path=str(DB_PATH))
    migrator.close()
    return sqlite3.connect(DB_PATH)


def list_experiments(conn):
    """List all experiments."""
    print("=" * 80)
    print("EXPERIMENTS")
    print("=" * 80)
    print(f"Database: {DB_PATH.resolve()}")
    print()

    rows = conn.execute("""
        SELECT id, timestamp, dataset, preset, intervention, seed
        FROM experiments ORDER BY id DESC
    """).fetchall()

    if not rows:
        print("No experiments found.")
        return

    print(f"{'ID':>4} | {'Timestamp':<20} | {'Dataset':<15} | {'Preset':<12} | {'Intervention':<12} | Seed")
    print("-" * 80)
    for row in rows:
        print(f"{row[0]:>4} | {row[1][:20]:<20} | {row[2] or '-':<15} | {row[3] or '-':<12} | {row[4] or '-':<12} | {row[5] or '-'}")


def show_experiment(conn, exp_id):
    """Show experiment details."""
    row = conn.execute("""
        SELECT id, timestamp, dataset, preset, intervention, config_json, seed
        FROM experiments WHERE id = ?
    """, (exp_id,)).fetchone()

    if not row:
        print(f"Experiment {exp_id} not found.")
        return

    print("=" * 80)
    print(f"EXPERIMENT {exp_id}")
    print("=" * 80)
    print(f"Database:     {DB_PATH.resolve()}")
    print(f"Timestamp:    {row[1]}")
    print(f"Dataset:      {row[2]}")
    print(f"Preset:       {row[3] or '-'}")
    print(f"Intervention: {row[4] or '-'}")
    print(f"Seed:         {row[6] or '-'}")
    print("\nConfig:")

    if row[5]:
        config = json.loads(row[5])
        for k, v in sorted(config.items()):
            print(f"  {k}: {v}")


def show_metrics(conn, exp_id):
    """Show metrics for an experiment."""
    print("=" * 80)
    print(f"METRICS (Experiment {exp_id})")
    print("=" * 80)

    # Group by split
    for split in ["train", "val", "test"]:
        rows = conn.execute("""
            SELECT epoch, metric_name, metric_value, timestamp
            FROM metrics
            WHERE experiment_id = ? AND split = ?
            ORDER BY epoch, metric_name, timestamp
        """, (exp_id, split)).fetchall()

        if rows:
            print(f"\n{split.upper()}:")
            print(f"  {'Epoch':>5} | {'Metric':<20} | {'Value':>10} | Timestamp")
            print("  " + "-" * 90)
            for row in rows:
                epoch_str = str(row[0]) if row[0] is not None else "final"
                print(f"  {epoch_str:>5} | {row[1]:<20} | {row[2]:>10.4f} | {row[3]}")


def show_profiling(conn, exp_id):
    """Show profiling breakdown for an experiment."""
    print("=" * 80)
    print(f"PROFILING (Experiment {exp_id})")
    print("=" * 80)

    # Summary by stage
    rows = conn.execute("""
        SELECT stage,
               SUM(duration_ms) as total_ms,
               AVG(duration_ms) as avg_epoch_ms,
               MIN(duration_ms) as min_epoch_ms,
               MAX(duration_ms) as max_epoch_ms,
               SUM(stage_call_count) as total_calls,
               SUM(duration_ms) / NULLIF(SUM(stage_call_count), 0) as avg_call_ms,
               MAX(vram_peak_mb) as peak_vram_mb,
               COUNT(*) as profiled_epochs,
               MIN(timestamp) as first_logged_at,
               MAX(timestamp) as last_logged_at
        FROM profiling
        WHERE experiment_id = ?
        GROUP BY stage
        ORDER BY total_ms DESC
    """, (exp_id,)).fetchall()

    if not rows:
        print("No profiling data found.")
        return

    total_ms = sum(row[1] for row in rows)

    print(
        f"\n{'Stage':<15} | {'Total/Epoch':>11} | {'Avg/Call':>10} | {'Calls':>6} | {'Epochs':>6} | {'%':>6} | Peak VRAM | Last Logged"
    )
    print("-" * 150)
    for row in rows:
        pct = (row[1] / total_ms * 100) if total_ms > 0 else 0
        peak_str = f"{row[7]:.0f} MB" if row[7] else "-"
        print(
            f"{row[0]:<15} | {row[2]:>11.1f} | {row[6]:>10.1f} | {row[5]:>6} | {row[8]:>6} | {pct:>5.1f}% | {peak_str:<9} | {row[10]}"
        )

    print("-" * 150)
    print(f"{'TOTAL':<15} | {total_ms:>11.1f}")


def show_alpha_drift(conn, exp_id):
    """Show alpha_pos/alpha_neg values over epochs."""
    print("=" * 80)
    print(f"ALPHA DRIFT (Experiment {exp_id})")
    print("=" * 80)

    rows = conn.execute("""
        SELECT epoch, metric_name, metric_value
        FROM metrics
        WHERE experiment_id = ? AND metric_name LIKE 'alpha%'
        ORDER BY epoch, metric_name
    """, (exp_id,)).fetchall()

    if not rows:
        print("No alpha values found. (Sign-aware mode may be disabled)")
        return

    print(f"\n{'Epoch':>5} | {'alpha_pos':>10} | {'alpha_neg':>10}")
    print("-" * 30)

    # Group by epoch
    from collections import defaultdict
    epochs = defaultdict(dict)
    for row in rows:
        epochs[row[0]][row[1]] = row[2]

    for epoch in sorted(epochs.keys()):
        alpha_pos = epochs[epoch].get("alpha_pos", float("nan"))
        alpha_neg = epochs[epoch].get("alpha_neg", float("nan"))
        print(f"{epoch:>5} | {alpha_pos:>10.4f} | {alpha_neg:>10.4f}")


def show_bottleneck(conn, exp_id):
    """Show bottleneck analysis (which stage takes most time)."""
    print("=" * 80)
    print(f"BOTTLENECK ANALYSIS (Experiment {exp_id})")
    print("=" * 80)

    rows = conn.execute("""
        SELECT stage,
               SUM(duration_ms) as total_ms,
             SUM(stage_call_count) as n_calls
        FROM profiling
        WHERE experiment_id = ?
        GROUP BY stage
        ORDER BY total_ms DESC
    """, (exp_id,)).fetchall()

    if not rows:
        print("No profiling data found.")
        return

    grand_total = sum(row[1] for row in rows)

    print(f"\n{'Rank':>4} | {'Stage':<15} | {'Total (ms)':>12} | {'Calls':>6} | {'% of Total':>10}")
    print("-" * 60)
    for i, row in enumerate(rows, 1):
        pct = (row[1] / grand_total * 100) if grand_total > 0 else 0
        print(f"{i:>4} | {row[0]:<15} | {row[1]:>12.1f} | {row[2]:>6} | {pct:>9.1f}%")

    print("-" * 60)
    print(f"\nBottleneck: {rows[0][0]} ({rows[0][1] / grand_total * 100:.1f}% of total time)")


def main():
    parser = argparse.ArgumentParser(description="Query experiment results")
    parser.add_argument("--exp", type=int, help="Show experiment details")
    parser.add_argument("--metrics", type=int, help="Show metrics for experiment")
    parser.add_argument("--profiling", type=int, help="Show profiling for experiment")
    parser.add_argument("--alpha", type=int, help="Show alpha drift for experiment")
    parser.add_argument("--bottleneck", type=int, help="Show bottleneck analysis for experiment")
    args = parser.parse_args()

    conn = connect()

    if args.exp:
        show_experiment(conn, args.exp)
    elif args.metrics:
        show_metrics(conn, args.metrics)
    elif args.profiling:
        show_profiling(conn, args.profiling)
    elif args.alpha:
        show_alpha_drift(conn, args.alpha)
    elif args.bottleneck:
        show_bottleneck(conn, args.bottleneck)
    else:
        list_experiments(conn)

    conn.close()


if __name__ == "__main__":
    main()
