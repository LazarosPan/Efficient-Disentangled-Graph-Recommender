#!/usr/bin/env python
"""Query experiment results from SQLite database.

Usage:
    python scripts/query_results.py                    # Show all experiments
    python scripts/query_results.py --view completed   # Show only completed runs
    python scripts/query_results.py --view attention   # Show failed, OOM, running, or unknown runs
    python scripts/query_results.py --view errors      # Show only failed and OOM runs
    python scripts/query_results.py --batch-id foo    # Show one batch only
    python scripts/query_results.py --status oom      # Show only OOM rows
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

VIEW_TABLES = {
    "all": "experiments",
    "completed": "experiment_completed_summary",
    "attention": "experiment_attention_summary",
    "errors": "experiment_error_summary",
}


def connect():
    """Connect to experiment database."""
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH.resolve()}")
        print(
            "Run a real experiment first to create the persistent database in results/."
        )
        print(
            "The verify scripts use temporary .db files and remove them when they finish."
        )
        sys.exit(1)

    from src.utils.experiment_logger import ExperimentLogger

    migrator = ExperimentLogger(db_path=str(DB_PATH))
    migrator.close()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def list_experiments(
    conn: sqlite3.Connection,
    *,
    batch_id: str | None = None,
    status: str | None = None,
    view_name: str = "all",
) -> None:
    """List all experiments."""
    print("=" * 80)
    print("EXPERIMENTS")
    print("=" * 80)
    print(f"Database: {DB_PATH.resolve()}")
    print(f"View: {view_name}")
    print()

    where_clauses: list[str] = []
    params: list[object] = []
    if batch_id:
        where_clauses.append("batch_id = ?")
        params.append(batch_id)
    if status:
        where_clauses.append("status = ?")
        params.append(status)

    source_table = VIEW_TABLES[view_name]
    sql = f"""
         SELECT id, timestamp, dataset, preset, intervention, seed, status,
             batch_id, profile_name, training_mode, graph_method, oom_flag
        FROM {source_table}
    """
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += " ORDER BY id DESC"

    rows = conn.execute(sql, params).fetchall()

    if not rows:
        print("No experiments found.")
        return

    print(
        f"{'ID':>4} | {'Status':<10} | {'Dataset':<14} | {'Preset':<12} | {'Profile':<8} | {'Mode':<18} | {'Graph':<8} | {'Batch':<22} | Seed"
    )
    print("-" * 126)
    for row in rows:
        batch_label = row["batch_id"] or "-"
        if len(batch_label) > 22:
            batch_label = f"{batch_label[:19]}..."
        status_label = row["status"] or ("oom" if row["oom_flag"] else "unknown")
        print(
            f"{row['id']:>4} | {status_label:<10} | {row['dataset'] or '-':<14} | {row['preset'] or '-':<12} | {row['profile_name'] or '-':<8} | {row['training_mode'] or '-':<18} | {row['graph_method'] or '-':<8} | {batch_label:<22} | {row['seed'] or '-'}"
        )


def show_experiment(conn: sqlite3.Connection, exp_id: int) -> None:
    """Show experiment details."""
    row = conn.execute(
        """
        SELECT id, timestamp, dataset, preset, intervention, config_json, seed,
             status, failure_reason, oom_flag, batch_id, profile_name,
             gpu_name, gpu_vram_gb, training_mode, graph_method, updated_at
        FROM experiments WHERE id = ?
    """,
        (exp_id,),
    ).fetchone()

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
    print(f"Status:       {row[7] or '-'}")
    print(f"Training:     {row[13] or '-'}")
    print(f"Graph:        {row[14] or '-'}")
    print(f"Batch ID:     {row[10] or '-'}")
    print(f"Profile:      {row[11] or '-'}")
    print(f"GPU:          {row[12] or '-'}")
    print(f"VRAM (GiB):   {row[13] if row[13] is not None else '-'}")
    print(f"Updated:      {row[16] or '-'}")
    if row[9]:
        print(f"OOM Flag:     {bool(row[9])}")
    if row[8]:
        print(f"Failure:      {row[8]}")
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
        rows = conn.execute(
            """
            SELECT epoch, metric_name, metric_value, timestamp
            FROM metrics
            WHERE experiment_id = ? AND split = ?
            ORDER BY epoch, metric_name, timestamp
        """,
            (exp_id, split),
        ).fetchall()

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
    rows = conn.execute(
        """
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
    """,
        (exp_id,),
    ).fetchall()

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

    rows = conn.execute(
        """
        SELECT epoch, metric_name, metric_value
        FROM metrics
        WHERE experiment_id = ? AND metric_name LIKE 'alpha%'
        ORDER BY epoch, metric_name
    """,
        (exp_id,),
    ).fetchall()

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

    rows = conn.execute(
        """
        SELECT stage,
               SUM(duration_ms) as total_ms,
             SUM(stage_call_count) as n_calls
        FROM profiling
        WHERE experiment_id = ?
        GROUP BY stage
        ORDER BY total_ms DESC
    """,
        (exp_id,),
    ).fetchall()

    if not rows:
        print("No profiling data found.")
        return

    grand_total = sum(row[1] for row in rows)

    print(
        f"\n{'Rank':>4} | {'Stage':<15} | {'Total (ms)':>12} | {'Calls':>6} | {'% of Total':>10}"
    )
    print("-" * 60)
    for i, row in enumerate(rows, 1):
        pct = (row[1] / grand_total * 100) if grand_total > 0 else 0
        print(f"{i:>4} | {row[0]:<15} | {row[1]:>12.1f} | {row[2]:>6} | {pct:>9.1f}%")

    print("-" * 60)
    print(
        f"\nBottleneck: {rows[0][0]} ({rows[0][1] / grand_total * 100:.1f}% of total time)"
    )


def main():
    parser = argparse.ArgumentParser(description="Query experiment results")
    parser.add_argument(
        "--view",
        choices=["all", "completed", "attention", "errors"],
        default="all",
        help="Select a convenience exploration view before applying any extra filters.",
    )
    parser.add_argument("--batch-id", help="Filter experiment list to one batch id")
    parser.add_argument(
        "--status",
        choices=["running", "completed", "oom", "failed", "unknown"],
        help="Filter experiment list by status",
    )
    parser.add_argument("--exp", type=int, help="Show experiment details")
    parser.add_argument("--metrics", type=int, help="Show metrics for experiment")
    parser.add_argument("--profiling", type=int, help="Show profiling for experiment")
    parser.add_argument("--alpha", type=int, help="Show alpha drift for experiment")
    parser.add_argument(
        "--bottleneck", type=int, help="Show bottleneck analysis for experiment"
    )
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
        list_experiments(
            conn,
            batch_id=args.batch_id,
            status=args.status,
            view_name=args.view,
        )

    conn.close()


if __name__ == "__main__":
    main()
