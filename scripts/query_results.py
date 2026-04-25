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

from collections import defaultdict
import json
import sqlite3
import sys
from pathlib import Path

from src.utils.experiment_logger import ExperimentLogger
from src.utils.cli_parsers import build_query_results_parser


REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "results" / "thesis_experiments.db"
VIEW_TABLES = ExperimentLogger.VIEW_TABLES


def connect() -> sqlite3.Connection:
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

    profile_width = 28
    print(
        f"{'ID':>4} | {'Status':<10} | {'Dataset':<14} | {'Preset':<12} | {'Profile':<{profile_width}} | {'Mode':<18} | {'Graph':<8} | {'Batch':<22} | Seed"
    )
    print("-" * (146 + profile_width - 8))
    for row in rows:
        batch_label = row["batch_id"] or "-"
        if len(batch_label) > 22:
            batch_label = f"{batch_label[:19]}..."
        profile_label = row["profile_name"] or "-"
        if len(profile_label) > profile_width:
            profile_label = f"{profile_label[: profile_width - 3]}..."
        status_label = row["status"] or ("oom" if row["oom_flag"] else "unknown")
        print(
            f"{row['id']:>4} | {status_label:<10} | {row['dataset'] or '-':<14} | {row['preset'] or '-':<12} | {profile_label:<{profile_width}} | {row['training_mode'] or '-':<18} | {row['graph_method'] or '-':<8} | {batch_label:<22} | {row['seed'] or '-'}"
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
    print(f"Timestamp:    {row['timestamp']}")
    print(f"Dataset:      {row['dataset']}")
    print(f"Preset:       {row['preset'] or '-'}")
    print(f"Intervention: {row['intervention'] or '-'}")
    print(f"Seed:         {row['seed'] or '-'}")
    print(f"Status:       {row['status'] or '-'}")
    print(f"Training:     {row['training_mode'] or '-'}")
    print(f"Graph:        {row['graph_method'] or '-'}")
    print(f"Batch ID:     {row['batch_id'] or '-'}")
    print(f"Profile:      {row['profile_name'] or '-'}")
    print(f"GPU:          {row['gpu_name'] or '-'}")
    print(
        f"VRAM (GiB):   {row['gpu_vram_gb'] if row['gpu_vram_gb'] is not None else '-'}"
    )
    print(f"Updated:      {row['updated_at'] or '-'}")
    if row["oom_flag"]:
        print(f"OOM Flag:     {bool(row['oom_flag'])}")
    if row["failure_reason"]:
        print(f"Failure:      {row['failure_reason']}")
    print("\nConfig:")

    if row["config_json"]:
        config = json.loads(row["config_json"])
        for k, v in sorted(config.items()):
            print(f"  {k}: {v}")


def show_metrics(conn: sqlite3.Connection, exp_id: int) -> None:
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
                epoch_str = str(row["epoch"]) if row["epoch"] is not None else "final"
                print(
                    f"  {epoch_str:>5} | {row['metric_name']:<20} | "
                    f"{row['metric_value']:>10.4f} | {row['timestamp']}"
                )


def show_profiling(conn: sqlite3.Connection, exp_id: int) -> None:
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

    total_ms = sum(float(row["total_ms"]) for row in rows)

    print(
        f"\n{'Stage':<15} | {'Total/Epoch':>11} | {'Avg/Call':>10} | {'Calls':>6} | {'Epochs':>6} | {'%':>6} | Peak VRAM | Last Logged"
    )
    print("-" * 150)
    for row in rows:
        stage_total_ms = float(row["total_ms"])
        pct = (stage_total_ms / total_ms * 100) if total_ms > 0 else 0
        peak_vram_mb = row["peak_vram_mb"]
        peak_str = f"{peak_vram_mb:.0f} MB" if peak_vram_mb else "-"
        print(
            f"{row['stage']:<15} | {row['avg_epoch_ms']:>11.1f} | "
            f"{row['avg_call_ms']:>10.1f} | {row['total_calls']:>6} | "
            f"{row['profiled_epochs']:>6} | {pct:>5.1f}% | "
            f"{peak_str:<9} | {row['last_logged_at']}"
        )

    print("-" * 150)
    print(f"{'TOTAL':<15} | {total_ms:>11.1f}")


def show_alpha_drift(conn: sqlite3.Connection, exp_id: int) -> None:
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

    epochs: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        epochs[row["epoch"]][row["metric_name"]] = row["metric_value"]

    for epoch in sorted(epochs.keys()):
        alpha_pos = epochs[epoch].get("alpha_pos", float("nan"))
        alpha_neg = epochs[epoch].get("alpha_neg", float("nan"))
        print(f"{epoch:>5} | {alpha_pos:>10.4f} | {alpha_neg:>10.4f}")


def show_bottleneck(conn: sqlite3.Connection, exp_id: int) -> None:
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

    grand_total = sum(float(row["total_ms"]) for row in rows)

    print(
        f"\n{'Rank':>4} | {'Stage':<15} | {'Total (ms)':>12} | {'Calls':>6} | {'% of Total':>10}"
    )
    print("-" * 60)
    for i, row in enumerate(rows, 1):
        stage_total_ms = float(row["total_ms"])
        pct = (stage_total_ms / grand_total * 100) if grand_total > 0 else 0
        print(
            f"{i:>4} | {row['stage']:<15} | {stage_total_ms:>12.1f} | "
            f"{row['n_calls']:>6} | {pct:>9.1f}%"
        )

    print("-" * 60)
    print(
        f"\nBottleneck: {rows[0]['stage']} "
        f"({float(rows[0]['total_ms']) / grand_total * 100:.1f}% of total time)"
    )


def main() -> int:
    """Parse arguments and print the requested experiment view."""
    args = build_query_results_parser().parse_args()

    conn = connect()
    try:
        if args.exp is not None:
            show_experiment(conn, args.exp)
        elif args.metrics is not None:
            show_metrics(conn, args.metrics)
        elif args.profiling is not None:
            show_profiling(conn, args.profiling)
        elif args.alpha is not None:
            show_alpha_drift(conn, args.alpha)
        elif args.bottleneck is not None:
            show_bottleneck(conn, args.bottleneck)
        else:
            list_experiments(
                conn,
                batch_id=args.batch_id,
                status=args.status,
                view_name=args.view,
            )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
