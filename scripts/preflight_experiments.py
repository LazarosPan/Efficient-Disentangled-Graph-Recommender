#!/usr/bin/env python
"""Representative preflight harness for formal U-CaGNN training runs."""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_experiment import build_config, run_experiment


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "results" / "thesis_experiments.db"

DEFAULT_RUNS = [
    ("full", "full_graph", "dense"),
    ("full", "cached_propagation", "cagra"),
    ("full", "mini_batch", "knn"),
]

FAST_SMOKE_RUNS = [
    ("full", "full_graph", "dense"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run representative preflight experiments"
    )
    parser.add_argument(
        "--dataset", default="movielens1m", help="Dataset to use for preflight"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--device", default="cuda", help="Execution device")
    parser.add_argument(
        "--epochs", type=int, default=2, help="Epochs per representative run"
    )
    parser.add_argument(
        "--sample-interactions",
        type=int,
        default=20000,
        help="Interaction budget for sampled preflight runs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch-size override for all preflight runs",
    )
    parser.add_argument(
        "--loader-max-rows",
        type=int,
        default=None,
        help="Optional early row cap for dataset loading during fast smoke/preflight runs.",
    )
    parser.add_argument(
        "--profile",
        choices=["representative", "fast"],
        default="representative",
        help="Representative runs for deeper coverage, or fast for a single ultra-light smoke run.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="ucagnn-preflight",
        help="MLflow experiment name used for preflight runs",
    )
    parser.add_argument(
        "--reset-sqlite-after",
        action="store_true",
        help="Reset the thesis SQLite database after all runs succeed.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the representative plan and exit"
    )
    return parser.parse_args()


def _sqlite_peak_vram(exp_id: int, db_path: Path) -> float | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT MAX(vram_peak_mb) FROM profiling WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row[0] is None:
        return None
    return float(row[0])


def _checkpoint_reload_ok(checkpoint_path: Path) -> bool:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_state", "optimizer_state", "loss_suite_state", "config"}
    return required.issubset(payload)


def _rerun_resume_probe(config, preset: str, checkpoint_path: Path) -> bool:
    result = run_experiment(
        config,
        preset=preset,
        save_checkpoint=True,
        enable_mlflow=False,
        mlflow_experiment_name="ucagnn-preflight",
        checkpoint_path=str(checkpoint_path),
        auto_resume=True,
    )
    return bool(result.get("resumed"))


def _losses_finite(history: dict) -> bool:
    losses = history.get("train_loss", [])
    return bool(losses) and all(math.isfinite(value) for value in losses)


def _print_plan(args: argparse.Namespace) -> None:
    run_plan = DEFAULT_RUNS if args.profile == "representative" else FAST_SMOKE_RUNS
    print("=" * 78)
    print("PREFLIGHT PLAN")
    print("=" * 78)
    print(f"Dataset: {args.dataset}")
    print(f"Profile: {args.profile}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Sample interactions: {args.sample_interactions}")
    print(
        f"Batch size override: {args.batch_size if args.batch_size is not None else 'config default'}"
    )
    print(
        f"Loader max rows: {args.loader_max_rows if args.loader_max_rows is not None else 'full dataset'}"
    )
    print(f"MLflow experiment: {args.mlflow_experiment_name}")
    print("Representative runs:")
    for index, (preset, training_mode, graph_method) in enumerate(run_plan, start=1):
        print(
            f"  {index}. preset={preset}, training_mode={training_mode}, graph_method={graph_method}"
        )


def main() -> int:
    args = parse_args()
    _print_plan(args)
    if args.dry_run:
        return 0

    process = psutil.Process()
    results: list[dict[str, object]] = []
    run_plan = DEFAULT_RUNS if args.profile == "representative" else FAST_SMOKE_RUNS

    for preset, training_mode, graph_method in run_plan:
        namespace = argparse.Namespace(
            dataset=args.dataset,
            recipe=None,
            preset=preset,
            seed=13,
            epochs=args.epochs,
            batch_size=args.batch_size,
            embed_dim=None,
            lr=None,
            graph_method=graph_method,
            training_mode=training_mode,
            num_neighbors=None,
            sample_interactions=args.sample_interactions,
            loader_max_rows=args.loader_max_rows,
            device=args.device,
            data_dir=args.data_dir,
            intervention="preflight",
            eval_scoring_mode=None,
            scoring_weight_mode=None,
            use_features=None,
        )
        config = build_config(namespace)
        config.enable_profiling = True
        config.profiling_cadence = 1

        rss_before_mb = process.memory_info().rss / (1024 * 1024)
        result = run_experiment(
            config,
            preset=preset,
            intervention="preflight",
            save_checkpoint=True,
            enable_mlflow=True,
            mlflow_experiment_name=args.mlflow_experiment_name,
            auto_resume=True,
        )
        rss_after_mb = process.memory_info().rss / (1024 * 1024)

        exp_id = int(result["exp_id"])
        checkpoint_path = Path(result["checkpoint_path"])
        peak_vram_mb = _sqlite_peak_vram(exp_id, DEFAULT_DB_PATH)
        checkpoint_exists = checkpoint_path.exists()
        checkpoint_reload_ok = checkpoint_exists and _checkpoint_reload_ok(
            checkpoint_path
        )
        resume_ok = checkpoint_exists and _rerun_resume_probe(
            config, preset, checkpoint_path
        )
        losses_finite = _losses_finite(result["history"])

        summary = {
            "preset": preset,
            "training_mode": training_mode,
            "graph_method": graph_method,
            "exp_id": exp_id,
            "peak_vram_mb": peak_vram_mb,
            "rss_before_mb": rss_before_mb,
            "rss_after_mb": rss_after_mb,
            "checkpoint_exists": checkpoint_exists,
            "checkpoint_reload_ok": checkpoint_reload_ok,
            "resume_ok": resume_ok,
            "losses_finite": losses_finite,
        }
        results.append(summary)

        print("-" * 78)
        print(
            f"{preset} / {training_mode} / {graph_method}: "
            f"exp_id={exp_id}, peak_vram_mb={peak_vram_mb}, "
            f"rss_before_mb={rss_before_mb:.1f}, rss_after_mb={rss_after_mb:.1f}, "
            f"checkpoint_exists={checkpoint_exists}, reload_ok={checkpoint_reload_ok}, "
            f"resume_ok={resume_ok}, finite_losses={losses_finite}"
        )

    all_ok = all(
        row["checkpoint_exists"]
        and row["checkpoint_reload_ok"]
        and row["resume_ok"]
        and row["losses_finite"]
        for row in results
    )

    print("=" * 78)
    print("PREFLIGHT SUMMARY")
    print("=" * 78)
    for row in results:
        print(
            f"preset={row['preset']}, mode={row['training_mode']}, graph={row['graph_method']}, "
            f"peak_vram_mb={row['peak_vram_mb']}, rss_after_mb={row['rss_after_mb']:.1f}, "
            f"reload_ok={row['checkpoint_reload_ok']}, resume_ok={row['resume_ok']}"
        )

    if args.reset_sqlite_after and all_ok:
        from scripts.reset_experiment_db import delete_rows, ordered_tables, TABLES

        conn = sqlite3.connect(DEFAULT_DB_PATH)
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            delete_rows(conn, ordered_tables(list(TABLES)))
            conn.commit()
        finally:
            conn.close()
        print("SQLite thesis database reset after successful preflight.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
