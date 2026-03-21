#!/usr/bin/env python
"""End-to-end pipeline sanity check (3 epochs, small batch)."""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="End-to-end pipeline sanity check")
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep the verification SQLite database instead of deleting it.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Optional SQLite path to use when --keep-db is set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from src.utils.config import UCaGNNConfig
    from src.utils.experiment_logger import ExperimentLogger
    from src.data.loaders import load_dataset
    from src.data.graph_builder import build_graph
    from src.models.ucagnn import UCaGNN
    from src.losses.loss_suite import LossSuite
    from src.training.trainer import Trainer
    from src.profiling.gpu_profiler import GPUProfiler

    print("=" * 60)
    print("PIPELINE SANITY CHECK (3 epochs)")
    print("=" * 60)

    # Use a temp DB by default; optionally keep a persistent copy for inspection.
    if args.keep_db:
        db_path = args.db_path or str(
            Path(__file__).parent.parent / "results" / "verify_pipeline.db"
        )
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

    try:
        # Config: minimal settings for fast test
        print("\n1. Creating config...")
        config = UCaGNNConfig(
            dataset="movielens1m",
            data_dir="data",
            epochs=3,
            batch_size=512,
            embed_dim=32,
            n_gnn_layers=1,
            patience=99,  # Disable early stopping
            lr=5e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            profiling_cadence=2,
        )
        print(f"   Device: {config.device}")
        print(f"   Epochs: {config.epochs}")
        print(f"   Seed: {config.seed}")
        print("   ✓ Config created")

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Data loading
        print("\n2. Loading dataset...")
        canonical = load_dataset(config.dataset, config.data_dir)
        print(f"   Users: {canonical.n_users:,}, Items: {canonical.n_items:,}")
        print(f"   Interactions: {len(canonical):,}")
        print("   ✓ Dataset loaded")

        # Graph construction
        print("\n3. Building graph...")
        data = build_graph(canonical, config, embeddings=None)
        print(f"   Nodes: {data.num_nodes:,}, Edges: {data.edge_index.size(1):,}")
        print(f"   Train: {data.train_mask.sum():,}, Val: {data.val_mask.sum():,}, Test: {data.test_mask.sum():,}")
        print("   ✓ Graph built")

        # Model
        print("\n4. Creating model...")
        model = UCaGNN(canonical.n_users, canonical.n_items, config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {n_params:,}")
        print("   ✓ Model created")

        # Loss suite
        print("\n5. Creating loss suite...")
        loss_suite = LossSuite(config)
        print("   ✓ Loss suite created")

        # Profiler
        print("\n6. Creating profiler...")
        profiler = GPUProfiler() if torch.cuda.is_available() else None
        print(f"   GPU profiling: {'enabled' if profiler else 'disabled (CPU mode)'}")
        print("   ✓ Profiler created")

        # Experiment logger
        print("\n7. Creating experiment logger...")
        experiment_logger = ExperimentLogger(db_path=db_path)
        exp_id = experiment_logger.log_experiment(config.dataset, config, preset="sanity_check")
        print(f"   Experiment ID: {exp_id}")
        print(f"   Temp DB path: {db_path}")
        print("   ✓ Logger created")

        # Trainer
        print("\n8. Creating trainer...")
        trainer = Trainer(
            model, loss_suite, data, config, profiler,
            experiment_logger=experiment_logger, exp_id=exp_id
        )
        print("   ✓ Trainer created")

        # Training
        print("\n9. Training (3 epochs)...")
        history = trainer.train()
        print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
        print("   ✓ Training complete")

        # Evaluation
        print("\n10. Running test evaluation...")
        test_metrics = trainer.evaluator.evaluate(model, data, data.test_mask)
        for metric, value in sorted(test_metrics.items()):
            print(f"    {metric}: {value:.4f}")
            experiment_logger.log_metric(exp_id, metric, value, split="test")
        print("   ✓ Evaluation complete")

        experiment_logger.close()

        # Verify SQLite contents
        print("\n11. Verifying SQLite contents...")
        import sqlite3
        conn = sqlite3.connect(db_path)

        exp_count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        metric_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        prof_count = conn.execute("SELECT COUNT(*) FROM profiling").fetchone()[0]
        epoch_time_count = conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE metric_name = 'epoch_time_s'"
        ).fetchone()[0]
        profiled_epochs = conn.execute(
            "SELECT COUNT(DISTINCT epoch) FROM profiling"
        ).fetchone()[0]
        experiment_row = conn.execute(
            "SELECT training_mode, graph_method, timestamp FROM experiments WHERE id = ?",
            (exp_id,),
        ).fetchone()

        print(f"   Experiments: {exp_count}")
        print(f"   Metrics: {metric_count}")
        print(f"   Profiling rows: {prof_count}")
        print(f"   epoch_time_s rows: {epoch_time_count}")
        print(f"   Profiled epochs: {profiled_epochs}")
        print(f"   Experiment metadata: {experiment_row}")

        assert epoch_time_count == config.epochs, (
            f"Expected {config.epochs} epoch_time_s rows, got {epoch_time_count}"
        )
        if profiler is not None:
            assert profiled_epochs == 1, f"Expected 1 profiled epoch, got {profiled_epochs}"
            assert prof_count > 0, "Expected aggregated profiling rows for the profiled epoch"
            stage_call_count_sum = conn.execute(
                "SELECT COALESCE(SUM(stage_call_count), 0) FROM profiling WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()[0]
            assert stage_call_count_sum >= prof_count, (
                "Aggregated profiling rows must preserve call counts"
            )
        else:
            assert profiled_epochs == 0, f"Expected 0 profiled epochs on CPU, got {profiled_epochs}"
            assert prof_count == 0, f"Expected 0 profiling rows on CPU, got {prof_count}"
        assert experiment_row is not None, "Missing experiment row"
        assert experiment_row[:2] == (config.training_mode, config.graph_method), (
            f"Unexpected experiment metadata: {experiment_row}"
        )
        assert experiment_row[2], "Experiment timestamp should be populated"

        for table_name in ("experiments", "metrics", "profiling"):
            table_info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            assert table_info[-1][1] == "timestamp", (
                f"{table_name}.timestamp must be the last column"
            )

        timestamp_counts = conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM metrics WHERE timestamp IS NOT NULL), "
            "(SELECT COUNT(*) FROM profiling WHERE timestamp IS NOT NULL)"
        ).fetchone()
        assert timestamp_counts is not None
        assert timestamp_counts[0] == metric_count, "All metric rows must have timestamps"
        assert timestamp_counts[1] == prof_count, "All profiling rows must have timestamps"

        summary_row = conn.execute(
            "SELECT avg_epoch_time_s, best_ndcg_20 FROM experiment_summary WHERE id = ?",
            (exp_id,),
        ).fetchone()
        assert summary_row is not None, "experiment_summary view returned no row"
        assert summary_row[0] is not None and summary_row[0] > 0, (
            f"Unexpected avg_epoch_time_s: {summary_row[0]}"
        )

        # Show sample metrics
        rows = conn.execute(
            "SELECT epoch, split, metric_name, metric_value FROM metrics ORDER BY epoch, split LIMIT 5"
        ).fetchall()
        print("   Sample metrics:")
        for row in rows:
            print(f"      epoch={row[0]}, split={row[1]}, {row[2]}={row[3]:.4f}")

        conn.close()
        print("   ✓ SQLite verified")

        print("\n" + "=" * 60)
        print("✓ PIPELINE SANITY CHECK PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if args.keep_db:
            print(f"\nKept verification DB at: {Path(db_path).resolve()}")
        else:
            # Cleanup temp DB. This script intentionally does not leave a persistent
            # SQLite file in results/ because it is only a sanity check.
            Path(db_path).unlink(missing_ok=True)
            Path(db_path + "-wal").unlink(missing_ok=True)
            Path(db_path + "-shm").unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
