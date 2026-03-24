#!/usr/bin/env python
"""Verify SQLite experiment tracking is working correctly."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify SQLite experiment tracking")
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


def check_sqlite_version():
    """Check SQLite version supports required features."""
    print("1. SQLite Version:")
    version = sqlite3.sqlite_version
    print(f"   SQLite version: {version}")

    # WAL mode requires SQLite 3.7.0+
    major, minor, _ = map(int, version.split("."))
    if (major, minor) >= (3, 7):
        print("   ✓ WAL mode supported (3.7.0+)")
        return True
    else:
        print("   ✗ WAL mode requires SQLite 3.7.0+")
        return False


def check_experiment_logger_import():
    """Check ExperimentLogger can be imported."""
    print("\n2. ExperimentLogger Import:")
    try:
        from src.utils.experiment_logger import ExperimentLogger

        print("   ✓ ExperimentLogger imported successfully")
        return True
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False


def check_experiment_logger_operations(args):
    """Test ExperimentLogger CRUD operations."""
    print("\n3. ExperimentLogger Operations:")

    from src.utils.experiment_logger import ExperimentLogger
    from src.utils.config import UCaGNNConfig

    if args.keep_db:
        db_path = args.db_path or str(
            Path(__file__).parent.parent / "results" / "verify_sqlite.db"
        )
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
    print(f"   Temporary verification DB: {db_path}")

    try:
        # Create logger
        logger = ExperimentLogger(db_path=db_path)
        print("   ✓ Logger created")

        # Log experiment
        config = UCaGNNConfig(
            dataset="test_dataset",
            epochs=3,
            training_mode="cached_propagation",
            graph_method="dense",
            profiling_cadence=2,
        )
        exp_id = logger.log_experiment(
            "test_dataset", config, preset="test", intervention="baseline"
        )
        print(f"   ✓ Experiment logged (id={exp_id})")

        # Log metrics
        logger.log_metric(exp_id, "loss", 0.5, epoch=0, split="train")
        logger.log_metric(exp_id, "epoch_time_s", 1.25, epoch=0, split="train")
        logger.log_metric(exp_id, "Recall@20", 0.15, epoch=0, split="val")
        logger.log_metric(exp_id, "NDCG@20", 0.12, epoch=0, split="val")
        print("   ✓ Metrics logged")

        # Log profiling (mock StageMetrics)
        class MockStage:
            name = "forward"
            elapsed_ms = 123.4
            vram_before_mb = 1000.0
            vram_after_mb = 1200.0
            vram_peak_mb = 1500.0

        logger.log_profiling(exp_id, epoch=0, stage=MockStage(), stage_call_count=2)
        print("   ✓ Profiling logged")
        logger.flush()

        # Verify data exists
        conn = sqlite3.connect(db_path)

        exp_count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        assert exp_count == 1, f"Expected 1 experiment, got {exp_count}"

        metric_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        assert metric_count == 4, f"Expected 4 metrics, got {metric_count}"

        prof_count = conn.execute("SELECT COUNT(*) FROM profiling").fetchone()[0]
        assert prof_count == 1, f"Expected 1 profiling row, got {prof_count}"

        profiling_row = conn.execute(
            "SELECT duration_ms, stage_call_count, timestamp FROM profiling WHERE experiment_id = ? AND stage = 'forward'",
            (exp_id,),
        ).fetchone()
        assert profiling_row is not None, "Missing profiling row"
        assert profiling_row[:2] == (123.4, 2), (
            f"Unexpected profiling row: {profiling_row}"
        )
        assert profiling_row[2], "Profiling timestamp should be populated"

        metric_timestamp = conn.execute(
            "SELECT timestamp FROM metrics WHERE experiment_id = ? ORDER BY id LIMIT 1",
            (exp_id,),
        ).fetchone()
        assert metric_timestamp is not None and metric_timestamp[0], (
            "Metric timestamp should be populated"
        )

        experiment_row = conn.execute(
            "SELECT training_mode, graph_method, timestamp FROM experiments WHERE id = ?",
            (exp_id,),
        ).fetchone()
        assert experiment_row is not None, "Missing experiment row"
        assert experiment_row[:2] == ("cached_propagation", "dense"), (
            f"Unexpected experiment metadata: {experiment_row}"
        )
        assert experiment_row[2], "Experiment timestamp should be populated"

        summary_row = conn.execute(
            "SELECT avg_epoch_time_s, best_recall_20, best_ndcg_20, avg_forward_ms, peak_vram_mb "
            "FROM experiment_summary WHERE id = ?",
            (exp_id,),
        ).fetchone()
        assert summary_row is not None, "experiment_summary view returned no row"
        assert abs(summary_row[0] - 1.25) < 1e-9, (
            f"Unexpected avg_epoch_time_s: {summary_row[0]}"
        )
        assert abs(summary_row[1] - 0.15) < 1e-9, (
            f"Unexpected best_recall_20: {summary_row[1]}"
        )
        assert abs(summary_row[2] - 0.12) < 1e-9, (
            f"Unexpected best_ndcg_20: {summary_row[2]}"
        )
        assert abs(summary_row[3] - 61.7) < 1e-9, (
            f"Unexpected avg_forward_ms: {summary_row[3]}"
        )
        assert abs(summary_row[4] - 1500.0) < 1e-9, (
            f"Unexpected peak_vram_mb: {summary_row[4]}"
        )

        profiling_columns = conn.execute("PRAGMA table_info(profiling)").fetchall()
        assert [row[1] for row in profiling_columns][-1] == "timestamp", (
            "profiling.timestamp must be the last column"
        )
        assert "stage_call_count" in {row[1] for row in profiling_columns}, (
            "Missing stage_call_count column"
        )

        metric_columns = conn.execute("PRAGMA table_info(metrics)").fetchall()
        assert [row[1] for row in metric_columns][-1] == "timestamp", (
            "metrics.timestamp must be the last column"
        )

        experiment_columns = conn.execute("PRAGMA table_info(experiments)").fetchall()
        assert [row[1] for row in experiment_columns][-1] == "timestamp", (
            "experiments.timestamp must be the last column"
        )

        index_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        for expected in {
            "idx_experiments_lookup",
            "idx_metrics_exp_split_name_epoch",
            "idx_profiling_exp_stage_epoch",
        }:
            assert expected in index_names, f"Missing index: {expected}"

        conn.close()
        print("   ✓ Data verified in database")

        logger.close()
        return True

    except Exception as e:
        print(f"   ✗ Operation failed: {e}")
        return False
    finally:
        if args.keep_db:
            print(f"   Kept verification DB at: {Path(db_path).resolve()}")
        else:
            Path(db_path).unlink(missing_ok=True)
            Path(db_path + "-wal").unlink(missing_ok=True)
            Path(db_path + "-shm").unlink(missing_ok=True)


def check_results_directory():
    """Check results directory can be created."""
    print("\n4. Results Directory:")
    results_path = Path(__file__).parent.parent / "results"

    if results_path.exists():
        print(f"   ✓ results/ directory exists")
    else:
        try:
            results_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ results/ directory created")
        except Exception as e:
            print(f"   ✗ Failed to create results/: {e}")
            return False

    # Check write permissions
    test_file = results_path / ".write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print("   ✓ results/ is writable")
        return True
    except Exception as e:
        print(f"   ✗ results/ not writable: {e}")
        return False


def check_query_examples():
    """Show example SQLite queries."""
    print("\n5. Example Queries (for reference):")
    print("   # View all experiments:")
    print('   sqlite3 results/thesis_experiments.db "SELECT * FROM experiments;"')
    print("\n   # Validation metrics over epochs:")
    print(
        "   sqlite3 results/thesis_experiments.db \"SELECT epoch, metric_name, metric_value, timestamp FROM metrics WHERE split='val' ORDER BY epoch, timestamp;\""
    )
    print("\n   # Profiling breakdown:")
    print(
        '   sqlite3 results/thesis_experiments.db "SELECT stage, SUM(duration_ms) AS total_ms, SUM(stage_call_count) AS calls FROM profiling GROUP BY stage ORDER BY total_ms DESC;"'
    )
    print("\n   # Per-experiment summary view:")
    print(
        '   sqlite3 results/thesis_experiments.db "SELECT dataset, preset, training_mode, graph_method, best_ndcg_20, avg_epoch_time_s FROM experiment_summary;"'
    )
    return True


def main():
    args = parse_args()
    print("=" * 60)
    print("SQLITE EXPERIMENT TRACKING VERIFICATION")
    print("=" * 60)

    all_good = all(
        [
            check_sqlite_version(),
            check_experiment_logger_import(),
            check_experiment_logger_operations(args),
            check_results_directory(),
            check_query_examples(),
        ]
    )

    print("\n" + "=" * 60)
    if all_good:
        print("✓ ALL SQLITE CHECKS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME SQLITE CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
