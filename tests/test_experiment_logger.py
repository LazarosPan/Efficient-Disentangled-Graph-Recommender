"""Regression coverage for ExperimentLogger write-path behavior."""

from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import query_results
from src.profiling.gpu_profiler import StageMetrics
from src.utils.experiment_logger import ExperimentLogger


@dataclass
class _DummyConfig:
    """Small dataclass config for logger serialization tests."""

    seed: int


class ExperimentLoggerTests(unittest.TestCase):
    """Pin logger serialization and epoch-aggregation behavior."""

    def setUp(self) -> None:
        """Create a temporary SQLite logger for each test."""
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.logger = ExperimentLogger(
            db_path=str(Path(self.temp_dir.name) / "experiments.sqlite"),
        )
        self.addCleanup(self.logger.close)

    def test_log_experiment_serializes_dataclass_config(self) -> None:
        """Dataclass configs should still be serialized into config_json."""
        exp_id = self.logger.log_experiment(
            dataset="movielens1m",
            config=_DummyConfig(seed=7),
        )

        row = self.logger.conn.execute(
            "SELECT config_json, seed FROM experiments WHERE id = ?",
            (exp_id,),
        ).fetchone()
        assert row is not None

        self.assertEqual(row["seed"], 7)
        self.assertEqual(
            json.loads(row["config_json"]),
            {"seed": 7},
        )

    def test_log_epoch_aggregates_profiler_stages_by_name(self) -> None:
        """log_epoch should combine repeated profiler stages into one row per stage."""
        exp_id = self.logger.log_experiment(
            dataset="movielens1m",
            config=_DummyConfig(seed=13),
        )

        profiler_stages = [
            StageMetrics(
                name="forward",
                elapsed_ms=10.0,
                vram_before_mb=100.0,
                vram_after_mb=120.0,
                vram_peak_mb=130.0,
            ),
            StageMetrics(
                name="forward",
                elapsed_ms=15.0,
                vram_before_mb=110.0,
                vram_after_mb=118.0,
                vram_peak_mb=132.0,
            ),
            StageMetrics(
                name="loss",
                elapsed_ms=4.0,
                vram_before_mb=118.0,
                vram_after_mb=119.0,
                vram_peak_mb=125.0,
            ),
        ]

        self.logger.log_epoch(
            exp_id=exp_id,
            epoch=0,
            train_loss=1.25,
            epoch_time_s=0.5,
            val_metrics={"NDCG@20": 0.3},
            profiler_stages=profiler_stages,
            model=None,
        )

        rows = self.logger.conn.execute(
            """
            SELECT stage, duration_ms, vram_before_mb, vram_after_mb, vram_peak_mb, stage_call_count
            FROM profiling
            WHERE experiment_id = ?
            ORDER BY stage
            """,
            (exp_id,),
        ).fetchall()

        self.assertEqual(len(rows), 2)
        rows_by_stage = {row["stage"]: row for row in rows}
        loss_row = rows_by_stage["loss"]
        forward_row = rows_by_stage["forward"]

        self.assertEqual(loss_row["stage"], "loss")
        self.assertAlmostEqual(loss_row["duration_ms"], 4.0)
        self.assertAlmostEqual(loss_row["vram_before_mb"], 118.0)
        self.assertAlmostEqual(loss_row["vram_after_mb"], 119.0)
        self.assertAlmostEqual(loss_row["vram_peak_mb"], 125.0)
        self.assertEqual(loss_row["stage_call_count"], 1)

        self.assertEqual(forward_row["stage"], "forward")
        self.assertAlmostEqual(forward_row["duration_ms"], 25.0)
        self.assertAlmostEqual(forward_row["vram_before_mb"], 105.0)
        self.assertAlmostEqual(forward_row["vram_after_mb"], 119.0)
        self.assertAlmostEqual(forward_row["vram_peak_mb"], 132.0)
        self.assertEqual(forward_row["stage_call_count"], 2)

    def test_log_experiment_persists_provenance_columns(self) -> None:
        """Experiment rows should retain code-version provenance for quick diffs."""
        exp_id = self.logger.log_experiment(
            dataset="movielens1m",
            config=_DummyConfig(seed=7),
            profile_name="dev-ucagnn",
            project_version="1.2.3",
            git_commit="abc1234",
            training_hash="trainhash",
            evaluation_hash="evalhash",
            change_note="sparse eval cache",
        )

        row = self.logger.conn.execute(
            """
            SELECT profile_name, project_version, git_commit, training_hash,
                evaluation_hash, change_note
            FROM experiments
            WHERE id = ?
            """,
            (exp_id,),
        ).fetchone()
        assert row is not None

        self.assertEqual(row["profile_name"], "dev-ucagnn")
        self.assertEqual(row["project_version"], "1.2.3")
        self.assertEqual(row["git_commit"], "abc1234")
        self.assertEqual(row["training_hash"], "trainhash")
        self.assertEqual(row["evaluation_hash"], "evalhash")
        self.assertEqual(row["change_note"], "sparse eval cache")

    def test_log_experiment_accepts_training_mode(self) -> None:
        """Training mode should be retained when provided explicitly."""
        exp_id = self.logger.log_experiment(
            dataset="movielens1m",
            config=_DummyConfig(seed=42),
            training_mode="mini_batch",
        )

        row = self.logger.conn.execute(
            "SELECT training_mode FROM experiments WHERE id = ?",
            (exp_id,),
        ).fetchone()
        assert row is not None

        self.assertEqual(row["training_mode"], "mini_batch")

    def test_experiment_summary_uses_peak_vram_metric_when_profiling_missing(self) -> None:
        """The summary view should expose peak VRAM stored as a train metric."""
        exp_id = self.logger.log_experiment(
            dataset="movielens1m",
            config=_DummyConfig(seed=15),
        )
        self.logger.log_metric(exp_id, "peak_vram_mb", 999.0, split="train")
        self.logger.update_experiment_status(exp_id, status="completed")

        row = self.logger.conn.execute(
            "SELECT peak_vram_mb FROM experiment_summary WHERE id = ?",
            (exp_id,),
        ).fetchone()
        assert row is not None

        self.assertAlmostEqual(row["peak_vram_mb"], 999.0)

    def test_experiment_summary_tracks_total_training_time_and_completed_epochs(self) -> None:
        """The summary view should expose total training time and stopped epochs."""
        exp_id = self.logger.log_experiment(
            dataset="amazonbook",
            config={
                "dataset": "amazonbook",
                "epochs": 200,
                "batch_size": 8192,
                "embed_dim": 64,
                "use_dual_branch": False,
                "single_branch_gnn_layers": 2,
                "lr_scheduler": "plateau",
                "seed": 13,
            },
            preset="lightgcn",
        )
        self.logger.log_epoch(
            exp_id=exp_id,
            epoch=0,
            train_loss=1.0,
            epoch_time_s=0.2,
            val_metrics={"NDCG@20": 0.1},
            profiler_stages=[],
            model=None,
        )
        self.logger.log_epoch(
            exp_id=exp_id,
            epoch=1,
            train_loss=0.8,
            epoch_time_s=0.25,
            val_metrics={"NDCG@20": 0.15},
            profiler_stages=[],
            model=None,
        )
        self.logger.log_metric(exp_id, "training_time_s", 7.9, split="train")
        self.logger.update_experiment_status(exp_id, status="completed")

        row = self.logger.conn.execute(
            """
            SELECT training_time_s, completed_train_epochs
            FROM experiment_summary
            WHERE id = ?
            """,
            (exp_id,),
        ).fetchone()
        assert row is not None

        self.assertAlmostEqual(row["training_time_s"], 7.9)
        self.assertEqual(row["completed_train_epochs"], 2)

    def test_comparison_view_exposes_metric_deltas_for_same_semantic_run(self) -> None:
        """Comparison view should show deltas across repeated same-config runs."""
        first_exp = self.logger.log_experiment(
            dataset="movielens1m",
            config=_DummyConfig(seed=7),
            preset="ucagnn",
            training_hash="trainhash",
            evaluation_hash="evalhash",
            git_commit="abc1234",
            change_note="baseline",
        )
        self.logger.log_metric(first_exp, "NDCG@20", 0.1, split="test")
        self.logger.log_metric(first_exp, "Recall@20", 0.2, split="test")
        self.logger.log_metric(first_exp, "AveragePopularity@20", 0.3, split="test")
        self.logger.update_experiment_status(first_exp, status="completed")

        second_exp = self.logger.log_experiment(
            dataset="movielens1m",
            config=_DummyConfig(seed=7),
            preset="ucagnn",
            training_hash="trainhash",
            evaluation_hash="evalhash",
            git_commit="def5678",
            change_note="new sampler",
        )
        self.logger.log_metric(second_exp, "NDCG@20", 0.15, split="test")
        self.logger.log_metric(second_exp, "Recall@20", 0.25, split="test")
        self.logger.log_metric(second_exp, "AveragePopularity@20", 0.28, split="test")
        self.logger.update_experiment_status(second_exp, status="completed")

        row = self.logger.conn.execute(
            """
            SELECT git_commit, change_note, delta_test_ndcg_20,
                delta_test_recall_20, delta_test_average_popularity_20
            FROM experiment_code_comparison
            WHERE id = ?
            """,
            (second_exp,),
        ).fetchone()
        assert row is not None

        self.assertEqual(row["git_commit"], "def5678")
        self.assertEqual(row["change_note"], "new sampler")
        self.assertAlmostEqual(row["delta_test_ndcg_20"], 0.05)
        self.assertAlmostEqual(row["delta_test_recall_20"], 0.05)
        self.assertAlmostEqual(row["delta_test_average_popularity_20"], -0.02)

    def test_query_results_top_completed_uses_training_time_and_actual_epochs(self) -> None:
        """Top-results output should show total training time and stopped epochs."""
        completed_exp = self.logger.log_experiment(
            dataset="amazonbook",
            config={
                "dataset": "amazonbook",
                "epochs": 200,
                "batch_size": 8192,
                "embed_dim": 64,
                "use_dual_branch": False,
                "single_branch_gnn_layers": 2,
                "num_neighbors": [10, 5],
                "lr_scheduler": "plateau",
                "seed": 13,
            },
            preset="lightgcn",
        )
        self.logger.log_epoch(
            exp_id=completed_exp,
            epoch=0,
            train_loss=1.0,
            epoch_time_s=0.4,
            val_metrics={"NDCG@20": 0.1},
            profiler_stages=[],
            model=None,
        )
        self.logger.log_metric(completed_exp, "training_time_s", 7.9, split="train")
        self.logger.log_metric(completed_exp, "NDCG@20", 0.1, split="test")
        self.logger.log_metric(completed_exp, "Recall@20", 0.2, split="test")
        self.logger.log_metric(completed_exp, "AveragePopularity@20", 0.3, split="test")
        self.logger.update_experiment_status(completed_exp, status="completed")

        failed_exp = self.logger.log_experiment(
            dataset="amazonbook",
            config={
                "dataset": "amazonbook",
                "epochs": 200,
                "batch_size": 8192,
                "embed_dim": 64,
                "use_dual_branch": False,
                "single_branch_gnn_layers": 2,
                "lr_scheduler": "plateau",
                "seed": 13,
            },
            preset="ucagnn",
        )
        self.logger.log_epoch(
            exp_id=failed_exp,
            epoch=0,
            train_loss=0.5,
            epoch_time_s=0.3,
            val_metrics={"NDCG@20": 0.9},
            profiler_stages=[],
            model=None,
        )
        self.logger.log_metric(failed_exp, "training_time_s", 2.0, split="train")
        self.logger.log_metric(failed_exp, "NDCG@20", 0.9, split="test")
        self.logger.update_experiment_status(
            failed_exp,
            status="failed",
            failure_reason="synthetic failure",
        )

        buffer = StringIO()
        temp_db_path = Path(self.temp_dir.name) / "experiments.sqlite"
        with patch.object(query_results, "DB_PATH", temp_db_path), redirect_stdout(buffer):
            query_results.list_top_completed(self.logger.conn, n=20)

        output = buffer.getvalue()
        self.assertIn("Training Time", output)
        self.assertIn("lightgcn", output)
        self.assertNotIn("ucagnn", output)

        completed_line = next(
            line for line in output.splitlines() if line.strip().startswith("amazonbook")
        )
        columns = [column.strip() for column in completed_line.split("|")]
        self.assertEqual(columns[10], "7.9s")
        self.assertEqual(columns[11], "1")

    def test_schema_mismatch_requires_current_tables(self) -> None:
        """ExperimentLogger should reject databases that predate the current schema."""
        db_path = Path(self.temp_dir.name) / "old-schema.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                preset TEXT,
                intervention TEXT,
                config_json TEXT,
                seed INTEGER,
                training_mode TEXT,
                status TEXT NOT NULL DEFAULT 'unknown',
                failure_reason TEXT,
                oom_flag INTEGER NOT NULL DEFAULT 0,
                batch_id TEXT,
                profile_name TEXT,
                project_version TEXT,
                git_commit TEXT,
                training_hash TEXT,
                evaluation_hash TEXT,
                change_note TEXT,
                gpu_name TEXT,
                gpu_vram_gb REAL,
                timestamp TEXT NOT NULL
            )
            """,
        )
        conn.commit()
        conn.close()

        with self.assertRaisesRegex(RuntimeError, "current schema"):
            ExperimentLogger(db_path=str(db_path))
