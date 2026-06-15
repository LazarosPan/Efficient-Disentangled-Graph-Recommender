"""Regression coverage for ExperimentLogger write-path behavior."""

from __future__ import annotations

import json
import sqlite3
import sys
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

    def test_existing_database_logger_open_does_not_require_schema_write_lock(self) -> None:
        """Opening a current WAL database should not need DDL while another writer exists."""
        db_path = Path(self.temp_dir.name) / "parallel.sqlite"
        bootstrap_logger = ExperimentLogger(db_path=str(db_path))
        bootstrap_logger.close()

        writer = sqlite3.connect(db_path)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("BEGIN IMMEDIATE")
        try:
            parallel_logger = ExperimentLogger(db_path=str(db_path))
            parallel_logger.close()
        finally:
            writer.rollback()
            writer.close()

    def test_experiment_logger_has_no_unsupported_metric_compatibility_lists(self) -> None:
        """Logger schema checks should use current required columns, not stale metric lists."""
        unsupported_class_vars = [
            name for name in vars(ExperimentLogger) if name.startswith("_UNSUPPORTED_")
        ]

        self.assertEqual(unsupported_class_vars, [])

    def test_query_results_connect_is_read_only_without_logger_migration(self) -> None:
        """Reporting should read existing views without taking logger write locks."""
        db_path = Path(self.temp_dir.name) / "query-readonly.sqlite"
        bootstrap_logger = ExperimentLogger(db_path=str(db_path))
        bootstrap_logger.close()

        with (
            patch.object(query_results, "DB_PATH", db_path),
            patch.object(query_results, "ExperimentLogger") as logger_cls,
        ):
            conn = query_results.connect()
            self.addCleanup(conn.close)

        logger_cls.assert_not_called()
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE query_results_should_be_read_only(id INTEGER)")

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

    def test_experiment_logger_no_longer_owns_optuna_search_trials(self) -> None:
        """Optuna trial metadata should stay in Optuna RDB storage."""
        self.assertFalse(hasattr(self.logger, "log_optuna_search_trial"))
        row = self.logger.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'optuna_search_trials'
            """
        ).fetchone()

        self.assertIsNone(row)

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

    def test_experiment_summary_tracks_average_gpu_utilization(self) -> None:
        """The summary view should expose the averaged train-split GPU utilization metric."""
        exp_id = self.logger.log_experiment(
            dataset="movielens1m",
            config=_DummyConfig(seed=21),
        )
        self.logger.log_metric(exp_id, "gpu_utilization_pct", 55.0, epoch=0, split="train")
        self.logger.log_metric(exp_id, "gpu_utilization_pct", 65.0, epoch=1, split="train")
        self.logger.update_experiment_status(exp_id, status="completed")

        row = self.logger.conn.execute(
            (
                "SELECT avg_gpu_utilization_pct, max_gpu_utilization_pct "
                "FROM experiment_summary WHERE id = ?"
            ),
            (exp_id,),
        ).fetchone()
        assert row is not None

        self.assertAlmostEqual(row["avg_gpu_utilization_pct"], 60.0)
        self.assertAlmostEqual(row["max_gpu_utilization_pct"], 65.0)

    def test_experiment_summary_uses_training_max_gpu_utilization(self) -> None:
        """The summary view should separate train-average and train-peak utilization."""
        exp_id = self.logger.log_experiment(
            dataset="movielens1m",
            config=_DummyConfig(seed=22),
        )
        self.logger.log_metric(exp_id, "gpu_utilization_pct", 33.0, epoch=0, split="train")
        self.logger.log_metric(
            exp_id,
            "max_gpu_utilization_pct",
            66.0,
            epoch=0,
            split="train",
        )
        self.logger.update_experiment_status(exp_id, status="completed")

        row = self.logger.conn.execute(
            (
                "SELECT avg_gpu_utilization_pct, max_gpu_utilization_pct "
                "FROM experiment_summary WHERE id = ?"
            ),
            (exp_id,),
        ).fetchone()
        assert row is not None

        self.assertAlmostEqual(row["avg_gpu_utilization_pct"], 33.0)
        self.assertAlmostEqual(row["max_gpu_utilization_pct"], 66.0)

    def test_get_metrics_for_split_retains_evaluation_diagnostics(self) -> None:
        """Arbitrary evaluator diagnostics should round-trip through the SQLite metric store."""
        exp_id = self.logger.log_experiment(
            dataset="movielens1m",
            config=_DummyConfig(seed=33),
        )
        val_metrics = {
            "NDCG@20": 0.3,
            "score_mix_interest_mean": 0.4,
            "score_mix_interest_std": 0.1,
            "interest_contribution@20": -1.25,
            "context_popularity_spearman@20": 0.8,
        }

        self.logger.log_epoch(
            exp_id=exp_id,
            epoch=0,
            train_loss=1.0,
            epoch_time_s=0.2,
            val_metrics=val_metrics,
            profiler_stages=[],
            model=None,
        )
        self.logger.log_metric(
            exp_id,
            "interest_conformity_cosine_mean",
            0.5,
            split="test",
        )
        self.logger.log_metric(
            exp_id,
            "conformity_contribution@40",
            1.1,
            split="test",
        )

        self.assertEqual(
            self.logger.get_metrics_for_split(exp_id, split="val"),
            val_metrics,
        )
        self.assertEqual(
            self.logger.get_metrics_for_split(exp_id, split="test"),
            {
                "conformity_contribution@40": 1.1,
                "interest_conformity_cosine_mean": 0.5,
            },
        )

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

    def test_compute_dataset_crru_scores_is_dataset_local_and_k_specific(self) -> None:
        """CRRU scores should normalize per dataset and differ between @20 and @40."""
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE report_rows (
                id INTEGER PRIMARY KEY,
                dataset TEXT NOT NULL,
                test_ndcg_20 REAL,
                test_recall_20 REAL,
                test_hit_ratio_20 REAL,
                test_personalization_20 REAL,
                test_average_popularity_20 REAL,
                test_ndcg_40 REAL,
                test_recall_40 REAL,
                test_hit_ratio_40 REAL,
                test_personalization_40 REAL,
                test_average_popularity_40 REAL,
                peak_vram_mb REAL,
                training_time_s REAL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO report_rows (
                id, dataset, test_ndcg_20, test_recall_20, test_hit_ratio_20,
                test_personalization_20, test_average_popularity_20,
                test_ndcg_40, test_recall_40, test_hit_ratio_40,
                test_personalization_40, test_average_popularity_40,
                peak_vram_mb, training_time_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    1,
                    "amazonbook",
                    0.90,
                    0.80,
                    0.70,
                    0.60,
                    0.20,
                    0.10,
                    0.20,
                    0.30,
                    0.40,
                    0.80,
                    1000.0,
                    100.0,
                ),
                (
                    2,
                    "amazonbook",
                    0.20,
                    0.30,
                    0.40,
                    0.50,
                    0.70,
                    0.95,
                    0.85,
                    0.75,
                    0.65,
                    0.10,
                    4000.0,
                    400.0,
                ),
                (
                    3,
                    "movielens1m",
                    0.09,
                    0.08,
                    0.07,
                    0.06,
                    0.02,
                    0.01,
                    0.02,
                    0.03,
                    0.04,
                    0.08,
                    500.0,
                    50.0,
                ),
                (
                    4,
                    "movielens1m",
                    0.02,
                    0.03,
                    0.04,
                    0.05,
                    0.07,
                    0.10,
                    0.09,
                    0.08,
                    0.07,
                    0.01,
                    1500.0,
                    150.0,
                ),
            ],
        )
        all_rows = conn.execute("SELECT * FROM report_rows ORDER BY id").fetchall()
        amazon_rows = conn.execute(
            "SELECT * FROM report_rows WHERE dataset = 'amazonbook' ORDER BY id"
        ).fetchall()

        all_scores = query_results._compute_dataset_crru_scores(all_rows)
        amazon_scores = query_results._compute_dataset_crru_scores(amazon_rows)

        self.assertGreater(all_scores[1][20], all_scores[2][20])
        self.assertLess(all_scores[1][40], all_scores[2][40])
        self.assertAlmostEqual(all_scores[1][20], amazon_scores[1][20])
        self.assertAlmostEqual(all_scores[1][40], amazon_scores[1][40])
        self.assertAlmostEqual(all_scores[2][20], amazon_scores[2][20])
        self.assertAlmostEqual(all_scores[2][40], amazon_scores[2][40])

    def test_compute_efficiency_scores_prefers_epoch_time_over_total_time(self) -> None:
        """CRRU efficiency should compare throughput per epoch, not early-stop wall time."""
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE report_rows (
                id INTEGER PRIMARY KEY,
                dataset TEXT NOT NULL,
                peak_vram_mb REAL,
                training_time_s REAL,
                completed_train_epochs REAL,
                avg_epoch_time_s REAL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO report_rows (
                id, dataset, peak_vram_mb, training_time_s,
                completed_train_epochs, avg_epoch_time_s
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (1, "amazonbook", 1000.0, 100.0, 100.0, 1.0),
                (2, "amazonbook", 1000.0, 10.0, 1.0, 10.0),
            ],
        )
        rows = conn.execute("SELECT * FROM report_rows ORDER BY id").fetchall()

        efficiency_scores = query_results.compute_crru_efficiency_scores(
            [row["peak_vram_mb"] for row in rows],
            [query_results._crru_epoch_time_s(row) for row in rows],
        )

        self.assertGreater(efficiency_scores[0], efficiency_scores[1])

    def test_query_results_duration_formatter_always_uses_seconds(self) -> None:
        """Runtime report durations should stay in seconds for direct comparison."""
        self.assertEqual(query_results._format_duration(None), "-")
        self.assertEqual(query_results._format_duration(12.34), "12.3s")
        self.assertEqual(query_results._format_duration(138.0), "138.0s")
        self.assertEqual(query_results._format_duration(122000.0), "122000.0s")

    def test_query_results_orders_rows_by_crru_within_dataset(self) -> None:
        """The summary tables should sort each dataset by CRRU@20 then CRRU@40."""
        high_accuracy_low_utility = self.logger.log_experiment(
            dataset="amazonbook",
            config={
                "dataset": "amazonbook",
                "epochs": 100,
                "batch_size": 4096,
                "embed_dim": 64,
                "use_dual_branch": False,
                "single_branch_gnn_layers": 2,
                "num_neighbors": [10, 5],
                "lr_scheduler": "plateau",
                "seed": 11,
            },
            preset="lightgcn",
            batch_id="formal-order-1",
            profile_name="high-accuracy-low-utility",
        )
        self.logger.log_metric(high_accuracy_low_utility, "NDCG@20", 0.95, split="test")
        self.logger.log_metric(high_accuracy_low_utility, "Recall@20", 0.05, split="test")
        self.logger.log_metric(high_accuracy_low_utility, "HitRatio@20", 0.05, split="test")
        self.logger.log_metric(high_accuracy_low_utility, "Personalization@20", 0.05, split="test")
        self.logger.log_metric(
            high_accuracy_low_utility,
            "AveragePopularity@20",
            0.95,
            split="test",
        )
        self.logger.log_metric(high_accuracy_low_utility, "NDCG@40", 0.90, split="test")
        self.logger.log_metric(high_accuracy_low_utility, "Recall@40", 0.05, split="test")
        self.logger.log_metric(high_accuracy_low_utility, "HitRatio@40", 0.05, split="test")
        self.logger.log_metric(high_accuracy_low_utility, "Personalization@40", 0.05, split="test")
        self.logger.log_metric(
            high_accuracy_low_utility,
            "AveragePopularity@40",
            0.95,
            split="test",
        )
        self.logger.log_metric(high_accuracy_low_utility, "training_time_s", 900.0, split="train")
        self.logger.log_metric(high_accuracy_low_utility, "peak_vram_mb", 9000.0, split="train")
        self.logger.update_experiment_status(high_accuracy_low_utility, status="completed")

        low_accuracy_high_utility = self.logger.log_experiment(
            dataset="amazonbook",
            config={
                "dataset": "amazonbook",
                "epochs": 100,
                "batch_size": 4096,
                "embed_dim": 64,
                "use_dual_branch": False,
                "single_branch_gnn_layers": 2,
                "num_neighbors": [10, 5],
                "lr_scheduler": "plateau",
                "seed": 12,
            },
            preset="lightgcn",
            batch_id="formal-order-2",
            profile_name="low-accuracy-high-utility",
        )
        self.logger.log_metric(low_accuracy_high_utility, "NDCG@20", 0.10, split="test")
        self.logger.log_metric(low_accuracy_high_utility, "Recall@20", 0.95, split="test")
        self.logger.log_metric(low_accuracy_high_utility, "HitRatio@20", 0.95, split="test")
        self.logger.log_metric(low_accuracy_high_utility, "Personalization@20", 0.95, split="test")
        self.logger.log_metric(
            low_accuracy_high_utility,
            "AveragePopularity@20",
            0.05,
            split="test",
        )
        self.logger.log_metric(low_accuracy_high_utility, "NDCG@40", 0.10, split="test")
        self.logger.log_metric(low_accuracy_high_utility, "Recall@40", 0.95, split="test")
        self.logger.log_metric(low_accuracy_high_utility, "HitRatio@40", 0.95, split="test")
        self.logger.log_metric(low_accuracy_high_utility, "Personalization@40", 0.95, split="test")
        self.logger.log_metric(
            low_accuracy_high_utility,
            "AveragePopularity@40",
            0.05,
            split="test",
        )
        self.logger.log_metric(low_accuracy_high_utility, "training_time_s", 10.0, split="train")
        self.logger.log_metric(low_accuracy_high_utility, "peak_vram_mb", 1000.0, split="train")
        self.logger.update_experiment_status(low_accuracy_high_utility, status="completed")

        buffer = StringIO()
        temp_db_path = Path(self.temp_dir.name) / "experiments.sqlite"
        with patch.object(query_results, "DB_PATH", temp_db_path), redirect_stdout(buffer):
            query_results.list_top_completed(self.logger.conn, n=20)

        output = buffer.getvalue()
        self.assertLess(
            output.index("low-accuracy-high-utility"),
            output.index("high-accuracy-low-utility"),
        )

    def test_query_results_shows_tiny_crru_values_without_rounding_them_to_zero(
        self,
    ) -> None:
        """Tiny CRRU values should render in scientific notation instead of 0.0000."""
        exp_id = self.logger.log_experiment(
            dataset="movielens1m",
            config={
                "dataset": "movielens1m",
                "epochs": 100,
                "batch_size": 4096,
                "embed_dim": 64,
                "use_dual_branch": False,
                "single_branch_gnn_layers": 2,
                "num_neighbors": [10, 5],
                "lr_scheduler": "plateau",
                "seed": 99,
            },
            preset="lightgcn",
            batch_id="formal-tiny-crru",
            profile_name="tiny-crru",
        )
        self.logger.log_metric(exp_id, "NDCG@20", 0.11, split="test")
        self.logger.log_metric(exp_id, "Recall@20", 0.22, split="test")
        self.logger.log_metric(exp_id, "HitRatio@20", 0.33, split="test")
        self.logger.log_metric(exp_id, "Personalization@20", 0.44, split="test")
        self.logger.log_metric(exp_id, "AveragePopularity@20", 0.55, split="test")
        self.logger.log_metric(exp_id, "NDCG@40", 0.66, split="test")
        self.logger.log_metric(exp_id, "Recall@40", 0.77, split="test")
        self.logger.log_metric(exp_id, "HitRatio@40", 0.88, split="test")
        self.logger.log_metric(exp_id, "Personalization@40", 0.99, split="test")
        self.logger.log_metric(exp_id, "AveragePopularity@40", 1.0, split="test")
        self.logger.log_metric(exp_id, "training_time_s", 11.2, split="train")
        self.logger.log_metric(exp_id, "peak_vram_mb", 2048.0, split="train")
        self.logger.update_experiment_status(exp_id, status="completed")
        buffer = StringIO()
        temp_db_path = Path(self.temp_dir.name) / "experiments.sqlite"
        with patch.object(query_results, "DB_PATH", temp_db_path), redirect_stdout(buffer):
            query_results.list_top_completed(self.logger.conn, n=20)

        output = buffer.getvalue()
        self.assertIn("tiny-crru", output)
        self.assertIn("e-06", output)
        self.assertNotIn("|   0.0000 |   0.0000", output)

    def test_query_results_top_completed_shows_only_formal_and_ablation_test_runs(self) -> None:
        """Default results output should exclude ad-hoc and smoke-test runs."""
        formal_exp = self.logger.log_experiment(
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
            batch_id="formal-dev-profile-20260515T000000Z",
            profile_name="dev-profile",
            training_hash="formalhash",
        )
        self.logger.log_epoch(
            exp_id=formal_exp,
            epoch=0,
            train_loss=1.0,
            epoch_time_s=0.4,
            val_metrics={"NDCG@20": 0.1},
            profiler_stages=[],
            model=None,
        )
        self.logger.log_metric(formal_exp, "NDCG@20", 0.1, split="test")
        self.logger.log_metric(formal_exp, "Recall@20", 0.2, split="test")
        self.logger.log_metric(formal_exp, "HitRatio@20", 0.21, split="test")
        self.logger.log_metric(formal_exp, "Personalization@20", 0.22, split="test")
        self.logger.log_metric(formal_exp, "AveragePopularity@20", 0.3, split="test")
        self.logger.log_metric(formal_exp, "NDCG@40", 0.4, split="test")
        self.logger.log_metric(formal_exp, "Recall@40", 0.5, split="test")
        self.logger.log_metric(formal_exp, "HitRatio@40", 0.51, split="test")
        self.logger.log_metric(formal_exp, "Personalization@40", 0.52, split="test")
        self.logger.log_metric(formal_exp, "AveragePopularity@40", 0.6, split="test")
        self.logger.log_metric(formal_exp, "context_contribution@20", 0.11, split="test")
        self.logger.log_metric(formal_exp, "context_contribution@40", 0.12, split="test")
        self.logger.log_metric(
            formal_exp,
            "final_popularity_spearman@20",
            0.13,
            split="test",
        )
        self.logger.log_metric(
            formal_exp,
            "final_popularity_spearman@40",
            0.14,
            split="test",
        )
        self.logger.log_metric(formal_exp, "interest_branch_NDCG@20", 0.31, split="test")
        self.logger.log_metric(formal_exp, "interest_branch_Recall@20", 0.32, split="test")
        self.logger.log_metric(
            formal_exp,
            "interest_branch_AveragePopularity@20",
            0.33,
            split="test",
        )
        self.logger.log_metric(formal_exp, "interest_branch_NDCG@40", 0.34, split="test")
        self.logger.log_metric(formal_exp, "interest_branch_Recall@40", 0.35, split="test")
        self.logger.log_metric(
            formal_exp,
            "interest_branch_AveragePopularity@40",
            0.36,
            split="test",
        )
        self.logger.log_metric(formal_exp, "conformity_branch_NDCG@20", 0.41, split="test")
        self.logger.log_metric(formal_exp, "conformity_branch_Recall@20", 0.42, split="test")
        self.logger.log_metric(
            formal_exp,
            "conformity_branch_AveragePopularity@20",
            0.43,
            split="test",
        )
        self.logger.log_metric(formal_exp, "conformity_branch_NDCG@40", 0.44, split="test")
        self.logger.log_metric(formal_exp, "conformity_branch_Recall@40", 0.45, split="test")
        self.logger.log_metric(
            formal_exp,
            "conformity_branch_AveragePopularity@40",
            0.46,
            split="test",
        )
        self.logger.log_metric(formal_exp, "training_time_s", 7.9, split="train")
        self.logger.log_metric(formal_exp, "peak_vram_mb", 1234.0, split="train")
        self.logger.update_experiment_status(formal_exp, status="completed")

        runtime_probe_exp = self.logger.log_experiment(
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
                "seed": 14,
            },
            preset="lightgcn",
            batch_id="formal-runtime-probe-20260515T000000Z",
            profile_name="runtime-probe-profile",
            training_hash="probehash",
        )
        self.logger.log_metric(runtime_probe_exp, "NDCG@20", 0.99, split="test")
        self.logger.log_metric(runtime_probe_exp, "Recall@20", 0.99, split="test")
        self.logger.log_metric(runtime_probe_exp, "HitRatio@20", 0.99, split="test")
        self.logger.log_metric(runtime_probe_exp, "Personalization@20", 0.99, split="test")
        self.logger.log_metric(runtime_probe_exp, "AveragePopularity@20", 0.01, split="test")
        self.logger.log_metric(runtime_probe_exp, "NDCG@40", 0.99, split="test")
        self.logger.log_metric(runtime_probe_exp, "Recall@40", 0.99, split="test")
        self.logger.log_metric(runtime_probe_exp, "HitRatio@40", 0.99, split="test")
        self.logger.log_metric(runtime_probe_exp, "Personalization@40", 0.99, split="test")
        self.logger.log_metric(runtime_probe_exp, "AveragePopularity@40", 0.01, split="test")
        self.logger.log_metric(runtime_probe_exp, "training_time_s", 7.9, split="train")
        self.logger.log_metric(runtime_probe_exp, "peak_vram_mb", 1234.0, split="train")
        self.logger.log_metric(
            runtime_probe_exp,
            "runtime_probe_target_epochs",
            200.0,
            split="approximation",
        )
        self.logger.log_metric(
            runtime_probe_exp,
            "runtime_probe_observed_epochs",
            1.0,
            split="approximation",
        )
        self.logger.log_metric(
            runtime_probe_exp,
            "runtime_probe_train_batches_per_epoch",
            1745.0,
            split="approximation",
        )
        self.logger.log_metric(
            runtime_probe_exp,
            "runtime_probe_observed_batches_per_second",
            2.86,
            split="approximation",
        )
        self.logger.log_metric(
            runtime_probe_exp,
            "runtime_probe_estimated_train_time_s",
            122000.0,
            split="approximation",
        )
        self.logger.update_experiment_status(runtime_probe_exp, status="completed")

        ablation_exp = self.logger.log_experiment(
            dataset="amazonbook",
            config={
                "dataset": "amazonbook",
                "epochs": 300,
                "batch_size": 4096,
                "embed_dim": 64,
                "use_dual_branch": True,
                "interest_gnn_layers": 1,
                "conformity_gnn_layers": 2,
                "num_neighbors": [20, 10],
                "scoring_weight_mode": "learned",
                "use_features": True,
                "lr_scheduler": "cosine",
                "seed": 13,
            },
            preset="ucagnn",
            intervention="no_independence",
            batch_id="ablation-20260515T000000Z",
            training_hash="ablationhash",
        )
        self.logger.log_metric(ablation_exp, "NDCG@20", 0.4, split="test")
        self.logger.log_metric(ablation_exp, "Recall@20", 0.5, split="test")
        self.logger.log_metric(ablation_exp, "HitRatio@20", 0.55, split="test")
        self.logger.log_metric(ablation_exp, "Personalization@20", 0.56, split="test")
        self.logger.log_metric(ablation_exp, "AveragePopularity@20", 0.6, split="test")
        self.logger.log_metric(ablation_exp, "NDCG@40", 0.7, split="test")
        self.logger.log_metric(ablation_exp, "Recall@40", 0.8, split="test")
        self.logger.log_metric(ablation_exp, "HitRatio@40", 0.81, split="test")
        self.logger.log_metric(ablation_exp, "Personalization@40", 0.82, split="test")
        self.logger.log_metric(ablation_exp, "AveragePopularity@40", 0.9, split="test")
        self.logger.log_epoch(
            exp_id=ablation_exp,
            epoch=0,
            train_loss=0.7,
            epoch_time_s=0.6,
            val_metrics={"NDCG@20": 0.35},
            profiler_stages=[],
            model=None,
        )
        self.logger.log_metric(ablation_exp, "training_time_s", 12.3, split="train")
        self.logger.log_metric(ablation_exp, "peak_vram_mb", 2345.0, split="train")
        self.logger.update_experiment_status(ablation_exp, status="completed")

        legacy_ablation_exp = self.logger.log_experiment(
            dataset="amazonbook",
            config={
                "dataset": "amazonbook",
                "epochs": 300,
                "batch_size": 4096,
                "embed_dim": 64,
                "use_dual_branch": True,
                "interest_gnn_layers": 1,
                "conformity_gnn_layers": 2,
                "num_neighbors": [20, 10],
                "use_features": True,
                "lr_scheduler": "cosine",
                "seed": 13,
            },
            preset="ucagnn",
            intervention="no_ipw",
            batch_id="ablation-legacy-20260515T000000Z",
        )
        self.logger.log_metric(legacy_ablation_exp, "NDCG@20", 0.9, split="test")
        self.logger.log_metric(legacy_ablation_exp, "Recall@20", 0.9, split="test")
        self.logger.log_metric(legacy_ablation_exp, "HitRatio@20", 0.9, split="test")
        self.logger.log_metric(legacy_ablation_exp, "Personalization@20", 0.9, split="test")
        self.logger.log_metric(legacy_ablation_exp, "AveragePopularity@20", 0.1, split="test")
        self.logger.log_metric(legacy_ablation_exp, "NDCG@40", 0.9, split="test")
        self.logger.log_metric(legacy_ablation_exp, "Recall@40", 0.9, split="test")
        self.logger.log_metric(legacy_ablation_exp, "HitRatio@40", 0.9, split="test")
        self.logger.log_metric(legacy_ablation_exp, "Personalization@40", 0.9, split="test")
        self.logger.log_metric(legacy_ablation_exp, "AveragePopularity@40", 0.1, split="test")
        self.logger.update_experiment_status(legacy_ablation_exp, status="completed")

        ad_hoc_exp = self.logger.log_experiment(
            dataset="amazonbook",
            config={
                "dataset": "amazonbook",
                "epochs": 100,
                "batch_size": 1024,
                "embed_dim": 64,
                "use_dual_branch": False,
                "single_branch_gnn_layers": 2,
                "num_neighbors": [10, 5],
                "lr_scheduler": "plateau",
                "seed": 13,
            },
            preset="ucagnn",
        )
        self.logger.log_metric(ad_hoc_exp, "NDCG@20", 0.9, split="test")
        self.logger.update_experiment_status(ad_hoc_exp, status="completed")

        smoke_exp = self.logger.log_experiment(
            dataset="amazonbook",
            config={
                "dataset": "amazonbook",
                "epochs": 100,
                "batch_size": 1024,
                "embed_dim": 64,
                "use_dual_branch": False,
                "single_branch_gnn_layers": 2,
                "num_neighbors": [10, 5],
                "lr_scheduler": "plateau",
                "sample_interactions": 100,
                "loader_max_rows": 100,
                "seed": 13,
            },
            preset="lightgcn",
            batch_id="formal-smoke-20260515T000000Z",
            profile_name="smoke-profile",
        )
        self.logger.log_metric(smoke_exp, "NDCG@20", 0.8, split="test")
        self.logger.update_experiment_status(smoke_exp, status="completed")

        buffer = StringIO()
        temp_db_path = Path(self.temp_dir.name) / "experiments.sqlite"
        with patch.object(query_results, "DB_PATH", temp_db_path), redirect_stdout(buffer):
            query_results.list_top_completed(self.logger.conn, n=20)

        output = buffer.getvalue()
        self.assertIn("FINAL FORMAL FULL-DATA TEST RUNS", output)
        self.assertIn("SUPPORTING FORMAL FULL-DATA RUNS", output)
        self.assertIn("ABLATION FULL-DATA TEST RUNS", output)
        self.assertIn("currently supported variants", output)
        self.assertIn("Composite Resource-aware Recommendation Utility at K", output)
        self.assertIn("CRRU is not a causal-effect estimator", output)
        self.assertIn("section-row min-max", output)
        self.assertNotIn("report-row min-max", output)
        self.assertIn("CRRU@20", output)
        self.assertIn("CRRU@40", output)
        self.assertIn("(1-log(1+time/epoch)_n)^0.50", output)
        self.assertIn("dev-profile", output)
        self.assertIn("no_independence", output)
        self.assertIn("Hit@20", output)
        self.assertIn("Pers@40", output)
        self.assertIn("context_contrib={20: 0.1100, 40: 0.1200}", output)
        self.assertIn("Final={20: 0.1300, 40: 0.1400}", output)
        self.assertIn(
            "Branch Rank: Interest NDCG={20: 0.3100, 40: 0.3400}",
            output,
        )
        self.assertIn(
            "Conformity NDCG={20: 0.4100, 40: 0.4400}",
            output,
        )
        self.assertNotIn("runtime-probe-profile", output)
        self.assertNotIn("Approximation: full_train=122000.0s", output)
        self.assertNotIn("target_epochs=200", output)
        self.assertNotIn("throughput=2.86 batch/s", output)
        self.assertIn(
            "Resources:  time=7.9s",
            output,
        )
        self.assertIn(
            "Resources:  time=12.3s",
            output,
        )
        self.assertIn("Resources:  time=12.3s", output)
        self.assertIn(
            "amazonbook_lightgcn_ep200_bs8192_dim64_layers2_nbr10-5_lr-plateau_seed13",
            output,
        )
        self.assertIn(
            "amazonbook_ucagnn_ep300_bs4096_dim64_layers2_branchL1-2_nbr20-10_feat_lr-cosine_no_independence_seed13",
            output,
        )
        self.assertNotIn("_train-formalhash", output)
        self.assertNotIn("_train-ablationhash", output)
        self.assertNotIn("CRRU COMPOSITE METRIC", output)
        self.assertNotIn("no_ipw", output)
        self.assertNotIn("smoke-profile", output)
        self.assertNotIn(
            "amazonbook_ucagnn_ep100_bs1024_dim64_layers2_nbr10-5_lr-plateau_seed13",
            output,
        )

    def test_query_results_main_writes_default_summary_markdown(self) -> None:
        """The base CLI should persist the thesis summary to results/query_results.md."""
        exp_id = self.logger.log_experiment(
            dataset="amazonbook",
            config={
                "dataset": "amazonbook",
                "epochs": 100,
                "batch_size": 4096,
                "embed_dim": 64,
                "use_dual_branch": False,
                "single_branch_gnn_layers": 2,
                "num_neighbors": [10, 5],
                "lr_scheduler": "plateau",
                "seed": 13,
            },
            preset="lightgcn",
            batch_id="formal-dev-profile-20260515T000000Z",
            profile_name="dev-profile",
        )
        self.logger.log_epoch(
            exp_id=exp_id,
            epoch=0,
            train_loss=1.0,
            epoch_time_s=0.4,
            val_metrics={"NDCG@20": 0.1},
            profiler_stages=[],
            model=None,
        )
        self.logger.log_metric(exp_id, "NDCG@20", 0.1, split="test")
        self.logger.log_metric(exp_id, "Recall@20", 0.2, split="test")
        self.logger.log_metric(exp_id, "HitRatio@20", 0.3, split="test")
        self.logger.log_metric(exp_id, "Personalization@20", 0.4, split="test")
        self.logger.log_metric(exp_id, "AveragePopularity@20", 0.5, split="test")
        self.logger.log_metric(exp_id, "NDCG@40", 0.6, split="test")
        self.logger.log_metric(exp_id, "Recall@40", 0.7, split="test")
        self.logger.log_metric(exp_id, "HitRatio@40", 0.8, split="test")
        self.logger.log_metric(exp_id, "Personalization@40", 0.9, split="test")
        self.logger.log_metric(exp_id, "AveragePopularity@40", 1.0, split="test")
        self.logger.log_metric(exp_id, "training_time_s", 11.2, split="train")
        self.logger.log_metric(exp_id, "peak_vram_mb", 2048.0, split="train")
        self.logger.update_experiment_status(exp_id, status="completed")
        temp_db_path = Path(self.temp_dir.name) / "experiments.sqlite"
        output_path = Path(self.temp_dir.name) / "query_results.md"
        buffer = StringIO()
        with (
            patch.object(query_results, "DB_PATH", temp_db_path),
            patch.object(query_results, "QUERY_RESULTS_MARKDOWN_PATH", output_path),
            patch.object(sys, "argv", ["query_results.py"]),
            redirect_stdout(buffer),
        ):
            exit_code = query_results.main()

        self.assertEqual(exit_code, 0)
        stdout_text = buffer.getvalue()
        self.assertIn(
            f"Wrote default results summary to {output_path.resolve()}",
            stdout_text,
        )
        self.assertNotIn("THESIS TEST RESULTS", stdout_text)
        self.assertNotIn("dev-profile", stdout_text)
        self.assertTrue(output_path.exists())
        markdown_output = output_path.read_text(encoding="utf-8")
        self.assertIn("# Query Results", markdown_output)
        self.assertNotIn("```text", markdown_output)
        self.assertIn("| Run | Dataset | Preset | ScoreMix | Neighbors |", markdown_output)
        self.assertIn("| ---: | --- | --- | --- | --- |", markdown_output)
        self.assertIn("THESIS TEST RESULTS", markdown_output)
        self.assertIn(
            "Composite Resource-aware Recommendation Utility at K",
            markdown_output,
        )
        self.assertIn("CRRU is not a causal-effect estimator", markdown_output)
        self.assertIn("CRRU@20", markdown_output)
        self.assertIn("CRRU@40", markdown_output)
        self.assertIn("dev-profile", markdown_output)
        self.assertIn("OPTUNA U-CaGNN SEARCH REPORT", markdown_output)
        self.assertIn("optuna_optimization.md", markdown_output)
        self.assertIn(
            "amazonbook_lightgcn_ep100_bs4096_dim64_layers2_nbr10-5_lr-plateau_seed13",
            markdown_output,
        )
        self.assertNotIn("CRRU COMPOSITE METRIC", markdown_output)

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
