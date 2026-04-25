"""Regression coverage for ExperimentLogger write-path behavior."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from src.profiling.gpu_profiler import StageMetrics
from src.utils.experiment_logger import ExperimentLogger


@dataclass
class _DummyConfig:
    """Small dataclass config for logger serialization tests."""

    seed: int
    graph_method: str = "cagra"


class ExperimentLoggerTests(unittest.TestCase):
    """Pin logger serialization and epoch-aggregation behavior."""

    def setUp(self) -> None:
        """Create a temporary SQLite logger for each test."""
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.logger = ExperimentLogger(
            db_path=str(Path(self.temp_dir.name) / "experiments.sqlite")
        )
        self.addCleanup(self.logger.close)

    def test_log_experiment_serializes_dataclass_config(self) -> None:
        """Dataclass configs should still be serialized into config_json."""
        exp_id = self.logger.log_experiment(
            dataset="movielens1m",
            config=_DummyConfig(seed=7),
        )

        row = self.logger.conn.execute(
            "SELECT config_json, seed, graph_method FROM experiments WHERE id = ?",
            (exp_id,),
        ).fetchone()
        assert row is not None

        self.assertEqual(row["seed"], 7)
        self.assertEqual(row["graph_method"], "cagra")
        self.assertEqual(
            json.loads(row["config_json"]),
            {"seed": 7, "graph_method": "cagra"},
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
                graph_method TEXT,
                status TEXT NOT NULL DEFAULT 'unknown',
                failure_reason TEXT,
                oom_flag INTEGER NOT NULL DEFAULT 0,
                batch_id TEXT,
                profile_name TEXT,
                gpu_name TEXT,
                gpu_vram_gb REAL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

        with self.assertRaisesRegex(RuntimeError, "current schema"):
            ExperimentLogger(db_path=str(db_path))
