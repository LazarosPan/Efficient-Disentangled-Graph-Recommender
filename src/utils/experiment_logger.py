"""ExperimentLogger: lightweight SQLite experiment tracker -- no MLflow dependency."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path


class ExperimentLogger:
    """Persist experiment configs, per-epoch metrics, and profiling data to SQLite."""

    TERMINAL_STATUSES = ("completed", "failed", "oom")

    _SQLITE_NOW_UTC = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
    _EXPECTED_EXPERIMENT_COLUMNS = [
        "id",
        "dataset",
        "preset",
        "intervention",
        "config_json",
        "seed",
        "training_mode",
        "graph_method",
        "status",
        "failure_reason",
        "oom_flag",
        "batch_id",
        "profile_name",
        "gpu_name",
        "gpu_vram_gb",
        "timestamp",
        "updated_at",
    ]
    _EXPECTED_PROFILING_COLUMNS = [
        "id",
        "experiment_id",
        "epoch",
        "stage",
        "duration_ms",
        "vram_before_mb",
        "vram_after_mb",
        "vram_peak_mb",
        "stage_call_count",
        "timestamp",
    ]
    _EXPECTED_METRIC_COLUMNS = [
        "id",
        "experiment_id",
        "epoch",
        "split",
        "metric_name",
        "metric_value",
        "timestamp",
    ]

    def __init__(self, db_path: str = "results/thesis_experiments.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    # ── Schema ────────────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        self._ensure_current_schema()
        self._repair_experiment_statuses()
        self._create_indexes_and_views()
        self.conn.commit()

    def _ensure_current_schema(self) -> None:
        self.conn.execute("PRAGMA foreign_keys=OFF")
        try:
            self.conn.execute("DROP VIEW IF EXISTS experiment_summary")
            self.conn.execute("DROP VIEW IF EXISTS experiment_completed_summary")
            self.conn.execute("DROP VIEW IF EXISTS experiment_attention_summary")
            self.conn.execute("DROP VIEW IF EXISTS experiment_error_summary")
            self._ensure_experiments_table()
            self._ensure_profiling_table()
            self._ensure_metrics_table()
            self._cleanup_legacy_tables()
            self.conn.commit()
        finally:
            self.conn.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _qualified_or_default(
        legacy_columns: set[str],
        qualified_name: str,
        default_sql: str,
    ) -> str:
        return (
            qualified_name
            if qualified_name.split(".")[-1] in legacy_columns
            else default_sql
        )

    def _table_exists(self, table_name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _table_columns(self, table_name: str) -> list[str]:
        return [row[1] for row in self.conn.execute(f"PRAGMA table_info({table_name})")]

    def _foreign_key_targets(self, table_name: str) -> set[str]:
        """Return referenced table names declared by a table's foreign keys."""
        return {
            row[2]
            for row in self.conn.execute(f"PRAGMA foreign_key_list({table_name})")
        }

    def _next_legacy_name(self, base_name: str) -> str:
        candidate = f"{base_name}__legacy"
        suffix = 2
        while self._table_exists(candidate):
            candidate = f"{base_name}__legacy_{suffix}"
            suffix += 1
        return candidate

    def _cleanup_legacy_tables(self) -> None:
        for table_name in ("metrics", "profiling", "experiments"):
            legacy_rows = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name GLOB ?",
                (f"{table_name}__legacy*",),
            ).fetchall()
            for (legacy_name,) in legacy_rows:
                if self._table_exists(legacy_name):
                    self.conn.execute(f"DROP TABLE {legacy_name}")

    def _create_experiments_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE experiments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset      TEXT    NOT NULL,
                preset       TEXT,
                intervention TEXT,
                config_json  TEXT,
                seed         INTEGER,
                training_mode TEXT,
                graph_method TEXT,
                status       TEXT    NOT NULL DEFAULT 'unknown',
                failure_reason TEXT,
                oom_flag     INTEGER NOT NULL DEFAULT 0,
                batch_id     TEXT,
                profile_name TEXT,
                gpu_name     TEXT,
                gpu_vram_gb  REAL,
                timestamp    TEXT    NOT NULL
                ,updated_at  TEXT    NOT NULL
            )
        """)

    def _create_profiling_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE profiling (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id    INTEGER NOT NULL REFERENCES experiments(id),
                epoch            INTEGER NOT NULL,
                stage            TEXT    NOT NULL,
                duration_ms      REAL    NOT NULL,
                vram_before_mb   REAL,
                vram_after_mb    REAL,
                vram_peak_mb     REAL,
                stage_call_count INTEGER NOT NULL DEFAULT 1,
                timestamp        TEXT    NOT NULL
            )
        """)

    def _create_metrics_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE metrics (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL REFERENCES experiments(id),
                epoch         INTEGER,
                split         TEXT    NOT NULL,
                metric_name   TEXT    NOT NULL,
                metric_value  REAL    NOT NULL,
                timestamp     TEXT    NOT NULL
            )
        """)

    def _ensure_experiments_table(self) -> None:
        if not self._table_exists("experiments"):
            self._create_experiments_table()
            return

        current_columns = self._table_columns("experiments")
        if current_columns == self._EXPECTED_EXPERIMENT_COLUMNS:
            return

        legacy_table = self._next_legacy_name("experiments")
        self.conn.execute(f"ALTER TABLE experiments RENAME TO {legacy_table}")
        self._create_experiments_table()

        legacy_columns = set(self._table_columns(legacy_table))
        self.conn.execute(
            f"""
            INSERT INTO experiments (
                id,
                dataset,
                preset,
                intervention,
                config_json,
                seed,
                training_mode,
                graph_method,
                status,
                failure_reason,
                oom_flag,
                batch_id,
                profile_name,
                gpu_name,
                gpu_vram_gb,
                timestamp,
                updated_at
            )
            SELECT
                id,
                dataset,
                preset,
                intervention,
                config_json,
                seed,
                {self._qualified_or_default(legacy_columns, "e.training_mode", "NULL")} AS training_mode,
                {self._qualified_or_default(legacy_columns, "e.graph_method", "NULL")} AS graph_method,
                {self._qualified_or_default(legacy_columns, "e.status", "'unknown'")} AS status,
                {self._qualified_or_default(legacy_columns, "e.failure_reason", "NULL")} AS failure_reason,
                {self._qualified_or_default(legacy_columns, "e.oom_flag", "0")} AS oom_flag,
                {self._qualified_or_default(legacy_columns, "e.batch_id", "NULL")} AS batch_id,
                {self._qualified_or_default(legacy_columns, "e.profile_name", "NULL")} AS profile_name,
                {self._qualified_or_default(legacy_columns, "e.gpu_name", "NULL")} AS gpu_name,
                {self._qualified_or_default(legacy_columns, "e.gpu_vram_gb", "NULL")} AS gpu_vram_gb,
                {self._qualified_or_default(legacy_columns, "e.timestamp", self._SQLITE_NOW_UTC)} AS timestamp,
                COALESCE(
                    {self._qualified_or_default(legacy_columns, "e.updated_at", "NULL")},
                    {self._qualified_or_default(legacy_columns, "e.timestamp", self._SQLITE_NOW_UTC)},
                    {self._SQLITE_NOW_UTC}
                ) AS updated_at
            FROM {legacy_table} e
            ORDER BY id
            """
        )
        self.conn.execute(f"DROP TABLE {legacy_table}")

    def _ensure_profiling_table(self) -> None:
        if not self._table_exists("profiling"):
            self._create_profiling_table()
            return

        current_columns = self._table_columns("profiling")
        fk_targets = self._foreign_key_targets("profiling")
        if (
            current_columns == self._EXPECTED_PROFILING_COLUMNS
            and fk_targets == {"experiments"}
        ):
            return

        legacy_table = self._next_legacy_name("profiling")
        self.conn.execute(f"ALTER TABLE profiling RENAME TO {legacy_table}")
        self._create_profiling_table()

        legacy_columns = set(self._table_columns(legacy_table))
        self.conn.execute(
            f"""
            INSERT INTO profiling (
                id,
                experiment_id,
                epoch,
                stage,
                duration_ms,
                vram_before_mb,
                vram_after_mb,
                vram_peak_mb,
                stage_call_count,
                timestamp
            )
            SELECT
                p.id,
                p.experiment_id,
                p.epoch,
                p.stage,
                p.duration_ms,
                p.vram_before_mb,
                p.vram_after_mb,
                p.vram_peak_mb,
                {self._qualified_or_default(legacy_columns, "p.stage_call_count", "1")} AS stage_call_count,
                COALESCE(
                    {self._qualified_or_default(legacy_columns, "p.timestamp", "NULL")},
                    e.timestamp,
                    {self._SQLITE_NOW_UTC}
                ) AS timestamp
            FROM {legacy_table} p
            LEFT JOIN experiments e ON e.id = p.experiment_id
            ORDER BY p.id
            """
        )
        self.conn.execute(f"DROP TABLE {legacy_table}")

    def _ensure_metrics_table(self) -> None:
        if not self._table_exists("metrics"):
            self._create_metrics_table()
            return

        current_columns = self._table_columns("metrics")
        fk_targets = self._foreign_key_targets("metrics")
        if current_columns == self._EXPECTED_METRIC_COLUMNS and fk_targets == {"experiments"}:
            return

        legacy_table = self._next_legacy_name("metrics")
        self.conn.execute(f"ALTER TABLE metrics RENAME TO {legacy_table}")
        self._create_metrics_table()

        legacy_columns = set(self._table_columns(legacy_table))
        self.conn.execute(
            f"""
            INSERT INTO metrics (
                id,
                experiment_id,
                epoch,
                split,
                metric_name,
                metric_value,
                timestamp
            )
            SELECT
                m.id,
                m.experiment_id,
                m.epoch,
                m.split,
                m.metric_name,
                m.metric_value,
                COALESCE(
                    {self._qualified_or_default(legacy_columns, "m.timestamp", "NULL")},
                    e.timestamp,
                    {self._SQLITE_NOW_UTC}
                ) AS timestamp
            FROM {legacy_table} m
            LEFT JOIN experiments e ON e.id = m.experiment_id
            ORDER BY m.id
            """
        )
        self.conn.execute(f"DROP TABLE {legacy_table}")

    def _create_indexes_and_views(self) -> None:
        self.conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_experiments_lookup
                ON experiments(dataset, preset, training_mode, graph_method);

            CREATE INDEX IF NOT EXISTS idx_experiments_batch_lookup
                ON experiments(batch_id, dataset, preset, intervention, training_mode, graph_method, seed, id DESC);

            CREATE INDEX IF NOT EXISTS idx_experiments_profile_updated
                ON experiments(profile_name, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_experiments_status
                ON experiments(status, oom_flag);

            CREATE INDEX IF NOT EXISTS idx_experiments_status_updated
                ON experiments(status, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_metrics_exp_split_name_epoch
                ON metrics(experiment_id, split, metric_name, epoch);

            CREATE INDEX IF NOT EXISTS idx_profiling_exp_stage_epoch
                ON profiling(experiment_id, stage, epoch);

            DROP VIEW IF EXISTS experiment_summary;

            CREATE VIEW experiment_summary AS
            SELECT
                e.id,
                e.timestamp,
                e.updated_at,
                e.dataset,
                e.preset,
                e.intervention,
                e.training_mode,
                e.graph_method,
                e.seed,
                e.status,
                e.failure_reason,
                e.oom_flag,
                e.batch_id,
                e.profile_name,
                e.gpu_name,
                e.gpu_vram_gb,
                AVG(CASE
                    WHEN m.metric_name = 'loss' AND m.split = 'train'
                    THEN m.metric_value
                END) AS avg_train_loss,
                AVG(CASE
                    WHEN m.metric_name = 'epoch_time_s' AND m.split = 'train'
                    THEN m.metric_value
                END) AS avg_epoch_time_s,
                MAX(CASE
                    WHEN m.metric_name = 'Recall@20' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_recall_20,
                MAX(CASE
                    WHEN m.metric_name = 'NDCG@20' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_ndcg_20,
                MIN(CASE
                    WHEN m.metric_name = 'AveragePopularity@20' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_average_popularity_20,
                MAX(CASE
                    WHEN m.metric_name = 'NDCG@50' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_ndcg_50,
                MAX(CASE
                    WHEN m.metric_name = 'Recall@50' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_recall_50,
                MIN(CASE
                    WHEN m.metric_name = 'AveragePopularity@50' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_average_popularity_50,
                AVG(CASE
                    WHEN p.stage = 'forward'
                    THEN p.duration_ms / NULLIF(p.stage_call_count, 0)
                END) AS avg_forward_ms,
                MAX(p.vram_peak_mb) AS peak_vram_mb
            FROM experiments e
            LEFT JOIN metrics m ON e.id = m.experiment_id
            LEFT JOIN profiling p ON e.id = p.experiment_id
            GROUP BY e.id
            ORDER BY e.timestamp DESC;

            DROP VIEW IF EXISTS experiment_completed_summary;
            CREATE VIEW experiment_completed_summary AS
            SELECT *
            FROM experiment_summary
            WHERE status = 'completed'
            ORDER BY updated_at DESC, id DESC;

            DROP VIEW IF EXISTS experiment_attention_summary;
            CREATE VIEW experiment_attention_summary AS
            SELECT *
            FROM experiment_summary
            WHERE status IS NULL OR status <> 'completed'
            ORDER BY updated_at DESC, id DESC;

            DROP VIEW IF EXISTS experiment_error_summary;
            CREATE VIEW experiment_error_summary AS
            SELECT *
            FROM experiment_summary
            WHERE status IN ('oom', 'failed')
            ORDER BY updated_at DESC, id DESC;
        """)

    def _repair_experiment_statuses(self) -> None:
        """Backfill stale status rows so exploration views reflect historical reality."""
        self.conn.execute(
            f"""
            UPDATE experiments
            SET status = 'completed',
                updated_at = COALESCE(updated_at, timestamp, {self._SQLITE_NOW_UTC})
            WHERE status IN ('running', 'unknown')
              AND EXISTS (
                  SELECT 1
                  FROM metrics m
                  WHERE m.experiment_id = experiments.id
                    AND m.split = 'test'
              )
            """
        )
        self.conn.execute(
            f"""
            UPDATE experiments
            SET status = 'oom',
                updated_at = COALESCE(updated_at, timestamp, {self._SQLITE_NOW_UTC})
            WHERE status IN ('running', 'unknown')
              AND oom_flag = 1
            """
        )
        self.conn.execute(
            f"""
            UPDATE experiments
            SET status = 'failed',
                updated_at = COALESCE(updated_at, timestamp, {self._SQLITE_NOW_UTC})
            WHERE status IN ('running', 'unknown')
              AND failure_reason IS NOT NULL
            """
        )

    def _serialize_config(self, config) -> str:
        if hasattr(config, "model_dump"):
            return json.dumps(config.model_dump(), default=str)
        if is_dataclass(config):
            return json.dumps(asdict(config), default=str)
        return json.dumps(config, default=str)

    @staticmethod
    def _mean_or_none(values: list[float | None]) -> float | None:
        finite_values = [float(value) for value in values if value is not None]
        if not finite_values:
            return None
        return sum(finite_values) / len(finite_values)

    def _aggregate_profiler_stages(
        self,
        profiler_stages: list,
    ) -> list[dict[str, float | int | str | None]]:
        aggregated: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "duration_ms": 0.0,
                "vram_before_mb": [],
                "vram_after_mb": [],
                "vram_peak_mb": [],
                "stage_call_count": 0,
            }
        )

        for stage in profiler_stages:
            bucket = aggregated[stage.name]
            bucket["duration_ms"] = float(bucket["duration_ms"]) + float(
                stage.elapsed_ms
            )
            bucket["vram_before_mb"].append(stage.vram_before_mb)
            bucket["vram_after_mb"].append(stage.vram_after_mb)
            bucket["vram_peak_mb"].append(stage.vram_peak_mb)
            bucket["stage_call_count"] = int(bucket["stage_call_count"]) + 1

        return [
            {
                "stage": stage_name,
                "duration_ms": float(values["duration_ms"]),
                "vram_before_mb": self._mean_or_none(values["vram_before_mb"]),
                "vram_after_mb": self._mean_or_none(values["vram_after_mb"]),
                "vram_peak_mb": max(
                    (
                        float(value)
                        for value in values["vram_peak_mb"]
                        if value is not None
                    ),
                    default=None,
                ),
                "stage_call_count": int(values["stage_call_count"]),
            }
            for stage_name, values in sorted(aggregated.items())
        ]

    # ── Public API ────────────────────────────────────────────────────────

    def log_experiment(
        self,
        dataset: str,
        config,
        preset: str | None = None,
        intervention: str | None = None,
        status: str = "running",
        batch_id: str | None = None,
        profile_name: str | None = None,
        gpu_name: str | None = None,
        gpu_vram_gb: float | None = None,
    ) -> int:
        """Create an experiment row and return its id."""
        config_json = self._serialize_config(config)
        seed = getattr(config, "seed", None)
        training_mode = getattr(config, "training_mode", None)
        graph_method = getattr(config, "graph_method", None)
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO experiments (dataset, preset, intervention, config_json, seed, training_mode, graph_method, status, failure_reason, oom_flag, batch_id, profile_name, gpu_name, gpu_vram_gb, timestamp, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dataset,
                preset,
                intervention,
                config_json,
                seed,
                training_mode,
                graph_method,
                status,
                None,
                0,
                batch_id,
                profile_name,
                gpu_name,
                gpu_vram_gb,
                now,
                now,
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_experiment_status(
        self,
        exp_id: int,
        *,
        status: str,
        failure_reason: str | None = None,
        oom_flag: bool | None = None,
    ) -> None:
        """Update terminal or intermediate state for an experiment row.

        Args:
            exp_id: Experiment id to update.
            status: New experiment status.
            failure_reason: Optional free-form failure detail.
            oom_flag: Optional explicit OOM indicator.
        """
        self.conn.execute(
            "UPDATE experiments SET status = ?, failure_reason = ?, oom_flag = COALESCE(?, oom_flag), updated_at = ? WHERE id = ?",
            (
                status,
                failure_reason,
                None if oom_flag is None else int(oom_flag),
                datetime.now(timezone.utc).isoformat(),
                exp_id,
            ),
        )
        self.conn.commit()

    def find_latest_batch_experiment(
        self,
        *,
        batch_id: str,
        dataset: str,
        preset: str | None,
        intervention: str | None,
        seed: int | None,
        training_mode: str | None,
        graph_method: str | None,
    ) -> sqlite3.Row | None:
        """Return the most recent experiment row for a batch-scoped matrix item."""
        filters = {
            "preset": preset,
            "intervention": intervention,
            "seed": seed,
            "training_mode": training_mode,
            "graph_method": graph_method,
        }
        where_clauses = ["batch_id = ?", "dataset = ?"]
        params: list[object] = [batch_id, dataset]

        for column_name, value in filters.items():
            if value is None:
                continue
            where_clauses.append(f"{column_name} = ?")
            params.append(value)

        sql = (
            "SELECT * FROM experiments WHERE "
            + " AND ".join(where_clauses)
            + " ORDER BY id DESC LIMIT 1"
        )
        return self.conn.execute(sql, params).fetchone()

    def get_metrics_for_split(
        self,
        exp_id: int,
        *,
        split: str,
    ) -> dict[str, float]:
        """Return the latest metric values for a split keyed by metric name."""
        rows = self.conn.execute(
            """
            SELECT metric_name, metric_value
            FROM metrics
            WHERE experiment_id = ? AND split = ?
            ORDER BY metric_name, COALESCE(epoch, -1) DESC, id DESC
            """,
            (exp_id, split),
        ).fetchall()

        metrics: dict[str, float] = {}
        for row in rows:
            metric_name = row[0]
            if metric_name not in metrics:
                metrics[metric_name] = float(row[1])
        return metrics

    def log_profiling(
        self,
        exp_id: int,
        epoch: int,
        stage,
        stage_call_count: int = 1,
    ) -> None:
        """Log a single profiling row.

        This method remains usable for direct inserts in verification scripts.
        Regular training should prefer ``log_epoch()``, which aggregates raw
        per-batch StageMetrics into one row per (epoch, stage).
        """
        self.conn.execute(
            "INSERT INTO profiling (experiment_id, epoch, stage, duration_ms, vram_before_mb, vram_after_mb, vram_peak_mb, stage_call_count, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exp_id,
                epoch,
                stage.name,
                stage.elapsed_ms,
                stage.vram_before_mb,
                stage.vram_after_mb,
                stage.vram_peak_mb,
                stage_call_count,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def log_metric(
        self,
        exp_id: int,
        metric_name: str,
        value: float,
        epoch: int | None = None,
        split: str = "test",
    ) -> None:
        """Log a single metric value."""
        if not math.isfinite(float(value)):
            raise ValueError(
                "Cannot log non-finite metric value "
                f"for experiment_id={exp_id}, split={split}, epoch={epoch}, "
                f"metric={metric_name}: {value!r}"
            )
        self.conn.execute(
            "INSERT INTO metrics (experiment_id, epoch, split, metric_name, metric_value, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                exp_id,
                epoch,
                split,
                metric_name,
                value,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def log_epoch(
        self,
        exp_id: int,
        epoch: int,
        train_loss: float,
        epoch_time_s: float,
        val_metrics: dict,
        profiler_stages: list,
        model=None,
    ) -> None:
        """Convenience: log all data for one epoch in a single call."""
        self.log_metric(exp_id, "loss", train_loss, epoch=epoch, split="train")
        self.log_metric(
            exp_id, "epoch_time_s", epoch_time_s, epoch=epoch, split="train"
        )

        for name, value in val_metrics.items():
            self.log_metric(exp_id, name, value, epoch=epoch, split="val")

        for stage_summary in self._aggregate_profiler_stages(profiler_stages):
            self.conn.execute(
                "INSERT INTO profiling (experiment_id, epoch, stage, duration_ms, vram_before_mb, vram_after_mb, vram_peak_mb, stage_call_count, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    exp_id,
                    epoch,
                    stage_summary["stage"],
                    stage_summary["duration_ms"],
                    stage_summary["vram_before_mb"],
                    stage_summary["vram_after_mb"],
                    stage_summary["vram_peak_mb"],
                    stage_summary["stage_call_count"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        # Log alpha_pos/alpha_neg if sign-aware model
        if model is not None:
            gcn = getattr(model, "gcn", None)
            if gcn is not None:
                for attr in ("alpha_pos", "alpha_neg"):
                    param = getattr(gcn, attr, None)
                    if param is not None:
                        self.log_metric(
                            exp_id, attr, param.item(), epoch=epoch, split="train"
                        )

        self.conn.commit()

    def close(self) -> None:
        self.flush()
        self.conn.close()

    def flush(self) -> None:
        """Flush pending writes to the database."""
        self.conn.commit()
