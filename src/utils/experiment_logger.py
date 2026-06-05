"""ExperimentLogger: lightweight SQLite experiment tracker -- no MLflow dependency."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar


class ExperimentLogger:
    """Persist experiment configs, per-epoch metrics, and profiling data to SQLite."""

    TERMINAL_STATUSES = ("completed", "failed", "oom")
    SUMMARY_VIEWS = (
        "experiment_summary",
        "experiment_completed_summary",
        "experiment_attention_summary",
        "experiment_error_summary",
        "experiment_code_comparison",
    )
    VIEW_TABLES: ClassVar[dict[str, str]] = {
        "all": "experiments",
        "completed": "experiment_completed_summary",
        "attention": "experiment_attention_summary",
        "errors": "experiment_error_summary",
        "comparison": "experiment_code_comparison",
    }

    _SQLITE_NOW_UTC = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
    _EXPECTED_EXPERIMENT_COLUMNS: ClassVar[tuple[str, ...]] = (
        "id",
        "dataset",
        "preset",
        "intervention",
        "config_json",
        "seed",
        "training_mode",
        "status",
        "failure_reason",
        "oom_flag",
        "batch_id",
        "profile_name",
        "gpu_name",
        "gpu_vram_gb",
        "timestamp",
        "updated_at",
        "project_version",
        "git_commit",
        "training_hash",
        "evaluation_hash",
        "change_note",
    )
    _LEGACY_EXPERIMENT_COLUMNS: ClassVar[tuple[str, ...]] = (
        "id",
        "dataset",
        "preset",
        "intervention",
        "config_json",
        "seed",
        "training_mode",
        "status",
        "failure_reason",
        "oom_flag",
        "batch_id",
        "profile_name",
        "gpu_name",
        "gpu_vram_gb",
        "timestamp",
        "updated_at",
    )
    _EXPECTED_PROFILING_COLUMNS: ClassVar[tuple[str, ...]] = (
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
    )
    _EXPECTED_METRIC_COLUMNS: ClassVar[tuple[str, ...]] = (
        "id",
        "experiment_id",
        "epoch",
        "split",
        "metric_name",
        "metric_value",
        "timestamp",
    )

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
        for view_name in self.SUMMARY_VIEWS:
            self.conn.execute(f"DROP VIEW IF EXISTS {view_name}")
        self._ensure_table(
            "experiments",
            self._EXPECTED_EXPERIMENT_COLUMNS,
            self._create_experiments_table,
        )
        self._ensure_table(
            "profiling",
            self._EXPECTED_PROFILING_COLUMNS,
            self._create_profiling_table,
            expected_fks=frozenset({"experiments"}),
        )
        self._ensure_table(
            "metrics",
            self._EXPECTED_METRIC_COLUMNS,
            self._create_metrics_table,
            expected_fks=frozenset({"experiments"}),
        )

    def _ensure_table(
        self,
        table_name: str,
        expected_columns: tuple[str, ...],
        create_fn: Callable[[], None],
        expected_fks: frozenset[str] | None = None,
    ) -> None:
        """Create *table_name* if absent, else validate its schema.

        Args:
            table_name: Name of the SQLite table to check or create.
            expected_columns: Ordered tuple of expected column names.
            create_fn: Zero-arg callable that creates the table when absent.
            expected_fks: Optional set of expected referenced table names.

        """
        if not self._table_exists(table_name):
            create_fn()
            return

        current_columns = self._table_columns(table_name)
        if expected_fks is not None:
            fk_targets = self._foreign_key_targets(table_name)
            if current_columns == expected_columns and fk_targets == expected_fks:
                return
            raise RuntimeError(
                (
                    f"{table_name} must already use the current schema. Expected "
                    f"columns {expected_columns} with foreign keys {sorted(expected_fks)}, "
                    f"found columns {current_columns} with foreign keys {sorted(fk_targets)}."
                ),
            )
        if current_columns == expected_columns:
            return
        if table_name == "experiments" and current_columns == self._LEGACY_EXPERIMENT_COLUMNS:
            self._migrate_legacy_experiments_table()
            current_columns = self._table_columns(table_name)
            if current_columns == expected_columns:
                return
        raise RuntimeError(
            (
                f"{table_name} must already use the current schema. Expected "
                f"columns {expected_columns}, found {current_columns}."
            ),
        )

    def _table_exists(self, table_name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _table_columns(self, table_name: str) -> tuple[str, ...]:
        return tuple(row[1] for row in self.conn.execute(f"PRAGMA table_info({table_name})"))

    def _foreign_key_targets(self, table_name: str) -> set[str]:
        """Return referenced table names declared by a table's foreign keys."""
        return {row[2] for row in self.conn.execute(f"PRAGMA foreign_key_list({table_name})")}

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
                status       TEXT    NOT NULL DEFAULT 'unknown',
                failure_reason TEXT,
                oom_flag     INTEGER NOT NULL DEFAULT 0,
                batch_id     TEXT,
                profile_name TEXT,
                gpu_name     TEXT,
                gpu_vram_gb  REAL,
                timestamp    TEXT    NOT NULL
                ,updated_at  TEXT    NOT NULL,
                project_version TEXT,
                git_commit   TEXT,
                training_hash TEXT,
                evaluation_hash TEXT,
                change_note  TEXT
            )
        """)

    def _migrate_legacy_experiments_table(self) -> None:
        """Add provenance columns to the immediately previous experiment schema."""
        for column_name in (
            "project_version",
            "git_commit",
            "training_hash",
            "evaluation_hash",
            "change_note",
        ):
            self.conn.execute(
                f"ALTER TABLE experiments ADD COLUMN {column_name} TEXT",
            )

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

    def _create_indexes_and_views(self) -> None:
        self.conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_experiments_lookup
                ON experiments(dataset, preset, training_mode);

            CREATE INDEX IF NOT EXISTS idx_experiments_batch_lookup
                ON experiments(
                    batch_id, dataset, preset, intervention,
                    training_mode, seed, id DESC
                );

            CREATE INDEX IF NOT EXISTS idx_experiments_profile_updated
                ON experiments(profile_name, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_experiments_training_hash_updated
                ON experiments(training_hash, updated_at DESC);

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
                e.seed,
                e.status,
                e.failure_reason,
                e.oom_flag,
                e.batch_id,
                e.profile_name,
                e.project_version,
                e.git_commit,
                e.training_hash,
                e.evaluation_hash,
                e.change_note,
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
                    WHEN m.metric_name = 'training_time_s' AND m.split = 'train'
                    THEN m.metric_value
                END) AS training_time_s,
                COALESCE(
                    MAX(CASE
                        WHEN m.metric_name = 'loss' AND m.split = 'train'
                        THEN m.epoch + 1
                    END),
                    MAX(CASE
                        WHEN m.metric_name = 'epoch_time_s' AND m.split = 'train'
                        THEN m.epoch + 1
                    END)
                ) AS completed_train_epochs,
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
                    WHEN m.metric_name = 'NDCG@40' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_ndcg_40,
                MAX(CASE
                    WHEN m.metric_name = 'Recall@40' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_recall_40,
                MIN(CASE
                    WHEN m.metric_name = 'AveragePopularity@40' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_average_popularity_40,
                MAX(CASE
                    WHEN m.metric_name = 'HitRatio@20' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_hit_ratio_20,
                MAX(CASE
                    WHEN m.metric_name = 'HitRatio@40' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_hit_ratio_40,
                MAX(CASE
                    WHEN m.metric_name = 'Personalization@20' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_personalization_20,
                MAX(CASE
                    WHEN m.metric_name = 'Personalization@40' AND m.split = 'val'
                    THEN m.metric_value
                END) AS best_personalization_40,
                MAX(CASE
                    WHEN m.metric_name = 'NDCG@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_ndcg_20,
                MAX(CASE
                    WHEN m.metric_name = 'NDCG@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_ndcg_40,
                MAX(CASE
                    WHEN m.metric_name = 'Recall@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_recall_20,
                MAX(CASE
                    WHEN m.metric_name = 'Recall@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_recall_40,
                MAX(CASE
                    WHEN m.metric_name = 'HitRatio@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_hit_ratio_20,
                MAX(CASE
                    WHEN m.metric_name = 'HitRatio@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_hit_ratio_40,
                MAX(CASE
                    WHEN m.metric_name = 'Personalization@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_personalization_20,
                MAX(CASE
                    WHEN m.metric_name = 'Personalization@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_personalization_40,
                MIN(CASE
                    WHEN m.metric_name = 'AveragePopularity@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_average_popularity_20,
                MIN(CASE
                    WHEN m.metric_name = 'AveragePopularity@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_average_popularity_40,
                MAX(CASE
                    WHEN m.metric_name = 'conformity_contribution@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_conformity_contribution_20,
                MAX(CASE
                    WHEN m.metric_name = 'conformity_contribution@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_conformity_contribution_40,
                MAX(CASE
                    WHEN m.metric_name = 'conformity_popularity_spearman@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_conformity_popularity_spearman_20,
                MAX(CASE
                    WHEN m.metric_name = 'conformity_popularity_spearman@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_conformity_popularity_spearman_40,
                MAX(CASE
                    WHEN m.metric_name = 'context_contribution@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_context_contribution_20,
                MAX(CASE
                    WHEN m.metric_name = 'context_contribution@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_context_contribution_40,
                MAX(CASE
                    WHEN m.metric_name = 'context_popularity_spearman@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_context_popularity_spearman_20,
                MAX(CASE
                    WHEN m.metric_name = 'context_popularity_spearman@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_context_popularity_spearman_40,
                MAX(CASE
                    WHEN m.metric_name = 'final_popularity_spearman@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_final_popularity_spearman_20,
                MAX(CASE
                    WHEN m.metric_name = 'final_popularity_spearman@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_final_popularity_spearman_40,
                MAX(CASE
                    WHEN m.metric_name = 'interest_conformity_cosine_mean' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_interest_conformity_cosine_mean,
                MAX(CASE
                    WHEN m.metric_name = 'interest_conformity_cosine_std' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_interest_conformity_cosine_std,
                MAX(CASE
                    WHEN m.metric_name = 'interest_contribution@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_interest_contribution_20,
                MAX(CASE
                    WHEN m.metric_name = 'interest_contribution@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_interest_contribution_40,
                MAX(CASE
                    WHEN m.metric_name = 'interest_popularity_spearman@20' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_interest_popularity_spearman_20,
                MAX(CASE
                    WHEN m.metric_name = 'interest_popularity_spearman@40' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_interest_popularity_spearman_40,
                MAX(CASE
                    WHEN m.metric_name = 'score_mix_conformity_mean' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_score_mix_conformity_mean,
                MAX(CASE
                    WHEN m.metric_name = 'score_mix_conformity_std' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_score_mix_conformity_std,
                MAX(CASE
                    WHEN m.metric_name = 'score_mix_context_mean' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_score_mix_context_mean,
                MAX(CASE
                    WHEN m.metric_name = 'score_mix_context_std' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_score_mix_context_std,
                MAX(CASE
                    WHEN m.metric_name = 'score_mix_interest_mean' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_score_mix_interest_mean,
                MAX(CASE
                    WHEN m.metric_name = 'score_mix_interest_std' AND m.split = 'test'
                    THEN m.metric_value
                END) AS test_score_mix_interest_std,
                AVG(CASE
                    WHEN p.stage = 'forward'
                    THEN p.duration_ms / NULLIF(p.stage_call_count, 0)
                END) AS avg_forward_ms,
                AVG(CASE
                    WHEN m.metric_name = 'gpu_utilization_pct' AND m.split = 'train'
                    THEN m.metric_value
                END) AS avg_gpu_utilization_pct,
                MAX(CASE
                    WHEN m.metric_name = 'gpu_utilization_pct' AND m.split = 'train'
                    THEN m.metric_value
                END) AS max_gpu_utilization_pct,
                COALESCE(
                    MAX(p.vram_peak_mb),
                    MAX(CASE WHEN m.metric_name = 'peak_vram_mb' THEN m.metric_value END)
                ) AS peak_vram_mb
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

            DROP VIEW IF EXISTS experiment_code_comparison;
            CREATE VIEW experiment_code_comparison AS
            SELECT
                s.id,
                s.timestamp,
                s.updated_at,
                s.dataset,
                s.preset,
                s.intervention,
                s.profile_name,
                s.project_version,
                s.git_commit,
                s.training_hash,
                s.evaluation_hash,
                s.change_note,
                s.test_ndcg_20,
                s.test_recall_20,
                s.test_average_popularity_20,
                s.test_ndcg_40,
                s.test_recall_40,
                s.test_average_popularity_40,
                s.avg_epoch_time_s,
                s.training_time_s,
                s.completed_train_epochs,
                s.peak_vram_mb,
                s.test_ndcg_20 - LAG(s.test_ndcg_20) OVER (
                    PARTITION BY s.dataset, s.preset, COALESCE(s.intervention, ''),
                    s.training_hash, s.evaluation_hash
                    ORDER BY s.updated_at, s.id
                ) AS delta_test_ndcg_20,
                s.test_recall_20 - LAG(s.test_recall_20) OVER (
                    PARTITION BY s.dataset, s.preset, COALESCE(s.intervention, ''),
                    s.training_hash, s.evaluation_hash
                    ORDER BY s.updated_at, s.id
                ) AS delta_test_recall_20,
                s.test_average_popularity_20 - LAG(s.test_average_popularity_20) OVER (
                    PARTITION BY s.dataset, s.preset, COALESCE(s.intervention, ''),
                    s.training_hash, s.evaluation_hash
                    ORDER BY s.updated_at, s.id
                ) AS delta_test_average_popularity_20
            FROM experiment_summary s
            ORDER BY s.updated_at DESC, s.id DESC;
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
            """,
        )
        self.conn.execute(
            f"""
            UPDATE experiments
            SET status = 'oom',
                updated_at = COALESCE(updated_at, timestamp, {self._SQLITE_NOW_UTC})
            WHERE status IN ('running', 'unknown')
              AND oom_flag = 1
            """,
        )
        self.conn.execute(
            f"""
            UPDATE experiments
            SET status = 'failed',
                updated_at = COALESCE(updated_at, timestamp, {self._SQLITE_NOW_UTC})
            WHERE status IN ('running', 'unknown')
              AND failure_reason IS NOT NULL
            """,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def log_experiment(
        self,
        dataset: str,
        config,
        preset: str | None = None,
        intervention: str | None = None,
        training_mode: str | None = None,
        status: str = "running",
        batch_id: str | None = None,
        profile_name: str | None = None,
        project_version: str | None = None,
        git_commit: str | None = None,
        training_hash: str | None = None,
        evaluation_hash: str | None = None,
        change_note: str | None = None,
        gpu_name: str | None = None,
        gpu_vram_gb: float | None = None,
    ) -> int:
        """Create an experiment row and return its id."""
        if hasattr(config, "model_dump"):
            config_json = json.dumps(config.model_dump(), default=str)
        elif is_dataclass(config):
            config_json = json.dumps(asdict(config), default=str)
        else:
            config_json = json.dumps(config, default=str)
        seed = getattr(config, "seed", None)
        if training_mode is None:
            training_mode = getattr(config, "training_mode", None)
        now = datetime.now(UTC).isoformat()
        cur = self.conn.execute(
            (
                "INSERT INTO experiments ("
                "dataset, preset, intervention, config_json, seed, training_mode, "
                "status, failure_reason, oom_flag, batch_id, profile_name, "
                "project_version, git_commit, training_hash, evaluation_hash, "
                "change_note, gpu_name, gpu_vram_gb, timestamp, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                dataset,
                preset,
                intervention,
                config_json,
                seed,
                training_mode,
                status,
                None,
                0,
                batch_id,
                profile_name,
                project_version,
                git_commit,
                training_hash,
                evaluation_hash,
                change_note,
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
            (
                "UPDATE experiments SET status = ?, failure_reason = ?, "
                "oom_flag = COALESCE(?, oom_flag), updated_at = ? WHERE id = ?"
            ),
            (
                status,
                failure_reason,
                None if oom_flag is None else int(oom_flag),
                datetime.now(UTC).isoformat(),
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
        config_filters: dict[str, object] | None = None,
    ) -> sqlite3.Row | None:
        """Return the most recent experiment row for a batch-scoped matrix item."""
        filters = {
            "preset": preset,
            "intervention": intervention,
            "seed": seed,
            "training_mode": training_mode,
        }
        where_clauses = ["batch_id = ?", "dataset = ?"]
        params: list[object] = [batch_id, dataset]

        for column_name, value in filters.items():
            if value is None:
                continue
            where_clauses.append(f"{column_name} = ?")
            params.append(value)

        if config_filters is not None:
            for field_name, value in config_filters.items():
                where_clauses.append("config_json LIKE ?")
                params.append(f'%"{field_name}": {json.dumps(value)}%')

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
            (
                "INSERT INTO profiling (experiment_id, epoch, stage, duration_ms, "
                "vram_before_mb, vram_after_mb, vram_peak_mb, stage_call_count, "
                "timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                exp_id,
                epoch,
                stage.name,
                stage.elapsed_ms,
                stage.vram_before_mb,
                stage.vram_after_mb,
                stage.vram_peak_mb,
                stage_call_count,
                datetime.now(UTC).isoformat(),
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
                (
                    f"Cannot log non-finite metric value for experiment_id={exp_id}, "
                    f"split={split}, epoch={epoch}, metric={metric_name}: {value!r}"
                ),
            )
        self.conn.execute(
            (
                "INSERT INTO metrics (experiment_id, epoch, split, metric_name, "
                "metric_value, timestamp) VALUES (?, ?, ?, ?, ?, ?)"
            ),
            (
                exp_id,
                epoch,
                split,
                metric_name,
                value,
                datetime.now(UTC).isoformat(),
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
            exp_id,
            "epoch_time_s",
            epoch_time_s,
            epoch=epoch,
            split="train",
        )

        for name, value in val_metrics.items():
            self.log_metric(exp_id, name, value, epoch=epoch, split="val")

        aggregated_stages: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "duration_ms": 0.0,
                "vram_before_mb": [],
                "vram_after_mb": [],
                "vram_peak_mb": [],
                "stage_call_count": 0,
            },
        )
        for stage in profiler_stages:
            stage_summary = aggregated_stages[stage.name]
            stage_summary["duration_ms"] = float(stage_summary["duration_ms"]) + float(
                stage.elapsed_ms,
            )
            stage_summary["vram_before_mb"].append(stage.vram_before_mb)
            stage_summary["vram_after_mb"].append(stage.vram_after_mb)
            stage_summary["vram_peak_mb"].append(stage.vram_peak_mb)
            stage_summary["stage_call_count"] = int(stage_summary["stage_call_count"]) + 1

        for stage_name, stage_values in sorted(aggregated_stages.items()):
            vram_before_values = [
                float(value) for value in stage_values["vram_before_mb"] if value is not None
            ]
            vram_after_values = [
                float(value) for value in stage_values["vram_after_mb"] if value is not None
            ]
            self.conn.execute(
                (
                    "INSERT INTO profiling (experiment_id, epoch, stage, duration_ms, "
                    "vram_before_mb, vram_after_mb, vram_peak_mb, stage_call_count, "
                    "timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    exp_id,
                    epoch,
                    stage_name,
                    float(stage_values["duration_ms"]),
                    (
                        sum(vram_before_values) / len(vram_before_values)
                        if vram_before_values
                        else None
                    ),
                    (
                        sum(vram_after_values) / len(vram_after_values)
                        if vram_after_values
                        else None
                    ),
                    max(
                        (
                            float(value)
                            for value in stage_values["vram_peak_mb"]
                            if value is not None
                        ),
                        default=None,
                    ),
                    int(stage_values["stage_call_count"]),
                    datetime.now(UTC).isoformat(),
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
                            exp_id,
                            attr,
                            param.item(),
                            epoch=epoch,
                            split="train",
                        )

        self.conn.commit()

    def close(self) -> None:
        """Flush pending writes and close the database connection."""
        self.conn.commit()
        self.conn.close()
