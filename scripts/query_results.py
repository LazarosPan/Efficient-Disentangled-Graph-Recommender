#!/usr/bin/env python
"""Query experiment results from SQLite database.

Usage:
    python scripts/query_results.py                    # Formal top runs + full-data ablations
    python scripts/query_results.py --view all         # Show all experiments
    python scripts/query_results.py --view completed   # Show only completed runs
    python scripts/query_results.py --view attention   # Show failed, OOM, running, or unknown runs
    python scripts/query_results.py --view errors      # Show only failed and OOM runs
    python scripts/query_results.py --view comparison  # Show summary comparison tables
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from experiments.ablation_configs import ABLATION_VARIANTS
from src.utils.cli_parsers import build_query_results_parser
from src.utils.experiment_logger import RUNTIME_PROBE_METRIC_NAMES, ExperimentLogger
from src.utils.experiment_naming import (
    build_canonical_experiment_name,
    format_num_neighbors_payload,
)
from src.utils.project_paths import RESULTS_DIR, THESIS_DB_PATH

DB_PATH = Path(os.environ.get("THESIS_DB_PATH_OVERRIDE", str(THESIS_DB_PATH)))
QUERY_RESULTS_MARKDOWN_PATH = RESULTS_DIR / "query_results.md"
VIEW_TABLES = ExperimentLogger.VIEW_TABLES
FORMAL_BATCH_PREFIX = "formal-"
ABLATION_BATCH_PREFIX = "ablation-"
ABLATION_VARIANT_ORDER = {
    variant_name: index for index, variant_name in enumerate(ABLATION_VARIANTS)
}
CRRU_EPSILON = 1e-8
FINAL_FORMAL_PROFILE_NAMES = frozenset(
    {
        "core-ucagnn-mainline",
        "core-paper-architecture-comparison",
        "paper-lightgcn-small-baselines",
        "paper-lightgcn-baselines",
    }
)
RUNTIME_PROBE_COLUMNS = RUNTIME_PROBE_METRIC_NAMES


def connect() -> sqlite3.Connection:
    """Connect to experiment database."""
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH.resolve()}")
        print(
            "Run a real experiment first to create the persistent database in results/.",
        )
        print(
            "Quick-validate uses the same database, but its smoke runs are filtered "
            "out of the default summary via sample_interactions/loader_max_rows.",
        )
        sys.exit(1)

    conn = sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
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
    print(f"Database: {THESIS_DB_PATH.resolve()}")
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
             batch_id, profile_name, training_mode, oom_flag
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
        (
            ""
            f"{'ID':>4} | {'Status':<10} | {'Dataset':<14} | {'Preset':<12} | "
            f"{'Profile':<{profile_width}} | {'Mode':<18} | {'Batch':<22} | Seed"
        ),
    )
    print("-" * (135 + profile_width - 8))
    for row in rows:
        batch_label = row["batch_id"] or "-"
        profile_label = row["profile_name"] or "-"
        status_label = row["status"] or ("oom" if row["oom_flag"] else "unknown")
        print(
            (
                f"{row['id']:>4} | {status_label:<10} | {row['dataset'] or '-':<14} | "
                f"{row['preset'] or '-':<12} | {profile_label:<{profile_width}} | "
                f"{row['training_mode'] or '-':<18} | {batch_label:<22} | "
                f"{row['seed'] or '-'}"
            ),
        )


def _load_config_json(config_json: str | None) -> dict[str, object]:
    """Parse stored config JSON into a Python mapping."""
    if not config_json:
        return {}
    try:
        return json.loads(config_json)
    except json.JSONDecodeError:
        return {}


def _format_metric_value(value: float | None) -> str:
    """Return a consistent display string for a metric cell."""
    return f"{value:.4f}" if value is not None else "-"


def _format_duration(value_s: float | None) -> str:
    """Return measured or estimated runtime in seconds."""
    if value_s is None:
        return "-"
    return f"{value_s:.1f}s"


def _format_count(value: float | None) -> str:
    """Return a whole-number display string for count-like metrics."""
    return "-" if value is None else f"{value:.0f}"


def _format_rate(value: float | None) -> str:
    """Return a compact batch/s display string."""
    return "-" if value is None else f"{value:.2f}"


def _format_crru_value(value: float | None) -> str:
    """Return a CRRU display string that preserves very small positive values."""
    if value is None:
        return "-"
    if value < 0.001:
        return f"{value:.3e}"
    return f"{value:.4f}"


def _format_neighbors(config: dict[str, object]) -> str:
    """Return the configured neighborhood fan-out as a compact label."""
    return format_num_neighbors_payload(config.get("num_neighbors")) or "-"


def _format_scoremix(config: dict[str, object]) -> str:
    """Return the stored scoring-weight mode, defaulting to the fixed thesis path."""
    legacy_mode = config.get("scoring_weight_mode")
    if legacy_mode is not None:
        return str(legacy_mode)
    return "learned" if bool(config.get("use_learned_score_mix", False)) else "fixed"


def _truncate_label(value: str | None, width: int) -> str:
    """Trim a long display label so table columns remain readable."""
    if not value:
        return "-"
    if len(value) <= width:
        return value
    return f"{value[: width - 3]}..."


def _query_report_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return completed full-data formal and ablation rows with test metrics."""
    summary_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(experiment_completed_summary)")
    }

    def summary_column(column_name: str) -> str:
        if column_name in summary_columns:
            return f"s.{column_name}"
        return f"NULL AS {column_name}"

    try:
        return conn.execute(
            f"""
            SELECT s.id, s.dataset, s.preset, s.updated_at,
                   s.training_time_s, s.completed_train_epochs,
                   {summary_column("avg_epoch_time_s")},
                   s.peak_vram_mb,
                   s.avg_gpu_utilization_pct,
                   {summary_column("runtime_probe_target_epochs")},
                   {summary_column("runtime_probe_observed_epochs")},
                   {summary_column("runtime_probe_train_batches_per_epoch")},
                   {summary_column("runtime_probe_observed_batches_per_second")},
                   {summary_column("runtime_probe_seconds_per_epoch")},
                   {summary_column("runtime_probe_estimated_train_time_s")},
                   {summary_column("runtime_probe_estimated_remaining_train_time_s")},
                   s.test_ndcg_20, s.test_ndcg_40,
                   s.test_recall_20, s.test_recall_40,
                   s.test_hit_ratio_20, s.test_hit_ratio_40,
                   s.test_personalization_20, s.test_personalization_40,
                   s.test_average_popularity_20, s.test_average_popularity_40,
                   {summary_column("test_interest_branch_ndcg_20")},
                   {summary_column("test_interest_branch_ndcg_40")},
                   {summary_column("test_interest_branch_recall_20")},
                   {summary_column("test_interest_branch_recall_40")},
                   {summary_column("test_interest_branch_average_popularity_20")},
                   {summary_column("test_interest_branch_average_popularity_40")},
                   {summary_column("test_conformity_branch_ndcg_20")},
                   {summary_column("test_conformity_branch_ndcg_40")},
                   {summary_column("test_conformity_branch_recall_20")},
                   {summary_column("test_conformity_branch_recall_40")},
                   {summary_column("test_conformity_branch_average_popularity_20")},
                   {summary_column("test_conformity_branch_average_popularity_40")},
                   s.test_conformity_contribution_20,
                   s.test_conformity_contribution_40,
                   s.test_conformity_popularity_spearman_20,
                   s.test_conformity_popularity_spearman_40,
                   s.test_context_contribution_20,
                   s.test_context_contribution_40,
                   s.test_context_popularity_spearman_20,
                   s.test_context_popularity_spearman_40,
                   s.test_final_popularity_spearman_20,
                   s.test_final_popularity_spearman_40,
                   s.test_interest_conformity_cosine_mean,
                   s.test_interest_conformity_cosine_std,
                   s.test_interest_contribution_20,
                   s.test_interest_contribution_40,
                   s.test_interest_popularity_spearman_20,
                   s.test_interest_popularity_spearman_40,
                   s.test_score_mix_conformity_mean,
                   s.test_score_mix_conformity_std,
                   s.test_score_mix_context_mean,
                   s.test_score_mix_context_std,
                   s.test_score_mix_interest_mean,
                   s.test_score_mix_interest_std,
                   e.intervention, e.batch_id, e.profile_name, e.config_json
            FROM experiment_completed_summary s
            JOIN experiments e ON s.id = e.id
            WHERE s.test_ndcg_20 IS NOT NULL
              AND json_extract(e.config_json, '$.sample_interactions') IS NULL
              AND json_extract(e.config_json, '$.loader_max_rows') IS NULL
              AND (
                  e.batch_id LIKE 'formal-%'
                  OR e.batch_id LIKE 'ablation-%'
              )
            ORDER BY s.dataset ASC, s.id DESC
            """,
        ).fetchall()
    except sqlite3.OperationalError as e:
        if "no such column" in str(e).lower():
            # Fallback to core metrics only if database views haven't been recreated/updated yet
            return conn.execute(
                """
                SELECT s.id, s.dataset, s.preset, s.updated_at,
                       s.training_time_s, s.completed_train_epochs,
                       NULL AS avg_epoch_time_s,
                       s.peak_vram_mb,
                       s.avg_gpu_utilization_pct,
                       NULL AS runtime_probe_target_epochs,
                       NULL AS runtime_probe_observed_epochs,
                       NULL AS runtime_probe_train_batches_per_epoch,
                       NULL AS runtime_probe_observed_batches_per_second,
                       NULL AS runtime_probe_seconds_per_epoch,
                       NULL AS runtime_probe_estimated_train_time_s,
                       NULL AS runtime_probe_estimated_remaining_train_time_s,
                       s.test_ndcg_20, s.test_ndcg_40,
                       s.test_recall_20, s.test_recall_40,
                       s.test_hit_ratio_20, s.test_hit_ratio_40,
                       s.test_personalization_20, s.test_personalization_40,
                       s.test_average_popularity_20, s.test_average_popularity_40,
                       e.intervention, e.batch_id, e.profile_name, e.config_json
                FROM experiment_completed_summary s
                JOIN experiments e ON s.id = e.id
                WHERE s.test_ndcg_20 IS NOT NULL
                  AND json_extract(e.config_json, '$.sample_interactions') IS NULL
                  AND json_extract(e.config_json, '$.loader_max_rows') IS NULL
                  AND (
                      e.batch_id LIKE 'formal-%'
                      OR e.batch_id LIKE 'ablation-%'
                  )
                ORDER BY s.dataset ASC, s.id DESC
                """,
            ).fetchall()
        raise


def _is_formal_row(row: sqlite3.Row) -> bool:
    """Return whether a report row came from the formal-run workflow."""
    batch_id = row["batch_id"]
    return isinstance(batch_id, str) and batch_id.startswith(FORMAL_BATCH_PREFIX)


def _is_runtime_probe_row(row: sqlite3.Row) -> bool:
    """Return whether a report row is a timing probe rather than a final run."""
    for column_name in RUNTIME_PROBE_COLUMNS:
        try:
            if row[column_name] is not None:
                return True
        except (IndexError, KeyError):
            continue
    return False


def _is_reportable_formal_row(row: sqlite3.Row) -> bool:
    """Return whether a formal row is a full-run row rather than a timing probe."""
    return _is_formal_row(row) and not _is_runtime_probe_row(row)


def _is_final_formal_row(row: sqlite3.Row) -> bool:
    """Return whether a formal row belongs in final thesis metric rankings."""
    profile_name = row["profile_name"]
    return (
        _is_reportable_formal_row(row)
        and isinstance(profile_name, str)
        and profile_name in FINAL_FORMAL_PROFILE_NAMES
    )


def _is_supporting_formal_row(row: sqlite3.Row) -> bool:
    """Return whether a formal row is supporting historical or diagnostic evidence."""
    return _is_reportable_formal_row(row) and not _is_final_formal_row(row)


def _is_ablation_row(row: sqlite3.Row) -> bool:
    """Return whether a report row came from the ablation workflow."""
    batch_id = row["batch_id"]
    return isinstance(batch_id, str) and batch_id.startswith(ABLATION_BATCH_PREFIX)


def _is_supported_ablation_row(row: sqlite3.Row) -> bool:
    """Return whether an ablation row belongs to the current public variant matrix."""
    return _is_ablation_row(row) and row["intervention"] in ABLATION_VARIANTS


def _is_default_report_row(row: sqlite3.Row) -> bool:
    """Return whether a row participates in the default thesis report."""
    return _is_reportable_formal_row(row) or (
        _is_supported_ablation_row(row) and not _is_runtime_probe_row(row)
    )


def _crru_sort_key(
    row: sqlite3.Row,
    crru_scores: dict[int, dict[int, float]],
) -> tuple[float, float, float, float, int]:
    """Return the sort key that orders rows by CRRU within a dataset."""
    row_scores = crru_scores[int(row["id"])]
    return (
        -row_scores[20],
        -row_scores[40],
        -(float(row["test_ndcg_20"]) if row["test_ndcg_20"] is not None else float("-inf")),
        -(float(row["test_ndcg_40"]) if row["test_ndcg_40"] is not None else float("-inf")),
        int(row["id"]),
    )


def _select_top_formal_rows(
    rows: list[sqlite3.Row],
    *,
    n: int,
    crru_scores: dict[int, dict[int, float]],
    supporting: bool = False,
) -> list[sqlite3.Row]:
    """Return the top completed formal rows per dataset ranked by CRRU."""
    row_filter = _is_supporting_formal_row if supporting else _is_final_formal_row
    ranked_rows = sorted(
        (row for row in rows if row_filter(row)),
        key=lambda row: (row["dataset"] or "-", *_crru_sort_key(row, crru_scores)),
    )

    top_rows: list[sqlite3.Row] = []
    dataset_counts: dict[str, int] = {}
    for row in ranked_rows:
        dataset = row["dataset"] or "-"
        if dataset_counts.get(dataset, 0) >= n:
            continue
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        top_rows.append(row)
    return top_rows


def _select_best_ablation_rows(
    rows: list[sqlite3.Row],
    *,
    crru_scores: dict[int, dict[int, float]],
) -> list[sqlite3.Row]:
    """Return one completed full-data ablation row per dataset and variant."""
    best_by_variant: dict[tuple[str, str], sqlite3.Row] = {}
    ranked_rows = sorted(
        (row for row in rows if _is_supported_ablation_row(row)),
        key=lambda row: (row["dataset"] or "-", *_crru_sort_key(row, crru_scores)),
    )
    for row in ranked_rows:
        dataset = row["dataset"] or "-"
        variant = row["intervention"] or "-"
        best_by_variant.setdefault((dataset, variant), row)

    return sorted(
        best_by_variant.values(),
        key=lambda row: (
            row["dataset"] or "-",
            *_crru_sort_key(row, crru_scores),
            ABLATION_VARIANT_ORDER.get(row["intervention"] or "", len(ABLATION_VARIANT_ORDER)),
        ),
    )


def _print_causal_diagnostics(row: sqlite3.Row) -> None:
    """Print causal diagnostic and score mix metrics under an experiment if they exist."""
    try:
        has_causal = (
            row["test_conformity_contribution_20"] is not None
            or row["test_interest_contribution_20"] is not None
            or row["test_context_contribution_20"] is not None
            or row["test_final_popularity_spearman_20"] is not None
            or row["test_score_mix_interest_mean"] is not None
            or _row_value(row, "test_interest_branch_ndcg_20") is not None
            or _row_value(row, "test_conformity_branch_ndcg_20") is not None
        )
    except (IndexError, KeyError):
        return

    if not has_causal:
        return

    def fmt_val(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "-"

    def fmt_pair(v20: float | None, v40: float | None) -> str:
        return f"{{20: {fmt_val(v20)}, 40: {fmt_val(v40)}}}"

    conformity_contrib = fmt_pair(
        row["test_conformity_contribution_20"], row["test_conformity_contribution_40"]
    )
    interest_contrib = fmt_pair(
        row["test_interest_contribution_20"], row["test_interest_contribution_40"]
    )
    context_contrib = fmt_pair(
        row["test_context_contribution_20"], row["test_context_contribution_40"]
    )

    print(
        f"  Diagnostics: conformity_contrib={conformity_contrib} | "
        f"interest_contrib={interest_contrib} | "
        f"context_contrib={context_contrib}"
    )

    interest_pop = fmt_pair(
        row["test_interest_popularity_spearman_20"], row["test_interest_popularity_spearman_40"]
    )
    conformity_pop = fmt_pair(
        row["test_conformity_popularity_spearman_20"],
        row["test_conformity_popularity_spearman_40"],
    )
    context_pop = fmt_pair(
        row["test_context_popularity_spearman_20"], row["test_context_popularity_spearman_40"]
    )
    final_pop = fmt_pair(
        row["test_final_popularity_spearman_20"], row["test_final_popularity_spearman_40"]
    )

    print(
        f"  Popularity:  Spearman (Interest={interest_pop} | "
        f"Conformity={conformity_pop} | "
        f"Context={context_pop} | "
        f"Final={final_pop})"
    )

    def fmt_mean_std(mean: float | None, std: float | None) -> str:
        if mean is None:
            return "-"
        if std is None:
            return f"{mean:.4f}"
        return f"{mean:.4f}±{std:.4f}"

    score_interest = fmt_mean_std(
        row["test_score_mix_interest_mean"], row["test_score_mix_interest_std"]
    )
    score_conformity = fmt_mean_std(
        row["test_score_mix_conformity_mean"], row["test_score_mix_conformity_std"]
    )
    score_context = fmt_mean_std(
        row["test_score_mix_context_mean"], row["test_score_mix_context_std"]
    )
    cosine_sim = fmt_mean_std(
        row["test_interest_conformity_cosine_mean"], row["test_interest_conformity_cosine_std"]
    )

    print(
        f"  Score Mix:   Interest={score_interest} | "
        f"Conformity={score_conformity} | "
        f"Context={score_context} | "
        f"Cosine={cosine_sim}"
    )

    interest_branch_ndcg = fmt_pair(
        _row_value(row, "test_interest_branch_ndcg_20"),
        _row_value(row, "test_interest_branch_ndcg_40"),
    )
    interest_branch_recall = fmt_pair(
        _row_value(row, "test_interest_branch_recall_20"),
        _row_value(row, "test_interest_branch_recall_40"),
    )
    interest_branch_pop = fmt_pair(
        _row_value(row, "test_interest_branch_average_popularity_20"),
        _row_value(row, "test_interest_branch_average_popularity_40"),
    )
    conformity_branch_ndcg = fmt_pair(
        _row_value(row, "test_conformity_branch_ndcg_20"),
        _row_value(row, "test_conformity_branch_ndcg_40"),
    )
    conformity_branch_recall = fmt_pair(
        _row_value(row, "test_conformity_branch_recall_20"),
        _row_value(row, "test_conformity_branch_recall_40"),
    )
    conformity_branch_pop = fmt_pair(
        _row_value(row, "test_conformity_branch_average_popularity_20"),
        _row_value(row, "test_conformity_branch_average_popularity_40"),
    )
    if interest_branch_ndcg == "{20: -, 40: -}" and conformity_branch_ndcg == "{20: -, 40: -}":
        return

    print(
        f"  Branch Rank: Interest NDCG={interest_branch_ndcg} | "
        f"Recall={interest_branch_recall} | AvgPop={interest_branch_pop} || "
        f"Conformity NDCG={conformity_branch_ndcg} | "
        f"Recall={conformity_branch_recall} | AvgPop={conformity_branch_pop}"
    )


def _row_value(row: sqlite3.Row, key: str) -> float | None:
    """Return a row value when present, else ``None`` for older DB views."""
    try:
        value = row[key]
    except (IndexError, KeyError):
        return None
    return None if value is None else float(value)


def _print_runtime_approximation(row: sqlite3.Row) -> None:
    """Print runtime-probe estimates under an experiment when available."""
    estimated_train_time_s = _row_value(row, "runtime_probe_estimated_train_time_s")
    if estimated_train_time_s is None:
        return

    target_epochs = _row_value(row, "runtime_probe_target_epochs")
    observed_epochs = _row_value(row, "runtime_probe_observed_epochs")
    batches_per_epoch = _row_value(row, "runtime_probe_train_batches_per_epoch")
    throughput = _row_value(row, "runtime_probe_observed_batches_per_second")
    seconds_per_epoch = _row_value(row, "runtime_probe_seconds_per_epoch")
    remaining_time_s = _row_value(row, "runtime_probe_estimated_remaining_train_time_s")

    print(
        "  Approximation: "
        f"full_train={_format_duration(estimated_train_time_s)} | "
        f"remaining={_format_duration(remaining_time_s)} | "
        f"target_epochs={_format_count(target_epochs)} | "
        f"observed_epochs={_format_count(observed_epochs)} | "
        f"batches/epoch={_format_count(batches_per_epoch)} | "
        f"throughput={_format_rate(throughput)} batch/s | "
        f"epoch_time={_format_duration(seconds_per_epoch)}",
    )


def _print_result_row_experiment(
    *,
    profile_name: str | None = None,
    canonical_name: str,
    training_time_s: float | None = None,
    completed_train_epochs: float | None = None,
    avg_epoch_time_s: float | None = None,
    peak_vram_mb: float | None = None,
    avg_gpu_utilization_pct: float | None = None,
    row: sqlite3.Row | None = None,
) -> None:
    """Print the full profile, resource, and experiment labels under a metric row."""
    if profile_name:
        print(f"  Profile:    {profile_name}")
    training_time = f"{training_time_s:.1f}s" if training_time_s is not None else "-"
    epochs = str(int(completed_train_epochs)) if completed_train_epochs is not None else "-"
    if avg_epoch_time_s is None and training_time_s is not None and completed_train_epochs:
        avg_epoch_time_s = training_time_s / completed_train_epochs
    epoch_time = _format_duration(avg_epoch_time_s)
    peak_vram = f"{peak_vram_mb:.0f}MB" if peak_vram_mb is not None else "-"
    gpu_utilization = (
        f"{avg_gpu_utilization_pct:.0f}%" if avg_gpu_utilization_pct is not None else "-"
    )
    print(
        "  Resources:  "
        f"time={training_time} | epochs={epochs} | time/epoch={epoch_time} | "
        f"peak_vram={peak_vram} | gpu_util={gpu_utilization}",
    )
    if row is not None:
        _print_causal_diagnostics(row)
        _print_runtime_approximation(row)
    print(f"  Experiment: {canonical_name}")


def _print_formal_rows(
    rows: list[sqlite3.Row],
    *,
    crru_scores: dict[int, dict[int, float]],
    title: str,
    empty_message: str,
) -> None:
    """Print the formal-run ranking table."""
    print("=" * 80)
    print(title)
    print("=" * 80)
    if not rows:
        print(empty_message)
        print()
        return

    print(
        (
            f"{'Dataset':<14} | {'Preset':<12} | {'ScoreMix':<8} | {'Neighbors':<10} | "
            f"{'NDCG@20':>8} | {'Recall@20':>10} | {'Hit@20':>8} | {'Pers@20':>9} | "
            f"{'AvgPop@20':>10} | {'NDCG@40':>8} | {'Recall@40':>10} | {'Hit@40':>8} | "
            f"{'Pers@40':>9} | {'AvgPop@40':>10} | {'CRRU@20':>8} | {'CRRU@40':>8}"
        ),
    )
    print("-" * 205)
    previous_dataset: str | None = None
    for row in rows:
        dataset = row["dataset"] or "-"
        if previous_dataset is not None and dataset != previous_dataset:
            print()
        previous_dataset = dataset
        row_crru_scores = crru_scores.get(int(row["id"]), {})
        config = _load_config_json(row["config_json"])
        canonical_name = build_canonical_experiment_name(config, row["preset"], None)
        print(
            (
                f"{dataset:<14} | {row['preset'] or '-':<12} | "
                f"{_format_scoremix(config):<8} | "
                f"{_format_neighbors(config):<10} | "
                f"{_format_metric_value(row['test_ndcg_20']):>8} | "
                f"{_format_metric_value(row['test_recall_20']):>10} | "
                f"{_format_metric_value(row['test_hit_ratio_20']):>8} | "
                f"{_format_metric_value(row['test_personalization_20']):>9} | "
                f"{_format_metric_value(row['test_average_popularity_20']):>10} | "
                f"{_format_metric_value(row['test_ndcg_40']):>8} | "
                f"{_format_metric_value(row['test_recall_40']):>10} | "
                f"{_format_metric_value(row['test_hit_ratio_40']):>8} | "
                f"{_format_metric_value(row['test_personalization_40']):>9} | "
                f"{_format_metric_value(row['test_average_popularity_40']):>10} | "
                f"{_format_crru_value(row_crru_scores.get(20)):>8} | "
                f"{_format_crru_value(row_crru_scores.get(40)):>8}"
            ),
        )
        _print_result_row_experiment(
            profile_name=row["profile_name"],
            canonical_name=canonical_name,
            training_time_s=row["training_time_s"],
            completed_train_epochs=row["completed_train_epochs"],
            avg_epoch_time_s=row["avg_epoch_time_s"],
            peak_vram_mb=row["peak_vram_mb"],
            avg_gpu_utilization_pct=row["avg_gpu_utilization_pct"],
            row=row,
        )
    print()


def _print_ablation_rows(
    rows: list[sqlite3.Row],
    *,
    crru_scores: dict[int, dict[int, float]],
) -> None:
    """Print the best full-data ablation rows per dataset and variant."""
    print("=" * 80)
    print(
        "ABLATION FULL-DATA TEST RUNS — currently supported variants, "
        "best run per dataset ranked by CRRU@20 then CRRU@40"
    )
    print("=" * 80)
    if not rows:
        print("No completed ablation full-data runs with test metrics found.")
        print()
        return

    print(
        (
            f"{'Dataset':<14} | {'Variant':<20} | {'ScoreMix':<8} | {'Neighbors':<10} | "
            f"{'NDCG@20':>8} | {'Recall@20':>10} | {'Hit@20':>8} | {'Pers@20':>9} | "
            f"{'AvgPop@20':>10} | {'NDCG@40':>8} | {'Recall@40':>10} | {'Hit@40':>8} | "
            f"{'Pers@40':>9} | {'AvgPop@40':>10} | {'CRRU@20':>8} | {'CRRU@40':>8}"
        ),
    )
    print("-" * 213)
    previous_dataset: str | None = None
    for row in rows:
        dataset = row["dataset"] or "-"
        if previous_dataset is not None and dataset != previous_dataset:
            print()
        previous_dataset = dataset
        row_crru_scores = crru_scores.get(int(row["id"]), {})
        config = _load_config_json(row["config_json"])
        intervention = row["intervention"]
        canonical_name = build_canonical_experiment_name(config, row["preset"], intervention)
        print(
            (
                f"{dataset:<14} | {intervention or '-':<20} | "
                f"{_format_scoremix(config):<8} | "
                f"{_format_neighbors(config):<10} | "
                f"{_format_metric_value(row['test_ndcg_20']):>8} | "
                f"{_format_metric_value(row['test_recall_20']):>10} | "
                f"{_format_metric_value(row['test_hit_ratio_20']):>8} | "
                f"{_format_metric_value(row['test_personalization_20']):>9} | "
                f"{_format_metric_value(row['test_average_popularity_20']):>10} | "
                f"{_format_metric_value(row['test_ndcg_40']):>8} | "
                f"{_format_metric_value(row['test_recall_40']):>10} | "
                f"{_format_metric_value(row['test_hit_ratio_40']):>8} | "
                f"{_format_metric_value(row['test_personalization_40']):>9} | "
                f"{_format_metric_value(row['test_average_popularity_40']):>10} | "
                f"{_format_crru_value(row_crru_scores.get(20)):>8} | "
                f"{_format_crru_value(row_crru_scores.get(40)):>8}"
            ),
        )
        _print_result_row_experiment(
            canonical_name=canonical_name,
            training_time_s=row["training_time_s"],
            completed_train_epochs=row["completed_train_epochs"],
            avg_epoch_time_s=row["avg_epoch_time_s"],
            peak_vram_mb=row["peak_vram_mb"],
            avg_gpu_utilization_pct=row["avg_gpu_utilization_pct"],
            row=row,
        )
    print()


def show_experiment(conn: sqlite3.Connection, exp_id: int) -> None:
    """Show experiment details."""
    row = conn.execute(
        """
        SELECT id, timestamp, dataset, preset, intervention, config_json, seed,
             status, failure_reason, oom_flag, batch_id, profile_name,
             gpu_name, gpu_vram_gb, training_mode, updated_at
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
    print(f"Database:     {THESIS_DB_PATH.resolve()}")
    print(f"Timestamp:    {row['timestamp']}")
    print(f"Dataset:      {row['dataset']}")
    print(f"Preset:       {row['preset'] or '-'}")
    print(f"Intervention: {row['intervention'] or '-'}")
    print(f"Seed:         {row['seed'] or '-'}")
    print(f"Status:       {row['status'] or '-'}")
    print(f"Training:     {row['training_mode'] or '-'}")
    print(f"Batch ID:     {row['batch_id'] or '-'}")
    print(f"Profile:      {row['profile_name'] or '-'}")
    print(f"GPU:          {row['gpu_name'] or '-'}")
    print(
        f"VRAM (GiB):   {row['gpu_vram_gb'] if row['gpu_vram_gb'] is not None else '-'}",
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
    for split in ["train", "val", "test", "approximation"]:
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
                    ""
                    f"  {epoch_str:>5} | {row['metric_name']:<20} | "
                    f"{row['metric_value']:>10.4f} | {row['timestamp']}",
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
        (
            ""
            f"\n{'Stage':<15} | {'Total/Epoch':>11} | {'Avg/Call':>10} | {'Calls':>6} | "
            f"{'Epochs':>6} | {'%':>6} | Peak VRAM | Last Logged"
        ),
    )
    print("-" * 150)
    for row in rows:
        stage_total_ms = float(row["total_ms"])
        pct = (stage_total_ms / total_ms * 100) if total_ms > 0 else 0
        peak_vram_mb = row["peak_vram_mb"]
        peak_str = f"{peak_vram_mb:.0f} MB" if peak_vram_mb else "-"
        print(
            (
                f"{row['stage']:<15} | {row['avg_epoch_ms']:>11.1f} | "
                f"{row['avg_call_ms']:>10.1f} | {row['total_calls']:>6} | "
                f"{row['profiled_epochs']:>6} | {pct:>5.1f}% | {peak_str:<9} | "
                f"{row['last_logged_at']}"
            ),
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
        f"\n{'Rank':>4} | {'Stage':<15} | {'Total (ms)':>12} | {'Calls':>6} | {'% of Total':>10}",
    )
    print("-" * 60)
    for i, row in enumerate(rows, 1):
        stage_total_ms = float(row["total_ms"])
        pct = (stage_total_ms / grand_total * 100) if grand_total > 0 else 0
        print(
            ""
            f"{i:>4} | {row['stage']:<15} | {stage_total_ms:>12.1f} | "
            f"{row['n_calls']:>6} | {pct:>9.1f}%",
        )

    print("-" * 60)
    print(
        ""
        f"\nBottleneck: {rows[0]['stage']} ({float(rows[0]['total_ms']) / grand_total * 100:.1f}% "
        "of total time)",
    )


def _minmax_normalize(
    values: list[float | None],
    *,
    lower_is_better: bool = False,
) -> list[float]:
    """Return clamped dataset-local min-max normalized values."""
    cleaned = [v if v is not None else 0.0 for v in values]
    lo, hi = min(cleaned), max(cleaned)
    scale = (hi - lo) + CRRU_EPSILON
    normalized = [(v - lo) / scale for v in cleaned]
    if lower_is_better:
        normalized = [1.0 - value for value in normalized]
    return [max(CRRU_EPSILON, min(1.0, value)) for value in normalized]


def _compute_efficiency_scores(rows: list[sqlite3.Row]) -> list[float]:
    """Compute the shared CRRU efficiency utility for one dataset."""
    vram_n = _minmax_normalize(
        [math.log1p(r["peak_vram_mb"]) if r["peak_vram_mb"] else None for r in rows],
        lower_is_better=True,
    )
    epoch_times_s = [_crru_epoch_time_s(r) for r in rows]
    time_n = _minmax_normalize(
        [math.log1p(epoch_time_s) if epoch_time_s else None for epoch_time_s in epoch_times_s],
        lower_is_better=True,
    )
    return [(vram**0.50) * (time**0.50) for vram, time in zip(vram_n, time_n, strict=True)]


def _crru_epoch_time_s(row: sqlite3.Row) -> float | None:
    """Return the per-epoch runtime used by CRRU's efficiency term."""
    avg_epoch_time_s = _row_value(row, "avg_epoch_time_s")
    if avg_epoch_time_s is not None and avg_epoch_time_s > 0:
        return avg_epoch_time_s

    runtime_probe_seconds_per_epoch = _row_value(row, "runtime_probe_seconds_per_epoch")
    if runtime_probe_seconds_per_epoch is not None and runtime_probe_seconds_per_epoch > 0:
        return runtime_probe_seconds_per_epoch

    training_time_s = _row_value(row, "training_time_s")
    completed_train_epochs = _row_value(row, "completed_train_epochs")
    if (
        training_time_s is not None
        and training_time_s > 0
        and completed_train_epochs is not None
        and completed_train_epochs > 0
    ):
        return training_time_s / completed_train_epochs

    return training_time_s if training_time_s is not None and training_time_s > 0 else None


def _compute_crru_for_k(
    rows: list[sqlite3.Row],
    *,
    k: int,
    efficiency_scores: list[float],
) -> list[float]:
    """Compute CRRU@K scores for one dataset."""
    ndcg_n = _minmax_normalize([r[f"test_ndcg_{k}"] for r in rows])
    recall_n = _minmax_normalize([r[f"test_recall_{k}"] for r in rows])
    hit_n = _minmax_normalize([r[f"test_hit_ratio_{k}"] for r in rows])
    pers_n = _minmax_normalize([r[f"test_personalization_{k}"] for r in rows])
    avg_pop_n = _minmax_normalize(
        [r[f"test_average_popularity_{k}"] for r in rows],
        lower_is_better=True,
    )

    scores: list[float] = []
    for ndcg, recall, hit, pers, avg_pop, efficiency in zip(
        ndcg_n,
        recall_n,
        hit_n,
        pers_n,
        avg_pop_n,
        efficiency_scores,
        strict=True,
    ):
        accuracy = (ndcg**0.50) * (recall**0.35) * (hit**0.15)
        bias = (pers**0.40) * (avg_pop**0.60)
        scores.append((accuracy**0.55) * (bias**0.30) * (efficiency**0.15))
    return scores


def _compute_dataset_crru_scores(rows: list[sqlite3.Row]) -> dict[int, dict[int, float]]:
    """Return dataset-local CRRU@20 and CRRU@40 scores keyed by experiment ID."""
    scores_by_id: dict[int, dict[int, float]] = {}
    rows_by_dataset: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        rows_by_dataset[row["dataset"] or "-"].append(row)

    for dataset_rows in rows_by_dataset.values():
        efficiency_scores = _compute_efficiency_scores(dataset_rows)
        crru_20 = _compute_crru_for_k(dataset_rows, k=20, efficiency_scores=efficiency_scores)
        crru_40 = _compute_crru_for_k(dataset_rows, k=40, efficiency_scores=efficiency_scores)
        for row, score_20, score_40 in zip(
            dataset_rows,
            crru_20,
            crru_40,
            strict=True,
        ):
            scores_by_id[int(row["id"])] = {20: score_20, 40: score_40}

    return scores_by_id


def _print_crru_summary() -> None:
    """Print the CRRU framing used by the default thesis summary."""
    print("CRRU@K — Composite Resource-aware Recommendation Utility at K")
    print("  Accuracy@K = NDCG@K^0.50 * Recall@K^0.35 * Hit@K^0.15")
    print("  Bias@K     = Pers@K^0.40 * (1-AvgPop@K_n)^0.60")
    print("  Efficiency = (1-log(1+VRAM)_n)^0.50 * (1-log(1+time/epoch)_n)^0.50")
    print("  CRRU@K     = Accuracy@K^0.55 * Bias@K^0.30 * Efficiency^0.15")
    print(f"  Normalization: dataset-local section-row min-max with epsilon={CRRU_EPSILON:g}")
    print("  Note: CRRU is not a causal-effect estimator.")
    print()


def list_top_completed(conn: sqlite3.Connection, *, n: int = 20) -> None:
    """Print the default thesis summary for formal and ablation test runs."""
    print("=" * 80)
    print("THESIS TEST RESULTS — formal and ablation runs only (full-data only)")
    print("=" * 80)
    print(f"Database: {THESIS_DB_PATH.resolve()}")
    print()

    rows = [row for row in _query_report_rows(conn) if _is_default_report_row(row)]
    final_formal_rows = [row for row in rows if _is_final_formal_row(row)]
    supporting_formal_rows = [row for row in rows if _is_supporting_formal_row(row)]
    ablation_rows = [row for row in rows if _is_supported_ablation_row(row)]
    if not (final_formal_rows or supporting_formal_rows or ablation_rows):
        print("No completed formal or ablation full-data runs with test metrics found.")
        print("Use --view all to inspect every logged experiment row.")
        return

    crru_scores: dict[int, dict[int, float]] = {}
    for section_rows in (final_formal_rows, supporting_formal_rows, ablation_rows):
        crru_scores.update(_compute_dataset_crru_scores(section_rows))
    _print_crru_summary()
    _print_formal_rows(
        _select_top_formal_rows(final_formal_rows, n=n, crru_scores=crru_scores),
        crru_scores=crru_scores,
        title=("FINAL FORMAL FULL-DATA TEST RUNS — thesis profiles ranked by CRRU@20 then CRRU@40"),
        empty_message="No completed final formal full-data runs with test metrics found.",
    )
    _print_formal_rows(
        _select_top_formal_rows(
            supporting_formal_rows,
            n=n,
            crru_scores=crru_scores,
            supporting=True,
        ),
        crru_scores=crru_scores,
        title=(
            "SUPPORTING FORMAL FULL-DATA RUNS — historical, diagnostic, and "
            "preprocessing-sweep rows"
        ),
        empty_message="No supporting formal full-data runs with test metrics found.",
    )
    _print_ablation_rows(
        _select_best_ablation_rows(ablation_rows, crru_scores=crru_scores),
        crru_scores=crru_scores,
    )


def _render_default_summary(conn: sqlite3.Connection, *, n: int = 20) -> str:
    """Render the default thesis summary into a text buffer."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        list_top_completed(conn, n=n)
    return buffer.getvalue().rstrip()


def _write_default_summary_markdown(report_text: str) -> None:
    """Persist the default thesis summary to the repository results folder."""
    QUERY_RESULTS_MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    markdown_report = (
        "# Query Results\n\n"
        "Generated by `uv run scripts/query_results.py`.\n\n"
        "```text\n"
        f"{report_text}\n"
        "```\n"
    )
    QUERY_RESULTS_MARKDOWN_PATH.write_text(markdown_report, encoding="utf-8")


def main() -> int:
    """Parse arguments and print the requested experiment view."""
    args = build_query_results_parser().parse_args()

    conn = connect()
    try:
        if args.view is None:
            report_text = _render_default_summary(conn)
            print(report_text)
            _write_default_summary_markdown(report_text)
            print(f"Wrote default results summary to {QUERY_RESULTS_MARKDOWN_PATH.resolve()}")
        else:
            list_experiments(
                conn,
                view_name=args.view or "all",
            )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
