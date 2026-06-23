#!/usr/bin/env python
"""Query experiment results from SQLite database.

Usage:
    python scripts/query_results.py                    # Write default test-set summary markdown
    python scripts/query_results.py --view all         # Show all experiments
    python scripts/query_results.py --view completed   # Show only completed runs
    python scripts/query_results.py --view attention   # Show failed, OOM, running, or unknown runs
    python scripts/query_results.py --view errors      # Show only failed and OOM runs
    python scripts/query_results.py --view comparison  # Show summary comparison tables
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from experiments.ablation_configs import ABLATION_VARIANTS
from src.reporting.feature_analysis import write_query_feature_analysis_reports
from src.utils.cli_parsers import build_query_results_parser
from src.utils.crru import (
    CRRU_REPORT_FORMULA_LINES,
    compute_crru_efficiency_scores,
    compute_crru_scores_for_k,
)
from src.utils.experiment_logger import RUNTIME_PROBE_METRIC_NAMES, ExperimentLogger
from src.utils.experiment_naming import (
    build_canonical_experiment_name,
    format_num_neighbors_payload,
)
from src.utils.method_naming import (
    display_method_label,
    public_method_identifier,
    public_preset_name,
)
from src.utils.project_paths import RESULTS_DIR, THESIS_DB_PATH

DB_PATH = Path(os.environ.get("THESIS_DB_PATH_OVERRIDE", str(THESIS_DB_PATH)))
QUERY_RESULTS_MARKDOWN_PATH = RESULTS_DIR / "query_results.md"
OPTUNA_OPTIMIZATION_MARKDOWN_PATH = RESULTS_DIR / "optuna_optimization.md"
VIEW_TABLES = ExperimentLogger.VIEW_TABLES
FORMAL_BATCH_PREFIX = "formal-"
ABLATION_BATCH_PREFIX = "ablation-"
FINAL_FORMAL_PROFILE_NAMES = frozenset(
    {
        "edgrec-compact-search-prior",
        "edgrec-lite-default",
        "amazonbook-edgrec-compact-candidate",
        "amazonbook-edgrec-deep-features-candidate",
        "core-edgrec-mainline",
        "core-paper-architecture-comparison",
        "paper-lightgcn-small-baselines",
        "paper-lightgcn-baselines",
    }
)
FINAL_FORMAL_PROFILE_PREFIXES = ("edgrec-global-top-",)
RUNTIME_PROBE_COLUMNS = RUNTIME_PROBE_METRIC_NAMES
PAPER_BASELINE_PRESETS = frozenset({"lightgcn_paper", "dice_paper"})


def _display_preset(preset: object | None) -> str:
    """Return the public report label for a stored preset token."""
    return display_method_label(preset)


def _display_profile(profile_name: object | None) -> str:
    """Return the public report label for a stored formal profile token."""
    if profile_name is None:
        return "-"
    return public_method_identifier(str(profile_name)) or str(profile_name)


def connect() -> sqlite3.Connection:
    """Connect to experiment database."""
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH.resolve()}")
        print(
            "Run a real experiment first to create the persistent database in results/.",
        )
        print("Quick-validate is non-persistent and does not write smoke rows.")
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
        profile_label = _display_profile(row["profile_name"])
        status_label = row["status"] or ("oom" if row["oom_flag"] else "unknown")
        print(
            (
                f"{row['id']:>4} | {status_label:<10} | {row['dataset'] or '-':<14} | "
                f"{_display_preset(row['preset']):<12} | {profile_label:<{profile_width}} | "
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
    if 0 <= value_s < 1:
        return f"{value_s:.3f}s"
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
                   (
                       SELECT metric_value
                       FROM metrics AS train_edge_metric
                       WHERE train_edge_metric.experiment_id = s.id
                         AND train_edge_metric.split = 'train'
                         AND train_edge_metric.metric_name = 'train_edge_count'
                       ORDER BY train_edge_metric.epoch DESC
                       LIMIT 1
                   ) AS train_edge_count,
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
                   {summary_column("test_conformity_branch_contribution_ratio_20")},
                   {summary_column("test_conformity_branch_contribution_ratio_40")},
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
                   {summary_column("test_interest_branch_contribution_ratio_20")},
                   {summary_column("test_interest_branch_contribution_ratio_40")},
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
    public_profile_name = (
        public_method_identifier(profile_name) if isinstance(profile_name, str) else None
    )
    return (
        _is_reportable_formal_row(row)
        and isinstance(profile_name, str)
        and (
            public_profile_name in FINAL_FORMAL_PROFILE_NAMES
            or (
                isinstance(public_profile_name, str)
                and public_profile_name.startswith(FINAL_FORMAL_PROFILE_PREFIXES)
            )
        )
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


def _evidence_label(row: sqlite3.Row) -> str:
    """Return the report-facing provenance label for one test-set row."""
    if _is_final_formal_row(row):
        return "thesis profile"
    if _is_supporting_formal_row(row):
        return "supporting"
    if _is_supported_ablation_row(row):
        return "ablation"
    return "other"


def _leaderboard_label(row: sqlite3.Row) -> str:
    """Return the preset or ablation variant label for the unified leaderboard."""
    if _is_supported_ablation_row(row):
        return row["intervention"] or "-"
    return _display_preset(row["preset"])


def _leaderboard_intervention(row: sqlite3.Row) -> str | None:
    """Return the intervention label only for ablation rows."""
    return row["intervention"] if _is_supported_ablation_row(row) else None


def _comparison_key(row: sqlite3.Row) -> tuple[str, str, str, str, str, str]:
    """Return a display-level key used to suppress duplicate support rows."""
    scoremix, neighbors = _scoremix_and_neighbors(row)
    return (
        row["dataset"] or "-",
        _evidence_label(row),
        _leaderboard_label(row),
        _display_profile(row["profile_name"]),
        scoremix,
        neighbors,
    )


def _select_top_test_rows(
    rows: list[sqlite3.Row],
    *,
    n: int,
    crru_scores: dict[int, dict[int, float]],
) -> list[sqlite3.Row]:
    """Return the top completed full-data test rows per dataset ranked by CRRU."""
    ranked_rows = sorted(
        (row for row in rows if _is_default_report_row(row)),
        key=lambda row: (row["dataset"] or "-", *_crru_sort_key(row, crru_scores)),
    )

    top_rows: list[sqlite3.Row] = []
    dataset_counts: dict[str, int] = {}
    seen_keys: set[tuple[str, str, str, str, str, str]] = set()
    for row in ranked_rows:
        key = _comparison_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        dataset = row["dataset"] or "-"
        if dataset_counts.get(dataset, 0) >= n:
            continue
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        top_rows.append(row)
    return top_rows


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
    interest_ratio = fmt_pair(
        _row_value(row, "test_interest_branch_contribution_ratio_20"),
        _row_value(row, "test_interest_branch_contribution_ratio_40"),
    )
    conformity_ratio = fmt_pair(
        _row_value(row, "test_conformity_branch_contribution_ratio_20"),
        _row_value(row, "test_conformity_branch_contribution_ratio_40"),
    )

    print(
        "  Diagnostics: exploratory, not causal proof | "
        f"conformity_contrib={conformity_contrib} | "
        f"interest_contrib={interest_contrib} | "
        f"context_contrib={context_contrib} | "
        f"branch_ratio(interest={interest_ratio}, conformity={conformity_ratio})"
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
    warnings = _branch_collapse_warning_label(row)
    warning_suffix = f" | Warnings={warnings}" if warnings != "-" else ""

    print(
        f"  Score Mix:   Interest={score_interest} | "
        f"Conformity={score_conformity} | "
        f"Context={score_context} | "
        f"Cosine={cosine_sim}{warning_suffix}"
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


def _branch_collapse_warning_label(row: sqlite3.Row) -> str:
    """Return compact branch-collapse warning labels for report rows."""
    warnings: list[str] = []
    conformity_mix_mean = _row_value(row, "test_score_mix_conformity_mean")
    interest_mix_mean = _row_value(row, "test_score_mix_interest_mean")
    branch_cosine_mean = _row_value(row, "test_interest_conformity_cosine_mean")
    if conformity_mix_mean is not None and conformity_mix_mean > 0.8:
        warnings.append("conformity_mix>0.8")
    if interest_mix_mean is not None and interest_mix_mean < 0.1:
        warnings.append("interest_mix<0.1")
    if branch_cosine_mean is not None and abs(branch_cosine_mean) > 0.9:
        warnings.append("|branch_cosine|>0.9")
    return ", ".join(warnings) if warnings else "-"


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


def _run_detail_cells(
    *,
    dataset_rank: int,
    row: sqlite3.Row,
    intervention_for_row: Callable[[sqlite3.Row], str | None],
    label_for_row: Callable[[sqlite3.Row], str],
) -> tuple[str, str, str, str, str, str, str, str, str, str, str]:
    """Collect one Markdown-safe run details table row."""
    config = _load_config_json(row["config_json"])
    intervention = intervention_for_row(row)
    preset_name = public_preset_name(row["preset"])
    canonical_name = build_canonical_experiment_name(config, preset_name, intervention)
    dataset = row["dataset"] or "-"
    label = label_for_row(row)
    profile_name = _display_profile(row["profile_name"])

    capture = StringIO()
    with redirect_stdout(capture):
        _print_result_row_experiment(
            profile_name=profile_name if profile_name != "-" else None,
            canonical_name=canonical_name,
            training_time_s=row["training_time_s"],
            completed_train_epochs=row["completed_train_epochs"],
            avg_epoch_time_s=row["avg_epoch_time_s"],
            peak_vram_mb=row["peak_vram_mb"],
            avg_gpu_utilization_pct=row["avg_gpu_utilization_pct"],
            row=row,
        )

    detail_lines = {
        line.strip().split(":", 1)[0]: line.strip().split(":", 1)[1].strip()
        for line in capture.getvalue().splitlines()
        if ":" in line
    }

    return (
        str(dataset_rank),
        dataset,
        label,
        detail_lines.get("Profile", profile_name),
        f"Resources:  {detail_lines.get('Resources', '-')}"
        if detail_lines.get("Resources", "-") != "-"
        else "Resources:  -",
        f"Diagnostics: {detail_lines.get('Diagnostics', '-')}"
        if detail_lines.get("Diagnostics", "-") != "-"
        else "Diagnostics: -",
        f"Popularity:  {detail_lines.get('Popularity', '-')}"
        if detail_lines.get("Popularity", "-") != "-"
        else "Popularity:  -",
        f"Score Mix: {detail_lines.get('Score Mix', '-')}"
        if detail_lines.get("Score Mix", "-") != "-"
        else "Score Mix: -",
        f"Branch Rank: {detail_lines.get('Branch Rank', '-')}"
        if detail_lines.get("Branch Rank", "-") != "-"
        else "Branch Rank: -",
        f"Approximation: {detail_lines.get('Approximation', '-')}"
        if detail_lines.get("Approximation", "-") != "-"
        else "Approximation: -",
        _wrap_identifier_for_markdown(detail_lines.get("Experiment", canonical_name)),
    )


def _format_markdown_cell(value: object) -> str:
    """Return one Markdown-safe table cell."""
    text = "-" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _wrap_identifier_for_markdown(value: str, *, max_line_length: int = 54) -> str:
    """Insert line breaks in long underscore-separated identifiers."""
    if len(value) <= max_line_length:
        return value

    lines: list[str] = []
    current = ""
    for part in value.split("_"):
        candidate = part if not current else f"{current}_{part}"
        if len(candidate) > max_line_length and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _markdown_alignment(align: str) -> str:
    """Return a Markdown table alignment marker."""
    if align == ">":
        return "---:"
    if align == "^":
        return ":---:"
    return "---"


def _print_metric_table(
    title: str,
    columns: Sequence[tuple[str, int, str]],
    rows: Sequence[Sequence[object]],
) -> None:
    """Print one compact Markdown result table."""
    print(f"### {title}")
    headers = [_format_markdown_cell(name) for name, _width, _align in columns]
    alignments = [_markdown_alignment(align) for _name, _width, align in columns]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(alignments) + " |")
    for row in rows:
        cells = [_format_markdown_cell(value) for value in row]
        print("| " + " | ".join(cells) + " |")
    print()


def _scoremix_and_neighbors(row: sqlite3.Row) -> tuple[str, str]:
    """Return report-facing score-mix and fan-out labels for one row."""
    config = _load_config_json(row["config_json"])
    return _format_scoremix(config), _format_neighbors(config)


def _peak_vram_label(row: sqlite3.Row) -> str:
    """Return peak VRAM as a compact report label."""
    value = _row_value(row, "peak_vram_mb")
    return _format_vram_value(value)


def _format_vram_value(value: float | None) -> str:
    """Return peak VRAM as a compact report value."""
    return "-" if value is None else f"{value:.0f}MB"


def _epoch_count_label(row: sqlite3.Row) -> str:
    """Return completed epoch count as a compact report label."""
    value = _row_value(row, "completed_train_epochs")
    return "-" if value is None else str(int(value))


def _dataset_rank_by_id(rows: Sequence[sqlite3.Row]) -> dict[int, int]:
    """Return dataset-local display ranks keyed by experiment id."""
    ranks_by_id: dict[int, int] = {}
    dataset_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        dataset = row["dataset"] or "-"
        dataset_counts[dataset] += 1
        ranks_by_id[int(row["id"])] = dataset_counts[dataset]
    return ranks_by_id


def _best_values_by_dataset(
    rows: Sequence[sqlite3.Row],
    value_for_row: Callable[[sqlite3.Row], float | None],
    *,
    direction: str,
) -> dict[str, float]:
    """Return the best raw value for each dataset under a max/min direction."""
    values_by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = value_for_row(row)
        if value is not None:
            values_by_dataset[row["dataset"] or "-"].append(value)

    if direction == "min":
        return {dataset: min(values) for dataset, values in values_by_dataset.items()}
    return {dataset: max(values) for dataset, values in values_by_dataset.items()}


def _format_best_cell(
    row: sqlite3.Row,
    value: float | None,
    formatted: str,
    best_by_dataset: dict[str, float],
) -> str:
    """Return a formatted table value, bolded when it is dataset-local best."""
    if value is None or formatted == "-":
        return formatted
    best = best_by_dataset.get(row["dataset"] or "-")
    if best is not None and abs(value - best) <= 1e-12:
        return f"**{formatted}**"
    return formatted


def _print_ranked_result_table(
    rows: list[sqlite3.Row],
    *,
    crru_scores: dict[int, dict[int, float]],
    label_header: str,
    label_width: int,
    label_for_row: Callable[[sqlite3.Row], str],
    evidence_for_row: Callable[[sqlite3.Row], str] | None = None,
) -> None:
    """Print result rows as one ranked metric/resource table."""
    columns: tuple[tuple[str, int, str], ...] = (
        ("DatasetRank", 11, ">"),
        ("ExpID", 6, ">"),
        ("Dataset", 14, "<"),
    )
    if evidence_for_row is not None:
        columns += (("Evidence", 14, "<"),)
    columns += (
        (label_header, label_width, "<"),
        ("Profile", 32, "<"),
        ("ScoreMix", 8, "<"),
        ("Neighbors", 10, "<"),
        ("CRRU@20", 8, ">"),
        ("CRRU@40", 8, ">"),
        ("NDCG@20", 8, ">"),
        ("Recall@20", 10, ">"),
        ("Hit@20", 8, ">"),
        ("Pers@20", 9, ">"),
        ("AvgPop@20", 10, ">"),
        ("NDCG@40", 8, ">"),
        ("Recall@40", 10, ">"),
        ("Hit@40", 8, ">"),
        ("Pers@40", 9, ">"),
        ("AvgPop@40", 10, ">"),
        ("Epochs", 6, ">"),
        ("Time/Ep", 10, ">"),
        ("PeakVRAM", 10, ">"),
    )

    table_rows: list[tuple[object, ...]] = []
    ranks_by_id = _dataset_rank_by_id(rows)
    metric_specs: tuple[
        tuple[str, Callable[[sqlite3.Row], float | None], Callable[[float | None], str], str],
        ...,
    ] = (
        (
            "CRRU@20",
            lambda row: crru_scores.get(int(row["id"]), {}).get(20),
            _format_crru_value,
            "max",
        ),
        (
            "CRRU@40",
            lambda row: crru_scores.get(int(row["id"]), {}).get(40),
            _format_crru_value,
            "max",
        ),
        ("NDCG@20", lambda row: _row_value(row, "test_ndcg_20"), _format_metric_value, "max"),
        (
            "Recall@20",
            lambda row: _row_value(row, "test_recall_20"),
            _format_metric_value,
            "max",
        ),
        (
            "Hit@20",
            lambda row: _row_value(row, "test_hit_ratio_20"),
            _format_metric_value,
            "max",
        ),
        (
            "Pers@20",
            lambda row: _row_value(row, "test_personalization_20"),
            _format_metric_value,
            "max",
        ),
        (
            "AvgPop@20",
            lambda row: _row_value(row, "test_average_popularity_20"),
            _format_metric_value,
            "min",
        ),
        ("NDCG@40", lambda row: _row_value(row, "test_ndcg_40"), _format_metric_value, "max"),
        (
            "Recall@40",
            lambda row: _row_value(row, "test_recall_40"),
            _format_metric_value,
            "max",
        ),
        (
            "Hit@40",
            lambda row: _row_value(row, "test_hit_ratio_40"),
            _format_metric_value,
            "max",
        ),
        (
            "Pers@40",
            lambda row: _row_value(row, "test_personalization_40"),
            _format_metric_value,
            "max",
        ),
        (
            "AvgPop@40",
            lambda row: _row_value(row, "test_average_popularity_40"),
            _format_metric_value,
            "min",
        ),
        ("Time/Ep", _crru_epoch_time_s, _format_duration, "min"),
        ("PeakVRAM", lambda row: _row_value(row, "peak_vram_mb"), _format_vram_value, "min"),
    )
    best_by_metric = {
        metric_name: _best_values_by_dataset(rows, value_getter, direction=direction)
        for metric_name, value_getter, _formatter, direction in metric_specs
    }

    for row in rows:
        dataset = row["dataset"] or "-"
        label = label_for_row(row)
        scoremix, neighbors = _scoremix_and_neighbors(row)
        base_cells: list[object] = [
            ranks_by_id[int(row["id"])],
            int(row["id"]),
            dataset,
        ]
        if evidence_for_row is not None:
            base_cells.append(evidence_for_row(row))
        metric_cells = []
        for metric_name, value_getter, formatter, _direction in metric_specs:
            raw_value = value_getter(row)
            metric_cells.append(
                _format_best_cell(
                    row,
                    raw_value,
                    formatter(raw_value),
                    best_by_metric[metric_name],
                ),
            )
        table_rows.append(
            (
                *base_cells,
                label,
                _display_profile(row["profile_name"]),
                scoremix,
                neighbors,
                *metric_cells[:12],
                _epoch_count_label(row),
                *metric_cells[12:],
            ),
        )

    _print_metric_table(
        "Ranked test metrics, CRRU, and resource use",
        columns,
        table_rows,
    )
    print(
        "Bold values mark the best shown value within each dataset. Lower is better for "
        "AvgPop, Time/Ep, and PeakVRAM; higher is better for CRRU, accuracy, and "
        "personalization. Epoch count is not bolded because fewer epochs is not always "
        "a better model result."
    )
    print()


def _select_paper_runtime_probe_rows(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    """Return the latest formal runtime-probe row for each paper baseline/dataset."""
    latest_by_key: dict[tuple[str, str], sqlite3.Row] = {}
    for row in rows:
        dataset = row["dataset"] or "-"
        preset = row["preset"] or "-"
        if preset not in PAPER_BASELINE_PRESETS:
            continue
        if not (_is_formal_row(row) and _is_runtime_probe_row(row)):
            continue
        key = (dataset, preset)
        previous = latest_by_key.get(key)
        if previous is None or int(row["id"]) > int(previous["id"]):
            latest_by_key[key] = row
    return sorted(latest_by_key.values(), key=lambda row: (row["dataset"] or "-", row["preset"]))


def _dataset_summary(rows: Sequence[sqlite3.Row]) -> str:
    """Return a compact comma-separated dataset list."""
    datasets = sorted({row["dataset"] or "-" for row in rows})
    return ", ".join(datasets) if datasets else "-"


def _paper_baseline_status_note(preset: str) -> str:
    """Return thesis-facing interpretation for one paper baseline preset."""
    if preset == "lightgcn_paper":
        return (
            "Paper-faithful LightGCN adapter; full test rows are ranking baselines, "
            "runtime probes are resource-only evidence."
        )
    if preset == "dice_paper":
        return (
            "Paper-faithful DICE adapter; current evidence is one-epoch runtime "
            "probes, so use time/epoch and VRAM only until a full run is scheduled."
        )
    return "-"


def _runtime_probe_note(row: sqlite3.Row) -> str:
    """Return a concise note for one runtime-probe row."""
    preset = row["preset"] or "-"
    if preset == "dice_paper":
        return "One-epoch DICE probe; accuracy is diagnostic only, not a final comparison."
    if preset == "lightgcn_paper":
        return "One-epoch LightGCN probe for large full-graph feasibility."
    return "One-epoch runtime probe; not part of final ranking."


def _print_paper_baseline_notes(
    *,
    reportable_formal_rows: list[sqlite3.Row],
    runtime_probe_rows: list[sqlite3.Row],
) -> None:
    """Print paper-baseline evidence status and runtime probes."""
    baseline_rows = [
        row for row in reportable_formal_rows if (row["preset"] or "-") in PAPER_BASELINE_PRESETS
    ]
    if not (baseline_rows or runtime_probe_rows):
        return

    print("## Paper Baseline Notes")
    print(
        "Paper baselines are kept visible here even when a full run is impractical. "
        "Runtime-probe rows are excluded from the test-set leaderboard."
    )
    print()

    status_rows: list[tuple[object, ...]] = []
    for preset in ("lightgcn_paper", "dice_paper"):
        full_rows = [row for row in baseline_rows if row["preset"] == preset]
        probe_rows = [row for row in runtime_probe_rows if row["preset"] == preset]
        if not (full_rows or probe_rows):
            continue
        status_rows.append(
            (
                preset,
                _dataset_summary(full_rows),
                _dataset_summary(probe_rows),
                _paper_baseline_status_note(preset),
            ),
        )
    _print_metric_table(
        "Baseline evidence status",
        (
            ("Preset", 14, "<"),
            ("Full test datasets", 24, "<"),
            ("Runtime-probe datasets", 24, "<"),
            ("Interpretation", 70, "<"),
        ),
        status_rows,
    )

    if not runtime_probe_rows:
        return

    probe_table_rows: list[tuple[object, ...]] = []
    for run_index, row in enumerate(runtime_probe_rows, start=1):
        probe_table_rows.append(
            (
                f"P{run_index}",
                row["dataset"] or "-",
                row["preset"] or "-",
                row["profile_name"] or "-",
                _epoch_count_label(row),
                _format_metric_value(row["test_ndcg_20"]),
                _format_metric_value(row["test_recall_20"]),
                _format_duration(_crru_epoch_time_s(row)),
                _peak_vram_label(row),
                _format_duration(_row_value(row, "runtime_probe_estimated_train_time_s")),
                _runtime_probe_note(row),
            ),
        )
    _print_metric_table(
        "Paper baseline runtime probes",
        (
            ("Probe", 5, "<"),
            ("Dataset", 14, "<"),
            ("Preset", 14, "<"),
            ("Profile", 32, "<"),
            ("Epochs", 6, ">"),
            ("NDCG@20", 8, ">"),
            ("Recall@20", 10, ">"),
            ("Time/Ep", 10, ">"),
            ("PeakVRAM", 10, ">"),
            ("Est. target train", 17, ">"),
            ("Note", 72, "<"),
        ),
        probe_table_rows,
    )


def _config_float(config: dict[str, object], key: str) -> float | None:
    """Return a numeric config value when it is scalar."""
    value = config.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _format_batch_label(config: dict[str, object]) -> str:
    """Return selected batch size as a compact label."""
    value = config.get("batch_size")
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.0f}"
    if isinstance(value, list):
        return ", ".join(f"{int(item):,}" for item in value if isinstance(item, int | float))
    return "-"


def _format_integer_label(value: float | None) -> str:
    """Return a whole-number metric as a compact label."""
    return "-" if value is None else f"{value:,.0f}"


def _timing_source_label(row: sqlite3.Row) -> str:
    """Return the source used for the reported time/epoch value."""
    if _row_value(row, "avg_epoch_time_s") is not None:
        return "avg epoch_time_s"
    if _row_value(row, "training_time_s") is not None and _row_value(
        row,
        "completed_train_epochs",
    ):
        return "train time / epochs"
    return "-"


def _print_resource_context_notes(rows: Sequence[sqlite3.Row]) -> None:
    """Print context for resource values that are easy to misread."""
    note_rows: list[tuple[object, ...]] = []
    for row in rows:
        config = _load_config_json(row["config_json"])
        item_policy = str(config.get("item_universe_policy") or "-")
        edge_keep = _config_float(config, "train_edge_keep_prob")
        batch_size = _config_float(config, "batch_size")
        epoch_time_s = _crru_epoch_time_s(row)
        train_edge_count = _row_value(row, "train_edge_count")

        reasons: list[str] = []
        if epoch_time_s is not None and epoch_time_s < 1.0:
            reasons.append("sub-second Time/Ep")
        if item_policy == "random_exposure_items_only":
            reasons.append("random-exposure item universe")
        if batch_size is not None and batch_size >= 1_000_000:
            reasons.append("million-scale batch")
        if edge_keep is not None and edge_keep < 1.0:
            reasons.append(f"train_edge_keep_prob={edge_keep:g}")
        if train_edge_count is not None and train_edge_count < 10_000:
            reasons.append("compact train graph")
        if not reasons:
            continue

        note_rows.append(
            (
                row["dataset"] or "-",
                int(row["id"]),
                _evidence_label(row),
                item_policy,
                _format_batch_label(config),
                _format_integer_label(train_edge_count),
                _format_duration(epoch_time_s),
                _timing_source_label(row),
                "; ".join(reasons),
            ),
        )

    if not note_rows:
        return

    _print_metric_table(
        "Resource comparability notes",
        (
            ("Dataset", 14, "<"),
            ("ExpID", 6, ">"),
            ("Evidence", 14, "<"),
            ("ItemScope", 28, "<"),
            ("Batch", 12, ">"),
            ("TrainEdges", 11, ">"),
            ("Time/Ep", 10, ">"),
            ("Timing source", 18, "<"),
            ("Why flagged", 68, "<"),
        ),
        note_rows,
    )


def _print_run_details(
    rows: list[sqlite3.Row],
    *,
    label_for_row: Callable[[sqlite3.Row], str],
    intervention_for_row: Callable[[sqlite3.Row], str | None],
) -> None:
    """Print per-row profile, diagnostic, and canonical-name details."""
    detail_rows: list[tuple[object, ...]] = []
    ranks_by_id = _dataset_rank_by_id(rows)
    for row in rows:
        detail_rows.append(
            _run_detail_cells(
                dataset_rank=ranks_by_id[int(row["id"])],
                row=row,
                intervention_for_row=intervention_for_row,
                label_for_row=label_for_row,
            ),
        )

    print("### Diagnostic Details")
    headers = (
        "DatasetRank",
        "Dataset",
        "Preset/Variant",
        "Profile",
        "Resources",
        "Diagnostics",
        "Popularity",
        "Score Mix",
        "Branch Rank",
        "Approximation",
        "Experiment",
    )
    aligns = (
        "---:",
        "---",
        "---",
        "---",
        "---",
        "---",
        "---",
        "---",
        "---",
        "---",
        "---",
    )
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(aligns) + " |")
    for detail_row in detail_rows:
        print("| " + " | ".join(_format_markdown_cell(value) for value in detail_row) + " |")
    print()


def _print_test_leaderboard(
    rows: list[sqlite3.Row],
    *,
    crru_scores: dict[int, dict[int, float]],
) -> None:
    """Print the unified full-data test-set leaderboard."""
    print("## TEST-SET RESULT LEADERBOARD - completed full-data rows")
    print(
        "Rows here all have test-set metrics and are ranked within each dataset by "
        "CRRU@20 then CRRU@40. The `Evidence` column marks whether a row is a named "
        "thesis profile, a supporting historical/diagnostic row, or a public ablation "
        "variant. Runtime probes remain separate because their accuracy is diagnostic."
    )
    print()
    if not rows:
        print("No completed full-data test rows with metrics found.")
        print()
        return

    _print_ranked_result_table(
        rows,
        crru_scores=crru_scores,
        label_header="Preset/Variant",
        label_width=20,
        label_for_row=_leaderboard_label,
        evidence_for_row=_evidence_label,
    )
    _print_resource_context_notes(rows)
    _print_run_details(
        rows,
        label_for_row=_leaderboard_label,
        intervention_for_row=_leaderboard_intervention,
    )


def _print_optuna_report_pointer() -> None:
    """Print where the dedicated Optuna report lives."""
    print("## OPTUNA EDGRec SEARCH REPORT")
    print(
        "Optuna search trials are reported from the Optuna RDB storage, not mirrored "
        "through the thesis experiment database."
    )
    print(f"Report: {OPTUNA_OPTIMIZATION_MARKDOWN_PATH}")
    print("Generate: `uv run scripts/report_optuna_optimization.py`")
    print("Figures: `uv run scripts/export_optuna_figures.py`")
    print("Dashboard: `uv run optuna-dashboard sqlite:///results/optuna_studies.db`")
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
    print(f"Preset:       {_display_preset(row['preset'])}")
    print(f"Intervention: {row['intervention'] or '-'}")
    print(f"Seed:         {row['seed'] or '-'}")
    print(f"Status:       {row['status'] or '-'}")
    print(f"Training:     {row['training_mode'] or '-'}")
    print(f"Batch ID:     {row['batch_id'] or '-'}")
    print(f"Profile:      {_display_profile(row['profile_name'])}")
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


def _compute_dataset_crru_scores(rows: list[sqlite3.Row]) -> dict[int, dict[int, float]]:
    """Return dataset-local CRRU@20 and CRRU@40 scores keyed by experiment ID."""
    scores_by_id: dict[int, dict[int, float]] = {}
    rows_by_dataset: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        rows_by_dataset[row["dataset"] or "-"].append(row)

    for dataset_rows in rows_by_dataset.values():
        efficiency_scores = compute_crru_efficiency_scores(
            [row["peak_vram_mb"] for row in dataset_rows],
            [_crru_epoch_time_s(row) for row in dataset_rows],
        )
        crru_20 = compute_crru_scores_for_k(
            ndcg=[row["test_ndcg_20"] for row in dataset_rows],
            recall=[row["test_recall_20"] for row in dataset_rows],
            hit=[row["test_hit_ratio_20"] for row in dataset_rows],
            personalization=[row["test_personalization_20"] for row in dataset_rows],
            average_popularity=[row["test_average_popularity_20"] for row in dataset_rows],
            efficiency_scores=efficiency_scores,
        )
        crru_40 = compute_crru_scores_for_k(
            ndcg=[row["test_ndcg_40"] for row in dataset_rows],
            recall=[row["test_recall_40"] for row in dataset_rows],
            hit=[row["test_hit_ratio_40"] for row in dataset_rows],
            personalization=[row["test_personalization_40"] for row in dataset_rows],
            average_popularity=[row["test_average_popularity_40"] for row in dataset_rows],
            efficiency_scores=efficiency_scores,
        )
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
    print("## CRRU Reporting Utility")
    for index, line in enumerate(CRRU_REPORT_FORMULA_LINES):
        if index == 0:
            print(f"**{line}**")
        else:
            print(f"- {line.strip()}")
    print()


def _print_evidence_role_notes() -> None:
    """Print report-wide evidence role and dataset-policy notes."""
    print("## Evidence Role Notes")
    print(
        "Rows in this report are evidence, not automatic profile promotion. "
        "Use the role labels below before turning a row into thesis wording."
    )
    print()
    _print_metric_table(
        "Evidence role legend",
        (
            ("Role", 26, "<"),
            ("Meaning", 90, "<"),
        ),
        (
            (
                "validation search candidate",
                "Optuna validation-row candidate from `results/optuna_optimization.md`; "
                "requires a full-data test rerun before test claims.",
            ),
            (
                "full-data test row",
                "Completed full-data row with test metrics in the thesis SQLite database.",
            ),
            (
                "runtime probe",
                "One-epoch or target-epoch feasibility row; use time/VRAM estimates, "
                "not final accuracy.",
            ),
            (
                "diagnostic-only evidence",
                "Score-mix, branch-rank, popularity, and contribution diagnostics; "
                "exploratory and not causal proof.",
            ),
        ),
    )
    _print_metric_table(
        "Dataset-conditioned profile policy",
        (
            ("Dataset", 14, "<"),
            ("Current policy", 94, "<"),
        ),
        (
            (
                "kuairec_v2",
                "Compact EDGRec is a default candidate because current evidence supports "
                "EDGRec accuracy plus speed.",
            ),
            (
                "movielens1m",
                "Compact EDGRec is a near-parity/speed candidate; keep the weaker "
                "popularity profile visible.",
            ),
            (
                "kuairand1k",
                "Stress-test and randomized-exposure diagnostic; do not headline as an "
                "accuracy win.",
            ),
            (
                "amazonbook",
                "Not in the shared compact default queue, but still an EDGRec target: "
                "compare compact and deeper/use_features candidates against LightGCN-paper.",
            ),
        ),
    )
    public_variants = ", ".join(ABLATION_VARIANTS)
    print(
        "Historical/internal Optuna labels such as `no_context_no_features` are search "
        "profile labels, not public ablation variants. Public ablations are: "
        f"{public_variants}."
    )
    print()


def list_top_completed(conn: sqlite3.Connection, *, n: int = 20) -> None:
    """Print the default thesis summary for full-data test runs."""
    print("## THESIS TEST RESULTS - completed full-data test rows")
    print()
    print(f"Database: `{THESIS_DB_PATH.resolve()}`")
    print()

    report_rows = _query_report_rows(conn)
    rows = [row for row in report_rows if _is_default_report_row(row)]
    reportable_formal_rows = [row for row in rows if _is_reportable_formal_row(row)]
    runtime_probe_rows = _select_paper_runtime_probe_rows(report_rows)
    if not (rows or runtime_probe_rows):
        print("No completed full-data test rows with metrics found.")
        print("Use --view all to inspect every logged experiment row.")
        print()
    else:
        crru_scores = _compute_dataset_crru_scores(rows)
        _print_crru_summary()
        _print_evidence_role_notes()
        _print_paper_baseline_notes(
            reportable_formal_rows=reportable_formal_rows,
            runtime_probe_rows=runtime_probe_rows,
        )
        _print_test_leaderboard(
            _select_top_test_rows(rows, n=n, crru_scores=crru_scores),
            crru_scores=crru_scores,
        )

    _print_optuna_report_pointer()


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
        f"# Query Results\n\nGenerated by `uv run scripts/query_results.py`.\n\n{report_text}\n"
    )
    QUERY_RESULTS_MARKDOWN_PATH.write_text(markdown_report, encoding="utf-8")


def main() -> int:
    """Parse arguments and write or print the requested experiment view."""
    args = build_query_results_parser().parse_args()

    conn = connect()
    try:
        if args.view is None:
            report_text = _render_default_summary(conn)
            _write_default_summary_markdown(report_text)
            write_query_feature_analysis_reports()
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
