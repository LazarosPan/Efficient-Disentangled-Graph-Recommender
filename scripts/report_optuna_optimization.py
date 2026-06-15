#!/usr/bin/env python
"""Write a thesis-facing Optuna optimization report from Optuna RDB storage."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from experiments.run_search import (
    DEFAULT_STORAGE,
    is_duplicate_pruned_trial,
    is_seeded_trial,
)
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.trial import FrozenTrial, TrialState
from src.utils.crru import (
    VALIDATION_ONLINE_CRRU_K_METRICS,
    VALIDATION_ONLINE_CRRU_METRIC,
    compute_crru_efficiency_scores,
    compute_crru_scores_for_k,
)
from src.utils.project_paths import RESULTS_DIR

OPTUNA_OPTIMIZATION_MARKDOWN_PATH = RESULTS_DIR / "optuna_optimization.md"
OPTUNA_FIGURES_DIR = RESULTS_DIR / "optuna_figures"
DEFAULT_TOP_N = 10
MIN_IMPORTANCE_TRIALS = 10
PAPER_FIGURE_FILENAMES = (
    "optuna_progress_by_dataset.png",
    "optuna_importance_by_dataset.png",
    "optuna_crru_components_by_dataset.png",
    "optuna_lr_branchmix_landscape.png",
    "optuna_branch_depth_heatmaps.png",
    "optuna_fanout_runtime_tradeoffs.png",
)
PARAMETER_PRIORITY = (
    "lr",
    "weight_decay",
    "batch_size",
    "lr_scheduler",
    "lr_scheduler_factor",
    "num_neighbors",
    "num_neighbors_depth_1",
    "num_neighbors_depth_2",
    "num_neighbors_depth_3",
    "interest_gnn_layers",
    "conformity_gnn_layers",
    "dropout",
    "score_mix_min_weight",
    "loss_weight_interest_bpr",
    "loss_weight_conformity_bpr",
    "loss_weight_independence",
    "loss_weight_contrastive",
    "loss_weight_popularity",
    "auxiliary_loss_schedule",
    "auxiliary_ramp_rate",
    "independence_ramp_rate",
    "auxiliary_losses_start_epoch",
    "popularity_supervision_start_epoch",
    "graph_policy",
    "cagra_candidate_k",
    "cagra_k",
    "cagra_out_degree",
    "cagra_initial_degree",
    "cagra_team_size",
    "cagra_metric",
    "cagra_itopk_size",
    "hard_negative_ratio",
    "dice_sampler_margin",
    "grad_clip_norm",
    "use_features",
    "use_popularity_head",
)
EFFECTIVE_CONFIG_EXTRA_PRIORITY = (
    "epochs",
    "patience",
    "use_early_stopping",
    "auto_batch_size",
    "batch_size_candidates",
    "baseline_family",
    "preset",
    "dataset",
    "preprocessing_preset",
    "feature_policy",
    "embedding_dim",
    "training_graph_mode",
    "negative_sampling_strategy",
    "n_negatives",
    "branch_loss_mode",
    "use_amp",
)
EFFECTIVE_CONFIG_PRIORITY = tuple(
    dict.fromkeys((*PARAMETER_PRIORITY, *EFFECTIVE_CONFIG_EXTRA_PRIORITY)),
)
DATASET_METRICS = (
    "NDCG@20",
    "Recall@20",
    "HitRatio@20",
    "Personalization@20",
    "AveragePopularity@20",
    "NDCG@40",
    "Recall@40",
    "HitRatio@40",
    "Personalization@40",
    "AveragePopularity@40",
    VALIDATION_ONLINE_CRRU_K_METRICS[20],
    VALIDATION_ONLINE_CRRU_K_METRICS[40],
    VALIDATION_ONLINE_CRRU_METRIC,
)


def load_studies(storage: str, study_name: str | None = None) -> list[optuna.Study]:
    """Load one or all studies from Optuna RDB storage."""
    if study_name:
        return [optuna.load_study(study_name=study_name, storage=storage)]
    return [
        optuna.load_study(study_name=summary.study_name, storage=storage)
        for summary in optuna.get_all_study_summaries(storage=storage)
    ]


def completed_trials(study: optuna.Study) -> list[FrozenTrial]:
    """Return completed single-objective trials with finite values."""
    return [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    ]


def trial_origin(trial: FrozenTrial) -> str:
    """Return a compact report label for trial provenance."""
    return "imported" if is_seeded_trial(trial) else "fresh"


def trial_search_revision(trial: FrozenTrial) -> str | None:
    """Return the stored search-space revision when available."""
    revision = trial.user_attrs.get("search_space_revision")
    if revision is None:
        return None
    return str(revision)


def format_revision_label(revision: str | None) -> str:
    """Return a clean report label for a stored search-space revision."""
    if not revision or revision == "legacy":
        return "unrevisioned"
    return revision


def trial_sort_key(study: optuna.Study, trial: FrozenTrial) -> tuple[float, int]:
    """Sort trials according to the study direction."""
    value = float(trial.value) if trial.value is not None else float("nan")
    direction = study.direction.name.lower()
    objective_rank = value if direction == "minimize" else -value
    return objective_rank, int(trial.number)


def dataset_trial_sort_key(
    study: optuna.Study,
    trial: FrozenTrial,
    *,
    dataset: str,
) -> tuple[float, int]:
    """Sort trials by one dataset-local objective."""
    value = dataset_metric(trial, dataset, "objective")
    if value is None:
        value = float("nan")
    direction = study.direction.name.lower()
    objective_rank = value if direction == "minimize" else -value
    return objective_rank, int(trial.number)


def ordered_params(params: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return sampled params in a stable thesis-friendly order."""
    keys = [key for key in PARAMETER_PRIORITY if key in params]
    keys.extend(sorted(key for key in params if key not in set(PARAMETER_PRIORITY)))
    return [(key, params[key]) for key in keys]


def ordered_effective_params(params: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return effective config params in a stable thesis-friendly order."""
    priority = set(EFFECTIVE_CONFIG_PRIORITY)
    keys = [key for key in EFFECTIVE_CONFIG_PRIORITY if key in params]
    keys.extend(sorted(key for key in params if key not in priority))
    return [(key, params[key]) for key in keys]


def logical_param_name(param_name: str) -> str:
    """Return the thesis-facing config field for an Optuna storage parameter."""
    base_name = param_name.split("__", 1)[0]
    if base_name.startswith("num_neighbors_depth_"):
        return "num_neighbors"
    return base_name


def logical_trial_params(trial: FrozenTrial) -> Mapping[str, Any]:
    """Return logical sampled params, preferring stored U-CaGNN config fields."""
    sampled_params = trial.user_attrs.get("sampled_params")
    if isinstance(sampled_params, Mapping):
        return sampled_params
    return {logical_param_name(key): value for key, value in trial.params.items()}


def logical_importances(importances: Mapping[str, float]) -> dict[str, float]:
    """Coalesce storage-level Optuna importance names into logical config fields."""
    coalesced: dict[str, float] = {}
    for name, importance in importances.items():
        logical_name = logical_param_name(name)
        coalesced[logical_name] = coalesced.get(logical_name, 0.0) + float(importance)
    return dict(sorted(coalesced.items(), key=lambda item: item[1], reverse=True))


def format_param_value(value: Any) -> str:
    """Return a full, non-truncated parameter value."""
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def format_params(params: Mapping[str, Any]) -> str:
    """Return all parameters without truncation."""
    if not params:
        return "-"
    return ", ".join(f"{key}={format_param_value(value)}" for key, value in ordered_params(params))


def trial_dataset_names_from_attrs(trial: FrozenTrial) -> list[str]:
    """Return dataset names visible through any dataset-scoped Optuna attrs."""
    declared = trial.user_attrs.get("datasets")
    declared_names = [str(dataset) for dataset in declared] if isinstance(declared, list) else []
    attr_names = {
        key.split(".", 1)[0]
        for key in trial.user_attrs
        if "." in key and not key.startswith("optuna_param_name.")
    }
    names = sorted(set(declared_names) | attr_names)
    return [name for name in names if name]


def trial_effective_config(trial: FrozenTrial, dataset: str) -> Mapping[str, Any]:
    """Return a stored dataset effective config when future trials provide one."""
    config = trial.user_attrs.get(f"{dataset}.effective_config")
    return config if isinstance(config, Mapping) else {}


def _shared_dataset_attr(trial: FrozenTrial, suffix: str) -> Any:
    """Return one dataset-scoped attr when all available values agree."""
    values = [
        trial.user_attrs[f"{dataset}.{suffix}"]
        for dataset in trial_dataset_names_from_attrs(trial)
        if f"{dataset}.{suffix}" in trial.user_attrs
    ]
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values) else None


def effective_trial_params(
    trial: FrozenTrial,
    *,
    dataset: str | None = None,
) -> Mapping[str, Any]:
    """Return report-facing sampled plus resolved runtime/config parameters."""
    if dataset is not None:
        config_params = dict(trial_effective_config(trial, dataset))
    else:
        dataset_configs = [
            trial_effective_config(trial, name) for name in trial_dataset_names_from_attrs(trial)
        ]
        populated = [config for config in dataset_configs if config]
        config_params = dict(populated[0]) if len(populated) == 1 else {}

    params = dict(logical_trial_params(trial))
    if config_params:
        selected_config = {
            key: config_params[key] for key in EFFECTIVE_CONFIG_PRIORITY if key in config_params
        }
        selected_config.update(params)
        params = selected_config

    attr_prefix = f"{dataset}." if dataset is not None else ""
    runtime_batch = (
        trial.user_attrs.get(f"{attr_prefix}batch_size")
        if dataset is not None
        else _shared_dataset_attr(trial, "batch_size")
    )
    runtime_auto_batch = (
        trial.user_attrs.get(f"{attr_prefix}auto_batch_size")
        if dataset is not None
        else _shared_dataset_attr(trial, "auto_batch_size")
    )
    if runtime_batch is not None:
        params["batch_size"] = runtime_batch
    if runtime_auto_batch is not None:
        params["auto_batch_size"] = runtime_auto_batch
    if dataset is not None:
        params["time_per_epoch_s"] = trial_epoch_time_s(trial, dataset)
        params["peak_vram_mb"] = attr_float(trial, f"{dataset}.peak_vram_mb")
    return {key: value for key, value in params.items() if value is not None}


def format_effective_params(
    trial: FrozenTrial,
    *,
    dataset: str | None = None,
) -> str:
    """Return the effective model/runtime params visible for one trial."""
    params = effective_trial_params(trial, dataset=dataset)
    if not params:
        return "-"
    return ", ".join(
        f"{key}={format_param_value(value)}" for key, value in ordered_effective_params(params)
    )


def attr_float(trial: FrozenTrial, key: str) -> float | None:
    """Return a finite float user attribute when present."""
    value = trial.user_attrs.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def dataset_names(trial: FrozenTrial) -> list[str]:
    """Return datasets represented by trial user attributes."""
    declared = trial.user_attrs.get("datasets")
    if isinstance(declared, list):
        names = [
            str(dataset)
            for dataset in declared
            if trial.user_attrs.get(f"{dataset}.objective") is not None
        ]
        if names:
            return sorted(names)
    names = {
        key.rsplit(".", 1)[0]
        for key in trial.user_attrs
        if key.endswith(".objective") and "." in key
    }
    return sorted(names)


def dataset_metric(trial: FrozenTrial, dataset: str, metric_name: str) -> float | None:
    """Return a dataset validation metric stored in Optuna user attrs."""
    if metric_name == "objective":
        return attr_float(trial, f"{dataset}.objective")
    return attr_float(trial, f"{dataset}.val.{metric_name}")


def average_trial_attr(trial: FrozenTrial, suffix: str) -> float | None:
    """Average one dataset-scoped user attribute suffix across available datasets."""
    values = [
        attr_float(trial, f"{dataset}.{suffix}")
        for dataset in dataset_names(trial)
        if attr_float(trial, f"{dataset}.{suffix}") is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def trial_epoch_time_s(trial: FrozenTrial, dataset: str) -> float | None:
    """Return the per-epoch runtime used by post-hoc validation CRRU."""
    explicit = attr_float(trial, f"{dataset}.avg_epoch_time_s")
    if explicit is not None and explicit > 0:
        return explicit
    training_time_s = attr_float(trial, f"{dataset}.training_time_s")
    epochs = attr_float(trial, f"{dataset}.epochs_stopped_at")
    if training_time_s is not None and epochs is not None and epochs > 0:
        return training_time_s / epochs
    return None


def compute_posthoc_validation_crru(
    trials: Sequence[FrozenTrial],
    *,
    dataset: str,
) -> dict[int, dict[int, float]]:
    """Return dataset-local post-hoc validation CRRU keyed by trial number."""
    dataset_trials = [
        trial
        for trial in trials
        if dataset_metric(trial, dataset, "NDCG@20") is not None
        and dataset_metric(trial, dataset, "Recall@20") is not None
        and dataset_metric(trial, dataset, "HitRatio@20") is not None
        and dataset_metric(trial, dataset, "Personalization@20") is not None
        and dataset_metric(trial, dataset, "AveragePopularity@20") is not None
        and dataset_metric(trial, dataset, "NDCG@40") is not None
        and dataset_metric(trial, dataset, "Recall@40") is not None
        and dataset_metric(trial, dataset, "HitRatio@40") is not None
        and dataset_metric(trial, dataset, "Personalization@40") is not None
        and dataset_metric(trial, dataset, "AveragePopularity@40") is not None
    ]
    if not dataset_trials:
        return {}

    efficiency_scores = compute_crru_efficiency_scores(
        [attr_float(trial, f"{dataset}.peak_vram_mb") for trial in dataset_trials],
        [trial_epoch_time_s(trial, dataset) for trial in dataset_trials],
    )
    crru_20 = compute_crru_scores_for_k(
        ndcg=[dataset_metric(trial, dataset, "NDCG@20") for trial in dataset_trials],
        recall=[dataset_metric(trial, dataset, "Recall@20") for trial in dataset_trials],
        hit=[dataset_metric(trial, dataset, "HitRatio@20") for trial in dataset_trials],
        personalization=[
            dataset_metric(trial, dataset, "Personalization@20") for trial in dataset_trials
        ],
        average_popularity=[
            dataset_metric(trial, dataset, "AveragePopularity@20") for trial in dataset_trials
        ],
        efficiency_scores=efficiency_scores,
    )
    crru_40 = compute_crru_scores_for_k(
        ndcg=[dataset_metric(trial, dataset, "NDCG@40") for trial in dataset_trials],
        recall=[dataset_metric(trial, dataset, "Recall@40") for trial in dataset_trials],
        hit=[dataset_metric(trial, dataset, "HitRatio@40") for trial in dataset_trials],
        personalization=[
            dataset_metric(trial, dataset, "Personalization@40") for trial in dataset_trials
        ],
        average_popularity=[
            dataset_metric(trial, dataset, "AveragePopularity@40") for trial in dataset_trials
        ],
        efficiency_scores=efficiency_scores,
    )
    return {
        int(trial.number): {20: value_20, 40: value_40}
        for trial, value_20, value_40 in zip(dataset_trials, crru_20, crru_40, strict=True)
    }


def format_float(value: float | None, digits: int = 6) -> str:
    """Return a compact numeric string."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


@dataclass(frozen=True)
class ImportanceResult:
    """One reliable-or-omitted Optuna importance result."""

    importances: dict[str, float]
    revision: str | None
    trial_count: int
    reason: str | None = None


def _importance_candidate_trials(
    study: optuna.Study,
    *,
    dataset: str | None = None,
) -> list[FrozenTrial]:
    """Return completed fresh trials that can support one importance target."""
    candidates: list[FrozenTrial] = []
    for trial in completed_trials(study):
        if is_seeded_trial(trial):
            continue
        revision = trial_search_revision(trial)
        if not revision or revision == "legacy":
            continue
        if dataset is not None and dataset_metric(trial, dataset, "objective") is None:
            continue
        candidates.append(trial)
    return candidates


def _varying_logical_param_names(trials: Sequence[FrozenTrial]) -> list[str]:
    """Return logical sampled parameters with at least two observed values."""
    names = set().union(*(logical_trial_params(trial) for trial in trials)) if trials else set()
    varying: list[str] = []
    missing = {"__missing__": True}
    for name in sorted(str(value) for value in names):
        values = {
            json.dumps(
                logical_trial_params(trial).get(name, missing),
                sort_keys=True,
                separators=(",", ":"),
            )
            for trial in trials
        }
        if len(values) > 1:
            varying.append(name)
    return varying


def _homogeneous_revision_subset(
    study: optuna.Study,
    *,
    dataset: str | None = None,
) -> tuple[str | None, list[FrozenTrial], str | None]:
    """Return the largest homogeneous non-imported revision subset for importances."""
    candidates = _importance_candidate_trials(study, dataset=dataset)
    if not candidates:
        return (
            None,
            [],
            "no completed fresh trials with a stored search_space_revision",
        )

    grouped: dict[str, list[FrozenTrial]] = {}
    for trial in candidates:
        revision = trial_search_revision(trial)
        if revision is None:
            continue
        grouped.setdefault(revision, []).append(trial)
    if not grouped:
        return None, [], "no homogeneous search_space_revision subset is available"

    revision, trials = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )[0]
    if len(trials) < MIN_IMPORTANCE_TRIALS:
        return (
            revision,
            trials,
            f"only {len(trials)} completed fresh trial(s) in revision {revision}; "
            f"need at least {MIN_IMPORTANCE_TRIALS}",
        )
    varying_params = _varying_logical_param_names(trials)
    if len(varying_params) < 2:
        return (
            revision,
            trials,
            (
                f"only {len(varying_params)} varying logical parameter(s) in revision "
                f"{revision}; importances would collapse to a one-parameter table"
            ),
        )
    return revision, trials, None


def _temporary_study_from_trials(
    study: optuna.Study,
    trials: Sequence[FrozenTrial],
) -> optuna.Study:
    """Create an in-memory study containing a homogeneous completed-trial subset."""
    temporary = optuna.create_study(direction=study.direction.name.lower())
    for trial in trials:
        temporary.add_trial(trial)
    return temporary


def safe_importances(
    study: optuna.Study,
    *,
    evaluator: object | None = None,
    target: object | None = None,
    trials: Sequence[FrozenTrial] | None = None,
) -> dict[str, float]:
    """Return Optuna parameter importances when enough completed trials exist."""
    target_study = study if trials is None else _temporary_study_from_trials(study, trials)
    if len(completed_trials(target_study)) < 2:
        return {}
    try:
        if evaluator is None:
            importances = get_param_importances(target_study, target=target)
        else:
            importances = get_param_importances(
                target_study,
                evaluator=evaluator,
                target=target,
            )
        return {name: float(value) for name, value in importances.items()}
    except Exception:
        return {}


def dashboard_importance_result(study: optuna.Study) -> ImportanceResult:
    """Return Optuna's default importance view for one homogeneous revision."""
    revision, trials, reason = _homogeneous_revision_subset(study)
    if reason is not None or not trials:
        return ImportanceResult({}, revision, len(trials), reason)
    importances = logical_importances(safe_importances(study, trials=trials))
    if not importances:
        return ImportanceResult(
            {},
            revision,
            len(trials),
            "Optuna could not compute importances for the homogeneous subset",
        )
    return ImportanceResult(importances, revision, len(trials))


def dashboard_importances(study: optuna.Study) -> dict[str, float]:
    """Return reliable dashboard-like importances for figure exporters."""
    return dashboard_importance_result(study).importances


def fanova_importances(study: optuna.Study) -> dict[str, float]:
    """Return deterministic fANOVA importances for sensitivity checking."""
    _revision, trials, reason = _homogeneous_revision_subset(study)
    if reason is not None or not trials:
        return {}
    return logical_importances(
        safe_importances(
            study,
            evaluator=FanovaImportanceEvaluator(seed=13),
            trials=trials,
        ),
    )


def dataset_importance_result(study: optuna.Study, dataset: str) -> ImportanceResult:
    """Return reliable importances for one dataset-local objective."""
    revision, trials, reason = _homogeneous_revision_subset(study, dataset=dataset)
    if reason is not None or not trials:
        return ImportanceResult({}, revision, len(trials), reason)

    def target(trial: FrozenTrial) -> float:
        value = dataset_metric(trial, dataset, "objective")
        if value is None:
            raise ValueError(f"Trial {trial.number} has no {dataset} objective.")
        return float(value)

    importances = logical_importances(safe_importances(study, target=target, trials=trials))
    if not importances:
        return ImportanceResult(
            {},
            revision,
            len(trials),
            "Optuna could not compute importances for the homogeneous subset",
        )
    return ImportanceResult(importances, revision, len(trials))


def dataset_importances(study: optuna.Study, dataset: str) -> dict[str, float]:
    """Return reliable dashboard-like importances for figure exporters."""
    return dataset_importance_result(study, dataset).importances


def failure_label(trial: FrozenTrial) -> str:
    """Return a compact failure reason for report grouping."""
    reason = trial.user_attrs.get("failure_reason")
    stage = trial.user_attrs.get("failure_stage")
    if reason:
        return f"{stage or 'unknown'}: {reason}"
    return "failure before stored attrs; exact exception unavailable in Optuna RDB"


def study_report_sort_key(study: optuna.Study) -> tuple[int, str]:
    """Sort active dataset-local CRRU studies before historical/global screens."""
    trials = completed_trials(study)
    datasets = sorted({dataset for trial in trials for dataset in dataset_names(trial)})
    objective_metric = str(trials[0].user_attrs.get("objective_metric", "")) if trials else ""
    if objective_metric == VALIDATION_ONLINE_CRRU_METRIC and len(datasets) == 1:
        return (0, study.study_name)
    if objective_metric == VALIDATION_ONLINE_CRRU_METRIC:
        return (1, study.study_name)
    return (2, study.study_name)


def render_failure_summary(study: optuna.Study) -> list[str]:
    """Render failed-trial diagnostics from stored Optuna attributes."""
    failed_trials = [trial for trial in study.trials if trial.state == TrialState.FAIL]
    if not failed_trials:
        return []
    counts = Counter(failure_label(trial) for trial in failed_trials)
    example_by_reason = {
        reason: next(trial.number for trial in failed_trials if failure_label(trial) == reason)
        for reason in counts
    }
    lines = [
        "### Failed-trial diagnostics",
        "",
        "| Count | Example trial | Stored reason |",
        "|---:|---:|---|",
    ]
    for reason, count in counts.most_common():
        lines.append(f"| {count} | {example_by_reason[reason]} | `{reason}` |")
    lines.extend(
        [
            "",
            "Legacy failures without stored attributes happened before the current failure "
            "recorder could write `failure_stage` / `failure_reason`; the old runs cannot "
            "be reconstructed from Optuna RDB alone.",
            "",
        ],
    )
    return lines


def render_trial_accounting(study: optuna.Study) -> list[str]:
    """Render fresh-vs-imported counts by Optuna state."""
    states = ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING")
    counts: dict[str, Counter[str]] = {
        "fresh": Counter(),
        "imported": Counter(),
    }
    fresh_duplicate_pruned = 0
    for trial in study.trials:
        origin = trial_origin(trial)
        state = trial.state.name
        counts[origin][state] += 1
        if (
            origin == "fresh"
            and trial.state == TrialState.PRUNED
            and is_duplicate_pruned_trial(trial)
        ):
            fresh_duplicate_pruned += 1

    fresh_informative = counts["fresh"]["COMPLETE"] + max(
        0,
        counts["fresh"]["PRUNED"] - fresh_duplicate_pruned,
    )
    lines = [
        "### Trial accounting",
        "",
        "| State | Fresh | Imported | Total |",
        "|---|---:|---:|---:|",
    ]
    for state in states:
        fresh = counts["fresh"][state]
        imported = counts["imported"][state]
        total = fresh + imported
        if total:
            lines.append(f"| {state} | {fresh} | {imported} | {total} |")
    lines.extend(
        [
            "",
            f"- Fresh informative budget count: `{fresh_informative}` "
            "(fresh COMPLETE + real fresh PRUNED).",
            f"- Duplicate-skip pruned trials excluded from that budget: "
            f"`{fresh_duplicate_pruned}`.",
            "",
        ],
    )
    return lines


def render_importance_result(title: str, result: ImportanceResult) -> list[str]:
    """Render a reliable importance table or an explicit omission note."""
    if not result.importances:
        return [
            f"### {title}",
            "",
            f"Importances omitted: {result.reason or 'not enough comparable trials'}.",
            "",
        ]
    lines = [
        f"### {title}",
        "",
        f"Subset: `{result.trial_count}` fresh completed trial(s) from "
        f"search-space revision `{result.revision}`.",
        "",
        "| Rank | Parameter | Importance |",
        "|---:|---|---:|",
    ]
    for rank, (name, importance) in enumerate(result.importances.items(), start=1):
        lines.append(f"| {rank} | `{name}` | {importance:.6f} |")
    lines.append("")
    return lines


def render_importance_table(title: str, importances: Mapping[str, float]) -> list[str]:
    """Render one importance table."""
    if not importances:
        return []
    lines = [
        f"### {title}",
        "",
        "| Rank | Parameter | Importance |",
        "|---:|---|---:|",
    ]
    for rank, (name, importance) in enumerate(importances.items(), start=1):
        lines.append(f"| {rank} | `{name}` | {importance:.6f} |")
    lines.append("")
    return lines


def render_dataset_best_trials(
    study: optuna.Study,
    *,
    datasets: Sequence[str],
    posthoc_crru: Mapping[str, Mapping[int, Mapping[int, float]]],
    top_n: int = 3,
) -> list[str]:
    """Render dataset-local best configurations."""
    lines = [
        "### Per-dataset best trials",
        "",
        "These are the configurations to inspect for formal dataset-specific reruns. "
        "For historical all-dataset studies, the `Study objective` column may be a global "
        "mean and should not replace the dataset-local objective.",
        "",
    ]
    for dataset in datasets:
        ranked = sorted(
            [
                trial
                for trial in completed_trials(study)
                if dataset_metric(trial, dataset, "objective") is not None
            ],
            key=lambda trial: dataset_trial_sort_key(study, trial, dataset=dataset),
        )
        if not ranked:
            continue
        lines.extend(
            [
                f"#### Dataset: `{dataset}`",
                "",
                "| Rank | Trial | Origin | Dataset objective | Study objective | "
                "PosthocCRRU@20 | PosthocCRRU@40 | Time/epoch (s) | Peak VRAM (MB) | Batch | "
                "Revision | Effective config |",
                "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ],
        )
        for rank, trial in enumerate(ranked[:top_n], start=1):
            trial_crru = posthoc_crru.get(dataset, {}).get(int(trial.number), {})
            lines.append(
                f"| {rank} | {trial.number} | {trial_origin(trial)} | "
                f"{format_float(dataset_metric(trial, dataset, 'objective'))} | "
                f"{format_float(float(trial.value) if trial.value is not None else None)} | "
                f"{format_float(trial_crru.get(20))} | "
                f"{format_float(trial_crru.get(40))} | "
                f"{format_float(trial_epoch_time_s(trial, dataset), digits=2)} | "
                f"{format_float(attr_float(trial, f'{dataset}.peak_vram_mb'), digits=1)} | "
                f"{format_float(attr_float(trial, f'{dataset}.batch_size'), digits=0)} | "
                f"`{format_revision_label(trial_search_revision(trial))}` | "
                f"`{format_effective_params(trial, dataset=dataset)}` |",
            )
        lines.append("")
    return lines


def render_dataset_importance_tables(
    study: optuna.Study,
    *,
    datasets: Sequence[str],
) -> list[str]:
    """Render per-dataset objective importance diagnostics."""
    lines: list[str] = []
    for dataset in datasets:
        result = dataset_importance_result(study, dataset)
        lines.extend(
            render_importance_result(
                f"Dashboard-like Optuna importances for `{dataset}` objective",
                result,
            ),
        )
    return lines


def render_study_report(study: optuna.Study, *, top_n: int) -> str:
    """Render one Optuna study as markdown."""
    trials = completed_trials(study)
    state_counts = Counter(trial.state.name.lower() for trial in study.trials)
    objective_metric = "-"
    objective_split = "-"
    if trials:
        objective_metric = str(trials[0].user_attrs.get("objective_metric", "-"))
        objective_split = str(trials[0].user_attrs.get("objective_split", "-"))

    lines = [
        f"## Study: `{study.study_name}`",
        "",
        f"- Direction: `{study.direction.name.lower()}`",
        f"- Objective: `{objective_split} {objective_metric}`",
        f"- Trials: {len(study.trials)} total, {len(trials)} completed, "
        f"{state_counts.get('fail', 0)} failed, {state_counts.get('running', 0)} running, "
        f"{state_counts.get('pruned', 0)} pruned",
        "",
    ]
    lines.extend(render_trial_accounting(study))
    if not trials:
        lines.extend(["No completed trials.", ""])
        return "\n".join(lines)

    best_trial = sorted(trials, key=lambda trial: trial_sort_key(study, trial))[0]
    all_datasets = sorted({dataset for trial in trials for dataset in dataset_names(trial)})
    posthoc_crru = {
        dataset: compute_posthoc_validation_crru(trials, dataset=dataset)
        for dataset in all_datasets
    }
    lines.extend(
        [
            "### Best study-level trial",
            "",
            f"- Trial: `{best_trial.number}`",
            f"- Objective value: `{format_float(float(best_trial.value))}`",
            f"- Effective config: `{format_effective_params(best_trial)}`",
            "",
        ],
    )

    datasets = dataset_names(best_trial)
    if datasets:
        lines.extend(
            [
                "### Best trial dataset metrics",
                "",
                "| Dataset | Objective | NDCG@20 | Recall@20 | Hit@20 | Pers@20 | AvgPop@20 | "
                "NDCG@40 | Recall@40 | Hit@40 | Pers@40 | AvgPop@40 | "
                "ValCRRU@20 | ValCRRU@40 | ValCRRU@20_40 | PosthocCRRU@20 | "
                "PosthocCRRU@40 | Time/epoch (s) | Peak VRAM (MB) | Batch |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                "---:|---:|---:|---:|---:|",
            ],
        )
        for dataset in datasets:
            values = [dataset_metric(best_trial, dataset, metric) for metric in DATASET_METRICS]
            trial_crru = posthoc_crru.get(dataset, {}).get(int(best_trial.number), {})
            lines.append(
                f"| {dataset} | {format_float(dataset_metric(best_trial, dataset, 'objective'))} | "
                + " | ".join(format_float(value) for value in values[:-1])
                + f" | {format_float(values[-1])} | "
                f"{format_float(trial_crru.get(20))} | "
                f"{format_float(trial_crru.get(40))} | "
                f"{format_float(trial_epoch_time_s(best_trial, dataset), digits=2)} | "
                f"{format_float(attr_float(best_trial, f'{dataset}.peak_vram_mb'), digits=1)} | "
                f"{format_float(attr_float(best_trial, f'{dataset}.batch_size'), digits=0)} |",
            )
        lines.append("")

    if all_datasets:
        lines.extend(
            render_dataset_best_trials(
                study,
                datasets=all_datasets,
                posthoc_crru=posthoc_crru,
                top_n=3,
            ),
        )

    lines.extend(
        render_importance_result(
            "Dashboard-like Optuna importances",
            dashboard_importance_result(study),
        ),
    )
    fanova = fanova_importances(study)
    if fanova:
        lines.extend(render_importance_table("Deterministic fANOVA importance sensitivity", fanova))
    if all_datasets:
        lines.extend(render_dataset_importance_tables(study, datasets=all_datasets))

    lines.extend(render_failure_summary(study))

    lines.extend(
        [
            f"### Top {min(top_n, len(trials))} completed trials",
            "",
            "| Rank | Trial | Objective | Avg train time (s) | Avg peak VRAM (MB) | "
            "Effective config |",
            "|---:|---:|---:|---:|---:|---|",
        ],
    )
    for rank, trial in enumerate(
        sorted(trials, key=lambda item: trial_sort_key(study, item))[:top_n],
        start=1,
    ):
        lines.append(
            f"| {rank} | {trial.number} | {format_float(float(trial.value))} | "
            f"{format_float(average_trial_attr(trial, 'training_time_s'), digits=2)} | "
            f"{format_float(average_trial_attr(trial, 'peak_vram_mb'), digits=1)} | "
            f"`{format_effective_params(trial)}` |",
        )
    lines.append("")

    lines.extend(
        [
            "### Formal-promotion candidates",
            "",
            "Promote dataset-local winners into formal profiles only after checking runtime "
            "and whether popularity diagnostics are acceptable.",
            "",
        ],
    )
    if all_datasets:
        for dataset in all_datasets:
            ranked = sorted(
                [
                    trial
                    for trial in trials
                    if dataset_metric(trial, dataset, "objective") is not None
                ],
                key=lambda trial: dataset_trial_sort_key(study, trial, dataset=dataset),
            )
            for trial in ranked[:3]:
                trial_crru = posthoc_crru.get(dataset, {}).get(int(trial.number), {})
                lines.append(
                    f"- `{dataset}` trial `{trial.number}`: dataset objective "
                    f"`{format_float(dataset_metric(trial, dataset, 'objective'))}`; "
                    f"posthoc CRRU@20/40 "
                    f"`{format_float(trial_crru.get(20))}`/"
                    f"`{format_float(trial_crru.get(40))}`; "
                    f"time/epoch `{format_float(trial_epoch_time_s(trial, dataset), digits=2)}`s; "
                    f"peak VRAM "
                    f"`{format_float(attr_float(trial, f'{dataset}.peak_vram_mb'), digits=1)}`MB; "
                    f"batch "
                    f"`{format_float(attr_float(trial, f'{dataset}.batch_size'), digits=0)}`; "
                    f"study objective "
                    f"`{format_float(float(trial.value) if trial.value is not None else None)}`; "
                    f"effective config `{format_effective_params(trial, dataset=dataset)}`",
                )
    else:
        for trial in sorted(trials, key=lambda item: trial_sort_key(study, item))[:3]:
            lines.append(
                f"- Trial `{trial.number}`: objective `{format_float(float(trial.value))}`; "
                f"effective config `{format_effective_params(trial)}`",
            )
    lines.append("")
    return "\n".join(lines)


def render_report(studies: Sequence[optuna.Study], *, storage: str, top_n: int) -> str:
    """Render all Optuna studies as markdown."""
    lines = [
        "# Optuna Optimization Report",
        "",
        "Generated by `uv run scripts/report_optuna_optimization.py`.",
        "",
        f"- Storage: `{storage}`",
        f"- Figure directory: `{OPTUNA_FIGURES_DIR}`",
        "- Dashboard: `uv run optuna-dashboard sqlite:///results/optuna_studies.db`",
        "",
        "## Interpretation notes",
        "",
        "- Search objectives use validation metrics only; test metrics remain for formal reruns.",
        "- New search-space definitions use `ValidationOnlineCRRU@20_40`, an online validation "
        "proxy with the same CRRU component/exponent structure as thesis CRRU.",
        "- Exact thesis CRRU uses dataset-local section-row min-max normalization and is "
        "recomputed after rows exist; it is not a stable live Optuna objective because "
        "future trials change the min/max range.",
        "- Historical studies may still show `val NDCG@40` or other older objectives.",
        "- Lower average popularity and higher personalization are supporting diagnostics, "
        "not causal-effect estimates.",
        "- If Optuna dashboard importances differ, first confirm the same storage URI, "
        "study name, objective target, and completed-trial subset.",
        "- Optuna RDB storage is the canonical owner for search trials; the thesis SQLite "
        "database keeps formal experiment and training logs.",
        "- Current default multi-dataset searches expand into one independent study per "
        "dataset; older `*-all-*` studies are historical global-mean screens.",
        "- `--trials N` means N fresh informative finished trials for the current "
        "`search_space_revision`: COMPLETE plus real PRUNED, excluding FAIL/RUNNING, "
        "historically imported rows, duplicate-skip prunes, and other revision hashes. "
        "Imported rows are reported separately for provenance.",
        "- Hyperparameter importances are computed only within one homogeneous, "
        "non-imported `search_space_revision` subset with enough completed trials. "
        "Mixed unrevisioned/revision studies are marked unreliable instead of showing a "
        "misleading one-parameter importance table.",
        "- The current second-pass grid tunes active U-CaGNN loss weights and schedule "
        "knobs only; inactive DirectAU/IPW-only weights remain out of the default "
        "search to avoid changing the thesis model family without a separate ablation.",
        "- Schedule-conditioned parameters are sampled only when they affect the resolved "
        "training config: ramp rates for `linear_ramp`, start epochs for `phased`.",
        "- Candidate rows print an effective-config view, not just raw Optuna sampled "
        "parameters. Future trials store the resolved dataset config directly; older "
        "trials fall back to sampled params plus any runtime attrs present in Optuna.",
        "",
        "## Paper-ready figures",
        "",
        "Generated by `uv run scripts/export_optuna_figures.py`. The default exporter "
        "writes a compact figure set and removes stale per-study PNGs from the figure "
        "directory.",
        "",
        "The `optuna_importance_by_dataset.png` heatmap is an overview figure: rows marked "
        "`strict` use the same homogeneous fresh-revision subset as the report tables, while "
        "rows marked `exploratory` use normalized univariate association scores over all "
        "completed dataset-local trials so older datasets are still visually inspectable. "
        "Use only `strict` rows for strong importance claims.",
        "",
    ]
    for filename in PAPER_FIGURE_FILENAMES:
        lines.append(f"- `{OPTUNA_FIGURES_DIR / filename}`")
    lines.extend(
        [
            "",
            "## Candidate report figures not generated by default",
            "",
            "Keep the default figure set compact. If the thesis needs one more figure, choose "
            "one of these rather than exporting dozens of per-study diagnostics:",
            "",
            "- Dataset-local top-3 candidate profile plot: NDCG, Recall, personalization, "
            "low-popularity score, epoch time, and VRAM for the formal-promotion trials.",
            "- Loss-weight response surface: interest-BPR weight vs conformity-BPR weight, "
            "colored by validation CRRU, one panel per dataset.",
            "- Schedule ablation strip: `linear_ramp` vs `phased`, with active ramp/start "
            "knobs annotated beside each dataset's best trial.",
            "- CRRU frontier plot: accuracy component vs bias/diversity component, bubble "
            "size for efficiency and star markers for promotion candidates.",
            "",
        ],
    )
    lines.extend(
        [
            "## Studies",
            "",
            "Active dataset-local CRRU studies are listed first; historical/global screens "
            "remain below for provenance and failure diagnosis.",
            "",
        ],
    )
    if not studies:
        lines.extend(["No Optuna studies found.", ""])
    for study in sorted(studies, key=study_report_sort_key):
        lines.append(render_study_report(study, top_n=top_n))
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--output", type=Path, default=OPTUNA_OPTIMIZATION_MARKDOWN_PATH)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the Optuna optimization markdown report."""
    args = build_parser().parse_args(argv)
    studies = load_studies(args.storage, args.study_name)
    report = render_report(studies, storage=args.storage, top_n=args.top_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote Optuna optimization report to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
