#!/usr/bin/env python
"""Write a thesis-facing Optuna optimization report from Optuna RDB storage."""

from __future__ import annotations

import argparse
import hashlib
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
from optuna.importance import (
    MeanDecreaseImpurityImportanceEvaluator,
    get_param_importances,
)
from optuna.trial import FrozenTrial, TrialState
from src.reporting.feature_analysis import (
    build_feature_subset_result_rows,
    render_feature_subset_report_section,
    write_feature_subset_search_reports,
)
from src.utils.crru import (
    CRRU_RECOMMENDATION_METRIC_NAMES,
    CRRU_REPORT_FORMULA_LINES,
    VALIDATION_ACCURACY_METRIC,
    VALIDATION_CRRU_K_METRIC_ALIASES,
    VALIDATION_CRRU_K_METRICS,
    VALIDATION_CRRU_METRIC,
    VALIDATION_CRRU_METRIC_ALIASES,
    compute_validation_crru_metric_value,
    is_validation_crru_metric_name,
)
from src.utils.crru_popularity import (
    CRRUPopularityReconstructionError,
    resolve_largest_training_item_interaction_count,
)
from src.utils.method_naming import (
    display_method_label,
    method_identifier_aliases,
    public_method_identifier,
)
from src.utils.project_paths import RESULTS_DIR

OPTUNA_OPTIMIZATION_MARKDOWN_PATH = RESULTS_DIR / "optuna_optimization.md"
OPTUNA_FIGURES_DIR = RESULTS_DIR / "optuna_figures"
DEFAULT_TOP_N = 10
MIN_IMPORTANCE_TRIALS = 10
IMPORTANCE_CACHE_VERSION = 1
IMPORTANCE_CACHE_PATH = RESULTS_DIR / "optuna_importance_cache.json"
IMPORTANCE_EVALUATOR_NAME = "mean_decrease_impurity_seed_13"
INCLUDE_OPTUNA_IMPORTANCE_TABLES = False
DEFAULT_SOURCE_OBJECTIVE_METRICS = (
    *VALIDATION_CRRU_METRIC_ALIASES,
    VALIDATION_ACCURACY_METRIC,
)
CRRU_FIGURE_STEMS = (
    "component_correlations_by_dataset",
    "selection_frontier_by_dataset",
    "importance_by_dataset",
    "components_by_dataset",
    "lr_branchmix_landscape",
    "branch_depth_heatmaps",
    "fanout_runtime_tradeoffs",
)
PARAMETER_PRIORITY = (
    "lr",
    "weight_decay",
    "batch_size",
    "lr_scheduler",
    "lr_scheduler_factor",
    "embedding_optimizer",
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
    "item_universe_policy",
    "train_edge_keep_prob",
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
    *CRRU_RECOMMENDATION_METRIC_NAMES,
    VALIDATION_CRRU_K_METRICS[20],
    VALIDATION_CRRU_K_METRICS[40],
    VALIDATION_CRRU_METRIC,
)


def objective_metric_label(metric: str) -> str:
    """Return a compact display label for a validation objective metric."""
    if metric in VALIDATION_CRRU_METRIC_ALIASES:
        return "ValidationCRRU@20_40"
    for cutoff, aliases in VALIDATION_CRRU_K_METRIC_ALIASES.items():
        if metric in aliases:
            return f"ValidationCRRU@{cutoff}"
    return {
        VALIDATION_ACCURACY_METRIC: "Validation Accuracy",
    }.get(metric, metric)


def objective_split_metric_label(split: str, metric: str) -> str:
    """Return a display label for one objective split and metric name."""
    return f"{split} {objective_metric_label(metric)}"


PAPER_FIGURE_FILENAMES = tuple(f"optuna_crru_{stem}.png" for stem in CRRU_FIGURE_STEMS)


def display_study_name(study_name: str) -> str:
    """Return a thesis-facing display name for an Optuna study."""
    name = public_method_identifier(study_name) or study_name
    return name.replace("validationonlinecrru", "validationcrru")


def _completed_finite_trial(trial: FrozenTrial) -> bool:
    """Return whether an Optuna trial has a completed finite objective value."""
    return (
        trial.state == TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    )


def _raw_trial_objective_metric(trial: FrozenTrial) -> str:
    """Return the raw stored objective metric label for one trial."""
    metric = trial.user_attrs.get("objective_metric")
    return str(metric) if metric else "-"


def _study_has_default_objective_metric(study: optuna.Study) -> bool:
    """Return whether a study belongs in default thesis Optuna outputs."""
    return any(
        _completed_finite_trial(trial)
        and _raw_trial_objective_metric(trial) in DEFAULT_SOURCE_OBJECTIVE_METRICS
        for trial in study.trials
    )


def load_studies(storage: str, study_name: str | None = None) -> list[optuna.Study]:
    """Load one requested study or every default thesis study from Optuna storage."""
    summaries = sorted(
        optuna.get_all_study_summaries(storage=storage),
        key=lambda summary: summary.study_name,
    )
    if study_name:
        existing = {summary.study_name for summary in summaries}
        resolved_name = next(
            (alias for alias in method_identifier_aliases(study_name) if alias in existing),
            study_name,
        )
        return [optuna.load_study(study_name=resolved_name, storage=storage)]
    studies = [
        optuna.load_study(study_name=summary.study_name, storage=storage) for summary in summaries
    ]
    return [study for study in studies if _study_has_default_objective_metric(study)]


def completed_trials(study: optuna.Study) -> list[FrozenTrial]:
    """Return completed single-objective trials with finite values."""
    return [trial for trial in study.trials if _completed_finite_trial(trial)]


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
    _study: optuna.Study,
    trial: FrozenTrial,
    *,
    dataset: str,
) -> tuple[float, int]:
    """Sort trials by the default reconstructed validation CRRU objective."""
    value = dataset_metric(trial, dataset, VALIDATION_CRRU_METRIC)
    if value is None:
        value = float("nan")
    return -value, int(trial.number)


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
    """Return logical sampled params, preferring stored EDGRec config fields."""
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


def format_effective_param_value(key: str, value: Any) -> str:
    """Return a public report value for one effective config parameter."""
    if key in {"baseline_family", "preset"}:
        return display_method_label(value)
    return format_param_value(value)


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
        f"{key}={format_effective_param_value(key, value)}"
        for key, value in ordered_effective_params(params)
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


def trial_largest_training_item_interaction_count(
    trial: FrozenTrial,
    dataset: str,
) -> float | None:
    """Return stored or reconstructed CRRU train-popularity denominator."""
    try:
        return resolve_largest_training_item_interaction_count(
            stored_value=attr_float(
                trial,
                f"{dataset}.largest_training_item_interaction_count",
            ),
            config=effective_trial_params(trial, dataset=dataset),
            dataset=dataset,
        )
    except CRRUPopularityReconstructionError:
        return None


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
    if is_validation_crru_metric_name(metric_name):
        return _derived_validation_crru_metric_value(trial, dataset, metric_name)
    if metric_name == "objective":
        if is_validation_crru_metric_name(trial_objective_metric(trial)):
            return _derived_validation_crru_metric_value(trial, dataset, VALIDATION_CRRU_METRIC)
        return attr_float(trial, f"{dataset}.objective")
    stored = attr_float(trial, f"{dataset}.val.{metric_name}")
    if stored is not None:
        return stored
    return None


def _primary_validation_metrics(trial: FrozenTrial, dataset: str) -> dict[str, float]:
    """Return stored validation metrics needed to derive CRRU."""
    metrics: dict[str, float] = {}
    for name in CRRU_RECOMMENDATION_METRIC_NAMES:
        value = attr_float(trial, f"{dataset}.val.{name}")
        if value is None:
            return {}
        metrics[name] = value
    return metrics


def _validation_crru_inputs(
    trial: FrozenTrial,
    dataset: str,
) -> tuple[dict[str, float], float, float, float] | None:
    """Return complete formal CRRU inputs for one dataset trial."""
    metrics = _primary_validation_metrics(trial, dataset)
    peak_vram_mb = attr_float(trial, f"{dataset}.peak_vram_mb")
    epoch_time_s = trial_epoch_time_s(trial, dataset)
    largest_training_item_interaction_count = trial_largest_training_item_interaction_count(
        trial,
        dataset,
    )
    if (
        not metrics
        or peak_vram_mb is None
        or epoch_time_s is None
        or largest_training_item_interaction_count is None
    ):
        return None
    return (
        metrics,
        peak_vram_mb,
        epoch_time_s,
        largest_training_item_interaction_count,
    )


def _derived_validation_crru_metric_value(
    trial: FrozenTrial,
    dataset: str,
    metric_name: str,
) -> float | None:
    """Derive a ValidationCRRU metric from stored validation and runtime attrs."""
    inputs = _validation_crru_inputs(trial, dataset)
    if inputs is None:
        return None
    metrics, peak_vram_mb, epoch_time_s, largest_training_item_interaction_count = inputs
    try:
        return compute_validation_crru_metric_value(
            metric_name,
            metrics,
            peak_vram_mb=peak_vram_mb,
            epoch_time_s=epoch_time_s,
            largest_training_item_interaction_count=largest_training_item_interaction_count,
        )
    except ValueError:
        return None


def trial_objective_metric(trial: FrozenTrial) -> str:
    """Return the objective metric stored for one Optuna trial."""
    return _raw_trial_objective_metric(trial)


def trial_objective_split(trial: FrozenTrial) -> str:
    """Return the objective split stored for one Optuna trial."""
    split = trial.user_attrs.get("objective_split")
    return str(split) if split else "-"


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


def _trial_validation_crru_by_cutoff(
    trial: FrozenTrial,
    dataset: str,
) -> dict[int, float] | None:
    """Return formal ValidationCRRU@K values for one dataset trial."""
    inputs = _validation_crru_inputs(trial, dataset)
    if inputs is None:
        return None
    metrics, peak_vram_mb, epoch_time_s, largest_training_item_interaction_count = inputs
    try:
        return {
            cutoff: compute_validation_crru_metric_value(
                metric_name,
                metrics,
                peak_vram_mb=peak_vram_mb,
                epoch_time_s=epoch_time_s,
                largest_training_item_interaction_count=largest_training_item_interaction_count,
            )
            for cutoff, metric_name in VALIDATION_CRRU_K_METRICS.items()
        }
    except ValueError:
        return None


def compute_posthoc_validation_crru(
    trials: Sequence[FrozenTrial],
    *,
    dataset: str,
) -> dict[int, dict[int, float]]:
    """Return absolute post-hoc validation CRRU keyed by trial number."""
    output: dict[int, dict[int, float]] = {}
    for trial in trials:
        values_by_cutoff = _trial_validation_crru_by_cutoff(trial, dataset)
        if values_by_cutoff:
            output[int(trial.number)] = values_by_cutoff
    return output


def format_float(value: float | None, digits: int = 6) -> str:
    """Return a compact numeric string."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


@dataclass(frozen=True)
class DatasetTrialCandidate:
    """One dataset-scoped completed Optuna trial candidate."""

    dataset: str
    objective_metric: str
    objective_split: str
    source_objective_metric: str
    source_objective_split: str
    study_name: str
    study_display_name: str
    study_direction: str
    trial: FrozenTrial
    objective_value: float
    study_objective_value: float | None


@dataclass(frozen=True)
class ImportanceResult:
    """One reliable-or-omitted Optuna importance result."""

    importances: dict[str, float]
    revision: str | None
    trial_count: int
    reason: str | None = None
    subset_count: int = 0


_IMPORTANCE_CACHE: dict[str, dict[str, float]] | None = None


def _load_importance_cache() -> dict[str, dict[str, float]]:
    """Return cached deterministic Optuna importances."""
    global _IMPORTANCE_CACHE
    if _IMPORTANCE_CACHE is not None:
        return _IMPORTANCE_CACHE
    if not IMPORTANCE_CACHE_PATH.exists():
        _IMPORTANCE_CACHE = {}
        return _IMPORTANCE_CACHE
    try:
        payload = json.loads(IMPORTANCE_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _IMPORTANCE_CACHE = {}
        return _IMPORTANCE_CACHE
    if payload.get("version") != IMPORTANCE_CACHE_VERSION:
        _IMPORTANCE_CACHE = {}
        return _IMPORTANCE_CACHE
    entries = payload.get("entries", {})
    _IMPORTANCE_CACHE = {
        str(key): {str(name): float(value) for name, value in value.items()}
        for key, value in entries.items()
        if isinstance(value, Mapping)
    }
    return _IMPORTANCE_CACHE


def _store_importance_cache(cache: Mapping[str, Mapping[str, float]]) -> None:
    """Persist cached deterministic Optuna importances."""
    IMPORTANCE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": IMPORTANCE_CACHE_VERSION,
        "entries": {key: dict(sorted(value.items())) for key, value in sorted(cache.items())},
    }
    IMPORTANCE_CACHE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dataset_trial_candidates(studies: Sequence[optuna.Study]) -> list[DatasetTrialCandidate]:
    """Return every dataset-scoped completed candidate from loaded studies."""
    candidates: list[DatasetTrialCandidate] = []
    for study in studies:
        display_name = display_study_name(study.study_name)
        for trial in completed_trials(study):
            study_objective = float(trial.value) if trial.value is not None else None
            source_metric = trial_objective_metric(trial)
            source_split = trial_objective_split(trial)
            for dataset in dataset_names(trial):
                objective = dataset_metric(trial, dataset, VALIDATION_CRRU_METRIC)
                if objective is None or not math.isfinite(float(objective)):
                    continue
                candidates.append(
                    DatasetTrialCandidate(
                        dataset=dataset,
                        objective_metric=VALIDATION_CRRU_METRIC,
                        objective_split="val",
                        source_objective_metric=source_metric,
                        source_objective_split=source_split,
                        study_name=study.study_name,
                        study_display_name=display_name,
                        study_direction="maximize",
                        trial=trial,
                        objective_value=float(objective),
                        study_objective_value=study_objective,
                    ),
                )
    return candidates


def render_side_feature_analysis(
    studies: Sequence[optuna.Study],
    *,
    rows: Sequence[Mapping[str, object]] | None = None,
) -> list[str]:
    """Render dataset-local feature-subset search evidence."""
    feature_rows = rows if rows is not None else build_feature_subset_result_rows(studies)
    return render_feature_subset_report_section(feature_rows)


def objective_group_sort_key(group: tuple[str, str, str]) -> tuple[int, str, str]:
    """Return a stable order for objective groups in thesis reports."""
    metric, split, direction = group
    if metric in VALIDATION_CRRU_METRIC_ALIASES:
        priority = 0
    else:
        priority = {
            VALIDATION_ACCURACY_METRIC: 1,
            "NDCG@40": 2,
        }.get(metric, 10)
    return priority, split, direction


def candidate_sort_key(candidate: DatasetTrialCandidate) -> tuple[float, str, int]:
    """Sort candidates by their own objective direction and provenance."""
    objective_rank = (
        candidate.objective_value
        if candidate.study_direction == "minimize"
        else -candidate.objective_value
    )
    return objective_rank, candidate.study_name, int(candidate.trial.number)


def compute_global_posthoc_validation_crru(
    candidates: Sequence[DatasetTrialCandidate],
) -> dict[tuple[str, int, str], dict[int, float]]:
    """Return absolute post-hoc CRRU over every loaded candidate."""
    result: dict[tuple[str, int, str], dict[int, float]] = {}
    for candidate in candidates:
        values_by_cutoff = _trial_validation_crru_by_cutoff(candidate.trial, candidate.dataset)
        if values_by_cutoff:
            result[(candidate.study_name, int(candidate.trial.number), candidate.dataset)] = {
                20: values_by_cutoff[20],
                40: values_by_cutoff[40],
            }
    return result


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


def _eligible_revision_subsets(
    study: optuna.Study,
    *,
    dataset: str | None = None,
) -> tuple[list[tuple[str, list[FrozenTrial]]], str | None]:
    """Return comparable completed-trial groups keyed by search-space revision."""
    candidates = _importance_candidate_trials(study, dataset=dataset)
    if not candidates:
        return [], "no completed fresh trials with a stored search_space_revision"

    grouped: dict[str, list[FrozenTrial]] = {}
    for trial in candidates:
        revision = trial_search_revision(trial)
        if revision is not None:
            grouped.setdefault(revision, []).append(trial)

    eligible: list[tuple[str, list[FrozenTrial]]] = []
    skipped: list[str] = []
    for revision, trials in sorted(grouped.items()):
        if len(trials) < MIN_IMPORTANCE_TRIALS:
            skipped.append(f"{revision}: {len(trials)} completed trial(s)")
            continue
        varying_params = _varying_logical_param_names(trials)
        if len(varying_params) < 2:
            skipped.append(f"{revision}: {len(varying_params)} varying parameter(s)")
            continue
        eligible.append((revision, trials))

    if eligible:
        return eligible, None
    if skipped:
        return (
            [],
            "no revision subset has enough completed trials and varying parameters "
            f"(minimum {MIN_IMPORTANCE_TRIALS}); skipped {', '.join(skipped)}",
        )
    return [], "no homogeneous search_space_revision subset is available"


def _aggregate_revision_importances(
    study: optuna.Study,
    *,
    dataset: str | None = None,
) -> ImportanceResult:
    """Average deterministic per-revision importances for one study or dataset."""
    subsets, reason = _eligible_revision_subsets(study, dataset=dataset)
    if reason is not None or not subsets:
        total_trials = sum(len(trials) for _revision, trials in subsets)
        return ImportanceResult({}, None, total_trials, reason)

    totals: dict[str, float] = {}
    support: dict[str, int] = {}
    revisions: list[str] = []
    total_trials = 0
    for revision, trials in subsets:
        target = None
        if dataset is not None:

            def target(trial: FrozenTrial, dataset: str = dataset) -> float:
                value = dataset_metric(trial, dataset, "objective")
                if value is None:
                    raise ValueError(f"Trial {trial.number} has no {dataset} objective.")
                return float(value)

        importances = cached_logical_importances(
            study,
            revision=revision,
            target=target,
            trials=trials,
            dataset=dataset,
        )
        if not importances:
            continue
        revisions.append(revision)
        trial_count = len(trials)
        total_trials += trial_count
        for name, importance in importances.items():
            totals[name] = totals.get(name, 0.0) + float(importance) * trial_count
            support[name] = support.get(name, 0) + trial_count

    if not totals or total_trials == 0:
        return ImportanceResult(
            {},
            None,
            0,
            "Optuna could not compute importances for any eligible revision subset",
        )

    averaged = {name: totals[name] / support[name] for name in totals}
    normalizer = sum(averaged.values())
    if normalizer > 0:
        averaged = {name: value / normalizer for name, value in averaged.items()}
    ordered = dict(sorted(averaged.items(), key=lambda item: item[1], reverse=True))
    return ImportanceResult(
        ordered,
        ", ".join(revisions),
        total_trials,
        subset_count=len(revisions),
    )


def _temporary_study_from_trials(
    study: optuna.Study,
    trials: Sequence[FrozenTrial],
) -> optuna.Study:
    """Create an in-memory study containing a homogeneous completed-trial subset."""
    temporary = optuna.create_study(direction=study.direction.name.lower())
    for trial in trials:
        temporary.add_trial(trial)
    return temporary


def _importance_cache_key(
    study: optuna.Study,
    *,
    revision: str,
    trials: Sequence[FrozenTrial],
    dataset: str | None,
) -> str:
    """Return a stable cache key for one deterministic fANOVA subset."""
    trial_signature = [
        {
            "number": int(trial.number),
            "params": logical_trial_params(trial),
            "value": (
                dataset_metric(trial, dataset, "objective") if dataset is not None else trial.value
            ),
        }
        for trial in sorted(trials, key=lambda item: int(item.number))
    ]
    payload = {
        "dataset": dataset,
        "direction": study.direction.name,
        "evaluator": IMPORTANCE_EVALUATOR_NAME,
        "revision": revision,
        "study": study.study_name,
        "trials": trial_signature,
    }
    serialized = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cached_logical_importances(
    study: optuna.Study,
    *,
    revision: str,
    trials: Sequence[FrozenTrial],
    dataset: str | None = None,
    target: object | None = None,
) -> dict[str, float]:
    """Return deterministic logical importances from cache or fANOVA."""
    key = _importance_cache_key(
        study,
        revision=revision,
        trials=trials,
        dataset=dataset,
    )
    cache = _load_importance_cache()
    cached = cache.get(key)
    if cached is not None:
        return dict(cached)
    importances = logical_importances(
        safe_importances(
            study,
            evaluator=MeanDecreaseImpurityImportanceEvaluator(seed=13),
            target=target,
            trials=trials,
        ),
    )
    cache[key] = importances
    _store_importance_cache(cache)
    return importances


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
    """Return deterministic mean importances across eligible revision subsets."""
    return _aggregate_revision_importances(study)


def dashboard_importances(study: optuna.Study) -> dict[str, float]:
    """Return reliable dashboard-like importances for figure exporters."""
    return dashboard_importance_result(study).importances


def fanova_importances(study: optuna.Study) -> dict[str, float]:
    """Return deterministic fANOVA importances for sensitivity checking."""
    revision, trials, reason = _homogeneous_revision_subset(study)
    if reason is not None or not trials:
        return {}
    return cached_logical_importances(
        study,
        revision=revision or "",
        trials=trials,
    )


def dataset_importance_result(study: optuna.Study, dataset: str) -> ImportanceResult:
    """Return deterministic mean importances for one dataset-local objective."""
    return _aggregate_revision_importances(study, dataset=dataset)


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
    display_name = display_study_name(study.study_name)
    if objective_metric in VALIDATION_CRRU_METRIC_ALIASES and len(datasets) == 1:
        return (0, display_name)
    if objective_metric in VALIDATION_CRRU_METRIC_ALIASES:
        return (1, display_name)
    return (2, display_name)


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
            f"- Fresh informative target count: `{fresh_informative}` "
            "(fresh COMPLETE + real fresh PRUNED).",
            f"- Duplicate-skip pruned trials excluded from that target count: "
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
    ]
    if result.subset_count > 1:
        lines.extend(
            [
                f"Subset: `{result.trial_count}` fresh completed trial(s) across "
                f"`{result.subset_count}` eligible search-space revision subsets.",
                f"Revisions: `{result.revision}`.",
                "Aggregation: trial-weighted mean of deterministic per-revision fANOVA "
                "importances; parameters absent from a revision are not counted as zero.",
            ],
        )
    else:
        lines.append(
            f"Subset: `{result.trial_count}` fresh completed trial(s) from "
            f"search-space revision `{result.revision}`.",
        )
    lines.extend(
        [
            "",
            "| Rank | Parameter | Importance |",
            "|---:|---|---:|",
        ],
    )
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
                "| Rank | Trial | Origin | ValidationCRRU@20_40 | Study objective | "
                "PosthocCRRU@20 | PosthocCRRU@40 | Time/epoch (s) | Peak VRAM (MB) | Batch | "
                "Revision | Effective config |",
                "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ],
        )
        for rank, trial in enumerate(ranked[:top_n], start=1):
            trial_crru = posthoc_crru.get(dataset, {}).get(int(trial.number), {})
            lines.append(
                f"| {rank} | {trial.number} | {trial_origin(trial)} | "
                f"{format_float(dataset_metric(trial, dataset, VALIDATION_CRRU_METRIC))} | "
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
                f"Revision-mean Optuna importances for `{dataset}` objective",
                result,
            ),
        )
    return lines


def render_global_best_trials(
    studies: Sequence[optuna.Study],
) -> list[str]:
    """Render cross-study best completed trials by dataset and objective."""
    candidates = dataset_trial_candidates(studies)
    lines = [
        "## Global best trials by dataset",
        "",
        "This section scans every loaded study in the Optuna storage. Study names are "
        "provenance only; selection is by reconstructed `ValidationCRRU@20_40` "
        "from validation metrics, raw PyG AveragePopularity, stored or reconstructed "
        "largest train item count, time/epoch, and peak VRAM when all required inputs "
        "are available.",
        "",
        f"- Loaded studies: `{len(studies)}`",
        f"- Dataset-scoped completed candidates: `{len(candidates)}`",
        "",
    ]
    if not candidates:
        lines.extend(["No completed dataset-scoped candidates found.", ""])
        return lines

    posthoc_crru = compute_global_posthoc_validation_crru(candidates)
    grouped: dict[tuple[str, str, str], dict[str, list[DatasetTrialCandidate]]] = {}
    for candidate in candidates:
        objective_key = (
            candidate.objective_metric,
            candidate.objective_split,
            candidate.study_direction,
        )
        grouped.setdefault(objective_key, {}).setdefault(candidate.dataset, []).append(candidate)

    for objective_key, by_dataset in sorted(
        grouped.items(), key=lambda item: objective_group_sort_key(item[0])
    ):
        metric, split, direction = objective_key
        lines.extend(
            [
                f"### Objective: `{objective_split_metric_label(split, metric)}`",
                "",
                f"Direction: `{direction}`.",
                "",
                "| Dataset | Best objective | Source study | Trial | Origin | Revision | "
                "Source objective | Study objective | Global PosthocCRRU@20 | "
                "Global PosthocCRRU@40 | "
                "NDCG@40 | Recall@40 | Time/epoch (s) | Peak VRAM (MB) | Batch | "
                "Effective config |",
                "|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ],
        )
        for dataset, dataset_candidates in sorted(by_dataset.items()):
            best = sorted(dataset_candidates, key=candidate_sort_key)[0]
            trial_crru = posthoc_crru.get(
                (best.study_name, int(best.trial.number), dataset),
                {},
            )
            source_objective_label = objective_split_metric_label(
                best.source_objective_split,
                best.source_objective_metric,
            )
            lines.append(
                f"| {dataset} | {format_float(best.objective_value)} | "
                f"`{best.study_display_name}` | {best.trial.number} | "
                f"{trial_origin(best.trial)} | "
                f"`{format_revision_label(trial_search_revision(best.trial))}` | "
                f"`{source_objective_label}` | "
                f"{format_float(best.study_objective_value)} | "
                f"{format_float(trial_crru.get(20))} | "
                f"{format_float(trial_crru.get(40))} | "
                f"{format_float(dataset_metric(best.trial, dataset, 'NDCG@40'))} | "
                f"{format_float(dataset_metric(best.trial, dataset, 'Recall@40'))} | "
                f"{format_float(trial_epoch_time_s(best.trial, dataset), digits=2)} | "
                f"{format_float(attr_float(best.trial, f'{dataset}.peak_vram_mb'), digits=1)} | "
                f"{format_float(attr_float(best.trial, f'{dataset}.batch_size'), digits=0)} | "
                f"`{format_effective_params(best.trial, dataset=dataset)}` |",
            )
        lines.append("")
    return lines


def render_study_report(study: optuna.Study, *, top_n: int) -> str:
    """Render one Optuna study as markdown."""
    trials = completed_trials(study)
    state_counts = Counter(trial.state.name.lower() for trial in study.trials)
    objective_metric = "-"
    objective_split = "-"
    if trials:
        objective_metric = trial_objective_metric(trials[0])
        objective_split = trial_objective_split(trials[0])

    displayed_study_name = display_study_name(study.study_name)
    lines = [
        f"## Study: `{displayed_study_name}`",
        "",
        f"- Direction: `{study.direction.name.lower()}`",
        f"- Objective: `{objective_split_metric_label(objective_split, objective_metric)}`",
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
                "| Dataset | Source objective | NDCG@20 | Recall@20 | Hit@20 | Pers@20 | "
                "AvgPop@20 | "
                "NDCG@40 | Recall@40 | Hit@40 | Pers@40 | AvgPop@40 | "
                "ValidationCRRU@20 | ValidationCRRU@40 | ValidationCRRU@20_40 | PosthocCRRU@20 | "
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

    if INCLUDE_OPTUNA_IMPORTANCE_TABLES:
        revision_mean_importances = dashboard_importance_result(study)
        lines.extend(
            render_importance_result(
                "Revision-mean Optuna importances",
                revision_mean_importances,
            ),
        )
        fanova = fanova_importances(study)
        if fanova and fanova != revision_mean_importances.importances:
            lines.extend(
                render_importance_table(
                    "Deterministic fANOVA importance sensitivity",
                    fanova,
                ),
            )
        if all_datasets:
            lines.extend(render_dataset_importance_tables(study, datasets=all_datasets))
    else:
        lines.extend(
            [
                "### Optuna importances",
                "",
                "Skipped in the default report refresh because Optuna post-hoc "
                "importance evaluation is too slow for routine reporting. "
                "Feature-subset conclusions use completed profile metrics and deltas.",
                "",
            ],
        )

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
            "Promote dataset-local candidates into formal profiles only after checking runtime "
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
                crru_proxy = dataset_metric(trial, dataset, VALIDATION_CRRU_METRIC)
                lines.append(
                    f"- `{dataset}` trial `{trial.number}`: ValidationCRRU@20_40 "
                    f"`{format_float(crru_proxy)}`; "
                    f"study objective "
                    f"`{format_float(float(trial.value) if trial.value is not None else None)}`; "
                    f"posthoc CRRU@20/40 "
                    f"`{format_float(trial_crru.get(20))}`/"
                    f"`{format_float(trial_crru.get(40))}`; "
                    f"time/epoch `{format_float(trial_epoch_time_s(trial, dataset), digits=2)}`s; "
                    f"peak VRAM "
                    f"`{format_float(attr_float(trial, f'{dataset}.peak_vram_mb'), digits=1)}`MB; "
                    f"batch "
                    f"`{format_float(attr_float(trial, f'{dataset}.batch_size'), digits=0)}`; "
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


def render_report(
    studies: Sequence[optuna.Study],
    *,
    storage: str,
    top_n: int,
    feature_subset_rows: Sequence[Mapping[str, object]] | None = None,
) -> str:
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
        "- The default report scans every useful Optuna study in storage. Use `--study-name` "
        "only for narrow forensic inspection of excluded historical studies.",
        "- Historical Optuna rows stored under the former validation-CRRU objective alias "
        "are loaded as storage aliases and reported with the current `ValidationCRRU*` names.",
        "- `ValidationCRRU@20_40` is the single default ranking and figure metric. "
        "It combines validation ranking, popularity-aware personalization, peak VRAM, "
        "and time/epoch.",
        "- Accuracy-optimized studies are still loaded as provenance, but their candidates "
        "are compared through reconstructed `ValidationCRRU@20_40` so the report "
        "has one cross-study source of truth.",
        "- Exact thesis CRRU is an absolute per-trial utility computed from that trial's "
        "validation/test metrics, raw PyG AveragePopularity, stored or reconstructed "
        "largest train item count, peak VRAM, and seconds/epoch; it does not normalize "
        "against other rows.",
        "- Historical `val NDCG@40` studies are excluded from default thesis outputs because "
        "they are small early screens and are less useful than the CRRU and accuracy studies.",
        "- Reports reconstruct canonical `ValidationCRRU@20_40` when raw metrics, "
        "resource attrs, and either a logged or reconstructable largest train item "
        "count are available.",
        "- Default figures aggregate loaded trials by dataset under `ValidationCRRU@20_40`. "
        "Study names are provenance fields in the tables, not separate figure facets.",
        "- Inactive CAGRA-only knobs are omitted from overview figure heatmaps, while historical "
        "CAGRA trial provenance remains visible in the report tables.",
        "- Lower raw PyG AveragePopularity and higher personalization are supporting "
        "diagnostics, not standalone fairness or exposure-effect estimates; CRRU "
        "log-normalizes raw ARP internally.",
        "- Optuna parameter importances are post-hoc diagnostics computed from stored "
        "trial parameters and objectives. Deterministic importance subsets are cached "
        f"in `{IMPORTANCE_CACHE_PATH}` and overwritten on cache-key changes.",
        "- Optuna RDB storage is the canonical owner for search trials; the thesis SQLite "
        "database keeps formal experiment and training logs.",
        "- Current multi-dataset searches expand into one independent study per "
        "dataset; older `*-all-*` studies are historical global-mean screens.",
        "- Removed CAGRA graph-augmentation studies are included for provenance when "
        "present in storage, but they should not be promoted as active thesis candidates "
        "without a current-code rerun.",
        "- `--trials N` means N fresh informative finished trials for the current "
        "`search_space_revision`: COMPLETE plus real PRUNED, excluding FAIL/RUNNING, "
        "historically imported rows, duplicate-skip prunes, and other revision hashes. "
        "Imported rows are reported separately for provenance.",
        "- Hyperparameter importances are computed as dataset-local, trial-weighted means "
        "over eligible homogeneous, non-imported `search_space_revision` subsets with "
        "enough completed trials. Parameters absent from a revision are not counted as "
        "zero, and revision groups with too little support are omitted rather than "
        "reported as stable evidence.",
        "- The current second-pass grid tunes active EDGRec loss weights and schedule "
        "knobs only; inactive DirectAU/IPW-only weights remain out of the default "
        "search to avoid changing the thesis model family without a separate ablation.",
        "- Compact EDGRec settings are search priors and formal candidates, not final "
        "thesis selections. Current evidence supports compact candidates most clearly "
        "for KuaiRec_v2, "
        "allows a MovieLens1M near-parity/speed candidate with popularity diagnostics, "
        "and treats KuaiRand1K as required randomized-exposure evidence whose current "
        "accuracy remains unresolved. AmazonBook is excluded only "
        "from the shared compact default queue: it still needs a dataset-specific "
        "compact-vs-deep_features EDGRec comparison against the LightGCN-paper accuracy "
        "baseline before thesis promotion.",
        "- Search profile labels such as `no_context_no_features` are historical/internal "
        "Optuna mechanism labels. They are not public ablation variants unless they appear "
        "in `experiments/ablation_configs.py`.",
        "- Schedule-conditioned parameters are sampled only when they affect the resolved "
        "training config: ramp rates for `linear_ramp`, start epochs for `phased`.",
        "- Candidate rows print an effective-config view, not just raw Optuna sampled "
        "parameters. Future trials store the resolved dataset config directly; older "
        "trials fall back to sampled params plus any runtime attrs present in Optuna.",
        "",
        "## CRRU Reporting Utility",
        "",
    ]
    for index, line in enumerate(CRRU_REPORT_FORMULA_LINES):
        prefix = "**" if index == 0 else "- "
        suffix = "**" if index == 0 else ""
        clean_line = line.strip()
        lines.append(f"{prefix}{clean_line}{suffix}")
    lines.append("")
    lines.extend(render_global_best_trials(studies))
    lines.extend(render_side_feature_analysis(studies, rows=feature_subset_rows))
    lines.extend(
        [
            "## Paper-ready figures",
            "",
            "Generated by `uv run scripts/export_optuna_figures.py`. The default exporter "
            "writes one unified validation CRRU PNG figure set and removes stale generated "
            "figure files, including older PNG/HTML artifacts, from the figure directory. "
            "The canonical Optuna selection metric is `ValidationCRRU@20_40`. "
            "Accuracy-only and historical NDCG figure sets are not generated by default.",
            "",
            "The importance heatmap is a global overview diagnostic: each row "
            "aggregates all loaded completed trials for one dataset/objective and uses normalized "
            "univariate association scores. Use the study tables, not the overview heatmaps, for "
            "strict same-revision importance claims. Colored cells always print a numeric "
            "association value; gray cells mean no detected association among the displayed "
            "top parameters.",
            "The selection-frontier figure summarizes the trade-off behind each selected "
            "trial: ranking-accuracy component on x, popularity-aware personalization "
            "component on y, "
            "resource-efficiency percentile within the same dataset as color, and a gold "
            "star for the selected trial. Colors are not comparable as absolute efficiency "
            "values across datasets.",
            "The component-response figure plots each absolute reconstructed CRRU component "
            "score against absolute `ValidationCRRU@20_40`. Use it to inspect response shape "
            "and selected-trial placement, not as controlled attribution.",
            "The component-correlation figure reports Spearman rank associations between "
            "`ValidationCRRU@20_40` and its reconstructed ranking-accuracy, "
            "popularity-aware personalization, and resource components. Treat it as a "
            "descriptive association diagnostic, not a controlled attribution estimate.",
            "Absolute resource-efficiency component values can cluster below 0.3 because "
            "the search-time score uses inverse-log transforms of time/epoch and peak VRAM. "
            "The default final CRRU exponent for this component is 0.15, so low absolute "
            "efficiency values do not by themselves imply that resource cost dominates the "
            "selection.",
            "The branch-depth heatmap uses completed trials because pruned trials do not "
            "have a comparable final validation objective and cannot be selected as global "
            "best. Cells with an asterisk after `n` contain fewer than 10 completed trials "
            "and should be treated as low-support diagnostics.",
            "",
        ],
    )
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
            "colored by the selected validation objective, one panel per dataset.",
            "- Schedule ablation strip: `linear_ramp` vs `phased`, with active ramp/start "
            "knobs annotated beside each dataset's best trial.",
            "- Top-candidate effective-config strip: the selected LR, branch depths, fanout, "
            "batch size, schedule, and loss weights for the best trial in each dataset.",
            "",
        ],
    )
    lines.extend(
        [
            "## Studies",
            "",
            "Detailed per-study sections below are provenance and diagnostics. Use the "
            "global best section above as the cross-study candidate source of truth.",
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
    feature_subset_rows = build_feature_subset_result_rows(studies)
    report = render_report(
        studies,
        storage=args.storage,
        top_n=args.top_n,
        feature_subset_rows=feature_subset_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    write_feature_subset_search_reports(studies, rows=feature_subset_rows)
    print(f"Wrote Optuna optimization report to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
