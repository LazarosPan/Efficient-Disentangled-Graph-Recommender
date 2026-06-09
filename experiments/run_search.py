#!/usr/bin/env python
"""Optuna search controller for configured U-CaGNN spaces."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from scripts._workflow_helpers import configure_cli_logging
from src.utils.cli_parsers import (
    benchmark_dataset_lookup_keys,
    normalize_benchmark_datasets_arg,
    resolve_benchmark_datasets,
)
from src.utils.config import BENCHMARK_CONFIG_FIELDS, UCaGNNConfig
from src.utils.experiment_logger import ExperimentLogger
from src.utils.project_paths import THESIS_DB_PATH

from experiments.cli_parsers import build_search_parser
from experiments.recipes import (
    get_formal_profile,
    get_search_space,
    resolve_profile_num_neighbors,
    search_space_names,
)
from experiments.run_experiment import (
    build_benchmark_config_inputs,
    build_config,
    normalize_benchmark_config_overrides,
    run_experiment,
)

logger = logging.getLogger("ucagnn.search")

DEFAULT_STORAGE = "sqlite:///results/optuna_studies.db"
DEFAULT_OBJECTIVE_METRIC = "NDCG@40"
DEFAULT_OBJECTIVE_SPLIT = "val"
DEFAULT_MAX_EPOCHS = 80
DEFAULT_TRIALS = 40
SEARCH_PRESET = "ucagnn"
SEARCH_PARAMETER_FIELDS = frozenset(
    {
        "lr",
        "weight_decay",
        "lr_scheduler",
        "lr_scheduler_factor",
        "grad_clip_norm",
        "num_neighbors",
        "interest_gnn_layers",
        "conformity_gnn_layers",
        "dropout",
        "score_mix_min_weight",
        "loss_weight_independence",
        "loss_weight_contrastive",
        "loss_weight_popularity",
        "hard_negative_ratio",
        "dice_sampler_margin",
        "use_popularity_head",
        "use_features",
    },
)
SUPPORTED_PARAMETER_TYPES = frozenset(("float", "int", "categorical", "fanout"))
TRIAL_ATTRIBUTE_TRAIN_METRICS = (
    "training_time_s",
    "peak_vram_mb",
    "gpu_utilization_pct",
    "train_avg_gpu_utilization_pct",
    "max_gpu_utilization_pct",
    "train_max_gpu_utilization_pct",
    "train_peak_vram_allocated_mb",
    "train_peak_vram_reserved_mb",
    "train_peak_gpu_memory_used_mb",
)
TRIAL_ATTRIBUTE_TEST_PREFIXES = (
    "interest_branch_",
    "conformity_branch_",
    "score_mix_",
    "interest_contribution@",
    "conformity_contribution@",
    "context_contribution@",
)


@dataclass(frozen=True)
class ObjectiveSpec:
    """Resolved validation objective contract for one search space."""

    metric: str = DEFAULT_OBJECTIVE_METRIC
    split: str = DEFAULT_OBJECTIVE_SPLIT
    direction: str = "maximize"


@dataclass(frozen=True)
class SearchSpaceSpec:
    """Resolved catalog search-space definition."""

    name: str
    description: str
    base_profile: str
    datasets: tuple[str, ...]
    objective: ObjectiveSpec
    max_epochs: int
    trials: int
    config_overrides: dict[str, Any]
    parameters: dict[str, dict[str, Any]]


def _slugify_fragment(raw: object) -> str:
    """Return a filesystem/identifier-safe fragment for generated batch ids."""
    normalized = "".join(
        character.lower() if str(character).isalnum() else "-" for character in str(raw)
    )
    return "-".join(part for part in normalized.split("-") if part) or "search"


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe representation for Optuna user attributes."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _require_mapping(raw: object, *, field_name: str) -> Mapping[str, Any]:
    """Validate and return a mapping payload."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be an object.")
    return raw


def _validate_catalog_fields(
    payload: Mapping[str, Any],
    *,
    field_name: str,
    allowed_fields: set[str],
) -> None:
    """Reject catalog payload keys that cannot enter the shared config path."""
    unsupported = sorted(set(payload) - allowed_fields)
    if unsupported:
        raise ValueError(
            f"{field_name} contains unsupported config fields: {', '.join(unsupported)}",
        )


def _normalize_objective(raw_objective: object) -> ObjectiveSpec:
    """Resolve the objective spec, enforcing validation-only search."""
    if raw_objective is None:
        return ObjectiveSpec()
    if isinstance(raw_objective, str):
        return ObjectiveSpec(metric=raw_objective)
    objective = _require_mapping(raw_objective, field_name="objective")
    metric = str(objective.get("metric", DEFAULT_OBJECTIVE_METRIC))
    raw_split = str(objective.get("split", DEFAULT_OBJECTIVE_SPLIT)).lower()
    split = "val" if raw_split == "validation" else raw_split
    if split != "val":
        raise ValueError("Optuna search objectives must use validation metrics, never test.")
    direction = str(objective.get("direction", "maximize")).lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("objective.direction must be either 'maximize' or 'minimize'.")
    return ObjectiveSpec(metric=metric, split=split, direction=direction)


def _validate_numeric_parameter(field_name: str, spec: Mapping[str, Any]) -> None:
    """Validate an int/float Optuna parameter spec."""
    if "low" not in spec or "high" not in spec:
        raise ValueError(f"parameters.{field_name} must define low and high.")
    low = float(spec["low"])
    high = float(spec["high"])
    if low > high:
        raise ValueError(f"parameters.{field_name}.low must be <= high.")
    if bool(spec.get("log", False)) and low <= 0:
        raise ValueError(f"parameters.{field_name} uses log scale, so low must be > 0.")
    if "step" in spec and float(spec["step"]) <= 0:
        raise ValueError(f"parameters.{field_name}.step must be > 0 when provided.")


def _validate_categorical_parameter(field_name: str, spec: Mapping[str, Any]) -> None:
    """Validate a categorical Optuna parameter spec."""
    choices = spec.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"parameters.{field_name}.choices must be a non-empty list.")
    labels = [json.dumps(choice, sort_keys=True, separators=(",", ":")) for choice in choices]
    if len(set(labels)) != len(labels):
        raise ValueError(f"parameters.{field_name}.choices contains duplicate values.")


def _validate_fanout_parameter(field_name: str, spec: Mapping[str, Any]) -> None:
    """Validate a depth-keyed ``num_neighbors`` search parameter."""
    if field_name != "num_neighbors":
        raise ValueError("fanout parameter type is only supported for num_neighbors.")
    choices_by_depth = spec.get("choices_by_depth")
    if not isinstance(choices_by_depth, Mapping) or not choices_by_depth:
        raise ValueError("parameters.num_neighbors.choices_by_depth must be a non-empty object.")
    for raw_depth, raw_choices in choices_by_depth.items():
        depth = int(raw_depth)
        if depth < 1:
            raise ValueError("num_neighbors fan-out depth keys must be >= 1.")
        resolved = resolve_profile_num_neighbors({"num_neighbors": raw_choices})
        if resolved is None:
            raise ValueError(f"num_neighbors choices for depth {depth} cannot be empty.")
        for vector in resolved:
            if len(vector) != depth:
                raise ValueError(
                    f"num_neighbors choice {vector} does not match depth {depth}.",
                )


def _validate_parameter_specs(parameters: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate search parameter specs and return a plain dict copy."""
    _validate_catalog_fields(
        parameters,
        field_name="parameters",
        allowed_fields=set(SEARCH_PARAMETER_FIELDS),
    )
    normalized: dict[str, dict[str, Any]] = {}
    for field_name, raw_spec in parameters.items():
        spec = dict(_require_mapping(raw_spec, field_name=f"parameters.{field_name}"))
        parameter_type = str(spec.get("type", "")).lower()
        if parameter_type not in SUPPORTED_PARAMETER_TYPES:
            raise ValueError(
                (
                    f"parameters.{field_name}.type must be one of "
                    f"{', '.join(sorted(SUPPORTED_PARAMETER_TYPES))}."
                ),
            )
        if parameter_type in {"float", "int"}:
            _validate_numeric_parameter(field_name, spec)
        elif parameter_type == "categorical":
            _validate_categorical_parameter(field_name, spec)
        elif parameter_type == "fanout":
            _validate_fanout_parameter(field_name, spec)
        normalized[field_name] = spec
    return normalized


def resolve_search_space(
    space_name: str,
    *,
    dataset: str | None = None,
) -> SearchSpaceSpec:
    """Resolve and validate one catalog search-space definition."""
    raw_space = get_search_space(space_name)
    base_profile_name = raw_space.get("base_profile")
    if not isinstance(base_profile_name, str) or not base_profile_name:
        raise ValueError(f"search_spaces.{space_name}.base_profile must name a formal profile.")

    base_profile = get_formal_profile(base_profile_name)
    presets = list(base_profile["matrix"]["presets"])
    if presets != [SEARCH_PRESET]:
        raise ValueError(
            (
                f"search space '{space_name}' must use a U-CaGNN-only base profile; "
                f"got presets={presets!r}."
            ),
        )

    raw_datasets = raw_space.get("datasets") or base_profile["matrix"]["datasets"]
    datasets = resolve_benchmark_datasets(normalize_benchmark_datasets_arg(raw_datasets))
    if dataset is not None:
        if dataset not in datasets:
            raise ValueError(
                f"dataset '{dataset}' is not part of search space '{space_name}': {datasets}.",
            )
        datasets = [dataset]
    if not datasets:
        raise ValueError(f"search space '{space_name}' must resolve at least one dataset.")

    objective = _normalize_objective(raw_space.get("objective"))
    max_epochs = int(raw_space.get("max_epochs") or DEFAULT_MAX_EPOCHS)
    if max_epochs < 1:
        raise ValueError(f"search space '{space_name}' max_epochs must be >= 1.")
    trials = int(raw_space.get("trials") or DEFAULT_TRIALS)
    if trials < 1:
        raise ValueError(f"search space '{space_name}' trials must be >= 1.")

    config_overrides = dict(raw_space.get("config_overrides", {}))
    _validate_catalog_fields(
        config_overrides,
        field_name="config_overrides",
        allowed_fields=set(BENCHMARK_CONFIG_FIELDS),
    )
    parameters = _validate_parameter_specs(dict(raw_space.get("parameters", {})))
    if not parameters:
        raise ValueError(f"search space '{space_name}' must define at least one parameter.")

    return SearchSpaceSpec(
        name=space_name,
        description=str(raw_space.get("description", "")),
        base_profile=base_profile_name,
        datasets=tuple(datasets),
        objective=objective,
        max_epochs=max_epochs,
        trials=trials,
        config_overrides=config_overrides,
        parameters=parameters,
    )


def _build_base_benchmark_args(
    search_space: SearchSpaceSpec,
    *,
    device: str,
    data_dir: str,
) -> dict[str, Any]:
    """Return the fixed benchmark-style config mapping for a search space."""
    base_profile = get_formal_profile(search_space.base_profile)
    raw_overrides = dict(base_profile["config_overrides"])
    raw_overrides.update(search_space.config_overrides)
    raw_overrides["epochs"] = search_space.max_epochs
    raw_overrides.setdefault("use_early_stopping", True)
    benchmark_args = normalize_benchmark_config_overrides(raw_overrides)
    benchmark_args["device"] = device
    benchmark_args["data_dir"] = data_dir
    return benchmark_args


def _single_lr_scheduler(benchmark_args: Mapping[str, Any]) -> str:
    """Return one concrete scheduler for a search trial."""
    raw_scheduler = benchmark_args.get("lr_scheduler") or UCaGNNConfig().lr_scheduler
    if isinstance(raw_scheduler, list):
        if len(raw_scheduler) != 1:
            raise ValueError("Search spaces must resolve one lr_scheduler per trial.")
        raw_scheduler = raw_scheduler[0]
    if raw_scheduler == "all":
        raise ValueError("Search spaces cannot use lr_scheduler='all'.")
    return str(raw_scheduler)


def _single_graph_policy(benchmark_args: Mapping[str, Any]) -> str:
    """Return one concrete graph policy for a search trial."""
    graph_options = benchmark_args.get("graph_policy_options")
    if isinstance(graph_options, list) and graph_options:
        if len(graph_options) != 1:
            raise ValueError("Search spaces must resolve one graph_policy per trial.")
        return str(graph_options[0])
    graph_policy = benchmark_args.get("graph_policy")
    return str(graph_policy) if graph_policy is not None else UCaGNNConfig().graph_policy


def _single_preprocessing_preset(benchmark_args: Mapping[str, Any]) -> str | None:
    """Return one concrete preprocessing preset for a search trial."""
    preprocessing_options = benchmark_args.get("preprocessing_preset_options")
    if isinstance(preprocessing_options, list) and preprocessing_options:
        if len(preprocessing_options) != 1:
            raise ValueError("Search spaces must resolve one preprocessing_preset per trial.")
        return str(preprocessing_options[0])
    preprocessing_preset = benchmark_args.get("preprocessing_preset")
    return None if preprocessing_preset is None else str(preprocessing_preset)


def _select_num_neighbors(
    benchmark_args: Mapping[str, Any],
    *,
    dataset: str,
) -> list[int]:
    """Return one concrete fan-out vector for a dataset-specific search run."""
    raw_num_neighbors = benchmark_args.get("num_neighbors")
    if isinstance(raw_num_neighbors, Mapping):
        for lookup_key in benchmark_dataset_lookup_keys(dataset):
            selected = raw_num_neighbors.get(lookup_key)
            if selected is None:
                continue
            resolved = resolve_profile_num_neighbors({"num_neighbors": selected})
            if resolved:
                return list(resolved[0])
        available = ", ".join(sorted(str(key) for key in raw_num_neighbors))
        raise ValueError(
            f"No num_neighbors entry matches dataset '{dataset}'. Available keys: {available}",
        )

    resolved = resolve_profile_num_neighbors({"num_neighbors": raw_num_neighbors})
    if resolved:
        return list(resolved[0])
    return list(UCaGNNConfig().num_neighbors)


def build_search_config_inputs(
    search_space: SearchSpaceSpec,
    *,
    dataset: str,
    sampled_overrides: Mapping[str, Any] | None = None,
    device: str,
    data_dir: str,
) -> dict[str, Any]:
    """Build one trial's config-input mapping through the benchmark bridge."""
    benchmark_args = _build_base_benchmark_args(
        search_space,
        device=device,
        data_dir=data_dir,
    )
    if sampled_overrides:
        benchmark_args.update(sampled_overrides)

    return build_benchmark_config_inputs(
        benchmark_args,
        dataset=dataset,
        preset=SEARCH_PRESET,
        lr_scheduler=_single_lr_scheduler(benchmark_args),
        num_neighbors=_select_num_neighbors(benchmark_args, dataset=dataset),
        preprocessing_preset=_single_preprocessing_preset(benchmark_args),
        graph_policy=_single_graph_policy(benchmark_args),
    )


def build_search_config(
    search_space: SearchSpaceSpec,
    *,
    dataset: str,
    sampled_overrides: Mapping[str, Any] | None = None,
    device: str,
    data_dir: str,
) -> UCaGNNConfig:
    """Resolve one concrete U-CaGNN config for a search trial."""
    return build_config(
        build_search_config_inputs(
            search_space,
            dataset=dataset,
            sampled_overrides=sampled_overrides,
            device=device,
            data_dir=data_dir,
        ),
    )


def _choice_labels(choices: list[Any]) -> tuple[list[str], dict[str, Any]]:
    """Return stable labels for possibly structured categorical choices."""
    labels: list[str] = []
    values_by_label: dict[str, Any] = {}
    for choice in choices:
        label = json.dumps(choice, sort_keys=True, separators=(",", ":"))
        labels.append(label)
        values_by_label[label] = choice
    return labels, values_by_label


def _suggest_categorical_value(
    trial: optuna.Trial,
    field_name: str,
    choices: list[Any],
) -> Any:
    """Suggest a categorical value without passing structured choices to Optuna."""
    if all(choice is None or isinstance(choice, str | int | float | bool) for choice in choices):
        return trial.suggest_categorical(field_name, choices)
    labels, values_by_label = _choice_labels(choices)
    return values_by_label[trial.suggest_categorical(field_name, labels)]


def _suggest_parameter_value(
    trial: optuna.Trial,
    field_name: str,
    spec: Mapping[str, Any],
) -> Any:
    """Suggest one non-fanout parameter value from a validated spec."""
    parameter_type = str(spec["type"]).lower()
    if parameter_type == "float":
        return trial.suggest_float(
            field_name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=float(spec["step"]) if "step" in spec else None,
        )
    if parameter_type == "int":
        return trial.suggest_int(
            field_name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if parameter_type == "categorical":
        return _suggest_categorical_value(trial, field_name, list(spec["choices"]))
    raise ValueError(f"Unsupported non-fanout parameter type: {parameter_type}")


def _suggest_fanout_value(
    trial: optuna.Trial,
    spec: Mapping[str, Any],
    *,
    base_config: UCaGNNConfig,
    sampled_overrides: Mapping[str, Any],
) -> list[int]:
    """Suggest ``num_neighbors`` for the active sampled branch depth.

    Optuna requires a stable categorical value space for each parameter name
    within a study. The active fan-out depth depends on sampled branch-layer
    overrides, so the storage parameter name includes depth while the returned
    config override remains the normal ``num_neighbors`` field.
    """
    interest_layers = int(
        sampled_overrides.get("interest_gnn_layers", base_config.interest_gnn_layers),
    )
    conformity_layers = int(
        sampled_overrides.get("conformity_gnn_layers", base_config.conformity_gnn_layers),
    )
    depth = max(interest_layers, conformity_layers)
    choices_by_depth = _require_mapping(
        spec["choices_by_depth"],
        field_name="parameters.num_neighbors.choices_by_depth",
    )
    raw_choices = choices_by_depth.get(str(depth)) or choices_by_depth.get(depth)
    if raw_choices is None:
        raise ValueError(f"num_neighbors has no choices for active depth {depth}.")
    choices = resolve_profile_num_neighbors({"num_neighbors": raw_choices})
    if choices is None:
        raise ValueError(f"num_neighbors choices for active depth {depth} are empty.")
    labels, values_by_label = _choice_labels(choices)
    optuna_param_name = f"num_neighbors_depth_{depth}"
    selected_label = trial.suggest_categorical(optuna_param_name, labels)
    trial.set_user_attr("num_neighbors_param", optuna_param_name)
    return list(values_by_label[selected_label])


def suggest_trial_overrides(
    trial: optuna.Trial,
    search_space: SearchSpaceSpec,
    *,
    base_config: UCaGNNConfig,
) -> dict[str, Any]:
    """Suggest one trial override dict over existing ``UCaGNNConfig`` fields."""
    sampled: dict[str, Any] = {}
    for field_name, spec in search_space.parameters.items():
        if str(spec["type"]).lower() == "fanout":
            continue
        sampled[field_name] = _suggest_parameter_value(trial, field_name, spec)
    for field_name, spec in search_space.parameters.items():
        if str(spec["type"]).lower() != "fanout":
            continue
        sampled[field_name] = _suggest_fanout_value(
            trial,
            spec,
            base_config=base_config,
            sampled_overrides=sampled,
        )
    return sampled


def _best_validation_metrics(
    result: Mapping[str, Any],
    *,
    metric_name: str,
    direction: str = "maximize",
) -> dict[str, float]:
    """Return validation metrics from the epoch that optimizes ``metric_name``."""
    history = result.get("history")
    if not isinstance(history, Mapping):
        raise ValueError("Experiment result did not include a history mapping.")
    raw_val_metrics = history.get("val_metrics")
    if not isinstance(raw_val_metrics, list) or not raw_val_metrics:
        raise ValueError("Experiment result did not include validation metrics.")

    candidates: list[dict[str, float]] = []
    for raw_metrics in raw_val_metrics:
        if not isinstance(raw_metrics, Mapping) or metric_name not in raw_metrics:
            continue
        value = float(raw_metrics[metric_name])
        if math.isfinite(value):
            candidates.append(
                {
                    str(metric): float(metric_value)
                    for metric, metric_value in raw_metrics.items()
                    if math.isfinite(float(metric_value))
                },
            )
    if not candidates:
        raise ValueError(f"No finite validation {metric_name} values were produced.")

    reverse = direction == "maximize"
    return sorted(candidates, key=lambda metrics: metrics[metric_name], reverse=reverse)[0]


def extract_validation_objective(
    result: Mapping[str, Any],
    objective: ObjectiveSpec,
) -> float:
    """Extract the Optuna objective from validation metrics only."""
    best_metrics = _best_validation_metrics(
        result,
        metric_name=objective.metric,
        direction=objective.direction,
    )
    return float(best_metrics[objective.metric])


def _latest_train_metrics(exp_id: int) -> dict[str, float]:
    """Return train metrics from SQLite when the experiment row exists."""
    tracker = ExperimentLogger(db_path=str(THESIS_DB_PATH))
    try:
        return tracker.get_metrics_for_split(exp_id, split="train")
    finally:
        tracker.close()


def _finite_float_or_none(value: object) -> float | None:
    """Return finite numeric values as floats, otherwise None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _branch_diagnostics_from_result(result: Mapping[str, Any] | None) -> dict[str, object]:
    """Extract branch diagnostic metrics suitable for compact JSON storage."""
    if result is None:
        return {}
    test_metrics = result.get("test_metrics", {})
    if not isinstance(test_metrics, Mapping):
        return {}

    diagnostics: dict[str, object] = {}
    for metric_name, value in test_metrics.items():
        has_diagnostic_prefix = any(
            str(metric_name).startswith(prefix_)
            for prefix_ in TRIAL_ATTRIBUTE_TEST_PREFIXES
        )
        if has_diagnostic_prefix:
            diagnostics[str(metric_name)] = _json_safe(value)
    return diagnostics


def _best_validation_metrics_or_empty(
    result: Mapping[str, Any] | None,
    *,
    objective: ObjectiveSpec,
) -> dict[str, float]:
    """Return best validation metrics when a trial produced them."""
    if result is None:
        return {}
    try:
        return _best_validation_metrics(
            result,
            metric_name=objective.metric,
            direction=objective.direction,
        )
    except ValueError:
        return {}


def _log_search_trial_row(
    *,
    search_space: SearchSpaceSpec,
    study_name: str,
    trial_number: int,
    dataset: str,
    batch_id: str,
    sampled_overrides: Mapping[str, Any],
    state: str,
    result: Mapping[str, Any] | None = None,
    dataset_objective_value: float | None = None,
    objective_value: float | None = None,
    failure_reason: str | None = None,
) -> None:
    """Mirror one Optuna trial/dataset result into the thesis SQLite database."""
    best_val_metrics = _best_validation_metrics_or_empty(
        result,
        objective=search_space.objective,
    )
    exp_id = result.get("exp_id") if isinstance(result, Mapping) else None
    tracker = ExperimentLogger(db_path=str(THESIS_DB_PATH))
    try:
        tracker.log_optuna_search_trial(
            study_name=study_name,
            search_space=search_space.name,
            trial_number=trial_number,
            dataset=dataset,
            experiment_id=exp_id if isinstance(exp_id, int) else None,
            batch_id=batch_id,
            objective_metric=search_space.objective.metric,
            objective_split=search_space.objective.split,
            objective_direction=search_space.objective.direction,
            objective_value=objective_value,
            dataset_objective_value=dataset_objective_value,
            params=dict(_json_safe(sampled_overrides)),
            state=state,
            failure_reason=failure_reason,
            runtime_s=_finite_float_or_none(
                result.get("training_time_s") if isinstance(result, Mapping) else None,
            ),
            peak_vram_mb=_finite_float_or_none(
                result.get("peak_vram_mb") if isinstance(result, Mapping) else None,
            ),
            average_popularity_20=_finite_float_or_none(
                best_val_metrics.get("AveragePopularity@20"),
            ),
            average_popularity_40=_finite_float_or_none(
                best_val_metrics.get("AveragePopularity@40"),
            ),
            branch_diagnostics=_branch_diagnostics_from_result(result),
        )
    finally:
        tracker.close()


def _set_trial_attrs_from_result(
    trial: optuna.Trial,
    *,
    dataset: str,
    result: Mapping[str, Any],
    objective: ObjectiveSpec,
) -> None:
    """Store runtime, validation, and diagnostic metadata as Optuna attrs."""
    best_val_metrics = _best_validation_metrics(
        result,
        metric_name=objective.metric,
        direction=objective.direction,
    )
    prefix = f"{dataset}."
    trial.set_user_attr(prefix + "exp_id", result.get("exp_id"))
    trial.set_user_attr(prefix + "canonical_name", result.get("canonical_name"))
    trial.set_user_attr(prefix + "checkpoint_path", result.get("checkpoint_path"))
    trial.set_user_attr(prefix + "epochs_stopped_at", result.get("epochs_stopped_at"))
    trial.set_user_attr(prefix + "training_time_s", result.get("training_time_s"))
    trial.set_user_attr(prefix + "peak_vram_mb", result.get("peak_vram_mb"))

    for metric_name in ("NDCG@20", "NDCG@40", "AveragePopularity@20", "AveragePopularity@40"):
        if metric_name in best_val_metrics:
            trial.set_user_attr(prefix + f"val.{metric_name}", best_val_metrics[metric_name])

    exp_id = result.get("exp_id")
    if isinstance(exp_id, int):
        for metric_name, value in _latest_train_metrics(exp_id).items():
            if metric_name in TRIAL_ATTRIBUTE_TRAIN_METRICS:
                trial.set_user_attr(prefix + f"train.{metric_name}", value)

    test_metrics = result.get("test_metrics", {})
    if isinstance(test_metrics, Mapping):
        for metric_name, value in test_metrics.items():
            has_diagnostic_prefix = any(
                str(metric_name).startswith(prefix_)
                for prefix_ in TRIAL_ATTRIBUTE_TEST_PREFIXES
            )
            if has_diagnostic_prefix:
                trial.set_user_attr(prefix + f"test.{metric_name}", float(value))


def _trial_batch_id(study_name: str, trial_number: int) -> str:
    """Return the SQLite/MLflow batch id for one Optuna trial."""
    return f"optuna-{_slugify_fragment(study_name)}-trial-{trial_number}"


def _trial_change_note(
    *,
    search_space: SearchSpaceSpec,
    study_name: str,
    trial_number: int,
    sampled_overrides: Mapping[str, Any],
) -> str:
    """Return compact trial metadata for the existing change_note field."""
    return json.dumps(
        {
            "search_space": search_space.name,
            "study_name": study_name,
            "trial_number": trial_number,
            "sampled_params": _json_safe(sampled_overrides),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _objective_factory(
    search_space: SearchSpaceSpec,
    *,
    study_name: str,
    device: str,
    data_dir: str,
    enable_mlflow: bool,
    mlflow_tracking_uri: str | None,
    mlflow_experiment_name: str,
    overwrite_checkpoint: bool,
) -> Any:
    """Build the Optuna objective callable for one resolved search space."""

    def objective(trial: optuna.Trial) -> float:
        base_config = build_search_config(
            search_space,
            dataset=search_space.datasets[0],
            sampled_overrides=None,
            device=device,
            data_dir=data_dir,
        )
        sampled_overrides = suggest_trial_overrides(
            trial,
            search_space,
            base_config=base_config,
        )
        trial.set_user_attr("search_space", search_space.name)
        trial.set_user_attr("study_name", study_name)
        trial.set_user_attr("sampled_params", _json_safe(sampled_overrides))
        trial.set_user_attr("objective_metric", search_space.objective.metric)
        trial.set_user_attr("objective_split", search_space.objective.split)

        scores: list[float] = []
        completed_results: list[tuple[str, Mapping[str, Any], float]] = []
        batch_id = _trial_batch_id(study_name, trial.number)
        change_note = _trial_change_note(
            search_space=search_space,
            study_name=study_name,
            trial_number=trial.number,
            sampled_overrides=sampled_overrides,
        )
        for dataset in search_space.datasets:
            try:
                config = build_search_config(
                    search_space,
                    dataset=dataset,
                    sampled_overrides=sampled_overrides,
                    device=device,
                    data_dir=data_dir,
                )
                result = run_experiment(
                    config,
                    preset=SEARCH_PRESET,
                    save_checkpoint=False,
                    enable_mlflow=enable_mlflow,
                    mlflow_tracking_uri=mlflow_tracking_uri,
                    mlflow_experiment_name=mlflow_experiment_name,
                    batch_id=batch_id,
                    profile_name=search_space.name,
                    change_note=change_note,
                    checkpoint_every=0,
                    auto_resume=False,
                    overwrite_checkpoint=overwrite_checkpoint,
                    include_refined_diagnostics=False,
                    evaluate_test=False,
                )
                score = extract_validation_objective(result, search_space.objective)
                trial.set_user_attr(f"{dataset}.objective", score)
                _set_trial_attrs_from_result(
                    trial,
                    dataset=dataset,
                    result=result,
                    objective=search_space.objective,
                )
                _log_search_trial_row(
                    search_space=search_space,
                    study_name=study_name,
                    trial_number=trial.number,
                    dataset=dataset,
                    batch_id=batch_id,
                    sampled_overrides=sampled_overrides,
                    state="completed",
                    result=result,
                    dataset_objective_value=score,
                )
                completed_results.append((dataset, result, score))
                scores.append(score)
            except Exception as exc:
                failure_reason = f"{type(exc).__name__}: {exc}"
                trial.set_user_attr(f"{dataset}.failure_reason", failure_reason)
                trial.set_user_attr("failure_reason", failure_reason)
                _log_search_trial_row(
                    search_space=search_space,
                    study_name=study_name,
                    trial_number=trial.number,
                    dataset=dataset,
                    batch_id=batch_id,
                    sampled_overrides=sampled_overrides,
                    state="failed",
                    failure_reason=failure_reason,
                )
                raise

        if not scores:
            raise ValueError("Trial produced no dataset scores.")
        objective_value = float(sum(scores) / len(scores))
        trial.set_user_attr("objective_value", objective_value)
        for dataset, result, score in completed_results:
            _log_search_trial_row(
                search_space=search_space,
                study_name=study_name,
                trial_number=trial.number,
                dataset=dataset,
                batch_id=batch_id,
                sampled_overrides=sampled_overrides,
                state="completed",
                result=result,
                dataset_objective_value=score,
                objective_value=objective_value,
            )
        return objective_value

    return objective


def _ensure_storage_parent(storage: str) -> None:
    """Create the parent directory for local SQLite Optuna storage."""
    if not storage.startswith("sqlite:///") or storage.startswith("sqlite:///:memory:"):
        return
    raw_path = storage.removeprefix("sqlite:///").split("?", 1)[0]
    if not raw_path:
        return
    Path(raw_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def default_study_name(space_name: str, datasets: tuple[str, ...]) -> str:
    """Return the default study name for a space/dataset selection."""
    dataset_part = datasets[0] if len(datasets) == 1 else "all"
    return f"{space_name}-{dataset_part}"


def build_dry_run_payload(
    search_space: SearchSpaceSpec,
    *,
    study_name: str,
    storage: str,
    trials: int | None = None,
    device: str,
    data_dir: str,
) -> dict[str, Any]:
    """Resolve dry-run details without starting training."""
    base_configs: dict[str, dict[str, Any]] = {}
    for dataset in search_space.datasets:
        config = build_search_config(
            search_space,
            dataset=dataset,
            sampled_overrides=None,
            device=device,
            data_dir=data_dir,
        )
        base_configs[dataset] = dataclasses.asdict(config)

    return {
        "search_space": search_space.name,
        "description": search_space.description,
        "base_profile": search_space.base_profile,
        "datasets": list(search_space.datasets),
        "study_name": study_name,
        "storage": storage,
        "objective": dataclasses.asdict(search_space.objective),
        "max_epochs": search_space.max_epochs,
        "trials": int(trials or search_space.trials),
        "config_overrides": _json_safe(search_space.config_overrides),
        "parameters": _json_safe(search_space.parameters),
        "base_configs": base_configs,
    }


def run_search(args: argparse.Namespace) -> int:
    """Execute a resolved Optuna search from parsed CLI args."""
    if args.list_spaces:
        print("Available search spaces:")
        for space_name in search_space_names():
            space = get_search_space(space_name)
            description = space.get("description", "")
            print(f"  {space_name}: {description}")
        return 0
    if args.space is None:
        raise ValueError("--space is required unless --list-spaces is used.")

    search_space = resolve_search_space(args.space, dataset=args.dataset)
    study_name = args.study_name or default_study_name(args.space, search_space.datasets)
    n_trials = int(args.trials or search_space.trials)
    if n_trials < 1:
        raise ValueError("--trials must be >= 1.")

    if args.dry_run:
        print(
            json.dumps(
                build_dry_run_payload(
                    search_space,
                    study_name=study_name,
                    storage=args.storage,
                    trials=n_trials,
                    device=args.device,
                    data_dir=args.data_dir,
                ),
                indent=2,
                sort_keys=True,
            ),
        )
        return 0

    _ensure_storage_parent(args.storage)
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction=search_space.objective.direction,
        load_if_exists=True,
    )
    study.optimize(
        _objective_factory(
            search_space,
            study_name=study_name,
            device=args.device,
            data_dir=args.data_dir,
            enable_mlflow=not bool(args.no_mlflow),
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
            overwrite_checkpoint=bool(args.overwrite_checkpoint),
        ),
        n_trials=n_trials,
        catch=(Exception,),
    )

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed_trials:
        print(
            (
                f"Best trial: {study.best_trial.number} | "
                f"{search_space.objective.metric}={study.best_value:.6f}"
            ),
        )
        print("Best params:")
        print(json.dumps(_json_safe(study.best_trial.user_attrs.get("sampled_params", {}))))
        return 0

    print("No Optuna trials completed successfully.")
    return 1


def main() -> int:
    """Parse CLI arguments and run an Optuna search."""
    parser = build_search_parser()
    args = parser.parse_args()
    configure_cli_logging()
    try:
        return run_search(args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc.args[0] if exc.args else exc))
    except Exception:
        traceback.print_exc()
        return 1
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
