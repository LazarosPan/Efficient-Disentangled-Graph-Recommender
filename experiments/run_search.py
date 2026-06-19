#!/usr/bin/env python
"""Optuna search controller for configured EDGRec spaces."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import optuna
from scripts._workflow_helpers import configure_cli_logging
from src.training import THESIS_PRIMARY_METRICS
from src.utils.benchmark_datasets import (
    normalize_benchmark_datasets_arg,
    resolve_benchmark_datasets,
)
from src.utils.config import BENCHMARK_CONFIG_FIELDS, DEFAULT_SEED, EDGRecConfig
from src.utils.crru import (
    VALIDATION_ACCURACY_METRIC,
    VALIDATION_ONLINE_CRRU_K_METRICS,
    VALIDATION_ONLINE_CRRU_METRIC,
    compute_validation_accuracy_objective,
    compute_validation_online_crru_for_k,
    compute_validation_online_crru_objective,
)
from src.utils.experiment_logger import ExperimentLogger
from src.utils.method_naming import (
    EDGREC_DISPLAY_NAME,
    EDGREC_PUBLIC_PRESET,
    is_edgrec_token,
    legacy_method_identifier,
    method_identifier_aliases,
    public_method_identifier,
)
from src.utils.project_paths import THESIS_DB_PATH

from experiments.benchmark_resolvers import (
    resolve_benchmark_graph_policy_values,
    resolve_benchmark_lr_scheduler_values,
    resolve_benchmark_num_neighbor_values,
    resolve_benchmark_preprocessing_preset_values,
)
from experiments.cli_parsers import build_search_parser
from experiments.recipes import (
    get_formal_profile,
    get_search_space,
    resolve_profile_num_neighbors,
    search_space_names,
    slugify_fragment,
)
from experiments.run_experiment import (
    build_benchmark_config_inputs,
    build_config,
    normalize_benchmark_config_overrides,
    run_experiment,
)

logger = logging.getLogger("edgrec.search")

DEFAULT_STORAGE = "sqlite:///results/optuna_studies.db"
DEFAULT_OBJECTIVE_METRIC = VALIDATION_ONLINE_CRRU_METRIC
DEFAULT_OBJECTIVE_SPLIT = "val"
DEFAULT_MAX_EPOCHS = 80
DEFAULT_TRIALS = 40
SEARCH_SPACE_REVISION_HASH_LENGTH = 12
INFORMATIVE_TRIAL_STATES = frozenset(
    {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    },
)
CUDA_CONTEXT_POISONED_MARKERS = (
    "CUDA error: device-side assert triggered",
    "cudaErrorAssert",
)
SEARCH_PRESET = EDGREC_PUBLIC_PRESET
SEARCH_PARAMETER_FIELDS = frozenset(
    {
        "lr",
        "weight_decay",
        "batch_size",
        "lr_scheduler",
        "lr_scheduler_factor",
        "grad_clip_norm",
        "num_neighbors",
        "interest_gnn_layers",
        "conformity_gnn_layers",
        "dropout",
        "n_negatives",
        "embedding_optimizer",
        "train_edge_keep_prob",
        "item_universe_policy",
        "score_mix_min_weight",
        "score_fusion_profile",
        "item_branch_profile",
        "context_feature_profile",
        "loss_profile",
        "graph_profile",
        "use_learned_score_mix",
        "score_weight_interest",
        "score_weight_conformity",
        "score_weight_popularity",
        "separate_item_branch_embeddings",
        "loss_weight_interest_bpr",
        "loss_weight_conformity_bpr",
        "loss_weight_independence",
        "loss_weight_contrastive",
        "loss_weight_align",
        "loss_weight_uniform",
        "loss_weight_popularity",
        "loss_weight_propensity_calibration",
        "loss_normalization",
        "auxiliary_loss_schedule",
        "auxiliary_ramp_rate",
        "independence_ramp_rate",
        "auxiliary_losses_start_epoch",
        "popularity_supervision_start_epoch",
        "graph_policy",
        "hard_negative_ratio",
        "dice_sampler_margin",
        "dice_mask_reduction",
        "feature_gate_init",
        "use_popularity_head",
        "use_features",
    },
)
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
LINEAR_RAMP_ONLY_PARAMETER_FIELDS = frozenset(
    {
        "auxiliary_ramp_rate",
        "independence_ramp_rate",
    },
)
PHASED_ONLY_PARAMETER_FIELDS = frozenset(
    {
        "auxiliary_losses_start_epoch",
        "popularity_supervision_start_epoch",
    },
)


@dataclass(frozen=True)
class ObjectiveSpec:
    """Resolved validation objective contract for one search space."""

    metric: str = DEFAULT_OBJECTIVE_METRIC
    split: str = DEFAULT_OBJECTIVE_SPLIT
    direction: str = "maximize"


@dataclass(frozen=True)
class SamplerSpec:
    """Resolved Optuna sampler settings for one search space."""

    name: str = "tpe"
    seed: int = DEFAULT_SEED
    n_startup_trials: int = 12
    multivariate: bool = False
    group: bool = False
    constant_liar: bool = False


@dataclass(frozen=True)
class PrunerSpec:
    """Resolved Optuna pruner settings for one search space."""

    name: str = "hyperband"
    min_resource: int = 15
    reduction_factor: int = 3
    bootstrap_count: int = 4


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
    profile_overrides: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    sampler: SamplerSpec = field(default_factory=SamplerSpec)
    pruner: PrunerSpec = field(default_factory=PrunerSpec)


@dataclass(frozen=True)
class TrialParameterResolution:
    """Logical sampled params plus concrete config overrides for one trial."""

    sampled_params: dict[str, Any]
    config_overrides: dict[str, Any]


ParameterValidator = Callable[[str, Mapping[str, Any]], None]
DistributionPayloadBuilder = Callable[[str, Mapping[str, Any], int | None], dict[str, Any]]
ParameterSuggester = Callable[[optuna.Trial, str, str, Mapping[str, Any]], Any]
ProfileOverrides = Mapping[str, Mapping[str, Mapping[str, Any]]]

_PROFILE_PARAMETER_FIELDS = frozenset(
    {
        "score_fusion_profile",
        "item_branch_profile",
        "context_feature_profile",
        "loss_profile",
        "graph_profile",
    },
)


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


def _normalize_sampler(raw_sampler: object) -> SamplerSpec:
    """Resolve Optuna sampler settings from a search-space payload."""
    if raw_sampler is None:
        return SamplerSpec()
    sampler = _require_mapping(raw_sampler, field_name="sampler")
    name = str(sampler.get("name", "tpe")).lower()
    if name not in {"tpe", "random"}:
        raise ValueError("sampler.name must be one of: tpe, random.")
    n_startup_trials = int(sampler.get("n_startup_trials", 12))
    if n_startup_trials < 0:
        raise ValueError("sampler.n_startup_trials must be >= 0.")
    multivariate = bool(sampler.get("multivariate", True))
    group = bool(sampler.get("group", True))
    if group and not multivariate:
        raise ValueError("sampler.group=true requires sampler.multivariate=true.")
    return SamplerSpec(
        name=name,
        seed=int(sampler.get("seed", DEFAULT_SEED)),
        n_startup_trials=n_startup_trials,
        multivariate=multivariate,
        group=group,
        constant_liar=bool(sampler.get("constant_liar", False)),
    )


def _normalize_pruner(raw_pruner: object, *, max_epochs: int) -> PrunerSpec:
    """Resolve Optuna pruner settings from a search-space payload."""
    if raw_pruner is None:
        return PrunerSpec()
    pruner = _require_mapping(raw_pruner, field_name="pruner")
    name = str(pruner.get("name", "hyperband")).lower()
    if name not in {"hyperband", "successive_halving", "median", "none"}:
        raise ValueError(
            "pruner.name must be one of: hyperband, successive_halving, median, none.",
        )
    min_resource = int(pruner.get("min_resource", 15))
    reduction_factor = int(pruner.get("reduction_factor", 3))
    bootstrap_count = int(pruner.get("bootstrap_count", 4))
    if min_resource < 1:
        raise ValueError("pruner.min_resource must be >= 1.")
    if min_resource > max_epochs and name != "none":
        raise ValueError("pruner.min_resource must be <= max_epochs.")
    if reduction_factor < 2:
        raise ValueError("pruner.reduction_factor must be >= 2.")
    if bootstrap_count < 0:
        raise ValueError("pruner.bootstrap_count must be >= 0.")
    return PrunerSpec(
        name=name,
        min_resource=min_resource,
        reduction_factor=reduction_factor,
        bootstrap_count=bootstrap_count,
    )


def _build_sampler(spec: SamplerSpec) -> optuna.samplers.BaseSampler:
    """Instantiate the configured Optuna sampler."""
    if spec.name == "random":
        return optuna.samplers.RandomSampler(seed=spec.seed)
    kwargs: dict[str, Any] = {
        "seed": spec.seed,
        "n_startup_trials": spec.n_startup_trials,
        "constant_liar": spec.constant_liar,
    }
    if spec.multivariate:
        kwargs["multivariate"] = True
        if spec.group:
            kwargs["group"] = True
    return optuna.samplers.TPESampler(**kwargs)


def _build_pruner(spec: PrunerSpec, *, max_epochs: int) -> optuna.pruners.BasePruner:
    """Instantiate the configured Optuna pruner."""
    if spec.name == "none":
        return optuna.pruners.NopPruner()
    if spec.name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=spec.bootstrap_count,
            n_warmup_steps=spec.min_resource,
        )
    if spec.name == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=spec.min_resource,
            reduction_factor=spec.reduction_factor,
            bootstrap_count=spec.bootstrap_count,
        )
    return optuna.pruners.HyperbandPruner(
        min_resource=spec.min_resource,
        max_resource=max_epochs,
        reduction_factor=spec.reduction_factor,
        bootstrap_count=spec.bootstrap_count,
    )


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


def _decimal_grid_value(value: Decimal) -> float:
    """Return a stable float for a human-declared grid point."""
    return float(format(value.normalize(), "f"))


def _expand_grid_float_choices(field_name: str, spec: Mapping[str, Any]) -> list[float]:
    """Expand one or more inclusive linear float-grid segments."""
    raw_segments = spec.get("segments")
    if raw_segments is None:
        raw_segments = [spec]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"parameters.{field_name}.segments must be a non-empty list.")

    choices: list[float] = []
    seen: set[str] = set()
    for raw_segment in raw_segments:
        segment = _require_mapping(
            raw_segment,
            field_name=f"parameters.{field_name}.segments[]",
        )
        if not {"low", "high", "step"}.issubset(segment):
            raise ValueError(
                f"parameters.{field_name} grid segments must define low, high, and step.",
            )
        low = Decimal(str(segment["low"]))
        high = Decimal(str(segment["high"]))
        step = Decimal(str(segment["step"]))
        if low > high:
            raise ValueError(f"parameters.{field_name}.low must be <= high.")
        if step <= 0:
            raise ValueError(f"parameters.{field_name}.step must be > 0.")

        current = low
        while current <= high:
            label = format(current.normalize(), "f")
            if label not in seen:
                seen.add(label)
                choices.append(_decimal_grid_value(current))
            current += step

    if not choices:
        raise ValueError(f"parameters.{field_name} grid must produce at least one value.")
    return choices


def _validate_grid_float_parameter(field_name: str, spec: Mapping[str, Any]) -> None:
    """Validate a grid-float Optuna parameter spec."""
    _expand_grid_float_choices(field_name, spec)


def _validate_categorical_parameter(field_name: str, spec: Mapping[str, Any]) -> None:
    """Validate a categorical Optuna parameter spec."""
    choices = spec.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"parameters.{field_name}.choices must be a non-empty list.")
    labels = [json.dumps(choice, sort_keys=True, separators=(",", ":")) for choice in choices]
    if len(set(labels)) != len(labels):
        raise ValueError(f"parameters.{field_name}.choices contains duplicate values.")


def _normalize_profile_overrides(
    raw_space: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return validated search-layer profile-to-config mappings."""
    raw_overrides = _require_mapping(
        raw_space.get("profile_overrides", {}),
        field_name="profile_overrides",
    )
    _validate_catalog_fields(
        raw_overrides,
        field_name="profile_overrides",
        allowed_fields=set(_PROFILE_PARAMETER_FIELDS),
    )
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for field_name, raw_profiles in raw_overrides.items():
        profiles = _require_mapping(raw_profiles, field_name=f"profile_overrides.{field_name}")
        if not profiles:
            raise ValueError(f"profile_overrides.{field_name} must define at least one profile.")
        normalized[field_name] = {}
        for profile_name, raw_config_overrides in profiles.items():
            config_overrides = dict(
                _require_mapping(
                    raw_config_overrides,
                    field_name=f"profile_overrides.{field_name}.{profile_name}",
                ),
            )
            _validate_catalog_fields(
                config_overrides,
                field_name=f"profile_overrides.{field_name}.{profile_name}",
                allowed_fields=set(BENCHMARK_CONFIG_FIELDS),
            )
            normalized[field_name][str(profile_name)] = config_overrides
    return normalized


def _validate_search_profile_parameter(
    field_name: str,
    spec: Mapping[str, Any],
    *,
    profile_overrides: ProfileOverrides,
) -> None:
    """Validate a search-layer mechanism profile parameter."""
    if field_name not in _PROFILE_PARAMETER_FIELDS:
        return
    if str(spec.get("type", "")).lower() != "categorical":
        raise ValueError(f"parameters.{field_name} must be a categorical profile.")
    choices = spec.get("choices", [])
    if field_name not in profile_overrides:
        raise ValueError(f"profile_overrides.{field_name} must define profile choices.")
    allowed = set(profile_overrides[field_name])
    unsupported = sorted(str(choice) for choice in choices if str(choice) not in allowed)
    if unsupported:
        raise ValueError(
            f"parameters.{field_name} contains unknown profile choices: {', '.join(unsupported)}.",
        )


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


_PARAMETER_VALIDATORS: dict[str, ParameterValidator] = {
    "float": _validate_numeric_parameter,
    "grid_float": _validate_grid_float_parameter,
    "int": _validate_numeric_parameter,
    "categorical": _validate_categorical_parameter,
    "fanout": _validate_fanout_parameter,
}
SUPPORTED_PARAMETER_TYPES = frozenset(_PARAMETER_VALIDATORS)


def _validate_parameter_specs(
    parameters: Mapping[str, Any],
    *,
    profile_overrides: ProfileOverrides | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate search parameter specs and return a plain dict copy."""
    _validate_catalog_fields(
        parameters,
        field_name="parameters",
        allowed_fields=set(SEARCH_PARAMETER_FIELDS),
    )
    profile_overrides = profile_overrides or {}
    normalized: dict[str, dict[str, Any]] = {}
    for field_name, raw_spec in parameters.items():
        spec = dict(_require_mapping(raw_spec, field_name=f"parameters.{field_name}"))
        parameter_type = str(spec.get("type", "")).lower()
        validator = _PARAMETER_VALIDATORS.get(parameter_type)
        if validator is None:
            raise ValueError(
                (
                    f"parameters.{field_name}.type must be one of "
                    f"{', '.join(sorted(SUPPORTED_PARAMETER_TYPES))}."
                ),
            )
        validator(field_name, spec)
        _validate_search_profile_parameter(
            field_name,
            spec,
            profile_overrides=profile_overrides,
        )
        normalized[field_name] = spec
    return normalized


def _resolve_parameter_specs(
    raw_space: Mapping[str, Any],
    *,
    all_datasets: Sequence[str],
    active_dataset: str | None,
    profile_overrides: ProfileOverrides,
) -> dict[str, dict[str, Any]]:
    """Resolve base plus optional dataset-local parameter overrides.

    The study name remains stable across search-space edits. Dataset-local
    overrides keep one command (`--space edgrec-core-optimization`) while
    avoiding a single compromise grid for datasets with different Optuna basins.
    """
    raw_parameters = _require_mapping(raw_space.get("parameters", {}), field_name="parameters")
    base_parameters = dict(raw_parameters)
    raw_by_dataset = raw_space.get("parameters_by_dataset", {})
    by_dataset = _require_mapping(
        raw_by_dataset,
        field_name="parameters_by_dataset",
    )
    unknown_datasets = sorted(set(by_dataset) - set(all_datasets))
    if unknown_datasets:
        raise ValueError(
            "parameters_by_dataset contains datasets outside search_spaces.datasets: "
            f"{', '.join(str(dataset) for dataset in unknown_datasets)}",
        )

    for dataset, raw_overrides in by_dataset.items():
        overrides = dict(
            _require_mapping(
                raw_overrides,
                field_name=f"parameters_by_dataset.{dataset}",
            ),
        )
        _validate_parameter_specs(
            {**base_parameters, **overrides},
            profile_overrides=profile_overrides,
        )

    if active_dataset is None:
        return _validate_parameter_specs(
            base_parameters,
            profile_overrides=profile_overrides,
        )

    dataset_overrides = dict(
        _require_mapping(
            by_dataset.get(active_dataset, {}),
            field_name=f"parameters_by_dataset.{active_dataset}",
        ),
    )
    return _validate_parameter_specs(
        {**base_parameters, **dataset_overrides},
        profile_overrides=profile_overrides,
    )


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
    if len(presets) != 1 or not is_edgrec_token(presets[0]):
        raise ValueError(
            (
                f"search space '{space_name}' must use an {EDGREC_DISPLAY_NAME}-only base "
                "profile; "
                f"got presets={presets!r}."
            ),
        )

    raw_datasets = raw_space.get("datasets") or base_profile["matrix"]["datasets"]
    all_datasets = resolve_benchmark_datasets(normalize_benchmark_datasets_arg(raw_datasets))
    datasets = list(all_datasets)
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
    sampler = _normalize_sampler(raw_space.get("sampler"))
    pruner = _normalize_pruner(raw_space.get("pruner"), max_epochs=max_epochs)
    trials = int(raw_space.get("trials") or DEFAULT_TRIALS)
    if trials < 1:
        raise ValueError(f"search space '{space_name}' trials must be >= 1.")

    config_overrides = dict(raw_space.get("config_overrides", {}))
    _validate_catalog_fields(
        config_overrides,
        field_name="config_overrides",
        allowed_fields=set(BENCHMARK_CONFIG_FIELDS),
    )
    profile_overrides = _normalize_profile_overrides(raw_space)
    parameters = _resolve_parameter_specs(
        raw_space,
        all_datasets=all_datasets,
        active_dataset=dataset,
        profile_overrides=profile_overrides,
    )
    if not parameters:
        raise ValueError(f"search space '{space_name}' must define at least one parameter.")

    return SearchSpaceSpec(
        name=str(raw_space["name"]),
        description=str(raw_space.get("description", "")),
        base_profile=base_profile_name,
        datasets=tuple(datasets),
        objective=objective,
        sampler=sampler,
        pruner=pruner,
        max_epochs=max_epochs,
        trials=trials,
        config_overrides=config_overrides,
        parameters=parameters,
        profile_overrides=profile_overrides,
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


def _single_search_value[T](field_name: str, values: Sequence[T]) -> T:
    """Return one resolved benchmark value for a search trial."""
    if len(values) != 1:
        raise ValueError(f"Search spaces must resolve one {field_name} per trial.")
    return values[0]


def _first_search_value[T](field_name: str, values: Sequence[T]) -> T:
    """Return the first resolved benchmark value for a search base config."""
    if not values:
        raise ValueError(f"Search spaces must resolve at least one {field_name}.")
    return values[0]


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

    try:
        lr_scheduler_values = resolve_benchmark_lr_scheduler_values(
            benchmark_args,
            expand_all=False,
        )
    except ValueError as exc:
        if "lr_scheduler='all'" not in str(exc):
            raise
        raise ValueError("Search spaces cannot use lr_scheduler='all'.") from exc
    num_neighbor_values = resolve_benchmark_num_neighbor_values(
        benchmark_args,
        dataset=dataset,
    )

    return build_benchmark_config_inputs(
        benchmark_args,
        dataset=dataset,
        preset=SEARCH_PRESET,
        lr_scheduler=_single_search_value("lr_scheduler", lr_scheduler_values),
        num_neighbors=list(_first_search_value("num_neighbors", num_neighbor_values)),
        preprocessing_preset=_single_search_value(
            "preprocessing_preset",
            resolve_benchmark_preprocessing_preset_values(benchmark_args),
        ),
        graph_policy=_single_search_value(
            "graph_policy",
            resolve_benchmark_graph_policy_values(benchmark_args),
        ),
    )


def build_search_config(
    search_space: SearchSpaceSpec,
    *,
    dataset: str,
    sampled_overrides: Mapping[str, Any] | None = None,
    device: str,
    data_dir: str,
) -> EDGRecConfig:
    """Resolve one concrete EDGRec config for a search trial."""
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


def _grid_float_distribution_payload(
    field_name: str,
    spec: Mapping[str, Any],
    _depth: int | None,
) -> dict[str, Any]:
    """Return Optuna distribution payload for an expanded float grid."""
    return {
        "type": "categorical",
        "choices": _expand_grid_float_choices(field_name, spec),
    }


def _categorical_distribution_payload(
    _field_name: str,
    spec: Mapping[str, Any],
    _depth: int | None,
) -> dict[str, Any]:
    """Return Optuna distribution payload for categorical choices."""
    return {"type": "categorical", "choices": list(spec["choices"])}


def _float_distribution_payload(
    _field_name: str,
    spec: Mapping[str, Any],
    _depth: int | None,
) -> dict[str, Any]:
    """Return Optuna distribution payload for a float range."""
    return {
        "type": "float",
        "low": float(spec["low"]),
        "high": float(spec["high"]),
        "log": bool(spec.get("log", False)),
        "step": float(spec["step"]) if "step" in spec else None,
    }


def _int_distribution_payload(
    _field_name: str,
    spec: Mapping[str, Any],
    _depth: int | None,
) -> dict[str, Any]:
    """Return Optuna distribution payload for an integer range."""
    return {
        "type": "int",
        "low": int(spec["low"]),
        "high": int(spec["high"]),
        "log": bool(spec.get("log", False)),
        "step": int(spec.get("step", 1)),
    }


def _fanout_distribution_payload(
    _field_name: str,
    spec: Mapping[str, Any],
    depth: int | None,
) -> dict[str, Any]:
    """Return Optuna distribution payload for the active fan-out depth."""
    if depth is None:
        raise ValueError("fanout distribution payload requires a depth.")
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
    return {"type": "categorical", "depth": depth, "choices": choices}


_DISTRIBUTION_PAYLOAD_BUILDERS: dict[str, DistributionPayloadBuilder] = {
    "grid_float": _grid_float_distribution_payload,
    "categorical": _categorical_distribution_payload,
    "float": _float_distribution_payload,
    "int": _int_distribution_payload,
    "fanout": _fanout_distribution_payload,
}


def _parameter_distribution_payload(
    field_name: str,
    spec: Mapping[str, Any],
    *,
    depth: int | None = None,
) -> dict[str, Any]:
    """Return the Optuna distribution-relevant payload for one logical field.

    Optuna requires each stored parameter name to keep one compatible
    distribution forever. We keep the study name stable across search-space
    edits, so the storage-facing parameter name must change when the declared
    distribution changes. The logical field name still remains the config owner
    through ``sampled_params``.
    """
    parameter_type = str(spec["type"]).lower()
    builder = _DISTRIBUTION_PAYLOAD_BUILDERS.get(parameter_type)
    if builder is None:
        raise ValueError(f"Unsupported parameter type: {parameter_type}")
    return builder(field_name, spec, depth)


def _distribution_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a short stable ID for one Optuna distribution."""
    canonical = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]


def search_space_revision(search_space: SearchSpaceSpec) -> str:
    """Return a stable revision id for the resolved logical search contract."""
    payload = {
        "name": legacy_method_identifier(search_space.name) or search_space.name,
        "base_profile": legacy_method_identifier(search_space.base_profile)
        or search_space.base_profile,
        "datasets": list(search_space.datasets),
        "objective": dataclasses.asdict(search_space.objective),
        "sampler": dataclasses.asdict(search_space.sampler),
        "pruner": dataclasses.asdict(search_space.pruner),
        "max_epochs": search_space.max_epochs,
        "config_overrides": _json_safe(search_space.config_overrides),
        "parameters": _json_safe(search_space.parameters),
    }
    if search_space.profile_overrides:
        payload["profile_overrides"] = _json_safe(search_space.profile_overrides)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:SEARCH_SPACE_REVISION_HASH_LENGTH]


def _parameter_storage_name(
    field_name: str,
    spec: Mapping[str, Any],
    *,
    depth: int | None = None,
) -> str:
    """Return Optuna's storage-facing parameter name for one logical field."""
    payload = _parameter_distribution_payload(field_name, spec, depth=depth)
    suffix = _distribution_fingerprint(payload)
    if depth is None:
        return f"{field_name}__{suffix}"
    return f"{field_name}_depth_{depth}__{suffix}"


def _suggest_categorical_value(
    trial: optuna.Trial,
    optuna_param_name: str,
    choices: list[Any],
) -> Any:
    """Suggest a categorical value without passing structured choices to Optuna."""
    if all(choice is None or isinstance(choice, str | int | float | bool) for choice in choices):
        return trial.suggest_categorical(optuna_param_name, choices)
    labels, values_by_label = _choice_labels(choices)
    return values_by_label[trial.suggest_categorical(optuna_param_name, labels)]


def _suggest_float_parameter(
    trial: optuna.Trial,
    _field_name: str,
    optuna_param_name: str,
    spec: Mapping[str, Any],
) -> Any:
    """Suggest a float Optuna parameter value."""
    return trial.suggest_float(
        optuna_param_name,
        float(spec["low"]),
        float(spec["high"]),
        log=bool(spec.get("log", False)),
        step=float(spec["step"]) if "step" in spec else None,
    )


def _suggest_grid_float_parameter(
    trial: optuna.Trial,
    field_name: str,
    optuna_param_name: str,
    spec: Mapping[str, Any],
) -> Any:
    """Suggest a categorical value from an expanded float grid."""
    return _suggest_categorical_value(
        trial,
        optuna_param_name,
        _expand_grid_float_choices(field_name, spec),
    )


def _suggest_int_parameter(
    trial: optuna.Trial,
    _field_name: str,
    optuna_param_name: str,
    spec: Mapping[str, Any],
) -> Any:
    """Suggest an integer Optuna parameter value."""
    return trial.suggest_int(
        optuna_param_name,
        int(spec["low"]),
        int(spec["high"]),
        step=int(spec.get("step", 1)),
        log=bool(spec.get("log", False)),
    )


def _suggest_categorical_parameter(
    trial: optuna.Trial,
    _field_name: str,
    optuna_param_name: str,
    spec: Mapping[str, Any],
) -> Any:
    """Suggest a categorical Optuna parameter value."""
    return _suggest_categorical_value(trial, optuna_param_name, list(spec["choices"]))


_PARAMETER_SUGGESTERS: dict[str, ParameterSuggester] = {
    "float": _suggest_float_parameter,
    "grid_float": _suggest_grid_float_parameter,
    "int": _suggest_int_parameter,
    "categorical": _suggest_categorical_parameter,
}


def _suggest_parameter_value(
    trial: optuna.Trial,
    field_name: str,
    spec: Mapping[str, Any],
) -> Any:
    """Suggest one non-fanout parameter value from a validated spec."""
    parameter_type = str(spec["type"]).lower()
    optuna_param_name = _parameter_storage_name(field_name, spec)
    trial.set_user_attr(f"optuna_param_name.{field_name}", optuna_param_name)
    suggester = _PARAMETER_SUGGESTERS.get(parameter_type)
    if suggester is not None:
        return suggester(trial, field_name, optuna_param_name, spec)
    raise ValueError(f"Unsupported non-fanout parameter type: {parameter_type}")


def _suggest_fanout_value(
    trial: optuna.Trial,
    spec: Mapping[str, Any],
    *,
    base_config: EDGRecConfig,
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
    optuna_param_name = _parameter_storage_name("num_neighbors", spec, depth=depth)
    selected_label = trial.suggest_categorical(optuna_param_name, labels)
    trial.set_user_attr("num_neighbors_param", optuna_param_name)
    return list(values_by_label[selected_label])


def _parameter_is_conditionally_active(
    field_name: str,
    sampled_params: Mapping[str, Any],
) -> bool:
    """Return whether a conditional parameter changes the resolved config."""
    if field_name == "lr_scheduler_factor":
        return sampled_params.get("lr_scheduler") == "plateau"
    schedule = str(sampled_params.get("auxiliary_loss_schedule", "linear_ramp"))
    if field_name in LINEAR_RAMP_ONLY_PARAMETER_FIELDS:
        return schedule == "linear_ramp"
    if field_name in PHASED_ONLY_PARAMETER_FIELDS:
        return schedule == "phased"
    return True


def _profile_config_overrides(
    profile_overrides: ProfileOverrides,
    field_name: str,
    profile_name: Any,
) -> dict[str, Any]:
    """Return concrete config overrides for one search-layer profile choice."""
    try:
        return dict(profile_overrides[field_name][str(profile_name)])
    except KeyError as exc:
        raise ValueError(f"Unknown {field_name} profile: {profile_name!r}") from exc


def _resolved_config_overrides_from_sampled_params(
    sampled_params: Mapping[str, Any],
    *,
    profile_overrides: ProfileOverrides,
) -> dict[str, Any]:
    """Resolve logical sampled params into concrete config overrides."""
    resolved: dict[str, Any] = {}
    for field_name in _PROFILE_PARAMETER_FIELDS:
        if field_name in sampled_params:
            resolved.update(
                _profile_config_overrides(
                    profile_overrides,
                    field_name,
                    sampled_params[field_name],
                ),
            )
    for field_name, value in sampled_params.items():
        if field_name not in _PROFILE_PARAMETER_FIELDS:
            resolved[field_name] = value
    return resolved


def resolve_trial_parameters(
    trial: optuna.Trial,
    search_space: SearchSpaceSpec,
    *,
    base_config: EDGRecConfig,
) -> TrialParameterResolution:
    """Suggest logical trial params and resolve them to config overrides."""
    sampled_params: dict[str, Any] = {}
    config_overrides: dict[str, Any] = {}

    for field_name, spec in search_space.parameters.items():
        if field_name not in _PROFILE_PARAMETER_FIELDS:
            continue
        value = _suggest_parameter_value(trial, field_name, spec)
        sampled_params[field_name] = value
        config_overrides.update(
            _profile_config_overrides(search_space.profile_overrides, field_name, value),
        )

    for field_name, spec in search_space.parameters.items():
        if field_name in _PROFILE_PARAMETER_FIELDS:
            continue
        if str(spec["type"]).lower() == "fanout":
            continue
        active_context = {**config_overrides, **sampled_params}
        if not _parameter_is_conditionally_active(field_name, active_context):
            continue
        value = _suggest_parameter_value(trial, field_name, spec)
        sampled_params[field_name] = value
        config_overrides[field_name] = value

    for field_name, spec in search_space.parameters.items():
        if str(spec["type"]).lower() != "fanout":
            continue
        active_context = {**config_overrides, **sampled_params}
        if not _parameter_is_conditionally_active(field_name, active_context):
            continue
        value = _suggest_fanout_value(
            trial,
            spec,
            base_config=base_config,
            sampled_overrides=config_overrides,
        )
        sampled_params[field_name] = value
        config_overrides[field_name] = value
    return TrialParameterResolution(
        sampled_params=sampled_params,
        config_overrides=config_overrides,
    )


def suggest_trial_overrides(
    trial: optuna.Trial,
    search_space: SearchSpaceSpec,
    *,
    base_config: EDGRecConfig,
) -> dict[str, Any]:
    """Suggest one concrete trial override dict over ``EDGRecConfig`` fields."""
    return resolve_trial_parameters(
        trial,
        search_space,
        base_config=base_config,
    ).config_overrides


def _result_epoch_time_s(result: Mapping[str, Any]) -> float | None:
    """Return the trial-level seconds-per-epoch value used by validation CRRU."""
    avg_epoch_time_s = result.get("avg_epoch_time_s")
    if avg_epoch_time_s is not None and float(avg_epoch_time_s) > 0:
        return float(avg_epoch_time_s)

    training_time_s = result.get("training_time_s")
    epochs_stopped_at = result.get("epochs_stopped_at")
    if training_time_s is not None and epochs_stopped_at:
        return float(training_time_s) / float(epochs_stopped_at)
    return None


def _objective_metric_value(
    metrics: Mapping[str, float],
    metric_name: str,
    *,
    result: Mapping[str, Any],
) -> float:
    """Return the objective value for one validation epoch."""
    if metric_name == VALIDATION_ONLINE_CRRU_METRIC:
        return compute_validation_online_crru_objective(
            metrics,
            peak_vram_mb=result.get("peak_vram_mb"),
            epoch_time_s=_result_epoch_time_s(result),
        )
    if metric_name == VALIDATION_ACCURACY_METRIC:
        return compute_validation_accuracy_objective(metrics)
    if metric_name not in metrics:
        raise ValueError(f"Validation metrics did not include {metric_name}.")
    return float(metrics[metric_name])


def _build_pruning_epoch_callback(
    trial: optuna.Trial,
    *,
    search_space: SearchSpaceSpec,
    dataset: str,
    dataset_index: int,
) -> Callable[[int, Mapping[str, float], float], None]:
    """Return a trainer callback that reports validation objective values to Optuna."""

    def callback(epoch: int, val_metrics: Mapping[str, float], epoch_time_s: float) -> None:
        metrics = {
            str(metric): float(metric_value)
            for metric, metric_value in val_metrics.items()
            if math.isfinite(float(metric_value))
        }
        value = _objective_metric_value(
            metrics,
            search_space.objective.metric,
            result={"avg_epoch_time_s": epoch_time_s},
        )
        step = dataset_index * search_space.max_epochs + epoch + 1
        trial.report(value, step=step)
        trial.set_user_attr(f"{dataset}.last_pruning_epoch", epoch + 1)
        trial.set_user_attr(f"{dataset}.last_pruning_objective", value)
        if trial.should_prune():
            trial.set_user_attr(f"{dataset}.pruned_epoch", epoch + 1)
            trial.set_user_attr(f"{dataset}.pruned_objective", value)
            raise optuna.TrialPruned(
                f"{dataset} pruned at epoch {epoch + 1}: "
                f"{search_space.objective.metric}={value:.6f}",
            )

    return callback


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
        if not isinstance(raw_metrics, Mapping):
            continue
        metrics = {
            str(metric): float(metric_value)
            for metric, metric_value in raw_metrics.items()
            if math.isfinite(float(metric_value))
        }
        try:
            value = _objective_metric_value(metrics, metric_name, result=result)
        except ValueError:
            continue
        if math.isfinite(value):
            metrics[metric_name] = value
            candidates.append(metrics)
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


def _set_trial_attrs_from_result(
    trial: optuna.Trial,
    *,
    dataset: str,
    config: EDGRecConfig,
    result: Mapping[str, Any],
    objective: ObjectiveSpec,
) -> None:
    """Store runtime, validation, and diagnostic metadata as Optuna attrs."""
    best_val_metrics = _best_validation_metrics(
        result,
        metric_name=objective.metric,
        direction=objective.direction,
    )
    try:
        for k, metric_name in VALIDATION_ONLINE_CRRU_K_METRICS.items():
            best_val_metrics[metric_name] = compute_validation_online_crru_for_k(
                best_val_metrics,
                k=k,
                peak_vram_mb=result.get("peak_vram_mb"),
                epoch_time_s=_result_epoch_time_s(result),
            )
        best_val_metrics[VALIDATION_ONLINE_CRRU_METRIC] = compute_validation_online_crru_objective(
            best_val_metrics,
            peak_vram_mb=result.get("peak_vram_mb"),
            epoch_time_s=_result_epoch_time_s(result),
        )
    except ValueError:
        logger.debug("Skipping CRRU trial attrs because validation metrics are incomplete.")
    prefix = f"{dataset}."
    trial.set_user_attr(prefix + "exp_id", result.get("exp_id"))
    trial.set_user_attr(prefix + "canonical_name", result.get("canonical_name"))
    trial.set_user_attr(prefix + "checkpoint_path", result.get("checkpoint_path"))
    trial.set_user_attr(prefix + "epochs_stopped_at", result.get("epochs_stopped_at"))
    trial.set_user_attr(prefix + "training_time_s", result.get("training_time_s"))
    trial.set_user_attr(prefix + "avg_epoch_time_s", _result_epoch_time_s(result))
    trial.set_user_attr(prefix + "peak_vram_mb", result.get("peak_vram_mb"))
    trial.set_user_attr(prefix + "batch_size", result.get("batch_size"))
    trial.set_user_attr(prefix + "auto_batch_size", result.get("auto_batch_size"))
    effective_config = dataclasses.asdict(config)
    trial.set_user_attr(prefix + "effective_config", _json_safe(effective_config))
    trial.set_user_attr(
        prefix + "effective_config_json",
        json.dumps(_json_safe(effective_config), sort_keys=True, separators=(",", ":")),
    )

    for metric_name in (
        *THESIS_PRIMARY_METRICS,
        *VALIDATION_ONLINE_CRRU_K_METRICS.values(),
        VALIDATION_ONLINE_CRRU_METRIC,
        VALIDATION_ACCURACY_METRIC,
    ):
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
                str(metric_name).startswith(prefix_) for prefix_ in TRIAL_ATTRIBUTE_TEST_PREFIXES
            )
            if has_diagnostic_prefix:
                trial.set_user_attr(prefix + f"test.{metric_name}", float(value))


def _trial_batch_id(study_name: str, trial_number: int) -> str:
    """Return the SQLite/MLflow batch id for one Optuna trial."""
    study_slug = slugify_fragment(study_name, fallback="search")
    return f"optuna-{study_slug}-trial-{trial_number}"


def _trial_change_note(
    *,
    search_space: SearchSpaceSpec,
    study_name: str,
    trial_number: int,
    sampled_params: Mapping[str, Any],
    resolved_config_overrides: Mapping[str, Any],
) -> str:
    """Return compact trial metadata for the existing change_note field."""
    return json.dumps(
        {
            "search_space": search_space.name,
            "search_space_revision": search_space_revision(search_space),
            "study_name": study_name,
            "trial_number": trial_number,
            "sampled_params": _json_safe(sampled_params),
            "resolved_config_overrides": _json_safe(resolved_config_overrides),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _is_cuda_context_poisoned_exception(exc: BaseException) -> bool:
    """Return whether an exception indicates a process-local CUDA assert state."""
    messages: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    combined = "\n".join(messages)
    return any(marker in combined for marker in CUDA_CONTEXT_POISONED_MARKERS)


def _trial_has_poisoned_cuda_context_failure(trial: optuna.trial.FrozenTrial) -> bool:
    """Return whether a stored trial failure should abort this Python process."""
    if trial.user_attrs.get("fatal_failure") == "cuda_context_poisoned":
        return True
    failure_reason = str(trial.user_attrs.get("failure_reason", ""))
    return any(marker in failure_reason for marker in CUDA_CONTEXT_POISONED_MARKERS)


def _record_trial_failure(
    trial: optuna.Trial,
    *,
    stage: str,
    exc: Exception,
    dataset: str | None = None,
) -> None:
    """Store failure diagnostics in Optuna RDB before re-raising."""
    failure_reason = f"{type(exc).__name__}: {exc}"
    trial.set_user_attr("failure_stage", stage)
    trial.set_user_attr("failure_reason", failure_reason)
    if _is_cuda_context_poisoned_exception(exc):
        trial.set_user_attr("fatal_failure", "cuda_context_poisoned")
    if dataset is not None:
        trial.set_user_attr(f"{dataset}.failure_stage", stage)
        trial.set_user_attr(f"{dataset}.failure_reason", failure_reason)


def _objective_factory(
    search_space: SearchSpaceSpec,
    *,
    study_name: str,
    device: str,
    data_dir: str,
    enable_mlflow: bool,
    mlflow_tracking_uri: str | None,
    mlflow_experiment_name: str,
    seen_sampled_param_keys: set[str] | None = None,
) -> Any:
    """Build the Optuna objective callable for one resolved search space."""

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr("search_space", search_space.name)
        trial.set_user_attr("search_space_revision", search_space_revision(search_space))
        trial.set_user_attr("study_name", study_name)
        trial.set_user_attr("datasets", list(search_space.datasets))
        trial.set_user_attr("objective_metric", search_space.objective.metric)
        trial.set_user_attr("objective_split", search_space.objective.split)
        try:
            base_config = build_search_config(
                search_space,
                dataset=search_space.datasets[0],
                sampled_overrides=None,
                device=device,
                data_dir=data_dir,
            )
            resolution = resolve_trial_parameters(
                trial,
                search_space,
                base_config=base_config,
            )
        except Exception as exc:
            _record_trial_failure(trial, stage="suggest_overrides", exc=exc)
            raise

        sampled_key = _canonical_sampled_params(resolution.sampled_params)
        trial.set_user_attr("sampled_params", _json_safe(resolution.sampled_params))
        trial.set_user_attr("sampled_params_json", sampled_key)
        trial.set_user_attr(
            "resolved_config_overrides",
            _json_safe(resolution.config_overrides),
        )
        trial.set_user_attr(
            "resolved_config_overrides_json",
            json.dumps(
                _json_safe(resolution.config_overrides),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if seen_sampled_param_keys is not None and sampled_key in seen_sampled_param_keys:
            trial.set_user_attr("duplicate_sampled_params", True)
            trial.set_user_attr("duplicate_sampled_params_key", sampled_key)
            raise optuna.TrialPruned(
                "Duplicate fresh sampled_params; skipping training.",
            )

        configs_by_dataset: dict[str, EDGRecConfig] = {}
        for dataset in search_space.datasets:
            try:
                configs_by_dataset[dataset] = build_search_config(
                    search_space,
                    dataset=dataset,
                    sampled_overrides=resolution.config_overrides,
                    device=device,
                    data_dir=data_dir,
                )
            except Exception as exc:
                _record_trial_failure(trial, stage="build_config", exc=exc, dataset=dataset)
                raise

        scores: list[float] = []
        batch_id = _trial_batch_id(study_name, trial.number)
        change_note = _trial_change_note(
            search_space=search_space,
            study_name=study_name,
            trial_number=trial.number,
            sampled_params=resolution.sampled_params,
            resolved_config_overrides=resolution.config_overrides,
        )
        for dataset_index, (dataset, config) in enumerate(configs_by_dataset.items()):
            try:
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
                    include_refined_diagnostics=False,
                    evaluate_test=False,
                    training_epoch_callback=_build_pruning_epoch_callback(
                        trial,
                        search_space=search_space,
                        dataset=dataset,
                        dataset_index=dataset_index,
                    ),
                )
                score = extract_validation_objective(result, search_space.objective)
                trial.set_user_attr(f"{dataset}.objective", score)
                _set_trial_attrs_from_result(
                    trial,
                    dataset=dataset,
                    config=config,
                    result=result,
                    objective=search_space.objective,
                )
                scores.append(score)
            except optuna.TrialPruned:
                trial.set_user_attr(f"{dataset}.pruned", True)
                raise
            except Exception as exc:
                _record_trial_failure(trial, stage="run_experiment", exc=exc, dataset=dataset)
                raise

        if not scores:
            raise ValueError("Trial produced no dataset scores.")
        objective_value = float(sum(scores) / len(scores))
        trial.set_user_attr("objective_value", objective_value)
        if seen_sampled_param_keys is not None:
            seen_sampled_param_keys.add(sampled_key)
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


def default_study_name(
    space_name: str,
    datasets: tuple[str, ...],
    *,
    search_space: SearchSpaceSpec | None = None,
) -> str:
    """Return the default study name for a space/dataset selection."""
    dataset_part = datasets[0] if len(datasets) == 1 else "all"
    public_space_name = public_method_identifier(space_name) or space_name
    base_name = f"{public_space_name}-{dataset_part}"
    if search_space is None:
        return base_name
    objective_name = slugify_fragment(
        f"{search_space.objective.split}-{search_space.objective.metric}",
        fallback="search",
    )
    return f"{base_name}-{objective_name}"


def _storage_study_names(storage: str) -> set[str]:
    """Return existing study names in one Optuna storage."""
    return {summary.study_name for summary in optuna.get_all_study_summaries(storage=storage)}


def _resolve_existing_study_name(storage: str, study_name: str) -> str:
    """Reuse an existing public or legacy EDGRec study instead of starting over."""
    existing_names = _storage_study_names(storage)
    if study_name in existing_names:
        return study_name
    for alias in method_identifier_aliases(study_name):
        if alias in existing_names:
            return alias
    return study_name


def _canonical_sampled_params(sampled_params: Mapping[str, Any]) -> str:
    """Return a stable key for duplicate logical hyperparameter detection."""
    return json.dumps(_json_safe(sampled_params), sort_keys=True, separators=(",", ":"))


def is_seeded_trial(trial: optuna.trial.FrozenTrial) -> bool:
    """Return whether a trial was copied from another study instead of run fresh."""
    return trial.user_attrs.get("seeded_from_study") is not None


def is_duplicate_pruned_trial(trial: optuna.trial.FrozenTrial) -> bool:
    """Return whether a pruned trial only records duplicate-parameter avoidance."""
    return bool(trial.user_attrs.get("duplicate_sampled_params"))


def _choice_matches(value: Any, choices: list[Any]) -> bool:
    """Return whether a sampled value equals one declared choice."""
    value_label = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return any(
        value_label == json.dumps(_json_safe(choice), sort_keys=True, separators=(",", ":"))
        for choice in choices
    )


def _float_matches_grid(value: Any, choices: list[float]) -> bool:
    """Return whether a sampled value is one declared float grid point."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return any(
        math.isclose(numeric_value, choice, rel_tol=0.0, abs_tol=1e-12) for choice in choices
    )


def _sampled_value_matches_spec(value: Any, field_name: str, spec: Mapping[str, Any]) -> bool:
    """Return whether a logical sampled value is valid under the current spec."""
    parameter_type = str(spec["type"]).lower()
    if parameter_type == "grid_float":
        return _float_matches_grid(value, _expand_grid_float_choices(field_name, spec))
    if parameter_type == "categorical":
        return _choice_matches(value, list(spec["choices"]))
    if parameter_type == "float":
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return False
        if numeric_value < float(spec["low"]) or numeric_value > float(spec["high"]):
            return False
        if "step" not in spec:
            return True
        step = float(spec["step"])
        offset = (numeric_value - float(spec["low"])) / step
        return math.isclose(offset, round(offset), rel_tol=0.0, abs_tol=1e-9)
    if parameter_type == "int":
        if isinstance(value, bool):
            return False
        try:
            numeric_value = int(value)
        except (TypeError, ValueError):
            return False
        if numeric_value != value and not (
            isinstance(value, float) and numeric_value == float(value)
        ):
            return False
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        return low <= numeric_value <= high and (numeric_value - low) % step == 0
    raise ValueError(f"Unsupported non-fanout parameter type: {parameter_type}")


def _sampled_fanout_matches_spec(
    sampled_params: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    profile_overrides: ProfileOverrides,
) -> bool:
    """Return whether sampled ``num_neighbors`` is valid for current branch depth."""
    resolved_params = _resolved_config_overrides_from_sampled_params(
        sampled_params,
        profile_overrides=profile_overrides,
    )
    raw_value = resolved_params.get("num_neighbors")
    if not isinstance(raw_value, list):
        return False
    try:
        interest_layers = int(resolved_params["interest_gnn_layers"])
        conformity_layers = int(resolved_params["conformity_gnn_layers"])
    except (KeyError, TypeError, ValueError):
        return False
    depth = max(interest_layers, conformity_layers)
    choices_by_depth = _require_mapping(
        spec["choices_by_depth"],
        field_name="parameters.num_neighbors.choices_by_depth",
    )
    raw_choices = choices_by_depth.get(str(depth)) or choices_by_depth.get(depth)
    if raw_choices is None:
        return False
    choices = resolve_profile_num_neighbors({"num_neighbors": raw_choices})
    if choices is None:
        return False
    return any(list(raw_value) == list(choice) for choice in choices)


def _sampled_params_match_search_space(
    sampled_params: Mapping[str, Any],
    search_space: SearchSpaceSpec,
) -> bool:
    """Return whether a completed logical trial is reusable for current search space."""
    runtime_only_fields: set[str] = set()
    if bool(search_space.config_overrides.get("auto_batch_size")) and "batch_size" not in (
        search_space.parameters
    ):
        runtime_only_fields.add("batch_size")
    logical_sampled_params = {
        key: value for key, value in sampled_params.items() if key not in runtime_only_fields
    }
    try:
        resolved_logical_params = _resolved_config_overrides_from_sampled_params(
            logical_sampled_params,
            profile_overrides=search_space.profile_overrides,
        )
    except ValueError:
        return False
    allowed_keys = set(search_space.parameters)
    if set(logical_sampled_params) - allowed_keys:
        return False
    for field_name, spec in search_space.parameters.items():
        parameter_type = str(spec["type"]).lower()
        active_context = {**resolved_logical_params, **logical_sampled_params}
        if parameter_type == "fanout":
            if not _parameter_is_conditionally_active(field_name, active_context):
                if field_name in logical_sampled_params:
                    return False
                continue
            if not _sampled_fanout_matches_spec(
                logical_sampled_params,
                spec,
                profile_overrides=search_space.profile_overrides,
            ):
                return False
            continue
        if not _parameter_is_conditionally_active(field_name, active_context):
            if field_name in logical_sampled_params:
                return False
            continue
        if field_name not in logical_sampled_params:
            return False
        if not _sampled_value_matches_spec(logical_sampled_params[field_name], field_name, spec):
            return False
    return True


def _trial_matches_current_search_contract(
    trial: optuna.trial.FrozenTrial,
    search_space: SearchSpaceSpec,
) -> bool:
    """Return whether an Optuna trial belongs to the current logical contract."""
    if str(trial.user_attrs.get("search_space")) not in method_identifier_aliases(
        search_space.name,
    ):
        return False
    if trial.user_attrs.get("objective_metric") != search_space.objective.metric:
        return False
    if trial.user_attrs.get("objective_split") != search_space.objective.split:
        return False
    sampled_params = trial.user_attrs.get("sampled_params")
    if not isinstance(sampled_params, Mapping):
        return False
    return _sampled_params_match_search_space(sampled_params, search_space)


def _trial_matches_current_search_space(
    trial: optuna.trial.FrozenTrial,
    search_space: SearchSpaceSpec,
) -> bool:
    """Return whether a completed Optuna trial belongs to the current logical contract."""
    if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
        return False
    if not _trial_matches_current_search_contract(trial, search_space):
        return False
    for dataset in search_space.datasets:
        value = trial.user_attrs.get(f"{dataset}.objective")
        if value is None:
            return False
    if len(search_space.datasets) == 1:
        dataset = search_space.datasets[0]
        dataset_value = float(trial.user_attrs[f"{dataset}.objective"])
        if not math.isclose(float(trial.value), dataset_value, rel_tol=0.0, abs_tol=1e-12):
            return False
    return True


def _trial_matches_search_budget_scope(
    trial: optuna.trial.FrozenTrial,
    search_space: SearchSpaceSpec,
) -> bool:
    """Return whether a fresh finished trial counts against ``--trials``."""
    if trial.state not in INFORMATIVE_TRIAL_STATES:
        return False
    if is_duplicate_pruned_trial(trial):
        return False
    if str(trial.user_attrs.get("search_space")) not in method_identifier_aliases(
        search_space.name,
    ):
        return False
    if trial.user_attrs.get("search_space_revision") != search_space_revision(search_space):
        return False
    if trial.user_attrs.get("objective_metric") != search_space.objective.metric:
        return False
    if trial.user_attrs.get("objective_split") != search_space.objective.split:
        return False
    if trial.state == optuna.trial.TrialState.COMPLETE:
        return trial.value is not None and math.isfinite(float(trial.value))
    return any(
        trial.user_attrs.get(f"{dataset}.pruned")
        or trial.user_attrs.get(f"{dataset}.last_pruning_objective") is not None
        for dataset in search_space.datasets
    )


def _compatible_completed_trials(
    study: optuna.Study,
    search_space: SearchSpaceSpec,
) -> list[optuna.trial.FrozenTrial]:
    """Return completed trials reusable under the current logical search space."""
    return [
        trial for trial in study.trials if _trial_matches_current_search_space(trial, search_space)
    ]


def _budget_informative_trials(
    study: optuna.Study,
    search_space: SearchSpaceSpec,
) -> list[optuna.trial.FrozenTrial]:
    """Return finished informative trials for the trial-budget contract."""
    return [
        trial
        for trial in study.trials
        if _trial_matches_search_budget_scope(trial, search_space) and not is_seeded_trial(trial)
    ]


def _budget_count_fragment(
    study: optuna.Study,
    search_space: SearchSpaceSpec,
) -> str:
    """Return a compact human-readable budget accounting fragment."""
    fresh = _budget_informative_trials(study, search_space)
    seeded = [
        trial
        for trial in study.trials
        if is_seeded_trial(trial) and _trial_matches_search_budget_scope(trial, search_space)
    ]
    fresh_complete = sum(1 for trial in fresh if trial.state == optuna.trial.TrialState.COMPLETE)
    fresh_pruned = sum(1 for trial in fresh if trial.state == optuna.trial.TrialState.PRUNED)
    seeded_complete = sum(1 for trial in seeded if trial.state == optuna.trial.TrialState.COMPLETE)
    seeded_pruned = sum(1 for trial in seeded if trial.state == optuna.trial.TrialState.PRUNED)
    return (
        f"revision={search_space_revision(search_space)} "
        f"fresh_budget_count={len(fresh)} "
        f"(fresh complete={fresh_complete}, fresh pruned={fresh_pruned}); "
        f"imported_history=(complete={seeded_complete}, pruned={seeded_pruned})"
    )


def _best_trial_from_completed(
    trials: list[optuna.trial.FrozenTrial],
    *,
    direction: str,
) -> optuna.trial.FrozenTrial | None:
    """Return the best trial within a prefiltered comparable trial set."""
    if not trials:
        return None
    reverse = direction == "maximize"
    return sorted(trials, key=lambda trial: float(trial.value), reverse=reverse)[0]


def _print_best_trial_summary(
    best_trial: optuna.trial.FrozenTrial,
    *,
    objective_metric: str,
) -> None:
    """Print best comparable trial details from effective config when available."""
    print(
        (f"Best trial: {best_trial.number} | {objective_metric}={float(best_trial.value):.6f}"),
    )
    label, payload = _best_trial_config_payload(best_trial)
    print(label)
    print(json.dumps(_json_safe(payload), sort_keys=True))


def _trial_dataset_names(trial: optuna.trial.FrozenTrial) -> list[str]:
    """Return dataset names recorded on a trial."""
    raw_datasets = trial.user_attrs.get("datasets")
    if isinstance(raw_datasets, Sequence) and not isinstance(raw_datasets, (str, bytes)):
        return [str(dataset) for dataset in raw_datasets]
    return sorted(
        {
            key.split(".", 1)[0]
            for key in trial.user_attrs
            if key.endswith(".objective") and "." in key
        },
    )


def _best_trial_config_payload(
    trial: optuna.trial.FrozenTrial,
) -> tuple[str, Mapping[str, Any]]:
    """Return a truthful best-trial config payload for CLI display."""
    dataset_names = _trial_dataset_names(trial)
    if len(dataset_names) == 1:
        dataset = dataset_names[0]
        effective_config = trial.user_attrs.get(f"{dataset}.effective_config")
        if isinstance(effective_config, Mapping):
            return "Best effective config:", effective_config

    payload: dict[str, Any] = {}
    sampled_params = trial.user_attrs.get("sampled_params")
    if isinstance(sampled_params, Mapping):
        payload.update(sampled_params)
    if len(dataset_names) == 1:
        dataset = dataset_names[0]
        for attr_name in (
            "batch_size",
            "auto_batch_size",
            "avg_epoch_time_s",
            "peak_vram_mb",
            "epochs_stopped_at",
            "canonical_name",
        ):
            value = trial.user_attrs.get(f"{dataset}.{attr_name}")
            if value is not None:
                payload[attr_name] = value
    return "Best sampled/runtime params (historical trial lacks effective_config):", payload


def _print_best_trial_or_note(
    compatible_completed: Sequence[optuna.trial.FrozenTrial],
    *,
    objective_metric: str,
    direction: str,
) -> None:
    """Print a best trial, or explain why no best row can be reused."""
    best_trial = _best_trial_from_completed(
        list(compatible_completed),
        direction=direction,
    )
    if best_trial is not None:
        _print_best_trial_summary(
            best_trial,
            objective_metric=objective_metric,
        )
        return
    print(
        (
            "No completed trial matches the current logical search-space parameters; "
            "fresh budget may still be spent by pruned or older incompatible completed trials."
        ),
    )


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
    profile_overrides = {
        field_name: search_space.profile_overrides[field_name]
        for field_name in search_space.parameters
        if field_name in _PROFILE_PARAMETER_FIELDS
    }

    return {
        "search_space": search_space.name,
        "description": search_space.description,
        "base_profile": search_space.base_profile,
        "datasets": list(search_space.datasets),
        "study_name": study_name,
        "storage": storage,
        "objective": dataclasses.asdict(search_space.objective),
        "sampler": dataclasses.asdict(search_space.sampler),
        "pruner": dataclasses.asdict(search_space.pruner),
        "max_epochs": search_space.max_epochs,
        "search_space_revision": search_space_revision(search_space),
        "trials": int(trials or search_space.trials),
        "trial_budget": (
            "for this search_space_revision only: fresh informative finished trials, "
            "COMPLETE plus real PRUNED; excludes FAIL, RUNNING, historically imported rows, "
            "duplicate-skip prunes, and trials from other search-space revisions"
        ),
        "config_overrides": _json_safe(search_space.config_overrides),
        "parameters": _json_safe(search_space.parameters),
        "profile_overrides": _json_safe(profile_overrides),
        "base_configs": base_configs,
    }


def _run_single_search_study(
    args: argparse.Namespace,
    *,
    search_space: SearchSpaceSpec,
    study_name: str,
    n_trials: int,
) -> int:
    """Run one Optuna study for one resolved search-space dataset selection."""
    _ensure_storage_parent(args.storage)
    public_study_name = study_name
    storage_study_name = _resolve_existing_study_name(args.storage, public_study_name)
    if storage_study_name != public_study_name:
        print(f"Reusing legacy Optuna study {storage_study_name} for {public_study_name}.")
    study = optuna.create_study(
        study_name=storage_study_name,
        storage=args.storage,
        direction=search_space.objective.direction,
        sampler=_build_sampler(search_space.sampler),
        pruner=_build_pruner(search_space.pruner, max_epochs=search_space.max_epochs),
        load_if_exists=True,
    )
    compatible_completed = _compatible_completed_trials(study, search_space)
    budget_informative = _budget_informative_trials(study, search_space)
    print(
        f"Study {public_study_name} trial budget status: "
        f"{_budget_count_fragment(study, search_space)}; "
        f"target={n_trials}."
    )
    if len(budget_informative) >= n_trials:
        print(
            (
                f"Study {public_study_name} already has {len(budget_informative)} fresh "
                f"informative trial(s); target={n_trials}. Nothing to run."
            ),
        )
        _print_best_trial_or_note(
            compatible_completed,
            objective_metric=search_space.objective.metric,
            direction=search_space.objective.direction,
        )
        return 0

    starting_trial_count = len(study.trials)
    target_trials = n_trials
    remaining = target_trials - len(budget_informative)
    max_attempts = max(remaining, remaining * 4)
    seen_sampled_param_keys = {
        _canonical_sampled_params(trial.user_attrs["sampled_params"])
        for trial in _budget_informative_trials(study, search_space)
        if isinstance(trial.user_attrs.get("sampled_params"), Mapping)
    }
    objective = _objective_factory(
        search_space,
        study_name=public_study_name,
        device=args.device,
        data_dir=args.data_dir,
        enable_mlflow=not bool(args.no_mlflow),
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        seen_sampled_param_keys=seen_sampled_param_keys,
    )
    attempts = 0
    while len(budget_informative) < target_trials and attempts < max_attempts:
        before_trial_count = len(study.trials)
        study.optimize(
            objective,
            n_trials=1,
            catch=(Exception,),
        )
        attempted_trials = study.trials[before_trial_count:]
        attempts += max(1, len(study.trials) - before_trial_count)
        compatible_completed = _compatible_completed_trials(study, search_space)
        budget_informative = _budget_informative_trials(study, search_space)
        if any(_trial_has_poisoned_cuda_context_failure(trial) for trial in attempted_trials):
            print(
                (
                    "Aborting Optuna search after CUDA device-side assert. "
                    "CUDA context is poisoned in this Python process; restart before retrying."
                ),
            )
            break

    new_trials = study.trials[starting_trial_count:]
    if len(budget_informative) >= target_trials:
        _print_best_trial_or_note(
            compatible_completed,
            objective_metric=search_space.objective.metric,
            direction=search_space.objective.direction,
        )
        return 0

    failed_new_trials = [
        trial for trial in new_trials if trial.state == optuna.trial.TrialState.FAIL
    ]
    print(
        (
            "Optuna search did not reach the requested fresh informative-trial target: "
            f"{len(budget_informative)}/{target_trials}. "
            f"{_budget_count_fragment(study, search_space)}."
        ),
    )
    if failed_new_trials:
        print(f"Failed trials in this invocation: {len(failed_new_trials)}")
    return 1


def _dataset_study_name(
    explicit_study_name: str | None,
    *,
    space_name: str,
    dataset_space: SearchSpaceSpec,
) -> str:
    """Return the study name for one dataset-local search."""
    dataset = dataset_space.datasets[0]
    if explicit_study_name:
        return f"{explicit_study_name}-{dataset}"
    return default_study_name(space_name, dataset_space.datasets, search_space=dataset_space)


def _build_per_dataset_dry_run_payload(
    args: argparse.Namespace,
    *,
    space_name: str,
    search_space: SearchSpaceSpec,
    n_trials: int,
) -> dict[str, Any]:
    """Return dry-run details for the default dataset-local search expansion."""
    payloads: dict[str, Any] = {}
    for dataset in search_space.datasets:
        dataset_space = resolve_search_space(space_name, dataset=dataset)
        study_name = _dataset_study_name(
            args.study_name,
            space_name=space_name,
            dataset_space=dataset_space,
        )
        payloads[dataset] = build_dry_run_payload(
            dataset_space,
            study_name=study_name,
            storage=args.storage,
            trials=n_trials,
            device=args.device,
            data_dir=args.data_dir,
        )
    return {
        "search_mode": "per_dataset",
        "search_space": search_space.name,
        "datasets": list(search_space.datasets),
        "target_trials_per_dataset": n_trials,
        "storage": args.storage,
        "dataset_payloads": payloads,
    }


def _parse_search_space_sequence(raw_space: str | None) -> list[str]:
    """Return one or more search-space identifiers from CLI input."""
    if raw_space is None:
        return []
    space_names = [part.strip() for part in raw_space.split(",") if part.strip()]
    if not space_names:
        raise ValueError("--space must name at least one search space.")
    return space_names


def _build_single_space_dry_run_payload(
    args: argparse.Namespace,
    *,
    space_name: str,
) -> dict[str, Any]:
    """Return dry-run details for one queued search-space entry."""
    search_space = resolve_search_space(space_name, dataset=args.dataset)
    n_trials = int(args.trials or search_space.trials)
    if n_trials < 1:
        raise ValueError("--trials must be >= 1.")

    if args.dataset is None and len(search_space.datasets) > 1:
        return _build_per_dataset_dry_run_payload(
            args,
            space_name=space_name,
            search_space=search_space,
            n_trials=n_trials,
        )

    study_name = args.study_name or default_study_name(
        space_name,
        search_space.datasets,
        search_space=search_space,
    )
    return build_dry_run_payload(
        search_space,
        study_name=study_name,
        storage=args.storage,
        trials=n_trials,
        device=args.device,
        data_dir=args.data_dir,
    )


def _build_search_space_queue_dry_run_payload(
    args: argparse.Namespace,
    *,
    space_names: Sequence[str],
) -> dict[str, Any]:
    """Return one dry-run payload for a sequential search-space queue."""
    return {
        "search_mode": "search_space_queue",
        "search_spaces": list(space_names),
        "dataset": args.dataset,
        "storage": args.storage,
        "space_payloads": {
            space_name: _build_single_space_dry_run_payload(args, space_name=space_name)
            for space_name in space_names
        },
    }


def _validate_search_space_queue(
    args: argparse.Namespace,
    *,
    space_names: Sequence[str],
) -> None:
    """Validate every queued search-space entry before training starts."""
    for space_name in space_names:
        resolve_search_space(space_name, dataset=args.dataset)


def _run_search_space(args: argparse.Namespace, *, space_name: str) -> int:
    """Execute one resolved Optuna search-space entry from parsed CLI args."""
    search_space = resolve_search_space(space_name, dataset=args.dataset)
    n_trials = int(args.trials or search_space.trials)
    if n_trials < 1:
        raise ValueError("--trials must be >= 1.")

    if args.dataset is None and len(search_space.datasets) > 1:
        if args.dry_run:
            print(
                json.dumps(
                    _build_per_dataset_dry_run_payload(
                        args,
                        space_name=space_name,
                        search_space=search_space,
                        n_trials=n_trials,
                    ),
                    indent=2,
                    sort_keys=True,
                ),
            )
            return 0

        exit_codes: list[int] = []
        for dataset in search_space.datasets:
            dataset_space = resolve_search_space(space_name, dataset=dataset)
            study_name = _dataset_study_name(
                args.study_name,
                space_name=space_name,
                dataset_space=dataset_space,
            )
            print(f"Dataset-local Optuna search: {dataset} -> {study_name}")
            exit_codes.append(
                _run_single_search_study(
                    args,
                    search_space=dataset_space,
                    study_name=study_name,
                    n_trials=n_trials,
                ),
            )
        return 0 if all(code == 0 for code in exit_codes) else 1

    study_name = args.study_name or default_study_name(
        space_name,
        search_space.datasets,
        search_space=search_space,
    )
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
    return _run_single_search_study(
        args,
        search_space=search_space,
        study_name=study_name,
        n_trials=n_trials,
    )


def run_search(args: argparse.Namespace) -> int:
    """Execute one or more Optuna searches from parsed CLI args."""
    if args.list_spaces:
        print("Available search spaces:")
        for space_name in search_space_names():
            space = get_search_space(space_name)
            description = space.get("description", "")
            print(f"  {space_name}: {description}")
        return 0

    space_names = _parse_search_space_sequence(args.space)
    if not space_names:
        raise ValueError("--space is required unless --list-spaces is used.")

    if len(space_names) == 1:
        return _run_search_space(args, space_name=space_names[0])

    if args.study_name:
        raise ValueError("--study-name is ambiguous with multiple --space entries.")

    _validate_search_space_queue(args, space_names=space_names)

    if args.dry_run:
        print(
            json.dumps(
                _build_search_space_queue_dry_run_payload(args, space_names=space_names),
                indent=2,
                sort_keys=True,
            ),
        )
        return 0

    print("=" * 70)
    print("OPTUNA SEARCH SPACE QUEUE")
    for index, space_name in enumerate(space_names, 1):
        print(f"  {index}. {space_name}")
    print("=" * 70)

    exit_codes = [_run_search_space(args, space_name=space_name) for space_name in space_names]
    return 0 if all(exit_code == 0 for exit_code in exit_codes) else 1


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
