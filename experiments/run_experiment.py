#!/usr/bin/env python
"""Main single-experiment CLI runner for U-CaGNN.

Usage:
    python experiments/run_experiment.py --dataset movielens1m --preset lightgcn --epochs 3
    python experiments/run_experiment.py --dataset kuairec_v2 --preset ucagnn
"""

from __future__ import annotations

import argparse
import gc
import dataclasses
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np

from experiments.recipes import (
    get_recipe,
    recipe_names,
    recipe_summary_lines,
)
from src.utils.config import DEFAULT_SEED, UCaGNNConfig
from src.utils.experiment_logger import ExperimentLogger
from src.data.loaders import load_dataset
from src.data.graph_builder import build_graph
from src.models.ucagnn import UCaGNN
from src.losses.loss_suite import LossSuite
from src.training.mini_batch_trainer import MiniBatchTrainer
from src.profiling.gpu_profiler import GPUProfiler

logger = logging.getLogger("ucagnn")

DB_PATH = Path(__file__).parent.parent / "results" / "thesis_experiments.db"
MLFLOW_DB_PATH = Path(__file__).parent.parent / "results" / "mlflow.db"
CHECKPOINT_DIR = Path(__file__).parent.parent / "results" / "checkpoints"
REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "model_state",
        "optimizer_state",
        "loss_suite_state",
        "config",
    }
)

PRESETS = {
    "ucagnn": "preset_full",
    "lightgcn": "preset_lightgcn",
    "dice_like": "preset_dice_like",
}
DEFAULT_PRESET_ORDER = ["ucagnn", "lightgcn", "dice_like"]


def canonical_preset_name(preset: str | None) -> str | None:
    """Return the canonical public preset name."""
    return preset


def canonicalize_preset_names(presets: list[str]) -> list[str]:
    """Normalize and deduplicate preset names while preserving order."""
    normalized: list[str] = []
    for preset in presets:
        canonical = canonical_preset_name(preset)
        if canonical is None or canonical in normalized:
            continue
        normalized.append(canonical)
    return normalized


def _config_as_dict(config: UCaGNNConfig) -> dict:
    """Convert config dataclass to a comparable dictionary."""
    return dataclasses.asdict(config)


def _build_canonical_name(
    config: UCaGNNConfig,
    preset: str | None,
    intervention: str | None,
) -> str:
    """Build a descriptive canonical experiment name from the effective config."""
    preset = canonical_preset_name(preset)
    parts = [
        config.dataset,
        preset or "custom",
        config.graph_method,
        f"ep{config.epochs}",
        f"bs{config.batch_size}",
        f"dim{config.embed_dim}",
        f"layers{config.n_gnn_layers}",
    ]
    if config.use_dual_branch and (
        config.resolved_interest_gnn_layers != config.n_gnn_layers
        or config.resolved_conformity_gnn_layers != config.n_gnn_layers
    ):
        parts.append(
            f"branchL{config.resolved_interest_gnn_layers}-{config.resolved_conformity_gnn_layers}"
        )
    neighbor_str = "-".join(str(value) for value in config.num_neighbors)
    parts.append(f"nbr{neighbor_str}")
    if config.sample_interactions is not None:
        parts.append(f"sample{config.sample_interactions}")
    if config.loader_max_rows is not None:
        parts.append(f"loadrows{config.loader_max_rows}")
    if config.preprocessing_preset is not None:
        parts.append(f"ppreset{config.preprocessing_preset}")
    if config.derived_split_mode != "per_user_temporal":
        parts.append(f"split{config.derived_split_mode}")
    if config.popularity_window_seconds is not None:
        parts.append(f"popwin{config.popularity_window_seconds}")
    if config.use_features:
        parts.append("feat")
    if config.feature_policy != "thesis_default":
        parts.append(f"fpolicy{config.feature_policy}")
    if config.scoring_weight_mode != "fixed":
        parts.append(f"scoremix{config.scoring_weight_mode}")
    if getattr(config, "train_scoring_mode", "default") != "default":
        parts.append(f"trainscore{config.train_scoring_mode}")
    if config.eval_scoring_mode != "default":
        parts.append(f"score{config.eval_scoring_mode}")
    if intervention:
        parts.append(intervention)
    parts.append(f"seed{config.seed}")
    return "_".join(parts)


def _resolve_checkpoint_path(
    config: UCaGNNConfig,
    preset: str | None,
    intervention: str | None,
    checkpoint_path: str | None,
) -> Path:
    """Resolve the checkpoint path for a run."""
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    preset = canonical_preset_name(preset)
    return CHECKPOINT_DIR / f"{_build_canonical_name(config, preset, intervention)}.pt"


def checkpoint_payload_has_required_keys(payload: object) -> bool:
    """Return whether a checkpoint payload contains the required runtime keys."""
    return isinstance(payload, dict) and REQUIRED_CHECKPOINT_KEYS.issubset(payload)


def load_checkpoint_payload(
    path: str | Path,
    device: str,
    *,
    require_runtime_keys: bool = False,
    require_config: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint payload and optionally validate the shared schema."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary.")

    if require_runtime_keys and not checkpoint_payload_has_required_keys(payload):
        missing_keys = sorted(REQUIRED_CHECKPOINT_KEYS.difference(payload))
        raise ValueError(
            "checkpoint is missing required runtime keys: " + ", ".join(missing_keys)
        )

    if require_config:
        config = payload.get("config")
        if not isinstance(config, UCaGNNConfig):
            raise TypeError(
                "checkpoint does not contain a UCaGNNConfig under the 'config' field"
            )

    return payload


def _select_sample_counts(total: int, split_sizes: list[int]) -> list[int]:
    """Allocate an exact sample budget across splits while preserving coverage."""
    if total >= sum(split_sizes):
        return list(split_sizes)

    counts = [0] * len(split_sizes)
    remaining = total
    active = [index for index, size in enumerate(split_sizes) if size > 0]

    if remaining >= len(active):
        for index in active:
            counts[index] = 1
            remaining -= 1

    if remaining <= 0:
        return counts

    total_available = sum(split_sizes)
    raw_shares = [remaining * size / total_available for size in split_sizes]
    base = [int(share) for share in raw_shares]
    for index, value in enumerate(base):
        increment = min(value, split_sizes[index] - counts[index])
        counts[index] += increment
        remaining -= increment

    if remaining <= 0:
        return counts

    remainders = sorted(
        range(len(split_sizes)),
        key=lambda index: raw_shares[index] - int(raw_shares[index]),
        reverse=True,
    )
    for index in remainders:
        if remaining == 0:
            break
        spare_capacity = split_sizes[index] - counts[index]
        if spare_capacity <= 0:
            continue
        counts[index] += 1
        remaining -= 1

    return counts


def _sample_canonical_interactions(
    canonical,
    sample_interactions: int | None,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    derived_split_mode: str = "per_user_temporal",
):
    """Return a sampled CanonicalInteractions subset for fast preflight runs."""
    if sample_interactions is None or sample_interactions >= len(canonical):
        return canonical

    rng = np.random.default_rng(seed)
    train_mask, val_mask, test_mask = canonical.get_splits(
        train_ratio,
        val_ratio,
        derived_split_mode=derived_split_mode,
    )
    split_indices = [
        np.flatnonzero(train_mask),
        np.flatnonzero(val_mask),
        np.flatnonzero(test_mask),
    ]
    split_sizes = [len(indices) for indices in split_indices]
    sample_counts = _select_sample_counts(sample_interactions, split_sizes)

    chosen_parts: list[np.ndarray] = []
    for indices, count in zip(split_indices, sample_counts, strict=True):
        if count <= 0:
            continue
        if count >= len(indices):
            chosen = indices
        else:
            chosen = np.sort(rng.choice(indices, size=count, replace=False))
        chosen_parts.append(chosen)

    selected = (
        np.sort(np.concatenate(chosen_parts))
        if chosen_parts
        else np.array([], dtype=np.int64)
    )
    selected_train = np.isin(selected, split_indices[0])
    selected_val = np.isin(selected, split_indices[1])
    selected_test = np.isin(selected, split_indices[2])

    selected_users, user_inverse = np.unique(
        canonical.user_id[selected], return_inverse=True
    )
    selected_items, item_inverse = np.unique(
        canonical.item_id[selected], return_inverse=True
    )

    reverse_user_map = {value: key for key, value in canonical.user_map.items()}
    reverse_item_map = {value: key for key, value in canonical.item_map.items()}

    def _slice_entity_features(features: np.ndarray | None, entity_ids: np.ndarray):
        if features is None:
            return None
        return features[entity_ids]

    def _slice_metadata(metadata: dict | None) -> dict | None:
        if metadata is None:
            return None
        sliced: dict = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                if len(value) == len(canonical):
                    sliced[key] = value[selected]
                elif len(value) == canonical.n_users:
                    sliced[key] = value[selected_users]
                elif len(value) == canonical.n_items:
                    sliced[key] = value[selected_items]
                else:
                    sliced[key] = value
            else:
                sliced[key] = value
        return sliced

    return dataclasses.replace(
        canonical,
        user_id=user_inverse.astype(np.int64, copy=False),
        item_id=item_inverse.astype(np.int64, copy=False),
        label=canonical.label[selected],
        timestamp=canonical.timestamp[selected],
        sign=canonical.sign[selected],
        raw_target=(
            None if canonical.raw_target is None else canonical.raw_target[selected]
        ),
        behavior_type=(
            None
            if canonical.behavior_type is None
            else canonical.behavior_type[selected]
        ),
        exposure_flag=(
            None
            if canonical.exposure_flag is None
            else canonical.exposure_flag[selected]
        ),
        source_domain=(
            None
            if canonical.source_domain is None
            else canonical.source_domain[selected]
        ),
        popularity=canonical.popularity[selected_items],
        n_users=int(len(selected_users)),
        n_items=int(len(selected_items)),
        user_map={
            reverse_user_map[int(old_id)]: new_id
            for new_id, old_id in enumerate(selected_users.tolist())
        },
        item_map={
            reverse_item_map[int(old_id)]: new_id
            for new_id, old_id in enumerate(selected_items.tolist())
        },
        user_features=_slice_entity_features(canonical.user_features, selected_users),
        item_features=_slice_entity_features(canonical.item_features, selected_items),
        train_mask=selected_train,
        val_mask=selected_val,
        test_mask=selected_test,
        metadata=_slice_metadata(canonical.metadata),
    )


def load_runtime_data(config: UCaGNNConfig) -> tuple[Any, Any]:
    """Load canonical interactions and build the matching runtime graph."""
    canonical = load_dataset(
        config.dataset,
        config.data_dir,
        max_rows=config.loader_max_rows,
        include_optional_features=config.use_features,
        feature_policy=config.feature_policy,
        preprocessing_preset=config.preprocessing_preset,
    )
    canonical = _sample_canonical_interactions(
        canonical,
        config.sample_interactions,
        config.seed,
        config.train_ratio,
        config.val_ratio,
        config.derived_split_mode,
    )
    data = build_graph(canonical, config, embeddings=None)
    return canonical, data


def build_runtime_model(config: UCaGNNConfig, canonical: Any, data: Any) -> UCaGNN:
    """Instantiate the runtime model for a loaded canonical dataset and graph."""
    train_mask, _, _ = canonical.get_splits(
        config.train_ratio,
        config.val_ratio,
        derived_split_mode=config.derived_split_mode,
    )
    item_recency = torch.from_numpy(canonical.compute_item_recency(train_mask))
    return UCaGNN(
        canonical.n_users,
        canonical.n_items,
        config,
        item_features=getattr(data, "item_features", None),
        item_popularity=data.popularity,
        item_recency=item_recency,
    )


def _load_checkpoint_metadata(path: Path, device: str) -> dict | None:
    """Load a checkpoint payload if it exists."""
    if not path.exists():
        return None
    try:
        return load_checkpoint_payload(
            path,
            device,
            require_runtime_keys=True,
            require_config=True,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Ignoring checkpoint at %s because %s", path, exc)
        return None


def _log_mlflow_resume_tags(mlflow_module, checkpoint_state: dict | None) -> None:
    """Annotate MLflow runs with resume state when applicable."""
    if mlflow_module is None or checkpoint_state is None:
        return
    try:
        mlflow_module.set_tag("resumed", "true")
        mlflow_module.set_tag(
            "resume_completed_epoch",
            str(checkpoint_state.get("completed_epoch", -1)),
        )
    except Exception as exc:
        logger.warning("Failed to add MLflow resume tags: %s", exc)


def _build_trainer(
    config: UCaGNNConfig,
    model: UCaGNN,
    loss_suite: LossSuite,
    data,
    profiler: GPUProfiler | None,
    experiment_logger: ExperimentLogger,
    exp_id: int,
):
    """Construct the mini-batch trainer for the configured experiment."""
    return MiniBatchTrainer(
        model=model,
        loss_suite=loss_suite,
        data=data,
        config=config,
        profiler=profiler,
        experiment_logger=experiment_logger,
        exp_id=exp_id,
    )


def _default_mlflow_tracking_uri() -> str:
    """Return the MLflow tracking URI, preferring an explicit environment override."""
    if tracking_uri := os.environ.get("MLFLOW_TRACKING_URI"):
        return tracking_uri
    return f"sqlite:///{MLFLOW_DB_PATH.resolve()}"


def _build_mlflow_run_name(
    config: UCaGNNConfig,
    preset: str | None,
    intervention: str | None,
) -> str:
    """Build a deterministic MLflow run name for experiment discovery."""
    return _build_canonical_name(config, preset, intervention)


def _build_mlflow_tags(
    config: UCaGNNConfig,
    preset: str | None,
    intervention: str | None,
    experiment_id: str | None,
    recipe_name: str | None,
    batch_id: str | None,
    profile_name: str | None,
) -> dict[str, str]:
    """Build compact MLflow tags without duplicating parameter columns in the UI."""
    tags = {
        "status": "running",
    }
    if experiment_id:
        tags["experiment_id"] = experiment_id
    if recipe_name:
        tags["recipe"] = recipe_name
    if batch_id:
        tags["batch_id"] = batch_id
    if profile_name:
        tags["profile_name"] = profile_name
    return tags


def _set_mlflow_status_tags(
    mlflow_module,
    *,
    status: str,
    failure_reason: str | None = None,
    oom_flag: bool | None = None,
) -> None:
    """Mirror SQLite status fields into MLflow tags for easier filtering."""
    tags = {"status": status}
    if oom_flag is not None:
        tags["oom_flag"] = "true" if oom_flag else "false"
    if failure_reason:
        tags["failure_reason"] = failure_reason[:500]
        tags["failure_type"] = failure_reason.split(":", 1)[0]
    mlflow_module.set_tags(tags)


def _project_version() -> str:
    """Return the installed project version for MLflow metadata."""
    try:
        return importlib_metadata.version("causal-embeddings-for-recommendations")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _git_commit_short() -> str:
    """Return the current short git revision when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _run_started_at_utc() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _gpu_hardware_metadata(device: str) -> tuple[str | None, float | None]:
    """Return GPU name and VRAM size in GiB when running on CUDA."""
    if device != "cuda" or not torch.cuda.is_available():
        return None, None

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.name, props.total_memory / float(1024**3)


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return whether an exception represents a CUDA out-of-memory failure."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True

    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _build_mlflow_params(
    config: UCaGNNConfig,
    preset: str | None,
    intervention: str | None,
    recipe_name: str | None,
    run_started_at_utc: str,
    batch_id: str | None,
    profile_name: str | None,
) -> dict[str, str | int | float | bool]:
    """Select compact config fields to expose as searchable MLflow params."""
    preset = canonical_preset_name(preset)
    params: dict[str, str | int | float | bool] = {
        "dataset": config.dataset,
        "preset": preset or "custom",
        "training_mode": "mini_batch",
        "graph_method": config.graph_method,
        "seed": config.seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "embed_dim": config.embed_dim,
        "n_gnn_layers": config.n_gnn_layers,
        "interest_gnn_layers": config.resolved_interest_gnn_layers,
        "conformity_gnn_layers": config.resolved_conformity_gnn_layers,
        "train_scoring_mode": getattr(config, "train_scoring_mode", "default"),
        "eval_scoring_mode": config.eval_scoring_mode,
        "scoring_weight_mode": config.scoring_weight_mode,
        "sample_interactions": config.sample_interactions or 0,
        "loader_max_rows": config.loader_max_rows or 0,
        "lr": config.lr,
        "use_features": config.use_features,
        "feature_policy": config.feature_policy,
        "preprocessing_preset": config.preprocessing_preset or "default",
        "derived_split_mode": config.derived_split_mode,
        "popularity_window_seconds": config.popularity_window_seconds or 0,
        "use_dual_branch": config.use_dual_branch,
        "use_sign_aware": config.use_sign_aware,
        "use_counterfactual": config.use_counterfactual,
        "use_ipw": config.use_ipw,
        "enable_profiling": config.enable_profiling,
        "canonical_name": _build_canonical_name(config, preset, intervention),
        "run_started_at_utc": run_started_at_utc,
        "project_version": _project_version(),
        "git_commit": _git_commit_short(),
    }
    if intervention:
        params["intervention"] = intervention
    if recipe_name:
        params["recipe"] = recipe_name
    if batch_id:
        params["batch_id"] = batch_id
    if profile_name:
        params["profile_name"] = profile_name
    params["num_neighbors"] = "-".join(str(value) for value in config.num_neighbors)
    return params


def _sanitize_mlflow_metric_name(metric_name: str) -> str:
    """Normalize metric names to MLflow-compatible characters."""
    return metric_name.replace("@", "_at_")


def _start_mlflow_run(
    config: UCaGNNConfig,
    preset: str | None,
    intervention: str | None,
    experiment_id: str | None,
    tracking_uri: str | None,
    experiment_name: str,
    run_name: str | None,
    recipe_name: str | None,
    batch_id: str | None,
    profile_name: str | None,
):
    """Start an MLflow run and log its static metadata.

    Returns:
        Active MLflow module if startup succeeds, else ``None``.
    """
    try:
        import mlflow
    except ImportError as exc:
        logger.warning("MLflow requested but not available: %s", exc)
        return None

    resolved_tracking_uri = tracking_uri or _default_mlflow_tracking_uri()
    run_started_at_utc = _run_started_at_utc()
    try:
        MLFLOW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(resolved_tracking_uri)
        try:
            mlflow.enable_system_metrics_logging()
        except Exception as exc:
            logger.info("MLflow system metrics logging unavailable: %s", exc)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(
            run_name=run_name or _build_mlflow_run_name(config, preset, intervention)
        )
        mlflow.set_tags(
            _build_mlflow_tags(
                config,
                preset,
                intervention,
                experiment_id,
                recipe_name,
                batch_id,
                profile_name,
            )
        )
        mlflow.log_params(
            _build_mlflow_params(
                config,
                preset,
                intervention,
                recipe_name,
                run_started_at_utc,
                batch_id,
                profile_name,
            )
        )
        logger.info("MLflow tracking enabled: %s", resolved_tracking_uri)
        return mlflow
    except Exception as exc:
        logger.warning("Failed to initialize MLflow tracking: %s", exc)
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run(status="FAILED")
        except Exception:
            pass
        return None


def build_config(args: argparse.Namespace) -> UCaGNNConfig:
    """Build UCaGNNConfig from CLI args with explicit precedence.

    Precedence is: defaults -> recipe overrides -> explicit CLI flags -> preset -> recipe overrides.
    The final recipe-override pass ensures matrix-defining recipe values win over preset defaults.
    """
    kwargs: dict = {}

    recipe = get_recipe(args.recipe) if getattr(args, "recipe", None) else None
    effective_preset = canonical_preset_name(getattr(args, "preset", None))
    if recipe is not None:
        _validate_recipe_cli_conflicts(args, recipe)
        kwargs.update(recipe.get("overrides", {}))
        if effective_preset is None:
            effective_preset = canonical_preset_name(recipe.get("preset"))

    if effective_preset is not None and effective_preset not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"Unknown preset '{effective_preset}'. Available presets: {available}"
        )

    # Core args
    kwargs["dataset"] = args.dataset
    kwargs["data_dir"] = args.data_dir
    kwargs["seed"] = getattr(args, "seed", DEFAULT_SEED)
    kwargs["device"] = args.device

    if getattr(args, "epochs", None) is not None:
        kwargs["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        kwargs["batch_size"] = args.batch_size
    if getattr(args, "embed_dim", None) is not None:
        kwargs["embed_dim"] = args.embed_dim
    if getattr(args, "n_gnn_layers", None) is not None:
        kwargs["n_gnn_layers"] = args.n_gnn_layers
    if getattr(args, "interest_gnn_layers", None) is not None:
        kwargs["interest_gnn_layers"] = args.interest_gnn_layers
    if getattr(args, "conformity_gnn_layers", None) is not None:
        kwargs["conformity_gnn_layers"] = args.conformity_gnn_layers
    if getattr(args, "dropout", None) is not None:
        kwargs["dropout"] = args.dropout
    if getattr(args, "lr", None) is not None:
        kwargs["lr"] = args.lr
    if getattr(args, "use_early_stopping", None) is not None:
        kwargs["use_early_stopping"] = args.use_early_stopping
    if getattr(args, "eval_scoring_mode", None) is not None:
        kwargs["eval_scoring_mode"] = args.eval_scoring_mode
    if getattr(args, "scoring_weight_mode", None) is not None:
        kwargs["scoring_weight_mode"] = args.scoring_weight_mode
    if getattr(args, "use_features", None) is not None:
        kwargs["use_features"] = args.use_features
    if getattr(args, "feature_policy", None) is not None:
        kwargs["feature_policy"] = args.feature_policy
    if getattr(args, "preprocessing_preset", None) is not None:
        kwargs["preprocessing_preset"] = args.preprocessing_preset
    if getattr(args, "graph_method", None) is not None:
        kwargs["graph_method"] = args.graph_method
    if getattr(args, "derived_split_mode", None) is not None:
        kwargs["derived_split_mode"] = args.derived_split_mode
    if getattr(args, "popularity_window_seconds", None) is not None:
        kwargs["popularity_window_seconds"] = args.popularity_window_seconds
    if getattr(args, "num_neighbors", None) is not None:
        kwargs["num_neighbors"] = args.num_neighbors
    if getattr(args, "hard_negative_ratio", None) is not None:
        kwargs["hard_negative_ratio"] = args.hard_negative_ratio
    if getattr(args, "curriculum_phase1_end", None) is not None:
        kwargs["curriculum_phase1_end"] = args.curriculum_phase1_end
    if getattr(args, "curriculum_phase2_end", None) is not None:
        kwargs["curriculum_phase2_end"] = args.curriculum_phase2_end
    if getattr(args, "loss_schedule", None) is not None:
        kwargs["loss_schedule"] = args.loss_schedule
    if getattr(args, "sample_interactions", None) is not None:
        kwargs["sample_interactions"] = args.sample_interactions
    if getattr(args, "loader_max_rows", None) is not None:
        kwargs["loader_max_rows"] = args.loader_max_rows

    requested_loss_schedule = kwargs.get("loss_schedule")
    if requested_loss_schedule not in (None, "baseline"):
        raise ValueError(
            "loss_schedule no longer supports legacy staged BPR modes; "
            "fused BPR stays active from epoch 0."
        )

    config = UCaGNNConfig(**kwargs)

    # Apply preset
    if effective_preset in PRESETS:
        getattr(config, PRESETS[effective_preset])()

    if recipe is not None:
        for key, value in recipe.get("overrides", {}).items():
            setattr(config, key, value)

    return config


def _validate_recipe_cli_conflicts(
    args: argparse.Namespace,
    recipe: dict[str, object],
) -> None:
    """Reject CLI overrides that conflict with an explicitly selected recipe.

    Args:
        args: Parsed CLI arguments.
        recipe: Resolved recipe metadata from the experiment catalog.

    Raises:
        ValueError: If the user supplies a matrix-defining CLI flag that conflicts
            with the selected recipe.
    """
    conflicts: list[str] = []

    recipe_preset = canonical_preset_name(recipe.get("preset"))
    cli_preset = canonical_preset_name(getattr(args, "preset", None))
    if (
        cli_preset is not None
        and recipe_preset is not None
        and cli_preset != recipe_preset
    ):
        conflicts.append(
            f"preset={cli_preset!r} conflicts with recipe preset={recipe_preset!r}"
        )

    overrides = recipe.get("overrides", {})
    assert isinstance(overrides, dict)

    for field_name in ("graph_method",):
        cli_value = getattr(args, field_name, None)
        recipe_value = overrides.get(field_name)
        if (
            cli_value is not None
            and recipe_value is not None
            and cli_value != recipe_value
        ):
            cli_flag = field_name.replace("_", "-")
            conflicts.append(
                f"--{cli_flag}={cli_value!r} conflicts with recipe {field_name}={recipe_value!r}"
            )

    if conflicts:
        raise ValueError(
            "Selected recipe conflicts with explicit CLI matrix fields: "
            + "; ".join(conflicts)
            + ". Use the recipe as-is, choose a different recipe alias, or drop --recipe and pass --preset with explicit matrix flags."
        )


def run_experiment(
    config: UCaGNNConfig,
    preset: str | None = None,
    intervention: str | None = None,
    save_checkpoint: bool = True,
    enable_mlflow: bool = True,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ucagnn-thesis",
    mlflow_run_name: str | None = None,
    experiment_id: str | None = None,
    recipe_name: str | None = None,
    batch_id: str | None = None,
    profile_name: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    auto_resume: bool = True,
) -> dict:
    """Run a single experiment end-to-end. Returns test metrics dict."""
    preset = canonical_preset_name(preset)

    # Seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Device
    device = config.device if torch.cuda.is_available() else "cpu"
    config.device = device
    canonical_name = _build_canonical_name(config, preset, intervention)
    resolved_checkpoint_path = _resolve_checkpoint_path(
        config,
        preset,
        intervention,
        checkpoint_path,
    )
    checkpoint_state = None
    if auto_resume:
        checkpoint_state = _load_checkpoint_metadata(resolved_checkpoint_path, device)

    if checkpoint_state is not None:
        saved_config = checkpoint_state.get("config")
        if saved_config is not None and _config_as_dict(
            saved_config
        ) != _config_as_dict(config):
            raise ValueError(
                f"Checkpoint config mismatch for {resolved_checkpoint_path}. "
                "Use a different checkpoint path or disable auto-resume."
            )

    logger.info(
        "Dataset: %s | Preset: %s | Device: %s | Canonical name: %s",
        config.dataset,
        preset,
        device,
        canonical_name,
    )

    logger.info("Loading dataset and building graph...")
    canonical, data = load_runtime_data(config)
    logger.info(f"  {repr(canonical)}")
    logger.info(f"  Nodes: {data.num_nodes:,}, Edges: {data.edge_index.size(1):,}")
    logger.info(
        f"  Train: {data.train_mask.sum():,}, Val: {data.val_mask.sum():,}, Test: {data.test_mask.sum():,}"
    )

    model = build_runtime_model(config, canonical, data)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Loss + profiler
    loss_suite = LossSuite(config)
    profiler = (
        GPUProfiler() if torch.cuda.is_available() and config.enable_profiling else None
    )
    gpu_name, gpu_vram_gb = _gpu_hardware_metadata(device)

    # Experiment logger
    experiment_logger = ExperimentLogger(db_path=str(DB_PATH))
    exp_id = checkpoint_state.get("exp_id") if checkpoint_state is not None else None
    if exp_id is None:
        exp_id = experiment_logger.log_experiment(
            config.dataset,
            config,
            preset=preset,
            intervention=intervention,
            status="running",
            batch_id=batch_id,
            profile_name=profile_name,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
        )
    else:
        experiment_logger.update_experiment_status(exp_id, status="running")
    logger.info(f"Experiment ID: {exp_id}")

    if checkpoint_state is not None and checkpoint_state.get("is_complete"):
        experiment_logger.update_experiment_status(exp_id, status="completed")
        logger.info(
            "Checkpoint already marked complete. Returning cached result from %s",
            resolved_checkpoint_path,
        )
        experiment_logger.close()
        return {
            "exp_id": exp_id,
            "test_metrics": checkpoint_state.get("test_metrics", {}),
            "history": checkpoint_state.get(
                "history", {"train_loss": [], "val_metrics": []}
            ),
            "checkpoint_path": str(resolved_checkpoint_path),
            "canonical_name": canonical_name,
            "resumed": True,
        }

    mlflow_module = None
    mlflow_status = "FAILED"
    if enable_mlflow:
        mlflow_module = _start_mlflow_run(
            config=config,
            preset=preset,
            intervention=intervention,
            experiment_id=experiment_id,
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name,
            run_name=mlflow_run_name,
            recipe_name=recipe_name,
            batch_id=batch_id,
            profile_name=profile_name,
        )
        _log_mlflow_resume_tags(mlflow_module, checkpoint_state)

    # Trainer (routed by training_mode)
    try:
        trainer = _build_trainer(
            config,
            model,
            loss_suite,
            data,
            profiler,
            experiment_logger,
            exp_id,
        )

        start_epoch = 0
        history = None
        if checkpoint_state is not None:
            trainer.load_checkpoint(resolved_checkpoint_path)
            start_epoch = trainer.completed_epoch + 1
            history = trainer.resume_history
            logger.info(
                "Resuming from checkpoint %s at epoch %d/%d",
                resolved_checkpoint_path,
                start_epoch + 1,
                config.epochs,
            )

        # Train
        should_persist_checkpoint = save_checkpoint or auto_resume
        if start_epoch < config.epochs:
            logger.info(f"Training for {config.epochs} epochs...")
            history = trainer.train(
                start_epoch=start_epoch,
                history=history,
                checkpoint_path=resolved_checkpoint_path
                if should_persist_checkpoint
                else None,
                checkpoint_every=checkpoint_every,
            )
        else:
            history = history or {"train_loss": [], "val_metrics": []}
            logger.info(
                "Checkpoint already reached configured epoch budget; skipping training."
            )
        if history["train_loss"]:
            logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")

        # Test evaluation
        logger.info("Running test evaluation...")
        test_metrics = trainer.evaluator.evaluate(model, data, data.test_mask)
        for metric, value in sorted(test_metrics.items()):
            logger.info(f"  {metric}: {value:.4f}")
            experiment_logger.log_metric(exp_id, metric, value, split="test")

        checkpoint_path: Path | None = None
        if save_checkpoint or auto_resume:
            checkpoint_path = resolved_checkpoint_path
            trainer.save_checkpoint(
                checkpoint_path,
                history=history,
                is_complete=True,
                test_metrics=test_metrics,
                exp_id=exp_id,
                canonical_name=canonical_name,
            )

        if mlflow_module is not None:
            try:
                mlflow_module.log_metrics(
                    {
                        _sanitize_mlflow_metric_name(f"test_{metric}"): float(value)
                        for metric, value in test_metrics.items()
                    }
                )
                if checkpoint_path is not None:
                    mlflow_module.log_artifact(
                        str(checkpoint_path), artifact_path="checkpoints"
                    )
                _set_mlflow_status_tags(
                    mlflow_module,
                    status="completed",
                    oom_flag=False,
                )
            except Exception as exc:
                logger.warning("Failed to log MLflow metrics or artifacts: %s", exc)

        experiment_logger.update_experiment_status(exp_id, status="completed")
        mlflow_status = "FINISHED"
        return {
            "exp_id": exp_id,
            "test_metrics": test_metrics,
            "history": history,
            "checkpoint_path": str(checkpoint_path)
            if checkpoint_path is not None
            else None,
            "canonical_name": canonical_name,
            "resumed": checkpoint_state is not None,
        }
    except Exception as exc:
        is_oom = _is_cuda_oom(exc)
        experiment_logger.update_experiment_status(
            exp_id,
            status="oom" if is_oom else "failed",
            failure_reason=f"{type(exc).__name__}: {exc}",
            oom_flag=is_oom,
        )
        if mlflow_module is not None:
            try:
                _set_mlflow_status_tags(
                    mlflow_module,
                    status="oom" if is_oom else "failed",
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    oom_flag=is_oom,
                )
            except Exception:
                pass
        raise
    finally:
        experiment_logger.close()
        if mlflow_module is not None:
            try:
                mlflow_module.end_run(status=mlflow_status)
            except Exception as exc:
                logger.warning("Failed to close MLflow run cleanly: %s", exc)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run a U-CaGNN experiment")
    parser.add_argument("--dataset", default="movielens1m", help="Dataset name")
    parser.add_argument(
        "--recipe", choices=recipe_names(), help="Named experiment recipe"
    )
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Config preset")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--embed-dim", type=int, default=None, help="Override embed dim"
    )
    parser.add_argument(
        "--n-gnn-layers", type=int, default=None, help="Override shared GNN depth"
    )
    parser.add_argument(
        "--interest-gnn-layers",
        type=int,
        default=None,
        help="Optional interest-branch GNN depth override",
    )
    parser.add_argument(
        "--conformity-gnn-layers",
        type=int,
        default=None,
        help="Optional conformity-branch GNN depth override",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--eval-scoring-mode",
        choices=[
            "default",
            "interest_only",
            "conformity_suppressed",
        ],
        default=None,
        help="Evaluation-time scoring mode for Recall/NDCG",
    )
    parser.add_argument(
        "--scoring-weight-mode",
        choices=["fixed", "learned"],
        default=None,
        help="Scoring mixture mode for the default score: fixed config weights or learned simplex weights.",
    )
    parser.add_argument(
        "--use-features",
        dest="use_features",
        action="store_true",
        help="Enable dataset side features when available",
    )
    parser.add_argument(
        "--no-features",
        dest="use_features",
        action="store_false",
        help="Disable dataset side features even when available",
    )
    parser.add_argument(
        "--feature-policy",
        choices=["thesis_default", "all_optional"],
        default=None,
        help="Feature-loading policy: thesis_default enforces the safe thesis allowlist; all_optional restores the full optional side-feature scans.",
    )
    parser.add_argument("--graph-method", choices=["knn", "cagra"], default=None)
    parser.add_argument(
        "--num-neighbors",
        type=int,
        nargs="+",
        default=None,
        help="Fan-out per GNN layer for mini_batch mode (e.g., 10 10)",
    )
    parser.add_argument(
        "--sample-interactions",
        type=int,
        default=None,
        help="Optional interaction budget for sampled runs such as preflight.",
    )
    parser.add_argument(
        "--loader-max-rows",
        type=int,
        default=None,
        help="Optional early row cap for dataset loading during fast smoke/preflight runs.",
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--no-checkpoint", action="store_true", help="Skip saving checkpoint"
    )
    parser.add_argument(
        "--checkpoint-path", default=None, help="Optional explicit checkpoint path"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--auto-resume",
        dest="auto_resume",
        action="store_true",
        help="Resume automatically from a matching checkpoint",
    )
    parser.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
        help="Disable automatic checkpoint resume for this run",
    )
    parser.add_argument(
        "--intervention", default=None, help="Ablation intervention name"
    )
    parser.add_argument(
        "--enable-mlflow",
        dest="enable_mlflow",
        action="store_true",
        help="Explicitly enable MLflow tracking",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="enable_mlflow",
        action="store_false",
        help="Disable MLflow tracking for this run",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Override MLflow tracking URI (otherwise uses MLFLOW_TRACKING_URI or results/mlflow.db)",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="ucagnn-thesis",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow-run-name", default=None, help="Optional explicit MLflow run name"
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional thesis experiment identifier tag, e.g. E1",
    )
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help="Print available named recipes and exit",
    )
    parser.set_defaults(enable_mlflow=True)
    parser.set_defaults(auto_resume=True)
    parser.set_defaults(use_features=None)
    args = parser.parse_args()

    if args.list_recipes:
        print("Available experiment recipes:")
        print("\n".join(recipe_summary_lines()))
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = build_config(args)
    recipe = get_recipe(args.recipe) if args.recipe else None
    resolved_preset = canonical_preset_name(
        args.preset or (recipe.get("preset") if recipe else None)
    )
    result = run_experiment(
        config,
        preset=resolved_preset,
        intervention=args.intervention,
        save_checkpoint=not args.no_checkpoint,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_run_name=args.mlflow_run_name,
        experiment_id=args.experiment_id,
        recipe_name=args.recipe,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        auto_resume=args.auto_resume,
    )

    print(f"\nExperiment {result['exp_id']} complete.")
    print("Test metrics:")
    for k, v in sorted(result["test_metrics"].items()):
        print(f"  {k}: {v:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
