#!/usr/bin/env python
"""Main single-experiment CLI runner for EDGRec.

Usage:
    uv run experiment --list-recipes
    uv run experiment --dataset movielens1m --recipe edgrec
    uv run experiment --dataset kuairec_v2 --preset edgrec --overwrite-checkpoint
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import logging
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

if "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

import numpy as np
import torch
from scripts._workflow_helpers import configure_cli_logging
from src.data.canonical import filter_canonical_interactions, sample_canonical_interactions
from src.data.feature_groups import apply_graph_item_feature_subset
from src.data.graph_builder import build_graph
from src.data.loaders import default_preprocessing_preset, load_dataset
from src.losses.loss_suite import LossSuite
from src.models.baselines import PaperGCNDICE, PaperLightGCN
from src.models.edgrec import EDGRec
from src.profiling.gpu_profiler import (
    reset_cuda_peak_memory_stats as _reset_cuda_peak_memory_stats,
)
from src.training.mini_batch_trainer import MiniBatchTrainer
from src.utils.config import (
    BENCHMARK_CONFIG_FIELDS,
    CONFIG_OVERRIDE_FIELDS,
    CONFIG_PRESET_CHOICES,
    CONFIG_PRESET_METHODS,
    DEFAULT_SEED,
    EDGRecConfig,
)
from src.utils.experiment_logger import ExperimentLogger
from src.utils.experiment_naming import (
    build_canonical_experiment_name,
    format_num_neighbors_payload,
)
from src.utils.method_naming import (
    canonical_preset_for_identity,
    method_identifier_aliases,
    public_preset_name,
)
from src.utils.project_paths import (
    CHECKPOINT_DIR,
    MLFLOW_DB_PATH,
    THESIS_DB_PATH,
)
from src.utils.reproducibility import (
    build_torch_generator,
    configure_torch_runtime,
    seed_everything,
)
from src.utils.trainer_runtime import REQUIRED_CHECKPOINT_KEYS, is_cuda_oom_error

from experiments.benchmark_resolvers import (
    normalize_benchmark_lr_scheduler_override,
    normalize_benchmark_num_neighbors_override,
    normalize_benchmark_preprocessing_override,
    resolve_benchmark_item_universe_policy_value,
)
from experiments.cli_parsers import build_run_experiment_parser
from experiments.recipes import (
    get_recipe,
    recipe_summary_lines,
)

logger = logging.getLogger("edgrec")

DB_PATH = THESIS_DB_PATH


_CHECKPOINT_IDENTITY_VERSION = 1
_CHECKPOINT_HASH_LEN = 16
_CHECKPOINT_FILENAME_BYTE_LIMIT = 240
_TRAINING_IDENTITY_FIELDS = (
    "amp_dtype",
    "auxiliary_loss_schedule",
    "auxiliary_ramp_rate",
    "auto_batch_size",
    "batch_size",
    "batch_size_candidates",
    "conformity_gnn_layers",
    "contrastive_max_pairs",
    "contrastive_temperature",
    "distance_correlation_max_pairs",
    "uniformity_max_pairs",
    "auxiliary_losses_start_epoch",
    "popularity_supervision_start_epoch",
    "dataset",
    "derived_split_mode",
    "dropout",
    "ema_decay",
    "embed_dim",
    "embedding_optimizer",
    "epochs",
    "feature_policy",
    "feature_subset_mode",
    "feature_include_groups",
    "feature_exclude_groups",
    "graph_policy",
    "grad_clip_norm",
    "hard_negative_ratio",
    "score_mix_min_weight",
    "separate_item_branch_embeddings",
    "use_temporal_interest",
    "temporal_history_size",
    "paper_scaled_batch",
    "training_graph_mode",
    "branch_loss_mode",
    "recommendation_loss_mode",
    "negative_sampling_strategy",
    "dice_sampler_margin",
    "dice_sampler_pool",
    "dice_branch_margin",
    "dice_loss_decay",
    "dice_margin_decay",
    "dice_adaptive_decay",
    "independence_ramp_rate",
    "interest_gnn_layers",
    "loss_weight_align",
    "loss_weight_conformity_bpr",
    "loss_weight_contrastive",
    "loss_weight_independence",
    "loss_weight_interest_bpr",
    "loss_weight_popularity",
    "loss_weight_recommendation",
    "loss_weight_uniform",
    "loss_normalization",
    "loader_max_rows",
    "loss_schedule",
    "lr",
    "lr_scheduler",
    "lr_scheduler_factor",
    "lr_scheduler_patience",
    "n_negatives",
    "num_neighbors",
    "item_universe_policy",
    "popularity_embedding_dimensions",
    "preprocessing_preset",
    "propagation_backend",
    "propensity_clip_max",
    "propensity_clip_min",
    "propensity_hidden",
    "sample_interactions",
    "score_weight_conformity",
    "score_weight_interest",
    "score_weight_popularity",
    "seed",
    "single_branch_gnn_layers",
    "sampler_residency_policy",
    "train_edge_keep_prob",
    "train_ratio",
    "uniformity_temperature",
    "use_amp",
    "use_conformity_au",
    "use_dual_branch",
    "use_early_stopping",
    "use_ema",
    "use_features",
    "use_ipw",
    "use_learned_score_mix",
    "use_popularity_emb",
    "use_popularity_head",
    "use_sign_aware",
    "use_torch_compile",
    "validation_every_n_epochs",
    "val_ratio",
    "weight_decay",
)
_EVALUATION_IDENTITY_FIELDS: tuple[str, ...] = ()


def _stable_identity_hash(payload: dict[str, Any]) -> str:
    """Return a short deterministic hash for checkpoint identity payloads."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:_CHECKPOINT_HASH_LEN]


def _build_training_identity(
    config: EDGRecConfig,
    preset: str | None,
    intervention: str | None,
) -> tuple[dict[str, Any], str]:
    """Build the resume-compatibility identity for a training run."""
    config_values = dataclasses.asdict(config)
    config_identity = {
        field_name: config_values[field_name] for field_name in _TRAINING_IDENTITY_FIELDS
    }
    if config.dataset == "kuairand1k":
        config_identity["label_mode"] = config_values["label_mode"]
        config_identity["watch_ratio_proxy_threshold"] = config_values[
            "watch_ratio_proxy_threshold"
        ]

    identity = {
        "identity_version": _CHECKPOINT_IDENTITY_VERSION,
        "identity_kind": "training",
        "preset": canonical_preset_for_identity(preset) or "custom",
        "intervention": intervention,
        "training_mode": "mini_batch",
        "config": config_identity,
    }
    return identity, _stable_identity_hash(identity)


def _build_evaluation_identity(
    config: EDGRecConfig,
    training_hash: str,
) -> tuple[dict[str, Any], str]:
    """Build the comparability identity for same-checkpoint evaluation runs."""
    config_values = dataclasses.asdict(config)
    identity = {
        "identity_version": _CHECKPOINT_IDENTITY_VERSION,
        "identity_kind": "evaluation",
        "training_hash": training_hash,
        "config": {
            field_name: config_values[field_name] for field_name in _EVALUATION_IDENTITY_FIELDS
        },
    }
    return identity, _stable_identity_hash(identity)


def _checkpoint_filename(canonical_name: str, training_hash: str) -> str:
    """Return a checkpoint filename that fits common filesystem byte limits."""
    suffix = f"_train-{training_hash}.pt"
    filename = f"{canonical_name}{suffix}"
    if len(filename.encode("utf-8")) <= _CHECKPOINT_FILENAME_BYTE_LIMIT:
        return filename

    name_digest = hashlib.sha256(canonical_name.encode("utf-8")).hexdigest()[:8]
    shortened_suffix = f"_{name_digest}{suffix}"
    prefix_budget = _CHECKPOINT_FILENAME_BYTE_LIMIT - len(
        shortened_suffix.encode("utf-8"),
    )
    prefix = (
        canonical_name.encode("utf-8")[:prefix_budget]
        .decode("utf-8", errors="ignore")
        .rstrip("._-")
    )
    return f"{prefix}{shortened_suffix}"


def _default_checkpoint_path(
    config: EDGRecConfig,
    preset: str | None,
    intervention: str | None,
    training_hash: str,
) -> Path:
    """Return the default checkpoint path for a semantic training identity."""
    canonical_name = build_canonical_experiment_name(config, preset, intervention)
    return CHECKPOINT_DIR / _checkpoint_filename(canonical_name, training_hash)


def _default_checkpoint_path_candidates(
    config: EDGRecConfig,
    preset: str | None,
    intervention: str | None,
    training_hash: str,
) -> list[Path]:
    """Return public and legacy checkpoint paths for one semantic identity."""
    candidates = [_default_checkpoint_path(config, preset, intervention, training_hash)]
    if preset is None:
        return candidates
    for preset_alias in method_identifier_aliases(preset):
        alias_path = _default_checkpoint_path(config, preset_alias, intervention, training_hash)
        if alias_path not in candidates:
            candidates.append(alias_path)
    return candidates


def _load_checkpoint_payload(
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

    if require_runtime_keys:
        missing_keys = sorted(REQUIRED_CHECKPOINT_KEYS.difference(payload))
        if missing_keys:
            raise ValueError(
                "checkpoint is missing required runtime keys: " + ", ".join(missing_keys),
            )

    if require_config:
        config = payload.get("config")
        if not isinstance(config, EDGRecConfig):
            raise TypeError(
                "checkpoint does not contain an EDGRecConfig under the 'config' field",
            )

    return payload


def _apply_item_universe_policy(canonical: Any, config: EDGRecConfig) -> Any:
    """Apply configured item-universe compaction before graph/model construction."""
    policy = config.item_universe_policy
    if policy == "all_catalog_items":
        return canonical

    if policy == "observed_interaction_items":
        keep_mask = np.ones(len(canonical), dtype=bool)
    elif policy == "random_exposure_items_only":
        exposure_flag = getattr(canonical, "exposure_flag", None)
        if exposure_flag is None:
            raise ValueError(
                "item_universe_policy='random_exposure_items_only' requires exposure flags",
            )
        keep_mask = np.asarray(exposure_flag, dtype=bool)
    else:
        raise ValueError(f"Unsupported item_universe_policy {policy!r}.")

    if not np.any(keep_mask):
        raise ValueError(f"item_universe_policy={policy!r} selected no interactions")

    before_items = int(canonical.n_items)
    before_interactions = len(canonical)
    compacted = filter_canonical_interactions(
        canonical,
        keep_mask,
        metadata_overrides={
            "item_universe_policy": policy,
            "item_universe_original_n_items": before_items,
            "item_universe_original_interactions": before_interactions,
        },
    )
    if compacted.n_items != before_items or len(compacted) != before_interactions:
        logger.info(
            "Applied item_universe_policy=%s: interactions %d -> %d, items %d -> %d.",
            policy,
            before_interactions,
            len(compacted),
            before_items,
            compacted.n_items,
        )
    return compacted


def load_runtime_data(config: EDGRecConfig) -> tuple[Any, Any]:
    """Load canonical interactions and build the matching runtime graph."""
    label_kwargs = (
        {
            "label_mode": config.label_mode,
            "watch_ratio_proxy_threshold": config.watch_ratio_proxy_threshold,
        }
        if config.dataset == "kuairand1k"
        else {}
    )
    canonical = load_dataset(
        config.dataset,
        config.data_dir,
        max_rows=config.loader_max_rows,
        include_optional_features=config.use_features,
        feature_policy=config.feature_policy,
        preprocessing_preset=config.preprocessing_preset,
        **label_kwargs,
    )
    canonical = _apply_item_universe_policy(canonical, config)
    canonical = sample_canonical_interactions(
        canonical,
        config.sample_interactions,
        config.seed,
        config.train_ratio,
        config.val_ratio,
        config.derived_split_mode,
    )
    data = build_graph(canonical, config)
    apply_graph_item_feature_subset(data, config)
    return canonical, data


def _train_mask_numpy_from_data(data: Any) -> np.ndarray:
    """Return the runtime graph's train mask as a NumPy boolean array.

    Args:
        data: Runtime graph payload produced by ``build_graph``.

    Returns:
        np.ndarray: Boolean train mask aligned with the canonical interactions.

    Raises:
        ValueError: If the runtime graph does not expose a CPU-accessible train mask.

    """
    train_mask = getattr(data, "train_mask", None)
    if not isinstance(train_mask, torch.Tensor):
        raise ValueError("Runtime graph data must include a torch train_mask tensor.")
    return train_mask.detach().cpu().numpy()


def _train_mask_cache_key(data: Any) -> tuple[int, tuple[int, ...], int]:
    """Return a lightweight identity key for the immutable runtime train mask."""
    train_mask = getattr(data, "train_mask", None)
    if not isinstance(train_mask, torch.Tensor):
        raise ValueError("Runtime graph data must include a torch train_mask tensor.")
    return (id(train_mask), tuple(train_mask.shape), int(train_mask.sum().item()))


def _runtime_feature_cache(data: Any) -> dict[str, Any]:
    """Return the private runtime feature cache attached to a graph payload."""
    cache = getattr(data, "_edgrec_runtime_feature_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        object.__setattr__(data, "_edgrec_runtime_feature_cache", cache)
    return cache


def _train_derived_model_tensors(
    config: EDGRecConfig,
    canonical: Any,
    data: Any,
) -> dict[str, torch.Tensor | None]:
    """Return train-split model tensors, reusing cache across probe model builds."""
    cache = _runtime_feature_cache(data)
    cache_key = (
        id(canonical),
        _train_mask_cache_key(data),
        int(config.temporal_history_size),
    )
    cached = cache.get("train_derived_model_tensors")
    if isinstance(cached, dict) and cached.get("train_mask_key") == cache_key:
        return cached["tensors"]

    train_mask = _train_mask_numpy_from_data(data)
    item_recency = torch.from_numpy(canonical.compute_item_recency(train_mask))
    recent_train_items, recent_train_mask = canonical.build_recent_train_history(
        train_mask,
        history_size=int(config.temporal_history_size),
    )
    item_propensity_targets = (
        torch.from_numpy(canonical.item_propensity_targets)
        if canonical.item_propensity_targets is not None
        else None
    )
    tensors = {
        "item_recency": item_recency,
        "item_propensity_targets": item_propensity_targets,
        "recent_train_items": torch.from_numpy(recent_train_items),
        "recent_train_mask": torch.from_numpy(recent_train_mask),
    }
    cache["train_derived_model_tensors"] = {
        "train_mask_key": cache_key,
        "tensors": tensors,
    }
    return tensors


def build_runtime_model(
    config: EDGRecConfig,
    canonical: Any,
    data: Any,
) -> torch.nn.Module:
    """Instantiate the runtime model for a loaded canonical dataset and graph.

    Args:
        config: Runtime configuration.
        canonical: Canonical dataset.
        data: Runtime graph data.

    Returns:
        Instantiated model.
    """
    if config.baseline_family == "lightgcn_paper":
        return PaperLightGCN(
            canonical.n_users,
            canonical.n_items,
            config,
        )
    if config.baseline_family == "dice_paper":
        return PaperGCNDICE(
            canonical.n_users,
            canonical.n_items,
            config,
        )

    train_derived_tensors = _train_derived_model_tensors(config, canonical, data)
    apply_graph_item_feature_subset(data, config)
    return EDGRec(
        canonical.n_users,
        canonical.n_items,
        config,
        item_features=getattr(data, "item_features", None),
        item_popularity=getattr(data, "normalized_popularity", data.popularity),
        item_recency=train_derived_tensors["item_recency"],
        item_propensity_targets=train_derived_tensors["item_propensity_targets"],
        recent_train_items=train_derived_tensors["recent_train_items"],
        recent_train_mask=train_derived_tensors["recent_train_mask"],
    )


def _load_checkpoint_metadata(path: Path, device: str) -> dict | None:
    """Load a checkpoint payload if it exists."""
    if not path.exists():
        return None
    try:
        return _load_checkpoint_payload(
            path,
            device,
            require_runtime_keys=True,
            require_config=True,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Ignoring checkpoint at %s because %s", path, exc)
        return None


def _validate_resume_identity(
    checkpoint_state: dict[str, Any] | None,
    *,
    checkpoint_path: Path,
    explicit_checkpoint_path: bool,
    training_identity: dict[str, Any],
    training_hash: str,
) -> dict[str, Any] | None:
    """Return a resumable checkpoint only when its training identity matches exactly."""
    if checkpoint_state is None:
        return None

    saved_training_identity = checkpoint_state.get("training_identity")
    saved_training_hash = checkpoint_state.get("training_hash")
    if not isinstance(saved_training_identity, dict) or not isinstance(
        saved_training_hash,
        str,
    ):
        message = (
            f"Checkpoint at {checkpoint_path} lacks training identity metadata "
            "and cannot be auto-resumed safely."
        )
        if explicit_checkpoint_path:
            raise ValueError(message)
        logger.warning(message)
        return None

    if saved_training_hash != training_hash or saved_training_identity != training_identity:
        message = (
            f"Checkpoint training identity mismatch for {checkpoint_path}. "
            "A checkpoint only resumes when every training-defining parameter "
            "matches."
        )
        if explicit_checkpoint_path:
            raise ValueError(message)
        logger.info("%s Starting a fresh run.", message)
        return None

    return checkpoint_state


def _checkpoint_ready_for_evaluation(
    checkpoint_state: dict[str, Any],
    config: EDGRecConfig,
) -> bool:
    """Return whether a checkpoint can be evaluated without more training."""
    if bool(checkpoint_state.get("training_finished")):
        return True

    completed_epoch = checkpoint_state.get("completed_epoch", -1)
    if isinstance(completed_epoch, int) and completed_epoch + 1 >= int(config.epochs):
        return True

    if not bool(config.use_early_stopping):
        return False

    patience_counter = checkpoint_state.get("patience_counter", 0)
    return (
        isinstance(patience_counter, int)
        and patience_counter >= int(config.patience)
        and checkpoint_state.get("best_state") is not None
    )


def _checkpoint_lookup_configs(
    config: EDGRecConfig,
    *,
    checkpoint_path: str | Path | None = None,
) -> list[EDGRecConfig]:
    """Return config variants whose checkpoint identities are worth checking."""
    if checkpoint_path is not None or not bool(config.auto_batch_size):
        return [config]

    configs: list[EDGRecConfig] = []
    seen_batch_sizes: set[int] = set()
    for candidate_batch_size in _auto_batch_probe_candidates(config):
        batch_size = int(candidate_batch_size)
        if batch_size in seen_batch_sizes:
            continue
        seen_batch_sizes.add(batch_size)
        configs.append(dataclasses.replace(config, batch_size=batch_size))
    return configs


def recoverable_checkpoint_for_config(
    config: EDGRecConfig,
    *,
    preset: str | None = None,
    intervention: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[EDGRecConfig, Path] | None:
    """Return the compatible config/path pair for an evaluation-ready checkpoint."""
    for checkpoint_config in _checkpoint_lookup_configs(
        config,
        checkpoint_path=checkpoint_path,
    ):
        training_identity, training_hash = _build_training_identity(
            checkpoint_config,
            preset,
            intervention,
        )
        resolved_checkpoint_paths = (
            [Path(checkpoint_path)]
            if checkpoint_path is not None
            else _default_checkpoint_path_candidates(
                checkpoint_config,
                preset,
                intervention,
                training_hash,
            )
        )
        for resolved_checkpoint_path in resolved_checkpoint_paths:
            checkpoint_state = _validate_resume_identity(
                _load_checkpoint_metadata(resolved_checkpoint_path, "cpu"),
                checkpoint_path=resolved_checkpoint_path,
                explicit_checkpoint_path=checkpoint_path is not None,
                training_identity=training_identity,
                training_hash=training_hash,
            )
            if checkpoint_state is None:
                continue
            if not _checkpoint_ready_for_evaluation(checkpoint_state, checkpoint_config):
                continue
            return checkpoint_config, resolved_checkpoint_path
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


def _default_mlflow_tracking_uri() -> str:
    """Return the MLflow tracking URI, preferring an explicit environment override."""
    if tracking_uri := os.environ.get("MLFLOW_TRACKING_URI"):
        return tracking_uri
    return f"sqlite:///{MLFLOW_DB_PATH.resolve()}"


def _build_run_provenance(
    training_hash: str,
    evaluation_hash: str,
) -> dict[str, str]:
    """Return lightweight code-version provenance for one run."""
    try:
        project_version = importlib_metadata.version(
            "efficient-disentangled-graph-recommender",
        )
    except importlib_metadata.PackageNotFoundError:
        project_version = "unknown"

    try:
        git_process = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        git_commit = git_process.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        git_commit = "unknown"

    return {
        "training_hash": training_hash,
        "evaluation_hash": evaluation_hash,
        "project_version": project_version,
        "git_commit": git_commit,
    }


def _build_mlflow_tags(
    experiment_id: str | None,
    recipe_name: str | None,
    batch_id: str | None,
    profile_name: str | None,
    change_note: str | None,
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
    if change_note:
        tags["change_note"] = change_note
    return tags


def _gpu_hardware_metadata(device: str) -> tuple[str | None, float | None]:
    """Return GPU name and VRAM size in GiB when running on CUDA."""
    if device != "cuda" or not torch.cuda.is_available():
        return None, None

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.name, props.total_memory / float(1024**3)


def _exception_summary(exc: BaseException, *, max_chars: int = 500) -> str:
    """Return a compact one-line exception summary for run logs."""
    message = str(exc).strip()
    message = message.splitlines()[0] if message else repr(exc)
    if len(message) > max_chars:
        message = f"{message[: max_chars - 3]}..."
    return f"{type(exc).__name__}: {message}"


def _cuda_memory_snapshot() -> str:
    """Return PyTorch CUDA allocator state for diagnosing OOM decisions."""
    if not torch.cuda.is_available():
        return "cuda_memory=unavailable"
    try:
        mib = 1024**2
        allocated_mb = torch.cuda.memory_allocated() / mib
        reserved_mb = torch.cuda.memory_reserved() / mib
        peak_allocated_mb = torch.cuda.max_memory_allocated() / mib
        peak_reserved_mb = torch.cuda.max_memory_reserved() / mib
    except Exception as exc:
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
        return f"cuda_memory=unavailable ({type(exc).__name__}: {message})"
    return (
        "cuda_memory="
        f"allocated={allocated_mb:.0f}MB "
        f"reserved={reserved_mb:.0f}MB "
        f"peak_allocated={peak_allocated_mb:.0f}MB "
        f"peak_reserved={peak_reserved_mb:.0f}MB"
    )


_AUTO_BATCH_PROBE_STEPS = 3
_AUTO_BATCH_VERIFY_STEPS = 1


class _AutoBatchProbeOOMError(RuntimeError):
    """CUDA OOM annotated with the auto-batch probe stage and subgraph size."""

    def __init__(
        self,
        stage: str,
        candidate_batch_size: int,
        sub_batch: Any,
        exc: BaseException,
    ) -> None:
        self.stage = stage
        self.candidate_batch_size = candidate_batch_size
        self.original = exc
        details = _subgraph_batch_summary(sub_batch)
        super().__init__(
            (
                "CUDA out of memory during auto-batch probe "
                f"stage={stage} candidate_batch_size={candidate_batch_size} "
                f"{details} caused_by={_exception_summary(exc)}"
            ),
        )


def _subgraph_batch_summary(sub_batch: Any) -> str:
    """Return compact subgraph dimensions for probe OOM diagnostics."""
    if sub_batch is None:
        return "subgraph=unavailable"
    try:
        return (
            f"effective_batch={int(sub_batch.batch_user_local.numel())} "
            f"sub_users={int(sub_batch.n_sub_users)} "
            f"sub_items={int(sub_batch.n_sub_items)} "
            f"sub_edges={int(sub_batch.sub_edge_index.size(1))}"
        )
    except Exception:
        return "subgraph=unavailable"


def _release_cuda_probe_memory() -> None:
    """Collect Python garbage and release cached CUDA allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _auto_batch_probe_candidates(config: EDGRecConfig) -> list[int]:
    """Return batch-size probe candidates in descending order.

    Args:
        config: Runtime config before auto batch-size resolution.

    Returns:
        list[int]: Ordered batch sizes to try, largest first.

    """
    configured = [int(value) for value in config.batch_size_candidates]
    candidates = sorted(dict.fromkeys(configured), reverse=True)
    if int(config.batch_size) not in candidates:
        insert_index = next(
            (index for index, value in enumerate(candidates) if int(config.batch_size) > value),
            len(candidates),
        )
        candidates.insert(insert_index, int(config.batch_size))
    return candidates


def _auto_batch_probe_interactions(
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    config: EDGRecConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the epoch-0 shuffled training interactions used by auto-batch probing."""
    if train_users.numel() <= 1:
        return train_users, train_items
    perm = torch.randperm(
        train_users.size(0),
        generator=build_torch_generator(config.seed, train_users.device),
        device=train_users.device,
    )
    return train_users[perm], train_items[perm]


def _probe_batch_size_candidate(
    config: EDGRecConfig,
    canonical: Any,
    data: Any,
    candidate_batch_size: int,
    batch_users: torch.Tensor,
    batch_items: torch.Tensor,
    random_seed: int,
) -> None:
    """Run one real training step to verify that a batch size fits in VRAM.

    Args:
        config: Runtime config with the candidate batch size already applied.
        data: Built runtime graph data object.
        candidate_batch_size: Batch size under test.
        batch_users: User IDs for the representative probe batch.
        batch_items: Item IDs for the representative probe batch.
        random_seed: Deterministic seed matching the real batch sampler seed.

    Returns:
        None. The function raises on failure.

    """
    probe_model = None
    probe_loss_suite = None
    probe_trainer = None
    sub_batch = None
    losses = None
    stage = "build_runtime_model"
    try:
        probe_model = build_runtime_model(config, canonical, data)
        stage = "build_trainer"
        probe_loss_suite = LossSuite(config)
        probe_trainer = MiniBatchTrainer(
            model=probe_model,
            loss_suite=probe_loss_suite,
            data=data,
            config=config,
            profiler=None,
            experiment_logger=None,
            exp_id=None,
            mlflow_module=None,
        )
        batch_size = min(candidate_batch_size, int(batch_users.numel()))
        if batch_size <= 0:
            return

        probe_users = batch_users[:batch_size]
        probe_items = batch_items[:batch_size]
        if config.training_graph_mode == "full":
            stage = "forward_loss"
            _, losses = probe_trainer._run_full_graph_training_batch(
                probe_users,
                probe_items,
                probe_trainer.popularity,
                epoch=0,
                random_seed=random_seed,
            )
        else:
            stage = "prepare_batch"
            sub_batch = probe_trainer._prepare_batch(
                probe_users,
                probe_items,
                random_seed=random_seed,
                epoch=0,
            )
            stage = "forward_loss"
            _, losses = probe_trainer._run_training_batch(
                sub_batch,
                probe_trainer.popularity,
                epoch=0,
            )
        stage = "backward_step"
        probe_trainer._apply_optimization_step(losses["total"])
    except Exception as exc:
        if is_cuda_oom_error(exc):
            raise _AutoBatchProbeOOMError(
                stage,
                candidate_batch_size,
                sub_batch,
                exc,
            ) from exc
        raise
    finally:
        if probe_trainer is not None:
            probe_trainer.optimizer.zero_grad(set_to_none=True)
            probe_trainer.subgraph_sampler = None
            probe_trainer._release_full_graph_cache_for_eval()
        sub_batch = None
        losses = None
        del probe_trainer, probe_loss_suite, probe_model
        _release_cuda_probe_memory()


def _resolve_auto_batch_size(
    config: EDGRecConfig,
    canonical: Any,
    data: Any,
) -> None:
    """Update ``config.batch_size`` to the largest feasible dataset-aware value.

    Args:
        config: Mutable runtime config.
        data: Built runtime graph data object.

    Returns:
        None. ``config.batch_size`` is updated in place when probing succeeds.

    Raises:
        RuntimeError: If every candidate batch size fails with CUDA OOM.

    """
    if not config.auto_batch_size or config.device != "cuda" or not torch.cuda.is_available():
        return

    candidates = _auto_batch_probe_candidates(config)
    logger.info(
        "Auto batch-size probe for %s will try: %s",
        config.dataset,
        ", ".join(str(value) for value in candidates),
    )
    failures: list[int] = []
    original_batch_size = config.batch_size
    shuffled_users, shuffled_items = _auto_batch_probe_interactions(
        data.user_nodes[data.train_mask],
        data.item_nodes[data.train_mask] - data.n_users,
        config,
    )
    for candidate in candidates:
        config.batch_size = candidate
        _release_cuda_probe_memory()
        _reset_cuda_peak_memory_stats()
        try:
            for probe_index in range(_AUTO_BATCH_PROBE_STEPS):
                start = probe_index * candidate
                if start >= int(shuffled_users.numel()):
                    break
                end = min(start + candidate, int(shuffled_users.numel()))
                _probe_batch_size_candidate(
                    config,
                    canonical,
                    data,
                    candidate,
                    shuffled_users[start:end],
                    shuffled_items[start:end],
                    random_seed=config.seed + probe_index,
                )
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            failures.append(candidate)
            logger.info(
                "Auto batch-size probe rejected %d on %s due to CUDA OOM (%s; %s).",
                candidate,
                config.dataset,
                _exception_summary(exc),
                _cuda_memory_snapshot(),
            )
            continue

        logger.info(
            "Auto batch-size probe selected %d for %s after %d representative shuffled batches.",
            candidate,
            config.dataset,
            min(
                _AUTO_BATCH_PROBE_STEPS,
                max(1, (int(shuffled_users.numel()) + candidate - 1) // candidate),
            ),
        )
        return

    config.batch_size = original_batch_size
    tried = ", ".join(str(value) for value in failures)
    raise RuntimeError(
        f"Automatic batch-size probe exhausted all candidates for {config.dataset}: {tried}",
    )


def _verify_selected_auto_batch_size(
    config: EDGRecConfig,
    canonical: Any,
    data: Any,
) -> None:
    """Re-check the selected auto batch size on the current post-probe CUDA state.

    This guards against the handoff case where a candidate passes the probe but
    still fails on the first real training batch because allocator state changed
    across the sequence of earlier probe attempts.
    """
    if not config.auto_batch_size or config.device != "cuda" or not torch.cuda.is_available():
        return

    candidates = _auto_batch_probe_candidates(config)
    try:
        start_index = candidates.index(int(config.batch_size))
    except ValueError:
        start_index = 0

    selected_batch_size = int(config.batch_size)
    shuffled_users, shuffled_items = _auto_batch_probe_interactions(
        data.user_nodes[data.train_mask],
        data.item_nodes[data.train_mask] - data.n_users,
        config,
    )

    for candidate in candidates[start_index:]:
        config.batch_size = candidate
        _release_cuda_probe_memory()
        _reset_cuda_peak_memory_stats()
        try:
            for probe_index in range(_AUTO_BATCH_VERIFY_STEPS):
                start = probe_index * candidate
                if start >= int(shuffled_users.numel()):
                    break
                end = min(start + candidate, int(shuffled_users.numel()))
                _probe_batch_size_candidate(
                    config,
                    canonical,
                    data,
                    candidate,
                    shuffled_users[start:end],
                    shuffled_items[start:end],
                    random_seed=config.seed + probe_index,
                )
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            logger.warning(
                (
                    "Auto batch-size verification rejected %d on %s; "
                    "retrying a smaller candidate (%s; %s)."
                ),
                candidate,
                config.dataset,
                _exception_summary(exc),
                _cuda_memory_snapshot(),
            )
            continue

        if candidate != selected_batch_size:
            logger.info(
                "Auto batch-size verification corrected %s from %d to %d "
                "on the final runtime state.",
                config.dataset,
                selected_batch_size,
                candidate,
            )
        return

    raise RuntimeError(
        (
            "Auto batch-size verification exhausted all candidates for "
            f"{config.dataset} starting from {selected_batch_size}."
        ),
    )


def _resume_auto_batch_fallback(
    trainer: MiniBatchTrainer,
    checkpoint_path: Path,
) -> tuple[int, dict[str, list]]:
    """Load the last completed epoch before retrying a smaller batch size.

    Args:
        trainer: Fresh trainer configured with the smaller candidate batch size.
        checkpoint_path: Checkpoint written by the failed larger-batch attempt.

    Returns:
        Tuple of the next epoch index and recovered training history.

    """
    trainer.load_checkpoint(checkpoint_path)
    start_epoch = trainer.completed_epoch + 1
    logger.info(
        "Auto batch-size fallback resuming from %s at epoch %d/%d.",
        checkpoint_path,
        start_epoch + 1,
        trainer.config.epochs,
    )
    return start_epoch, trainer.resume_history


def _build_mlflow_params(
    config: EDGRecConfig,
    preset: str | None,
    intervention: str | None,
    recipe_name: str | None,
    run_started_at_utc: str,
    batch_id: str | None,
    profile_name: str | None,
    change_note: str | None,
) -> dict[str, str | int | float | bool]:
    """Select compact config fields to expose as searchable MLflow params."""
    _, training_hash = _build_training_identity(config, preset, intervention)
    _, evaluation_hash = _build_evaluation_identity(config, training_hash)
    provenance = _build_run_provenance(training_hash, evaluation_hash)
    params: dict[str, str | int | float | bool] = {
        "dataset": config.dataset,
        "preset": preset or "custom",
        "training_mode": "mini_batch",
        "seed": config.seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "auto_batch_size": config.auto_batch_size,
        "embed_dim": config.embed_dim,
        "max_gnn_layers": config.max_gnn_layers,
        "sample_interactions": config.sample_interactions or 0,
        "loader_max_rows": config.loader_max_rows or 0,
        "lr": config.lr,
        "use_features": config.use_features,
        "feature_policy": config.feature_policy,
        "feature_subset_mode": config.feature_subset_mode,
        "preprocessing_preset": config.preprocessing_preset or "default",
        "item_universe_policy": config.item_universe_policy,
        "derived_split_mode": config.derived_split_mode,
        "embedding_optimizer": config.embedding_optimizer,
        "embedding_sparse_optimizer": config.embedding_sparse_optimizer,
        "propagation_backend": config.propagation_backend,
        "validation_every_n_epochs": config.validation_every_n_epochs,
        "sampler_residency_policy": config.sampler_residency_policy,
        "train_edge_keep_prob": config.train_edge_keep_prob,
        "use_dual_branch": config.use_dual_branch,
        "use_sign_aware": config.use_sign_aware,
        "use_ipw": config.use_ipw,
        "canonical_name": build_canonical_experiment_name(config, preset, intervention),
        "run_started_at_utc": run_started_at_utc,
        **provenance,
    }
    if config.use_dual_branch:
        params["interest_gnn_layers"] = config.interest_gnn_layers
        params["conformity_gnn_layers"] = config.conformity_gnn_layers
    else:
        params["single_branch_gnn_layers"] = config.single_branch_gnn_layers
    if intervention:
        params["intervention"] = intervention
    if recipe_name:
        params["recipe"] = recipe_name
    if batch_id:
        params["batch_id"] = batch_id
    if profile_name:
        params["profile_name"] = profile_name
    if config.dataset == "kuairand1k":
        params["label_mode"] = config.label_mode
        params["watch_ratio_proxy_threshold"] = config.watch_ratio_proxy_threshold
    if change_note:
        params["change_note"] = change_note
    params["num_neighbors"] = format_num_neighbors_payload(config.num_neighbors) or "-"
    params["batch_size_candidates"] = "-".join(str(value) for value in config.batch_size_candidates)
    return params


def _embedding_table_parameter_count(model: torch.nn.Module) -> int:
    """Return number of trainable scalar parameters in embedding tables."""
    embedding = getattr(model, "embedding", None)
    if embedding is None:
        return 0
    total = 0
    for module in embedding.modules():
        if isinstance(module, torch.nn.Embedding):
            total += sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total


def _estimate_optimizer_state_mb(
    model: torch.nn.Module,
    loss_suite: LossSuite,
    config: EDGRecConfig,
) -> float:
    """Estimate persistent optimizer-state memory in MiB."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in loss_suite.parameters() if p.requires_grad)
    embedding_params = _embedding_table_parameter_count(model)
    state_scalars = 0

    if config.baseline_family in {"lightgcn_paper", "dice_paper"} or (
        config.embedding_optimizer == "adamw"
    ):
        state_scalars = 2 * total_params
    else:
        dense_params = max(0, total_params - embedding_params)
        state_scalars = 2 * dense_params
        if config.embedding_optimizer == "sparseadam":
            state_scalars += 2 * embedding_params

    return state_scalars * 4 / 1024**2


def _log_resource_contract(
    *,
    config: EDGRecConfig,
    model: torch.nn.Module,
    loss_suite: LossSuite,
    data: Any,
    experiment_logger: ExperimentLogger,
    exp_id: int,
) -> None:
    """Log static resource-shaping fields before training starts."""
    model_parameter_count = sum(p.numel() for p in model.parameters())
    optimizer_state_mb = _estimate_optimizer_state_mb(model, loss_suite, config)
    resource_fields = {
        "model_parameter_count": float(model_parameter_count),
        "optimizer_state_estimate_mb": optimizer_state_mb,
        "item_embedding_count": float(data.n_items),
        "train_edge_count": float(data.edge_index.size(1)),
        "largest_training_item_interaction_count": float(
            getattr(data, "largest_training_item_interaction_count", 0.0)
        ),
    }
    logger.info(
        (
            "Resource contract: item embeddings=%d | train edges=%d | "
            "model params=%d | optimizer state estimate=%.1f MB | "
            "embedding optimizer=%s"
        ),
        int(data.n_items),
        int(data.edge_index.size(1)),
        model_parameter_count,
        optimizer_state_mb,
        config.embedding_optimizer,
    )
    for metric_name, metric_value in resource_fields.items():
        experiment_logger.log_metric(exp_id, metric_name, metric_value, split="train")


def _start_mlflow_run(
    config: EDGRecConfig,
    preset: str | None,
    intervention: str | None,
    experiment_id: str | None,
    tracking_uri: str | None,
    experiment_name: str,
    run_name: str | None,
    recipe_name: str | None,
    batch_id: str | None,
    profile_name: str | None,
    change_note: str | None,
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
    run_started_at_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        MLFLOW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(resolved_tracking_uri)
        try:
            mlflow.enable_system_metrics_logging()
        except Exception as exc:
            logger.info("MLflow system metrics logging unavailable: %s", exc)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(
            run_name=run_name or build_canonical_experiment_name(config, preset, intervention),
        )
        mlflow.set_tags(
            _build_mlflow_tags(
                experiment_id,
                recipe_name,
                batch_id,
                profile_name,
                change_note,
            ),
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
                change_note,
            ),
        )
        logger.info("MLflow tracking enabled: %s", resolved_tracking_uri)
    except Exception as exc:
        logger.warning("Failed to initialize MLflow tracking: %s", exc)
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run(status="FAILED")
        except Exception as cleanup_exc:
            logger.debug("Failed to end MLflow run during cleanup: %s", cleanup_exc)
        return None
    else:
        return mlflow


def normalize_config_inputs(
    args: argparse.Namespace | Mapping[str, object] | object,
) -> dict[str, object]:
    """Return config inputs as a plain mapping."""
    return dict(args) if isinstance(args, Mapping) else vars(args).copy()


def _present_field_mapping(
    source: Mapping[str, object],
    fields: tuple[str, ...] | list[str],
) -> dict[str, object]:
    """Return the subset of ``fields`` whose values are explicitly present."""
    present_fields: dict[str, object] = {}
    for field_name in fields:
        field_value = source.get(field_name)
        if field_value is not None:
            present_fields[field_name] = field_value
    return present_fields


def _build_config_input_mapping(
    *,
    dataset: str,
    recipe: str | None = None,
    preset: str | None = None,
    seed: int = DEFAULT_SEED,
    data_dir: str | None = None,
    device: str | None = None,
    copied_fields: Mapping[str, object] | None = None,
    extra_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one plain config-input mapping for ``build_config()``."""
    config_inputs: dict[str, object] = {
        "dataset": dataset,
        "recipe": recipe,
        "preset": preset,
        "seed": seed,
    }
    if data_dir is not None:
        config_inputs["data_dir"] = data_dir
    if device is not None:
        config_inputs["device"] = device
    if copied_fields is not None:
        config_inputs.update(copied_fields)
    if extra_overrides is not None:
        config_inputs.update(_present_field_mapping(extra_overrides, list(extra_overrides)))
    return config_inputs


def _collect_explicit_config_overrides(
    config_inputs: Mapping[str, object],
) -> dict[str, object]:
    """Collect explicit config overrides from CLI- or mapping-style inputs."""
    explicit_overrides: dict[str, object] = {
        "dataset": config_inputs.get("dataset"),
        "data_dir": config_inputs.get("data_dir"),
        "seed": config_inputs.get("seed", DEFAULT_SEED),
        "device": config_inputs.get("device"),
    }
    explicit_overrides.update(_present_field_mapping(config_inputs, CONFIG_OVERRIDE_FIELDS))
    return explicit_overrides


def build_runtime_config_inputs(
    *,
    dataset: str,
    recipe: str | None = None,
    preset: str | None = None,
    seed: int = DEFAULT_SEED,
    data_dir: str | None = None,
    device: str | None = None,
    **overrides: object,
) -> dict[str, object]:
    """Build one plain config-input mapping for ``build_config()``.

    Args:
        dataset: Dataset name for the run.
        recipe: Optional named recipe.
        preset: Optional preset name.
        seed: Random seed for the run.
        data_dir: Optional data-directory override.
        device: Optional device override.
        **overrides: Additional config override fields. ``None`` values are
            omitted so the downstream config builder can distinguish unset fields
            from explicit overrides.

    Returns:
        JSON-safe mapping ready for ``build_config()``.

    """
    return _build_config_input_mapping(
        dataset=dataset,
        recipe=recipe,
        preset=preset,
        seed=seed,
        data_dir=data_dir,
        device=device,
        extra_overrides=overrides,
    )


def build_benchmark_config_inputs(
    benchmark_args: Mapping[str, object],
    *,
    dataset: str,
    preset: str,
    lr_scheduler: str,
    num_neighbors: list[int],
    preprocessing_preset: str | None = None,
    graph_policy: str | None = None,
    batch_size: int | None = None,
) -> dict[str, object]:
    """Build one run's config inputs from normalized benchmark arguments."""
    copied_fields = _present_field_mapping(
        benchmark_args,
        BENCHMARK_CONFIG_FIELDS,
    )
    item_universe_policy = resolve_benchmark_item_universe_policy_value(
        benchmark_args,
        dataset=dataset,
    )
    if item_universe_policy is not None:
        copied_fields["item_universe_policy"] = item_universe_policy
    return _build_config_input_mapping(
        dataset=dataset,
        preset=preset,
        seed=DEFAULT_SEED,
        copied_fields=copied_fields,
        extra_overrides={
            "lr_scheduler": lr_scheduler,
            "num_neighbors": num_neighbors,
            "preprocessing_preset": preprocessing_preset,
            "graph_policy": graph_policy,
            "batch_size": batch_size,
        },
    )


def normalize_benchmark_config_overrides(
    raw_config: Mapping[str, object],
) -> dict[str, object]:
    """Normalize the config-bearing portion of a benchmark or formal-run payload."""
    removed_fields = [
        field_name
        for field_name in ("num_neighbors_options", "popularity_window_seconds")
        if field_name in raw_config
    ]
    if removed_fields:
        raise ValueError(
            "Use the current config surface only; removed fields are not supported: "
            + ", ".join(removed_fields),
        )
    default_config = EDGRecConfig()
    normalized: dict[str, object] = {
        field_name: raw_config.get(field_name) for field_name in BENCHMARK_CONFIG_FIELDS
    }
    # Keep the benchmark plan explicit even when the catalog omits the stop policy.
    use_early_stopping = raw_config.get("use_early_stopping")
    normalized["use_early_stopping"] = (
        default_config.use_early_stopping if use_early_stopping is None else use_early_stopping
    )
    patience = raw_config.get("patience")
    normalized["patience"] = int(patience) if patience is not None else default_config.patience
    use_features = raw_config.get("use_features")
    normalized["use_features"] = bool(use_features) if use_features is not None else None
    feature_policy = raw_config.get("feature_policy")
    normalized["feature_policy"] = str(feature_policy) if feature_policy is not None else None
    graph_policy = raw_config.get("graph_policy")
    if graph_policy is not None and not isinstance(graph_policy, str):
        raise ValueError(
            "graph_policy must be a string; graph-policy sweeps are not supported.",
        )
    normalized["graph_policy"] = graph_policy
    (
        preprocessing_preset,
        preprocessing_preset_options,
    ) = normalize_benchmark_preprocessing_override(
        raw_config.get("preprocessing_preset"),
    )
    normalized["preprocessing_preset"] = preprocessing_preset
    normalized["preprocessing_preset_options"] = preprocessing_preset_options
    derived_split_mode = raw_config.get("derived_split_mode")
    normalized["derived_split_mode"] = (
        str(derived_split_mode) if derived_split_mode is not None else None
    )
    for string_field in (
        "training_graph_mode",
        "sampler_residency_policy",
        "propagation_backend",
        "branch_loss_mode",
        "dice_mask_reduction",
        "recommendation_loss_mode",
        "negative_sampling_strategy",
        "loss_normalization",
        "label_mode",
        "feature_subset_mode",
    ):
        value = raw_config.get(string_field)
        normalized[string_field] = str(value) if value is not None else None
    batch_size = raw_config.get("batch_size")
    if isinstance(batch_size, (list, tuple)):
        normalized["batch_size"] = [int(value) for value in batch_size]
    else:
        normalized["batch_size"] = (
            int(batch_size) if batch_size is not None else default_config.batch_size
        )
    normalized["auto_batch_size"] = bool(raw_config.get("auto_batch_size", True))
    batch_size_candidates = raw_config.get("batch_size_candidates")
    normalized["batch_size_candidates"] = (
        list(batch_size_candidates) if isinstance(batch_size_candidates, (list, tuple)) else None
    )
    temporal_history_size = raw_config.get("temporal_history_size")
    normalized["temporal_history_size"] = (
        int(temporal_history_size) if temporal_history_size is not None else None
    )
    for list_field in (
        "feature_include_groups",
        "feature_exclude_groups",
    ):
        list_value = raw_config.get(list_field)
        normalized[list_field] = list(list_value) if isinstance(list_value, (list, tuple)) else None
    lr_value = raw_config.get("lr")
    normalized["lr"] = float(lr_value) if lr_value is not None else None
    weight_decay = raw_config.get("weight_decay")
    normalized["weight_decay"] = float(weight_decay) if weight_decay is not None else None
    normalized["lr_scheduler"] = normalize_benchmark_lr_scheduler_override(
        raw_config.get("lr_scheduler"),
    )
    lr_scheduler_factor = raw_config.get("lr_scheduler_factor")
    normalized["lr_scheduler_factor"] = (
        float(lr_scheduler_factor) if lr_scheduler_factor is not None else None
    )
    lr_scheduler_patience = raw_config.get("lr_scheduler_patience")
    normalized["lr_scheduler_patience"] = (
        int(lr_scheduler_patience) if lr_scheduler_patience is not None else None
    )
    grad_clip_norm = raw_config.get("grad_clip_norm")
    normalized["grad_clip_norm"] = float(grad_clip_norm) if grad_clip_norm is not None else None
    single_branch_layers = raw_config.get("single_branch_gnn_layers")
    normalized["single_branch_gnn_layers"] = (
        int(single_branch_layers) if single_branch_layers is not None else None
    )
    interest_layers = raw_config.get("interest_gnn_layers")
    normalized["interest_gnn_layers"] = (
        int(interest_layers) if interest_layers is not None else None
    )
    conformity_layers = raw_config.get("conformity_gnn_layers")
    normalized["conformity_gnn_layers"] = (
        int(conformity_layers) if conformity_layers is not None else None
    )
    dropout = raw_config.get("dropout")
    normalized["dropout"] = float(dropout) if dropout is not None else None
    normalized["num_neighbors"] = normalize_benchmark_num_neighbors_override(
        raw_config.get("num_neighbors"),
    )
    embedding_optimizer = raw_config.get("embedding_optimizer")
    normalized["embedding_optimizer"] = (
        str(embedding_optimizer) if embedding_optimizer is not None else None
    )
    item_universe_policy = raw_config.get("item_universe_policy")
    if isinstance(item_universe_policy, Mapping):
        normalized["item_universe_policy"] = dict(item_universe_policy)
    else:
        normalized["item_universe_policy"] = (
            str(item_universe_policy) if item_universe_policy is not None else None
        )
    normalized["hard_negative_ratio"] = float(
        raw_config.get("hard_negative_ratio", default_config.hard_negative_ratio),
    )
    optional_float_fields = (
        "score_mix_min_weight",
        "score_weight_interest",
        "score_weight_conformity",
        "score_weight_popularity",
        "loss_weight_recommendation",
        "loss_weight_interest_bpr",
        "loss_weight_conformity_bpr",
        "loss_weight_independence",
        "loss_weight_contrastive",
        "loss_weight_align",
        "loss_weight_uniform",
        "loss_weight_popularity",
        "loss_weight_propensity_calibration",
        "auxiliary_ramp_rate",
        "independence_ramp_rate",
        "contrastive_temperature",
        "uniformity_temperature",
        "feature_gate_init",
        "train_edge_keep_prob",
        "watch_ratio_proxy_threshold",
    )
    for field_name in optional_float_fields:
        field_value = raw_config.get(field_name)
        normalized[field_name] = float(field_value) if field_value is not None else None

    optional_bool_fields = (
        "use_ipw",
        "use_conformity_au",
        "use_popularity_head",
        "use_learned_score_mix",
        "separate_item_branch_embeddings",
        "use_temporal_interest",
        "paper_scaled_batch",
        "embedding_sparse_optimizer",
        "profile_training_stages",
    )
    for field_name in optional_bool_fields:
        field_value = raw_config.get(field_name)
        normalized[field_name] = bool(field_value) if field_value is not None else None

    auxiliary_loss_schedule = raw_config.get("auxiliary_loss_schedule")
    normalized["auxiliary_loss_schedule"] = (
        str(auxiliary_loss_schedule) if auxiliary_loss_schedule is not None else None
    )

    n_negatives = raw_config.get("n_negatives")
    normalized["n_negatives"] = int(n_negatives) if n_negatives is not None else None
    dice_sampler_margin = raw_config.get("dice_sampler_margin")
    normalized["dice_sampler_margin"] = (
        float(dice_sampler_margin) if dice_sampler_margin is not None else None
    )
    dice_sampler_pool = raw_config.get("dice_sampler_pool")
    normalized["dice_sampler_pool"] = (
        int(dice_sampler_pool) if dice_sampler_pool is not None else None
    )
    dice_branch_margin = raw_config.get("dice_branch_margin")
    normalized["dice_branch_margin"] = (
        float(dice_branch_margin) if dice_branch_margin is not None else None
    )
    dice_loss_decay = raw_config.get("dice_loss_decay")
    normalized["dice_loss_decay"] = float(dice_loss_decay) if dice_loss_decay is not None else None
    dice_margin_decay = raw_config.get("dice_margin_decay")
    normalized["dice_margin_decay"] = (
        float(dice_margin_decay) if dice_margin_decay is not None else None
    )
    dice_adaptive_decay = raw_config.get("dice_adaptive_decay")
    normalized["dice_adaptive_decay"] = (
        bool(dice_adaptive_decay) if dice_adaptive_decay is not None else None
    )
    distance_correlation_max_pairs = raw_config.get("distance_correlation_max_pairs")
    normalized["distance_correlation_max_pairs"] = (
        int(distance_correlation_max_pairs) if distance_correlation_max_pairs is not None else None
    )
    contrastive_max_pairs = raw_config.get("contrastive_max_pairs")
    normalized["contrastive_max_pairs"] = (
        int(contrastive_max_pairs) if contrastive_max_pairs is not None else None
    )
    uniformity_max_pairs = raw_config.get("uniformity_max_pairs")
    normalized["uniformity_max_pairs"] = (
        int(uniformity_max_pairs) if uniformity_max_pairs is not None else None
    )
    normalized["auxiliary_losses_start_epoch"] = int(
        raw_config.get(
            "auxiliary_losses_start_epoch",
            default_config.auxiliary_losses_start_epoch,
        ),
    )
    normalized["popularity_supervision_start_epoch"] = int(
        raw_config.get(
            "popularity_supervision_start_epoch",
            default_config.popularity_supervision_start_epoch,
        ),
    )
    validation_every_n_epochs = raw_config.get("validation_every_n_epochs")
    normalized["validation_every_n_epochs"] = (
        int(validation_every_n_epochs) if validation_every_n_epochs is not None else None
    )
    normalized["loss_schedule"] = raw_config.get("loss_schedule")
    loader_max_rows = raw_config.get("loader_max_rows")
    normalized["loader_max_rows"] = int(loader_max_rows) if loader_max_rows is not None else None
    sample_interactions = raw_config.get("sample_interactions")
    normalized["sample_interactions"] = (
        int(sample_interactions) if sample_interactions is not None else None
    )
    return normalized


def build_config(args: argparse.Namespace | Mapping[str, object]) -> EDGRecConfig:
    """Build EDGRecConfig from CLI args or mapping-style overrides.

    Precedence is: defaults -> preset -> recipe overrides -> explicit CLI flags.
    Conflicts between recipe-owned overrides and explicit CLI/config overrides are
    rejected rather than silently choosing one side.
    """
    config_inputs = normalize_config_inputs(args)
    recipe_overrides: dict[str, object] = {}

    recipe_name = config_inputs.get("recipe")
    recipe = get_recipe(recipe_name) if recipe_name else None
    effective_preset = config_inputs.get("preset")
    if recipe is not None:
        cli_preset = config_inputs.get("preset")
        recipe_preset = recipe.get("preset")
        if (
            cli_preset is not None
            and recipe_preset is not None
            and canonical_preset_for_identity(str(cli_preset))
            != canonical_preset_for_identity(str(recipe_preset))
        ):
            raise ValueError(
                (
                    f"preset={cli_preset!r} conflicts with recipe preset={recipe_preset!r}. "
                    "Use the recipe as-is, choose a different recipe alias, or "
                    "drop --recipe and pass --preset directly."
                ),
            )
        recipe_overrides.update(recipe.get("overrides", {}))
        if effective_preset is None:
            effective_preset = recipe.get("preset")
    if effective_preset is not None:
        effective_preset = public_preset_name(str(effective_preset))

    if effective_preset is not None and effective_preset not in CONFIG_PRESET_METHODS:
        available = ", ".join(sorted(CONFIG_PRESET_CHOICES))
        raise ValueError(
            f"Unknown preset '{effective_preset}'. Available presets: {available}",
        )

    explicit_overrides = _collect_explicit_config_overrides(config_inputs)

    conflicting_recipe_fields = sorted(
        field_name
        for field_name, recipe_value in recipe_overrides.items()
        if field_name in explicit_overrides and explicit_overrides[field_name] != recipe_value
    )
    if conflicting_recipe_fields:
        conflict_preview = ", ".join(conflicting_recipe_fields)
        raise ValueError(
            (
                "Explicit overrides conflict with recipe-owned fields: "
                f"{conflict_preview}. Use the recipe as-is or drop --recipe."
            ),
        )

    requested_loss_schedule = explicit_overrides.get("loss_schedule") or recipe_overrides.get(
        "loss_schedule",
    )
    if requested_loss_schedule not in (None, "baseline"):
        raise ValueError(
            "loss_schedule only supports 'baseline'; fused BPR stays active from epoch 0.",
        )

    config = EDGRecConfig()

    # Apply preset
    if effective_preset in CONFIG_PRESET_METHODS:
        getattr(config, CONFIG_PRESET_METHODS[effective_preset])()

    for override_group in (recipe_overrides, explicit_overrides):
        for key, val in override_group.items():
            setattr(config, key, val)

    config.enforce_paper_baseline_contract()

    if config.preprocessing_preset is None:
        config.preprocessing_preset = default_preprocessing_preset(config.dataset)

    config.validate()
    return config


def run_experiment(
    config: EDGRecConfig,
    preset: str | None = None,
    intervention: str | None = None,
    save_checkpoint: bool = True,
    enable_mlflow: bool = True,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "edgrec-thesis",
    mlflow_run_name: str | None = None,
    experiment_id: str | None = None,
    recipe_name: str | None = None,
    batch_id: str | None = None,
    profile_name: str | None = None,
    change_note: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    auto_resume: bool = True,
    overwrite_checkpoint: bool = False,
    include_refined_diagnostics: bool = True,
    evaluate_test: bool = True,
    log_to_sqlite: bool = True,
    training_epoch_callback: Callable[[int, Mapping[str, float], float], None] | None = None,
) -> dict:
    """Run a single experiment end-to-end.

    Args:
        config: Experiment configuration including dataset, model hyperparameters,
            and training settings.
        preset: Recipe preset name used for canonical naming and MLflow tagging.
        intervention: Ablation intervention label for canonical naming and tagging.
        save_checkpoint: Whether to persist the final checkpoint to disk.
        enable_mlflow: Whether to log the run to MLflow.
        mlflow_tracking_uri: MLflow tracking server URI. Defaults to the local
            file-based store resolved by ``_default_mlflow_tracking_uri``.
        mlflow_experiment_name: MLflow experiment namespace to log the run under.
        mlflow_run_name: Optional explicit MLflow run name override.
        experiment_id: Existing SQLite experiment ID to resume rather than
            creating a new record.
        recipe_name: Recipe name tag written to MLflow.
        batch_id: Batch-run group identifier written to both SQLite and MLflow.
        profile_name: Hardware profile label written to SQLite.
        change_note: Optional short note describing the active code change or
            experiment intent. Logged to SQLite and MLflow without affecting
            checkpoint compatibility.
        checkpoint_path: Override the default checkpoint file path. When None the
            canonical name is used to derive the path under ``CHECKPOINT_DIR``.
        checkpoint_every: Save an intermediate checkpoint every N epochs.
        auto_resume: If True, load an existing checkpoint and continue training
            from where it left off.
        overwrite_checkpoint: If True, delete any existing checkpoint at the
            resolved path and force a fresh run.
        include_refined_diagnostics: If True, compute optional refined scorer
            diagnostics during final test evaluation. Quick smoke validation can
            disable this so undefined tiny-slice diagnostics do not mask runtime
            coverage.
        evaluate_test: If True, run the final test evaluation and log test
            metrics. Exploratory validation-only workflows can disable this so
            test data stays reserved for promoted formal runs.
        log_to_sqlite: If True, persist experiment metadata and metrics in the
            thesis SQLite database. Quick validation sets this to False so the
            shared runtime path can execute without creating smoke rows.
        training_epoch_callback: Optional callback invoked after each validation
            epoch with ``(epoch, val_metrics, epoch_time_s)``. Search workflows
            use this for Optuna pruning; normal experiment entry points leave it
            unset.

    Returns:
        Dict with keys:
            - ``exp_id``: SQLite experiment row ID, or None when
              ``log_to_sqlite=False``.
            - ``test_metrics``: Metric name → value mapping from the test split.
            - ``history``: Training history dict (``train_loss``, ``val_metrics``).
            - ``checkpoint_path``: Path to the saved checkpoint, or None.
            - ``canonical_name``: Derived experiment identifier string.
            - ``resumed``: True if training was resumed from an existing checkpoint.

    """
    if experiment_id is not None and not log_to_sqlite:
        raise ValueError("experiment_id requires log_to_sqlite=True")

    seed_everything(config.seed)
    configure_torch_runtime()

    config.device = config.device if torch.cuda.is_available() else "cpu"
    effective_auto_resume = auto_resume and not overwrite_checkpoint
    explicit_checkpoint_path = checkpoint_path is not None
    pre_resolved_checkpoint_path: Path | None = None
    if effective_auto_resume:
        recovered_checkpoint = recoverable_checkpoint_for_config(
            config,
            preset=preset,
            intervention=intervention,
            checkpoint_path=checkpoint_path,
        )
        if recovered_checkpoint is not None:
            recovered_config, pre_resolved_checkpoint_path = recovered_checkpoint
            if int(recovered_config.batch_size) != int(config.batch_size):
                logger.info(
                    (
                        "Found recoverable checkpoint %s with batch_size %d before "
                        "auto batch-size probing; using the saved batch size."
                    ),
                    pre_resolved_checkpoint_path,
                    int(recovered_config.batch_size),
                )
            else:
                logger.info(
                    "Found recoverable checkpoint %s before auto batch-size probing.",
                    pre_resolved_checkpoint_path,
                )
            config = recovered_config

    logger.info("Loading dataset and building graph...")
    canonical, data = load_runtime_data(config)
    if canonical.item_propensity_targets is not None:
        data.propensity_targets = torch.from_numpy(canonical.item_propensity_targets)
    if pre_resolved_checkpoint_path is None:
        _resolve_auto_batch_size(config, canonical, data)
        _verify_selected_auto_batch_size(config, canonical, data)
        _release_cuda_probe_memory()
    canonical_name = build_canonical_experiment_name(config, preset, intervention)
    training_identity, training_hash = _build_training_identity(
        config,
        preset,
        intervention,
    )
    evaluation_identity, evaluation_hash = _build_evaluation_identity(
        config,
        training_hash,
    )
    run_provenance = _build_run_provenance(training_hash, evaluation_hash)
    resolved_checkpoint_path = (
        pre_resolved_checkpoint_path
        if pre_resolved_checkpoint_path is not None
        else (
            Path(checkpoint_path)
            if explicit_checkpoint_path
            else _default_checkpoint_path(config, preset, intervention, training_hash)
        )
    )
    if overwrite_checkpoint and resolved_checkpoint_path.exists():
        resolved_checkpoint_path.unlink()
        logger.info(
            "Deleted existing checkpoint at %s before starting a fresh run.",
            resolved_checkpoint_path,
        )
    checkpoint_state = None
    if effective_auto_resume:
        candidate_paths = (
            [resolved_checkpoint_path]
            if explicit_checkpoint_path or pre_resolved_checkpoint_path is not None
            else _default_checkpoint_path_candidates(config, preset, intervention, training_hash)
        )
        for candidate_path in candidate_paths:
            checkpoint_state = _validate_resume_identity(
                _load_checkpoint_metadata(
                    candidate_path,
                    config.device,
                ),
                checkpoint_path=candidate_path,
                explicit_checkpoint_path=explicit_checkpoint_path,
                training_identity=training_identity,
                training_hash=training_hash,
            )
            if checkpoint_state is not None:
                resolved_checkpoint_path = candidate_path
                break
    checkpoint_ready_for_eval = (
        _checkpoint_ready_for_evaluation(checkpoint_state, config)
        if checkpoint_state is not None
        else False
    )

    logger.info(
        (
            "Dataset: %s | Preset: %s | Profile: %s | Device: %s | "
            "Canonical name: %s | training hash: %s"
        ),
        config.dataset,
        preset,
        profile_name or "direct",
        config.device,
        canonical_name,
        training_hash,
    )

    logger.info("  %r", canonical)
    logger.info(f"  Nodes: {data.num_nodes:,}, Edges: {data.edge_index.size(1):,}")
    logger.info(
        f"  Train: {data.train_mask.sum():,}, Val: {data.val_mask.sum():,}, "
        f"Test: {data.test_mask.sum():,}",
    )

    model = build_runtime_model(config, canonical, data)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    loss_suite = LossSuite(config)
    profiler = None
    gpu_name, gpu_vram_gb = _gpu_hardware_metadata(config.device)

    # Return early for already-complete checkpoints before opening the DB.
    if checkpoint_state is not None and checkpoint_state.get("is_complete"):
        logger.info(
            "Checkpoint already marked complete. Returning cached result from %s",
            resolved_checkpoint_path,
        )
        history = checkpoint_state.get(
            "history",
            {"train_loss": [], "val_metrics": []},
        )
        return {
            "exp_id": checkpoint_state.get("exp_id"),
            "test_metrics": checkpoint_state.get("test_metrics", {}),
            "history": history,
            "checkpoint_path": str(resolved_checkpoint_path),
            "canonical_name": canonical_name,
            "resumed": True,
            "peak_vram_mb": None,
            "largest_training_item_interaction_count": float(
                getattr(data, "largest_training_item_interaction_count", 0.0)
            ),
            "epochs_stopped_at": len(history.get("train_loss", [])),
            "training_time_s": None,
            "train_batches_per_epoch": None,
            "batch_size": config.batch_size,
            "auto_batch_size": config.auto_batch_size,
        }

    experiment_logger: ExperimentLogger | None = None
    exp_id = checkpoint_state.get("exp_id") if checkpoint_state is not None else None
    if log_to_sqlite:
        experiment_logger = ExperimentLogger(db_path=str(DB_PATH))
        if exp_id is None:
            exp_id = experiment_logger.log_experiment(
                config.dataset,
                config,
                preset=preset,
                intervention=intervention,
                training_mode="mini_batch",
                status="running",
                batch_id=batch_id,
                profile_name=profile_name,
                project_version=run_provenance["project_version"],
                git_commit=run_provenance["git_commit"],
                training_hash=training_hash,
                evaluation_hash=evaluation_hash,
                change_note=change_note,
                gpu_name=gpu_name,
                gpu_vram_gb=gpu_vram_gb,
            )
        else:
            experiment_logger.update_experiment_status(exp_id, status="running")
        logger.info(f"Experiment ID: {exp_id}")
        _log_resource_contract(
            config=config,
            model=model,
            loss_suite=loss_suite,
            data=data,
            experiment_logger=experiment_logger,
            exp_id=exp_id,
        )
    else:
        logger.info("SQLite experiment logging disabled for this run.")

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
            change_note=change_note,
        )
        _log_mlflow_resume_tags(mlflow_module, checkpoint_state)

    try:
        trainer = None
        start_epoch = 0
        history = None
        should_persist_checkpoint = save_checkpoint or effective_auto_resume
        cuda_ = config.device.startswith("cuda") and torch.cuda.is_available()
        if cuda_:
            _reset_cuda_peak_memory_stats()
        train_start_ = time.perf_counter()
        if checkpoint_state is not None:
            model = build_runtime_model(config, canonical, data)
            trainer = MiniBatchTrainer(
                model=model,
                loss_suite=loss_suite,
                data=data,
                config=config,
                profiler=profiler,
                experiment_logger=experiment_logger,
                exp_id=exp_id,
                mlflow_module=mlflow_module,
            )
            trainer.training_identity = training_identity
            trainer.training_hash = training_hash
            trainer.evaluation_identity = evaluation_identity
            trainer.evaluation_hash = evaluation_hash
            trainer.load_checkpoint(
                resolved_checkpoint_path,
                load_best_model=checkpoint_ready_for_eval,
            )
            start_epoch = trainer.completed_epoch + 1
            history = trainer.resume_history
            if checkpoint_ready_for_eval:
                logger.info(
                    (
                        "Checkpoint %s already contains finished training state; "
                        "re-evaluating its best validation model without retraining."
                    ),
                    resolved_checkpoint_path,
                )
            else:
                logger.info(
                    "Resuming from checkpoint %s at epoch %d/%d",
                    resolved_checkpoint_path,
                    start_epoch + 1,
                    config.epochs,
                )

        if not checkpoint_ready_for_eval and start_epoch < config.epochs:
            logger.info(f"Training for {config.epochs} epochs...")
            if config.auto_batch_size and cuda_ and checkpoint_state is None:
                # The parameter-count model above is only diagnostic on this path.
                # Release it before constructing candidate trainers so large catalogs
                # do not pay a transient duplicate-model CUDA footprint.
                model = None
                loss_suite = None
                trainer = None
                _release_cuda_probe_memory()
                candidates = _auto_batch_probe_candidates(config)
                try:
                    start_index = candidates.index(int(config.batch_size))
                except ValueError:
                    start_index = 0

                selected_batch_size = int(config.batch_size)
                fallback_checkpoint_path: Path | None = None
                for candidate in candidates[start_index:]:
                    config.batch_size = candidate
                    # A previous candidate may have failed while its exception
                    # traceback still held CUDA tensors. Purge here, outside the
                    # prior ``except`` scope, before allocating the next trainer.
                    _release_cuda_probe_memory()
                    training_identity, training_hash = _build_training_identity(
                        config,
                        preset,
                        intervention,
                    )
                    evaluation_identity, evaluation_hash = _build_evaluation_identity(
                        config,
                        training_hash,
                    )
                    current_checkpoint_path = (
                        Path(checkpoint_path)
                        if explicit_checkpoint_path
                        else _default_checkpoint_path(
                            config,
                            preset,
                            intervention,
                            training_hash,
                        )
                    )
                    model = build_runtime_model(config, canonical, data)
                    loss_suite = LossSuite(config)
                    trainer = MiniBatchTrainer(
                        model=model,
                        loss_suite=loss_suite,
                        data=data,
                        config=config,
                        profiler=profiler,
                        experiment_logger=experiment_logger,
                        exp_id=exp_id,
                        mlflow_module=mlflow_module,
                    )
                    trainer.training_identity = training_identity
                    trainer.training_hash = training_hash
                    trainer.evaluation_identity = evaluation_identity
                    trainer.evaluation_hash = evaluation_hash
                    if cuda_:
                        _reset_cuda_peak_memory_stats()
                    candidate_start_epoch = start_epoch
                    candidate_history = history
                    if fallback_checkpoint_path is not None:
                        candidate_start_epoch, candidate_history = _resume_auto_batch_fallback(
                            trainer,
                            fallback_checkpoint_path,
                        )
                    try:
                        history = trainer.train(
                            start_epoch=candidate_start_epoch,
                            history=candidate_history,
                            checkpoint_path=current_checkpoint_path
                            if should_persist_checkpoint
                            else None,
                            checkpoint_every=checkpoint_every,
                            epoch_callback=training_epoch_callback,
                        )
                        resolved_checkpoint_path = current_checkpoint_path
                        canonical_name = build_canonical_experiment_name(
                            config,
                            preset,
                            intervention,
                        )
                        if candidate != selected_batch_size:
                            logger.info(
                                (
                                    "Auto batch-size training fallback selected %d for %s "
                                    "after selected batch_size %d failed at runtime; "
                                    "canonical name now %s."
                                ),
                                candidate,
                                config.dataset,
                                selected_batch_size,
                                canonical_name,
                            )
                            if experiment_logger is not None and exp_id is not None:
                                experiment_logger.conn.execute(
                                    (
                                        "UPDATE experiments SET config_json = ?, "
                                        "training_hash = ?, evaluation_hash = ?, "
                                        "updated_at = ? WHERE id = ?"
                                    ),
                                    (
                                        json.dumps(
                                            dataclasses.asdict(config),
                                            default=str,
                                        ),
                                        training_hash,
                                        evaluation_hash,
                                        datetime.now(UTC).isoformat(),
                                        exp_id,
                                    ),
                                )
                                experiment_logger.conn.commit()
                            if mlflow_module is not None:
                                try:
                                    mlflow_module.set_tag("training_hash", training_hash)
                                    mlflow_module.set_tag("evaluation_hash", evaluation_hash)
                                    mlflow_module.set_tag("canonical_name", canonical_name)
                                except Exception:
                                    pass
                        break
                    except Exception as exc:
                        if not is_cuda_oom_error(exc):
                            raise
                        fallback_checkpoint_path = (
                            current_checkpoint_path
                            if should_persist_checkpoint and current_checkpoint_path.exists()
                            else None
                        )
                        logger.warning(
                            (
                                "Training with batch_size %d OOM on %s; trying next smaller "
                                "candidate%s (%s; %s)."
                            ),
                            candidate,
                            config.dataset,
                            (
                                f" from checkpoint {fallback_checkpoint_path}"
                                if fallback_checkpoint_path is not None
                                else " from epoch 1 because no recovery checkpoint exists"
                            ),
                            _exception_summary(exc),
                            _cuda_memory_snapshot(),
                        )
                        trainer.subgraph_sampler = None
                        trainer = None
                        model = None
                        loss_suite = None
                        continue
                else:
                    raise RuntimeError(
                        (
                            "Auto batch-size training fallback exhausted all "
                            f"candidates for {config.dataset} starting from {config.batch_size}."
                        ),
                    )
            else:
                if trainer is None:
                    model = build_runtime_model(config, canonical, data)
                    trainer = MiniBatchTrainer(
                        model=model,
                        loss_suite=loss_suite,
                        data=data,
                        config=config,
                        profiler=profiler,
                        experiment_logger=experiment_logger,
                        exp_id=exp_id,
                        mlflow_module=mlflow_module,
                    )
                    trainer.training_identity = training_identity
                    trainer.training_hash = training_hash
                    trainer.evaluation_identity = evaluation_identity
                    trainer.evaluation_hash = evaluation_hash
                history = trainer.train(
                    start_epoch=start_epoch,
                    history=history,
                    checkpoint_path=resolved_checkpoint_path if should_persist_checkpoint else None,
                    checkpoint_every=checkpoint_every,
                    epoch_callback=training_epoch_callback,
                )
        else:
            history = history or {"train_loss": [], "val_metrics": []}
            if not checkpoint_ready_for_eval:
                logger.info(
                    "Checkpoint already reached configured epoch budget; skipping training.",
                )
        total_training_time_s = time.perf_counter() - train_start_
        peak_vram_mb = 0.0
        if cuda_:
            training_peak_vram_mb = trainer.training_peak_vram_mb if trainer is not None else None
            peak_vram_mb = (
                training_peak_vram_mb
                if training_peak_vram_mb is not None
                else torch.cuda.max_memory_allocated() / 1024**2
            )
        if experiment_logger is not None and exp_id is not None:
            experiment_logger.log_metric(
                exp_id, "training_time_s", total_training_time_s, split="train"
            )
        if cuda_ and experiment_logger is not None and exp_id is not None:
            experiment_logger.log_metric(exp_id, "peak_vram_mb", peak_vram_mb, split="train")
        if history["train_loss"]:
            logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
        logger.info(
            "Training complete — time: %.1fs | peak GPU memory: %.0f MB",
            total_training_time_s,
            peak_vram_mb,
        )
        train_batches_per_epoch = None
        if trainer is not None and hasattr(trainer, "train_users"):
            n_train_interactions = int(trainer.train_users.size(0))
            train_batches_per_epoch = max(
                1,
                (n_train_interactions + int(config.batch_size) - 1) // int(config.batch_size),
            )

        final_checkpoint_path: Path | None = None
        if save_checkpoint or effective_auto_resume:
            final_checkpoint_path = resolved_checkpoint_path
            trainer.save_checkpoint(
                final_checkpoint_path,
                history=history,
                training_finished=True,
                exp_id=exp_id,
                canonical_name=canonical_name,
            )

        test_metrics: dict[str, float] = {}
        if evaluate_test:
            logger.info("Running test evaluation...")
            test_metrics = trainer._evaluate_split_metrics(
                data.test_mask,
                include_refined_diagnostics=include_refined_diagnostics,
            )
            for metric, value in sorted(test_metrics.items()):
                logger.info(f"  {metric}: {value:.4f}")
                if experiment_logger is not None and exp_id is not None:
                    experiment_logger.log_metric(exp_id, metric, value, split="test")
        else:
            logger.info("Skipping test evaluation for validation-only run.")

        if evaluate_test and (save_checkpoint or effective_auto_resume):
            final_checkpoint_path = resolved_checkpoint_path
            trainer.save_checkpoint(
                final_checkpoint_path,
                history=history,
                is_complete=True,
                test_metrics=test_metrics,
                exp_id=exp_id,
                canonical_name=canonical_name,
            )

        if mlflow_module is not None:
            try:
                resource_metrics: dict[str, float] = {
                    "training_time_s": total_training_time_s,
                }
                if cuda_:
                    resource_metrics["peak_vram_mb"] = peak_vram_mb
                mlflow_module.log_metrics(
                    {
                        **{
                            f"test_{metric}".replace("@", "_at_"): float(value)
                            for metric, value in test_metrics.items()
                        },
                        **resource_metrics,
                    },
                )
                if final_checkpoint_path is not None:
                    mlflow_module.log_artifact(
                        str(final_checkpoint_path),
                        artifact_path="checkpoints",
                    )
                mlflow_module.set_tags({"status": "completed", "oom_flag": "false"})
            except Exception as exc:
                logger.warning("Failed to log MLflow metrics or artifacts: %s", exc)

        if experiment_logger is not None and exp_id is not None:
            experiment_logger.update_experiment_status(exp_id, status="completed")
        mlflow_status = "FINISHED"
        return {
            "exp_id": exp_id,
            "test_metrics": test_metrics,
            "history": history,
            "checkpoint_path": str(final_checkpoint_path)
            if final_checkpoint_path is not None
            else None,
            "canonical_name": canonical_name,
            "resumed": checkpoint_state is not None,
            "peak_vram_mb": peak_vram_mb,
            "largest_training_item_interaction_count": float(
                getattr(data, "largest_training_item_interaction_count", 0.0)
            ),
            "epochs_stopped_at": len(history["train_loss"]) if history is not None else 0,
            "training_time_s": total_training_time_s,
            "train_batches_per_epoch": train_batches_per_epoch,
            "batch_size": config.batch_size,
            "auto_batch_size": config.auto_batch_size,
        }
    except Exception as exc:
        is_oom = is_cuda_oom_error(exc)
        is_pruned = exc.__class__.__name__ == "TrialPruned"
        failure_reason = f"{type(exc).__name__}: {exc}"
        if experiment_logger is not None and exp_id is not None:
            experiment_logger.update_experiment_status(
                exp_id,
                status="pruned" if is_pruned else ("oom" if is_oom else "failed"),
                failure_reason=failure_reason,
                oom_flag=is_oom,
            )
        if mlflow_module is not None:
            try:
                mlflow_module.set_tags(
                    {
                        "status": "pruned" if is_pruned else ("oom" if is_oom else "failed"),
                        "oom_flag": "true" if is_oom else "false",
                        "failure_reason": failure_reason[:500],
                        "failure_type": failure_reason.split(":", 1)[0],
                    },
                )
            except Exception as exc:
                logger.debug("Failed to set MLflow failure tags: %s", exc)
        raise
    finally:
        if experiment_logger is not None:
            experiment_logger.close()
        if mlflow_module is not None:
            try:
                if "is_pruned" in locals() and is_pruned:
                    mlflow_status = "KILLED"
                mlflow_module.end_run(status=mlflow_status)
            except Exception as exc:
                logger.warning("Failed to close MLflow run cleanly: %s", exc)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    """Parse CLI arguments and run one experiment."""
    parser = build_run_experiment_parser()
    args = parser.parse_args()

    if args.list_recipes:
        print("Available experiment recipes:")
        print("\n".join(recipe_summary_lines()))
        return 0

    configure_cli_logging()

    config = build_config(args)
    recipe = get_recipe(args.recipe) if args.recipe else None
    resolved_preset = args.preset or (recipe.get("preset") if recipe else None)
    if resolved_preset is not None:
        resolved_preset = public_preset_name(str(resolved_preset))
    recipe_name = public_preset_name(str(args.recipe)) if args.recipe else None
    result = run_experiment(
        config,
        preset=resolved_preset,
        intervention=args.intervention,
        save_checkpoint=True,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_run_name=args.mlflow_run_name,
        experiment_id=args.experiment_id,
        recipe_name=recipe_name,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        auto_resume=args.auto_resume,
        overwrite_checkpoint=args.overwrite_checkpoint,
        change_note=args.change_note,
    )

    print(f"\nExperiment {result['exp_id']} complete.")
    print("Test metrics:")
    for k, v in sorted(result["test_metrics"].items()):
        print(f"  {k}: {v:.4f}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
