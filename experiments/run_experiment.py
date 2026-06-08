#!/usr/bin/env python
"""Main single-experiment CLI runner for U-CaGNN.

Usage:
    uv run experiment --list-recipes
    uv run experiment --dataset movielens1m --recipe ucagnn
    uv run experiment --dataset kuairec_v2 --preset ucagnn --overwrite-checkpoint
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
from collections.abc import Mapping
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
from src.data.graph_builder import build_graph
from src.data.loaders import default_preprocessing_preset, load_dataset
from src.losses.loss_suite import LossSuite
from src.models.baselines import PaperGCNDICE, PaperLightGCN
from src.models.ucagnn import UCaGNN
from src.training.mini_batch_trainer import MiniBatchTrainer
from src.utils.config import (
    BENCHMARK_CONFIG_FIELDS,
    CONFIG_OVERRIDE_FIELDS,
    CONFIG_PRESET_METHODS,
    DEFAULT_SEED,
    GRAPH_POLICY_CHOICES,
    UCaGNNConfig,
)
from src.utils.experiment_logger import ExperimentLogger
from src.utils.experiment_naming import build_canonical_experiment_name
from src.utils.interaction_indexing import compute_normalized_popularity
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

from experiments.cli_parsers import build_run_experiment_parser
from experiments.recipes import (
    get_recipe,
    recipe_summary_lines,
    resolve_profile_num_neighbors,
)

logger = logging.getLogger("ucagnn")

DB_PATH = THESIS_DB_PATH
_CHECKPOINT_IDENTITY_VERSION = 1
_CHECKPOINT_HASH_LEN = 16
_TRAINING_IDENTITY_FIELDS = (
    "amp_dtype",
    "auxiliary_loss_schedule",
    "auxiliary_ramp_rate",
    "auto_batch_size",
    "batch_size",
    "batch_size_candidates",
    "cagra_initial_degree",
    "cagra_itopk_size",
    "cagra_k",
    "cagra_metric",
    "cagra_out_degree",
    "cagra_team_size",
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
    "epochs",
    "feature_policy",
    "graph_policy",
    "grad_clip_norm",
    "hard_negative_ratio",
    "score_mix_min_weight",
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
    "loader_max_rows",
    "loss_schedule",
    "lr",
    "lr_scheduler",
    "lr_scheduler_factor",
    "lr_scheduler_patience",
    "n_negatives",
    "num_neighbors",
    "popularity_embedding_dimensions",
    "preprocessing_preset",
    "propensity_clip_max",
    "propensity_clip_min",
    "propensity_hidden",
    "sample_interactions",
    "score_weight_conformity",
    "score_weight_interest",
    "score_weight_popularity",
    "seed",
    "single_branch_gnn_layers",
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
    "val_ratio",
    "weight_decay",
)
_EVALUATION_IDENTITY_FIELDS = ("eval_ks",)


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
    config: UCaGNNConfig,
    preset: str | None,
    intervention: str | None,
) -> tuple[dict[str, Any], str]:
    """Build the resume-compatibility identity for a training run."""
    config_values = dataclasses.asdict(config)
    identity = {
        "identity_version": _CHECKPOINT_IDENTITY_VERSION,
        "identity_kind": "training",
        "preset": preset or "custom",
        "intervention": intervention,
        "training_mode": "mini_batch",
        "config": {
            field_name: config_values[field_name] for field_name in _TRAINING_IDENTITY_FIELDS
        },
    }
    return identity, _stable_identity_hash(identity)


def _build_evaluation_identity(
    config: UCaGNNConfig,
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


def _default_checkpoint_path(
    config: UCaGNNConfig,
    preset: str | None,
    intervention: str | None,
    training_hash: str,
) -> Path:
    """Return the default checkpoint path for a semantic training identity."""
    canonical_name = build_canonical_experiment_name(config, preset, intervention)
    return CHECKPOINT_DIR / f"{canonical_name}_train-{training_hash}.pt"


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

    if require_runtime_keys:
        missing_keys = sorted(REQUIRED_CHECKPOINT_KEYS.difference(payload))
        if missing_keys:
            raise ValueError(
                "checkpoint is missing required runtime keys: " + ", ".join(missing_keys),
            )

    if require_config:
        config = payload.get("config")
        if not isinstance(config, UCaGNNConfig):
            raise TypeError(
                "checkpoint does not contain a UCaGNNConfig under the 'config' field",
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


def _slice_optional(arr: np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
    """Return arr[idx], or None if arr is None."""
    return None if arr is None else arr[idx]


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
        chosen = (
            indices
            if count >= len(indices)
            else np.sort(rng.choice(indices, size=count, replace=False))
        )
        chosen_parts.append(chosen)

    selected = (
        np.sort(np.concatenate(chosen_parts)) if chosen_parts else np.array([], dtype=np.int64)
    )
    selected_train = np.isin(selected, split_indices[0])
    selected_val = np.isin(selected, split_indices[1])
    selected_test = np.isin(selected, split_indices[2])

    selected_users, user_inverse = np.unique(
        canonical.user_id[selected],
        return_inverse=True,
    )
    selected_items, item_inverse = np.unique(
        canonical.item_id[selected],
        return_inverse=True,
    )

    reverse_user_map = {value: key for key, value in canonical.user_map.items()}
    reverse_item_map = {value: key for key, value in canonical.item_map.items()}
    sampled_popularity = compute_normalized_popularity(
        item_inverse.astype(np.int64, copy=False),
        len(selected_items),
    )

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
        raw_target=_slice_optional(canonical.raw_target, selected),
        behavior_type=_slice_optional(canonical.behavior_type, selected),
        exposure_flag=_slice_optional(canonical.exposure_flag, selected),
        source_domain=_slice_optional(canonical.source_domain, selected),
        popularity=sampled_popularity,
        n_users=len(selected_users),
        n_items=len(selected_items),
        user_map={
            reverse_user_map[int(old_id)]: new_id
            for new_id, old_id in enumerate(selected_users.tolist())
        },
        item_map={
            reverse_item_map[int(old_id)]: new_id
            for new_id, old_id in enumerate(selected_items.tolist())
        },
        user_features=_slice_optional(canonical.user_features, selected_users),
        item_features=_slice_optional(canonical.item_features, selected_items),
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
    observed_data = build_graph(canonical, config, embeddings=None)
    if config.graph_policy == "observed":
        return canonical, observed_data

    bootstrap_embeddings = _bootstrap_cagra_embeddings(config, canonical, observed_data)
    return canonical, build_graph(canonical, config, embeddings=bootstrap_embeddings)


def _bootstrap_cagra_embeddings(
    config: UCaGNNConfig,
    canonical: Any,
    observed_data: Any,
) -> torch.Tensor:
    """Return the bootstrap node embeddings used for CAGRA augmentation.

    Args:
        config: Active runtime configuration.
        canonical: Loaded canonical interactions.
        observed_data: Graph payload for the observed train-interaction graph.

    Returns:
        torch.Tensor: CPU float tensor with shape ``(n_users + n_items, d)``.

    Raises:
        ValueError: If the current bootstrap path lacks item features and would
            therefore build ANN edges from untrained ID-only embeddings.

    """
    item_features = getattr(observed_data, "item_features", None)
    if item_features is None or item_features.numel() == 0:
        raise ValueError(
            "graph_policy='cagra_augmented' combines CAGRA edges with the observed "
            "train-interaction graph, but the current bootstrap path still needs "
            "item features. Without them, load_runtime_data() would build ANN edges "
            "from untrained ID-only embeddings before training begins.",
        )

    bootstrap_model = build_runtime_model(config, canonical, observed_data)
    with torch.no_grad():
        bootstrap_embeddings = (
            bootstrap_model.embedding.get_stacked_embeddings().detach().float().cpu()
        )
    del bootstrap_model
    return bootstrap_embeddings


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


def build_runtime_model(
    config: UCaGNNConfig,
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

    train_mask = _train_mask_numpy_from_data(data)
    item_recency = torch.from_numpy(canonical.compute_item_recency(train_mask))
    recent_train_items, recent_train_mask = canonical.build_recent_train_history(train_mask)
    item_propensity_targets = (
        torch.from_numpy(canonical.item_propensity_targets)
        if canonical.item_propensity_targets is not None
        else None
    )
    return UCaGNN(
        canonical.n_users,
        canonical.n_items,
        config,
        item_features=getattr(data, "item_features", None),
        item_popularity=data.popularity,
        item_recency=item_recency,
        item_propensity_targets=item_propensity_targets,
        recent_train_items=torch.from_numpy(recent_train_items),
        recent_train_mask=torch.from_numpy(recent_train_mask),
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
    config: UCaGNNConfig,
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
    config: UCaGNNConfig,
    *,
    checkpoint_path: str | Path | None = None,
) -> list[UCaGNNConfig]:
    """Return config variants whose checkpoint identities are worth checking."""
    if checkpoint_path is not None or not bool(config.auto_batch_size):
        return [config]

    configs: list[UCaGNNConfig] = []
    seen_batch_sizes: set[int] = set()
    for candidate_batch_size in _auto_batch_probe_candidates(config):
        batch_size = int(candidate_batch_size)
        if batch_size in seen_batch_sizes:
            continue
        seen_batch_sizes.add(batch_size)
        configs.append(dataclasses.replace(config, batch_size=batch_size))
    return configs


def recoverable_checkpoint_for_config(
    config: UCaGNNConfig,
    *,
    preset: str | None = None,
    intervention: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[UCaGNNConfig, Path] | None:
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
        resolved_checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else _default_checkpoint_path(
                checkpoint_config,
                preset,
                intervention,
                training_hash,
            )
        )
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


def recoverable_checkpoint_path(
    config: UCaGNNConfig,
    *,
    preset: str | None = None,
    intervention: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> Path | None:
    """Return a compatible checkpoint path that is ready for test re-evaluation."""
    recovered = recoverable_checkpoint_for_config(
        config,
        preset=preset,
        intervention=intervention,
        checkpoint_path=checkpoint_path,
    )
    if recovered is None:
        return None
    return recovered[1]


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
            "causal-embeddings-for-recommendations",
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


def _reset_cuda_peak_memory_stats() -> None:
    """Reset CUDA peak stats when available without disturbing CPU-only tests."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


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


def _auto_batch_probe_candidates(config: UCaGNNConfig) -> list[int]:
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
    config: UCaGNNConfig,
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
    config: UCaGNNConfig,
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

        stage = "prepare_batch"
        sub_batch = probe_trainer._prepare_batch(
            batch_users[:batch_size],
            batch_items[:batch_size],
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
        sub_batch = None
        losses = None
        del probe_trainer, probe_loss_suite, probe_model
        _release_cuda_probe_memory()


def _resolve_auto_batch_size(
    config: UCaGNNConfig,
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
    config: UCaGNNConfig,
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
    config: UCaGNNConfig,
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
        "preprocessing_preset": config.preprocessing_preset or "default",
        "derived_split_mode": config.derived_split_mode,
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
    if change_note:
        params["change_note"] = change_note
    params["num_neighbors"] = "-".join(str(value) for value in config.num_neighbors)
    params["batch_size_candidates"] = "-".join(str(value) for value in config.batch_size_candidates)
    return params


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
) -> dict[str, object]:
    """Build one run's config inputs from normalized benchmark arguments."""
    return _build_config_input_mapping(
        dataset=dataset,
        preset=preset,
        seed=DEFAULT_SEED,
        copied_fields=_present_field_mapping(
            benchmark_args,
            BENCHMARK_CONFIG_FIELDS,
        ),
        extra_overrides={
            "lr_scheduler": lr_scheduler,
            "num_neighbors": num_neighbors,
            "preprocessing_preset": preprocessing_preset,
            "graph_policy": graph_policy,
        },
    )


def _normalize_benchmark_lr_scheduler_override(
    raw_value: object,
) -> list[str] | str:
    """Normalize benchmark ``lr_scheduler`` overrides to one string or a sweep list."""
    if raw_value is None:
        return "plateau"
    if isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.split(",") if part.strip()]
        return values[0] if len(values) == 1 else values
    if isinstance(raw_value, (list, tuple)):
        return [str(value) for value in raw_value]
    return "plateau"


def _normalize_benchmark_graph_policy_override(
    raw_value: object,
) -> tuple[str | None, list[str] | None]:
    """Normalize benchmark ``graph_policy`` overrides to one value or a sweep list."""
    if raw_value is None:
        return None, None
    if isinstance(raw_value, str):
        if raw_value not in GRAPH_POLICY_CHOICES:
            raise ValueError(
                f"graph_policy must be one of {GRAPH_POLICY_CHOICES}, got {raw_value!r}",
            )
        return raw_value, None
    if isinstance(raw_value, (list, tuple)):
        if not raw_value:
            raise ValueError("graph_policy sweep must be a non-empty list.")
        graph_policy_options: list[str] = []
        for index, value in enumerate(raw_value):
            graph_policy = str(value)
            if graph_policy not in GRAPH_POLICY_CHOICES:
                raise ValueError(
                    f"graph_policy[{index}] must be one of {GRAPH_POLICY_CHOICES}, "
                    f"got {graph_policy!r}",
                )
            if graph_policy not in graph_policy_options:
                graph_policy_options.append(graph_policy)
        return graph_policy_options[0], graph_policy_options
    raise ValueError(
        "graph_policy must be a string or a list of graph-policy strings.",
    )


def _normalize_benchmark_preprocessing_override(
    raw_value: object,
) -> tuple[str | None, list[str] | None]:
    """Normalize benchmark preprocessing overrides to one value or a sweep list."""
    if raw_value is None:
        return None, None
    if isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.split(",") if part.strip()]
    elif isinstance(raw_value, (list, tuple)):
        values = [str(value).strip() for value in raw_value if str(value).strip()]
    else:
        raise ValueError(
            "preprocessing_preset must be a string or a list of preprocessing preset names.",
        )
    if not values:
        raise ValueError("preprocessing_preset sweep must be a non-empty string or list.")
    deduped = list(dict.fromkeys(values))
    return deduped[0], deduped if len(deduped) > 1 else None


def _normalize_benchmark_num_neighbors_override(
    raw_value: object,
) -> list[list[int]] | dict[str, list[list[int]]] | None:
    """Normalize benchmark ``num_neighbors`` overrides for one profile payload."""
    if raw_value is None:
        return None
    if isinstance(raw_value, Mapping):
        normalized: dict[str, list[list[int]]] = {}
        for key, value in raw_value.items():
            if isinstance(value, Mapping):
                raise ValueError(
                    f"num_neighbors[{key}] must be a vector or a list of vectors, "
                    "not a nested mapping.",
                )
            resolved = resolve_profile_num_neighbors({"num_neighbors": value})
            if resolved is None:
                raise ValueError(
                    f"num_neighbors[{key}] must be a non-empty fan-out vector "
                    "or a non-empty list of vectors.",
                )
            normalized[str(key)] = resolved
        return normalized

    resolved = resolve_profile_num_neighbors({"num_neighbors": raw_value})
    if resolved is None:
        return None
    return resolved[0] if len(resolved) == 1 else resolved


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
    default_config = UCaGNNConfig()
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
    raw_graph_policy = raw_config.get("graph_policy_options")
    if raw_graph_policy is None:
        raw_graph_policy = raw_config.get("graph_policy")
    graph_policy, graph_policy_options = _normalize_benchmark_graph_policy_override(
        raw_graph_policy,
    )
    normalized["graph_policy"] = graph_policy
    normalized["graph_policy_options"] = graph_policy_options
    (
        preprocessing_preset,
        preprocessing_preset_options,
    ) = _normalize_benchmark_preprocessing_override(
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
        "branch_loss_mode",
        "recommendation_loss_mode",
        "negative_sampling_strategy",
    ):
        value = raw_config.get(string_field)
        normalized[string_field] = str(value) if value is not None else None
    batch_size = raw_config.get("batch_size")
    normalized["batch_size"] = (
        int(batch_size) if batch_size is not None else default_config.batch_size
    )
    normalized["auto_batch_size"] = bool(raw_config.get("auto_batch_size", True))
    batch_size_candidates = raw_config.get("batch_size_candidates")
    normalized["batch_size_candidates"] = (
        list(batch_size_candidates) if isinstance(batch_size_candidates, (list, tuple)) else None
    )
    lr_value = raw_config.get("lr")
    normalized["lr"] = float(lr_value) if lr_value is not None else None
    normalized["lr_scheduler"] = _normalize_benchmark_lr_scheduler_override(
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
    normalized["num_neighbors"] = _normalize_benchmark_num_neighbors_override(
        raw_config.get("num_neighbors"),
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
    )
    for field_name in optional_float_fields:
        field_value = raw_config.get(field_name)
        normalized[field_name] = float(field_value) if field_value is not None else None

    optional_bool_fields = ("use_ipw", "use_conformity_au")
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
    normalized["loss_schedule"] = raw_config.get("loss_schedule")
    loader_max_rows = raw_config.get("loader_max_rows")
    normalized["loader_max_rows"] = int(loader_max_rows) if loader_max_rows is not None else None
    sample_interactions = raw_config.get("sample_interactions")
    normalized["sample_interactions"] = (
        int(sample_interactions) if sample_interactions is not None else None
    )
    return normalized


def build_config(args: argparse.Namespace | Mapping[str, object]) -> UCaGNNConfig:
    """Build UCaGNNConfig from CLI args or mapping-style overrides.

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
        if cli_preset is not None and recipe_preset is not None and cli_preset != recipe_preset:
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

    if effective_preset is not None and effective_preset not in CONFIG_PRESET_METHODS:
        available = ", ".join(sorted(CONFIG_PRESET_METHODS))
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

    config = UCaGNNConfig()

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
    change_note: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    auto_resume: bool = True,
    overwrite_checkpoint: bool = False,
    include_refined_diagnostics: bool = True,
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

    Returns:
        Dict with keys:
            - ``exp_id``: SQLite experiment row ID.
            - ``test_metrics``: Metric name → value mapping from the test split.
            - ``history``: Training history dict (``train_loss``, ``val_metrics``).
            - ``checkpoint_path``: Path to the saved checkpoint, or None.
            - ``canonical_name``: Derived experiment identifier string.
            - ``resumed``: True if training was resumed from an existing checkpoint.

    """
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
        checkpoint_state = _validate_resume_identity(
            _load_checkpoint_metadata(
                resolved_checkpoint_path,
                config.device,
            ),
            checkpoint_path=resolved_checkpoint_path,
            explicit_checkpoint_path=explicit_checkpoint_path,
            training_identity=training_identity,
            training_hash=training_hash,
        )
    checkpoint_ready_for_eval = (
        _checkpoint_ready_for_evaluation(checkpoint_state, config)
        if checkpoint_state is not None
        else False
    )

    logger.info(
        "Dataset: %s | Preset: %s | Device: %s | Canonical name: %s | training hash: %s",
        config.dataset,
        preset,
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
            "epochs_stopped_at": len(history.get("train_loss", [])),
            "training_time_s": None,
            "train_batches_per_epoch": None,
        }

    experiment_logger = ExperimentLogger(db_path=str(DB_PATH))
    exp_id = checkpoint_state.get("exp_id") if checkpoint_state is not None else None
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
            torch.cuda.reset_peak_memory_stats()
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
                candidates = _auto_batch_probe_candidates(config)
                try:
                    start_index = candidates.index(int(config.batch_size))
                except ValueError:
                    start_index = 0

                selected_batch_size = int(config.batch_size)
                fallback_checkpoint_path: Path | None = None
                for candidate in candidates[start_index:]:
                    config.batch_size = candidate
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
                        torch.cuda.reset_peak_memory_stats()
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
                        )
                        resolved_checkpoint_path = current_checkpoint_path
                        canonical_name = build_canonical_experiment_name(
                            config,
                            preset,
                            intervention,
                        )
                        if candidate != selected_batch_size:
                            if exp_id is not None:
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
                        _release_cuda_probe_memory()
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
        experiment_logger.log_metric(
            exp_id, "training_time_s", total_training_time_s, split="train"
        )
        if cuda_:
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

        logger.info("Running test evaluation...")
        test_metrics = trainer._evaluate_split_metrics(
            data.test_mask,
            include_refined_diagnostics=include_refined_diagnostics,
        )
        for metric, value in sorted(test_metrics.items()):
            logger.info(f"  {metric}: {value:.4f}")
            experiment_logger.log_metric(exp_id, metric, value, split="test")

        if save_checkpoint or effective_auto_resume:
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
            "epochs_stopped_at": len(history["train_loss"]) if history is not None else 0,
            "training_time_s": total_training_time_s,
            "train_batches_per_epoch": train_batches_per_epoch,
        }
    except Exception as exc:
        is_oom = is_cuda_oom_error(exc)
        failure_reason = f"{type(exc).__name__}: {exc}"
        experiment_logger.update_experiment_status(
            exp_id,
            status="oom" if is_oom else "failed",
            failure_reason=failure_reason,
            oom_flag=is_oom,
        )
        if mlflow_module is not None:
            try:
                mlflow_module.set_tags(
                    {
                        "status": "oom" if is_oom else "failed",
                        "oom_flag": "true" if is_oom else "false",
                        "failure_reason": failure_reason[:500],
                        "failure_type": failure_reason.split(":", 1)[0],
                    },
                )
            except Exception as exc:
                logger.debug("Failed to set MLflow failure tags: %s", exc)
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
        recipe_name=args.recipe,
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
