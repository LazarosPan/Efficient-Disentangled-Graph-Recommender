#!/usr/bin/env python
"""Main single-experiment CLI runner for U-CaGNN.

Usage:
    python experiments/run_experiment.py --dataset movielens1m --preset lightgcn --epochs 3
    python experiments/run_experiment.py --dataset kuairec_v2 --preset full --seed 123 --device cuda
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from experiments.recipes import (
    get_recipe,
    recipe_names,
    recipe_summary_lines,
)
from src.utils.config import UCaGNNConfig
from src.utils.experiment_logger import ExperimentLogger
from src.data.loaders import load_dataset
from src.data.graph_builder import build_graph
from src.models.ucagnn import UCaGNN
from src.losses.loss_suite import LossSuite
from src.training.trainer import Trainer
from src.profiling.gpu_profiler import GPUProfiler

logger = logging.getLogger("ucagnn")

DB_PATH = Path(__file__).parent.parent / "results" / "thesis_experiments.db"
MLFLOW_DB_PATH = Path(__file__).parent.parent / "results" / "mlflow.db"
CHECKPOINT_DIR = Path(__file__).parent.parent / "results" / "checkpoints"

PRESETS = {
    "lightgcn": "preset_lightgcn",
    "dice_like": "preset_dice_like",
    "full": "preset_full",
}


def _config_as_dict(config: UCaGNNConfig) -> dict:
    """Convert config dataclass to a comparable dictionary."""
    return dataclasses.asdict(config)


def _build_canonical_name(
    config: UCaGNNConfig,
    preset: str | None,
    intervention: str | None,
) -> str:
    """Build a descriptive canonical experiment name from the effective config."""
    parts = [
        config.dataset,
        preset or "custom",
        config.training_mode,
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
    if config.training_mode == "mini_batch":
        neighbor_str = "-".join(str(value) for value in config.num_neighbors)
        parts.append(f"nbr{neighbor_str}")
    if config.sample_interactions is not None:
        parts.append(f"sample{config.sample_interactions}")
    if config.loader_max_rows is not None:
        parts.append(f"loadrows{config.loader_max_rows}")
    if config.use_features:
        parts.append("feat")
    if config.feature_policy != "thesis_default":
        parts.append(f"fpolicy{config.feature_policy}")
    if config.scoring_weight_mode != "fixed":
        parts.append(f"scoremix{config.scoring_weight_mode}")
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
    return CHECKPOINT_DIR / f"{_build_canonical_name(config, preset, intervention)}.pt"


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
):
    """Return a sampled CanonicalInteractions subset for fast preflight runs."""
    if sample_interactions is None or sample_interactions >= len(canonical):
        return canonical

    rng = np.random.default_rng(seed)
    train_mask, val_mask, test_mask = canonical.get_splits(train_ratio, val_ratio)
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

    selected = np.sort(np.concatenate(chosen_parts)) if chosen_parts else np.array([], dtype=np.int64)
    selected_train = np.isin(selected, split_indices[0])
    selected_val = np.isin(selected, split_indices[1])
    selected_test = np.isin(selected, split_indices[2])

    selected_users, user_inverse = np.unique(canonical.user_id[selected], return_inverse=True)
    selected_items, item_inverse = np.unique(canonical.item_id[selected], return_inverse=True)

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
        popularity=canonical.popularity[selected_items],
        n_users=int(len(selected_users)),
        n_items=int(len(selected_items)),
        user_map={reverse_user_map[int(old_id)]: new_id for new_id, old_id in enumerate(selected_users.tolist())},
        item_map={reverse_item_map[int(old_id)]: new_id for new_id, old_id in enumerate(selected_items.tolist())},
        user_features=_slice_entity_features(canonical.user_features, selected_users),
        item_features=_slice_entity_features(canonical.item_features, selected_items),
        train_mask=selected_train,
        val_mask=selected_val,
        test_mask=selected_test,
        metadata=_slice_metadata(canonical.metadata),
    )


def _load_checkpoint_metadata(path: Path, device: str) -> dict | None:
    """Load a checkpoint payload if it exists."""
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=False)


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
    """Construct the trainer matching the configured training mode."""
    trainer_kwargs = dict(
        model=model,
        loss_suite=loss_suite,
        data=data,
        config=config,
        profiler=profiler,
        experiment_logger=experiment_logger,
        exp_id=exp_id,
    )
    if config.training_mode == "mini_batch":
        from src.training.mini_batch_trainer import MiniBatchTrainer

        return MiniBatchTrainer(**trainer_kwargs)
    if config.training_mode == "cached_propagation":
        from src.training.cached_trainer import CachedPropagationTrainer

        return CachedPropagationTrainer(**trainer_kwargs)
    return Trainer(**trainer_kwargs)


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
) -> dict[str, str]:
    """Build compact MLflow tags without duplicating parameter columns in the UI."""
    tags = {
        "status": "running",
    }
    if experiment_id:
        tags["experiment_id"] = experiment_id
    if recipe_name:
        tags["recipe"] = recipe_name
    return tags


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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_mlflow_params(
    config: UCaGNNConfig,
    preset: str | None,
    intervention: str | None,
    recipe_name: str | None,
    run_started_at_utc: str,
) -> dict[str, str | int | float | bool]:
    """Select compact config fields to expose as searchable MLflow params."""
    params: dict[str, str | int | float | bool] = {
        "dataset": config.dataset,
        "preset": preset or "custom",
        "training_mode": config.training_mode,
        "graph_method": config.graph_method,
        "seed": config.seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "embed_dim": config.embed_dim,
        "n_gnn_layers": config.n_gnn_layers,
        "interest_gnn_layers": config.resolved_interest_gnn_layers,
        "conformity_gnn_layers": config.resolved_conformity_gnn_layers,
        "eval_scoring_mode": config.eval_scoring_mode,
        "scoring_weight_mode": config.scoring_weight_mode,
        "sample_interactions": config.sample_interactions or 0,
        "loader_max_rows": config.loader_max_rows or 0,
        "lr": config.lr,
        "use_features": config.use_features,
        "feature_policy": config.feature_policy,
        "use_dual_branch": config.use_dual_branch,
        "use_sign_aware": config.use_sign_aware,
        "use_counterfactual": config.use_counterfactual,
        "use_ipw": config.use_ipw,
        "use_popularity_emb": config.use_popularity_emb,
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
    if config.training_mode == "mini_batch":
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
        mlflow.start_run(run_name=run_name or _build_mlflow_run_name(config, preset, intervention))
        mlflow.set_tags(_build_mlflow_tags(config, preset, intervention, experiment_id, recipe_name))
        mlflow.log_params(_build_mlflow_params(config, preset, intervention, recipe_name, run_started_at_utc))
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
    effective_preset = getattr(args, "preset", None)
    if recipe is not None:
        kwargs.update(recipe.get("overrides", {}))
        if effective_preset is None:
            effective_preset = recipe.get("preset")

    # Core args
    kwargs["dataset"] = args.dataset
    kwargs["data_dir"] = args.data_dir
    kwargs["seed"] = args.seed
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
    if getattr(args, "lr", None) is not None:
        kwargs["lr"] = args.lr
    if getattr(args, "eval_scoring_mode", None) is not None:
        kwargs["eval_scoring_mode"] = args.eval_scoring_mode
    if getattr(args, "scoring_weight_mode", None) is not None:
        kwargs["scoring_weight_mode"] = args.scoring_weight_mode
    if getattr(args, "use_features", None) is not None:
        kwargs["use_features"] = args.use_features
    if getattr(args, "feature_policy", None) is not None:
        kwargs["feature_policy"] = args.feature_policy
    if getattr(args, "graph_method", None) is not None:
        kwargs["graph_method"] = args.graph_method
    if getattr(args, "training_mode", None) is not None:
        kwargs["training_mode"] = args.training_mode
    if getattr(args, "num_neighbors", None) is not None:
        kwargs["num_neighbors"] = args.num_neighbors
    if getattr(args, "sample_interactions", None) is not None:
        kwargs["sample_interactions"] = args.sample_interactions
    if getattr(args, "loader_max_rows", None) is not None:
        kwargs["loader_max_rows"] = args.loader_max_rows

    config = UCaGNNConfig(**kwargs)

    # Apply preset
    if effective_preset in PRESETS:
        getattr(config, PRESETS[effective_preset])()

    if recipe is not None:
        for key, value in recipe.get("overrides", {}).items():
            setattr(config, key, value)

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
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    auto_resume: bool = True,
) -> dict:
    """Run a single experiment end-to-end. Returns test metrics dict."""
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
        if saved_config is not None and _config_as_dict(saved_config) != _config_as_dict(config):
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

    # Load data
    logger.info("Loading dataset...")
    canonical = load_dataset(
        config.dataset,
        config.data_dir,
        max_rows=config.loader_max_rows,
        include_optional_features=config.use_features,
        feature_policy=config.feature_policy,
    )
    canonical = _sample_canonical_interactions(
        canonical,
        config.sample_interactions,
        config.seed,
        config.train_ratio,
        config.val_ratio,
    )
    logger.info(f"  {repr(canonical)}")

    # Build graph
    logger.info("Building graph...")
    data = build_graph(canonical, config, embeddings=None)
    logger.info(f"  Nodes: {data.num_nodes:,}, Edges: {data.edge_index.size(1):,}")
    logger.info(f"  Train: {data.train_mask.sum():,}, Val: {data.val_mask.sum():,}, Test: {data.test_mask.sum():,}")

    # Model
    model = UCaGNN(
        canonical.n_users,
        canonical.n_items,
        config,
        item_features=getattr(data, "item_features", None),
        item_popularity=data.popularity,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Loss + profiler
    loss_suite = LossSuite(config)
    profiler = (
        GPUProfiler()
        if torch.cuda.is_available() and config.enable_profiling
        else None
    )

    # Experiment logger
    experiment_logger = ExperimentLogger(db_path=str(DB_PATH))
    exp_id = checkpoint_state.get("exp_id") if checkpoint_state is not None else None
    if exp_id is None:
        exp_id = experiment_logger.log_experiment(
            config.dataset,
            config,
            preset=preset,
            intervention=intervention,
        )
    logger.info(f"Experiment ID: {exp_id}")

    if checkpoint_state is not None and checkpoint_state.get("is_complete"):
        logger.info("Checkpoint already marked complete. Returning cached result from %s", resolved_checkpoint_path)
        experiment_logger.close()
        return {
            "exp_id": exp_id,
            "test_metrics": checkpoint_state.get("test_metrics", {}),
            "history": checkpoint_state.get("history", {"train_loss": [], "val_metrics": []}),
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
                checkpoint_path=resolved_checkpoint_path if should_persist_checkpoint else None,
                checkpoint_every=checkpoint_every,
            )
        else:
            history = history or {"train_loss": [], "val_metrics": []}
            logger.info("Checkpoint already reached configured epoch budget; skipping training.")
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
                mlflow_module.log_metrics({
                    _sanitize_mlflow_metric_name(f"test_{metric}"): float(value)
                    for metric, value in test_metrics.items()
                })
                if checkpoint_path is not None:
                    mlflow_module.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
                mlflow_module.set_tag("status", "completed")
            except Exception as exc:
                logger.warning("Failed to log MLflow metrics or artifacts: %s", exc)

        mlflow_status = "FINISHED"
        return {
            "exp_id": exp_id,
            "test_metrics": test_metrics,
            "history": history,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "canonical_name": canonical_name,
            "resumed": checkpoint_state is not None,
        }
    except Exception:
        if mlflow_module is not None:
            try:
                mlflow_module.set_tag("status", "failed")
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


def main():
    parser = argparse.ArgumentParser(description="Run a U-CaGNN experiment")
    parser.add_argument("--dataset", default="movielens1m", help="Dataset name")
    parser.add_argument("--recipe", choices=recipe_names(), help="Named experiment recipe")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Config preset")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--embed-dim", type=int, default=None, help="Override embed dim")
    parser.add_argument("--n-gnn-layers", type=int, default=None, help="Override shared GNN depth")
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
            "conformity_only",
            "counterfactual_only",
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
    parser.add_argument("--graph-method", choices=["dense", "knn", "cagra"], default=None)
    parser.add_argument(
        "--training-mode",
        choices=["full_graph", "cached_propagation", "mini_batch"],
        default=None,
        help="Training mode: full_graph (default), cached_propagation, or mini_batch",
    )
    parser.add_argument(
        "--num-neighbors", type=int, nargs="+", default=None,
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
    parser.add_argument("--no-checkpoint", action="store_true", help="Skip saving checkpoint")
    parser.add_argument("--checkpoint-path", default=None, help="Optional explicit checkpoint path")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--auto-resume", dest="auto_resume", action="store_true", help="Resume automatically from a matching checkpoint")
    parser.add_argument("--no-auto-resume", dest="auto_resume", action="store_false", help="Disable automatic checkpoint resume for this run")
    parser.add_argument("--intervention", default=None, help="Ablation intervention name")
    parser.add_argument("--enable-mlflow", dest="enable_mlflow", action="store_true", help="Explicitly enable MLflow tracking")
    parser.add_argument("--no-mlflow", dest="enable_mlflow", action="store_false", help="Disable MLflow tracking for this run")
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Override MLflow tracking URI (otherwise uses MLFLOW_TRACKING_URI or results/mlflow.db)",
    )
    parser.add_argument("--mlflow-experiment-name", default="ucagnn-thesis", help="MLflow experiment name")
    parser.add_argument("--mlflow-run-name", default=None, help="Optional explicit MLflow run name")
    parser.add_argument("--experiment-id", default=None, help="Optional thesis experiment identifier tag, e.g. E1")
    parser.add_argument("--list-recipes", action="store_true", help="Print available named recipes and exit")
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
    result = run_experiment(
        config,
        preset=args.preset or (recipe.get("preset") if recipe else None),
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
