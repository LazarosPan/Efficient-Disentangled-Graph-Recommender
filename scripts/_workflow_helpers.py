"""Small shared utilities for lightweight experiment-oriented scripts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import time
from pathlib import Path
from typing import Any

import torch

from experiments.run_experiment import run_experiment


PROJECT_ROOT = Path(__file__).parent.parent
_TINY_SAMPLE_INTERACTIONS = {
    "amazonbook": 100,
    "movielens1m": 100,
    "movielens20m": 100,
    "kuairec_v2": 100,
    "taobao": 100,
    "kuairand1k": 100,
}
_TINY_LOADER_MAX_ROWS = {
    "amazonbook": 100,
    "movielens1m": 100,
    "movielens20m": 100,
    "kuairec_v2": 100,
    "taobao": 100,
    "kuairand1k": 100,
}


def dataset_limit(
    dataset: str,
    overrides: dict[str, int],
    *,
    default: int,
) -> int:
    """Resolve a dataset-specific limit with a fallback default.

    Args:
        dataset: Dataset name used as the lookup key.
        overrides: Dataset-specific numeric overrides.
        default: Fallback value when the dataset is not present.

    Returns:
        The dataset-specific override when present, otherwise the fallback.
    """
    return int(overrides.get(dataset, default))


def tiny_sample_interactions(dataset: str, *, default: int = 100) -> int:
    """Return the shared tiny-run interaction budget for a dataset."""
    return dataset_limit(dataset, _TINY_SAMPLE_INTERACTIONS, default=default)


def tiny_loader_max_rows(dataset: str, *, default: int = 100) -> int:
    """Return the shared tiny-run loader row cap for a dataset."""
    return dataset_limit(dataset, _TINY_LOADER_MAX_ROWS, default=default)


def default_runtime_device() -> str:
    """Return the preferred execution device for smoke and utility scripts."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_batch_id(provided: str | None, prefix: str) -> str:
    """Return an explicit or generated batch id for grouped runs."""
    if provided:
        return provided
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def is_cuda_oom(exc: BaseException) -> bool:
    """Return whether an exception represents a CUDA OOM failure."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def metric_value(metrics: dict[str, float], metric_name: str) -> float:
    """Return a metric value, falling back from @40 to @20 when needed."""
    if metric_name in metrics:
        return metrics[metric_name]
    if metric_name.endswith("@40"):
        fallback = metric_name.replace("@40", "@20")
        return metrics.get(fallback, 0.0)
    return 0.0


def timed_run_experiment(
    config: Any,
    *,
    preset: str | None,
    intervention: str,
    enable_mlflow: bool,
    mlflow_experiment_name: str,
    recipe_name: str | None = None,
    save_checkpoint: bool = False,
    auto_resume: bool = False,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = None,
) -> tuple[dict[str, Any], float]:
    """Run an experiment and return the result with elapsed time.

    Args:
        config: Runtime config passed to run_experiment.
        preset: Optional preset label for logging.
        intervention: Intervention label for the run.
        enable_mlflow: Whether to enable MLflow tracking.
        mlflow_experiment_name: MLflow experiment name.
        recipe_name: Optional recipe name for logging.
        save_checkpoint: Whether to save a checkpoint.
        auto_resume: Whether to allow automatic checkpoint resume.
        checkpoint_path: Optional checkpoint path override.
        checkpoint_every: Optional checkpoint cadence override.

    Returns:
        A tuple of (run_experiment result, elapsed seconds).
    """
    started = time.perf_counter()
    result = run_experiment(
        config,
        preset=preset,
        intervention=intervention,
        save_checkpoint=save_checkpoint,
        enable_mlflow=enable_mlflow,
        mlflow_experiment_name=mlflow_experiment_name,
        recipe_name=recipe_name,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        auto_resume=auto_resume,
    )
    return result, time.perf_counter() - started


def write_json_report(
    output_path: str | Path,
    payload: dict[str, Any],
    *,
    root: Path | None = None,
) -> Path:
    """Write a JSON report, resolving relative paths from the project root.

    Args:
        output_path: Target output path.
        payload: JSON-serializable payload to write.
        root: Optional root used to resolve relative paths.

    Returns:
        The resolved output path.
    """
    resolved = Path(output_path)
    if not resolved.is_absolute():
        resolved = (root or PROJECT_ROOT) / resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved
