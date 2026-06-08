"""Shared trainer runtime support for setup, checkpointing, and early stopping."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)

from ..data.interaction_masks import positive_interaction_mask
from ..data.negative_sampler import NegativeSampler
from ..losses.loss_suite import LossSuite
from ..models.ucagnn import UCaGNN
from ..profiling.gpu_profiler import GPUProfiler, TrainingResourceStats
from .config import UCaGNNConfig
from .reproducibility import configure_torch_runtime

logger = logging.getLogger(__name__)

# Key renames introduced during refactoring — map old checkpoint keys to current names.
_STATE_KEY_MIGRATIONS: dict[str, str] = {
    "propensity.mlp": "_propensity_mlp",
}
REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "model_state",
        "optimizer_state",
        "loss_suite_state",
        "config",
    },
)


def autocast_context(
    *,
    use_amp: bool,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> AbstractContextManager[None]:
    """Return the shared CUDA autocast context for AMP-enabled regions.

    Args:
        use_amp: Whether CUDA AMP should be enabled.
        amp_dtype: Autocast dtype used when AMP is enabled.

    Returns:
        Active CUDA autocast context when ``use_amp`` is true, else ``nullcontext``.

    """
    if use_amp:
        return torch.autocast("cuda", dtype=amp_dtype)
    return nullcontext()


def move_tensor_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Move a tensor with the runtime's shared non-blocking device policy.

    Args:
        tensor: Tensor to move.
        device: Destination device.
        dtype: Optional target dtype override.

    Returns:
        torch.Tensor: Tensor moved to ``device``.

    """
    move_kwargs: dict[str, object] = {
        "device": device,
        "non_blocking": device.type == "cuda",
    }
    if dtype is not None:
        move_kwargs["dtype"] = dtype
    return tensor.to(**move_kwargs)


def move_optional_tensor_to_device(
    tensor: torch.Tensor | None,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Move an optional tensor to a device when it exists.

    Args:
        tensor: Optional tensor to move.
        device: Destination device.
        dtype: Optional target dtype override.

    Returns:
        torch.Tensor | None: Moved tensor, or ``None`` when ``tensor`` is absent.

    """
    if tensor is None:
        return None
    return move_tensor_to_device(tensor, device, dtype=dtype)


def stage_graph_tensors_for_device(
    data: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Stage edge tensors on a device for sampler or evaluator use.

    Args:
        data: Graph-like object carrying ``edge_index`` and optional edge fields.
        device: Destination device for the staged tensors.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        ``(edge_index, edge_sign, edge_norm)`` on ``device``.

    """
    return (
        move_tensor_to_device(data.edge_index, device),
        move_optional_tensor_to_device(getattr(data, "edge_sign", None), device),
        move_optional_tensor_to_device(getattr(data, "edge_norm", None), device),
    )


def model_device(module: torch.nn.Module) -> torch.device:
    """Return the device of the first parameter owned by ``module``.

    Args:
        module: Module whose active parameter device should be reported.

    Returns:
        torch.device: Device of the first parameter in ``module``.

    """
    return next(module.parameters()).device


def empty_cuda_cache(device: torch.device) -> None:
    """Clear the CUDA allocator cache when the target device is CUDA.

    Args:
        device: Device whose allocator cache should be considered.

    Returns:
        None.

    """
    if device.type == "cuda":
        torch.cuda.empty_cache()


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Return whether an exception represents a CUDA out-of-memory failure."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _migrate_model_state(state: dict[str, Any]) -> dict[str, Any]:
    """Rewrite legacy parameter-name prefixes so old checkpoints load cleanly."""
    migrated: dict[str, Any] = {}
    for key, value in state.items():
        new_key = key
        for old_prefix, new_prefix in _STATE_KEY_MIGRATIONS.items():
            if key.startswith(old_prefix + ".") or key == old_prefix:
                new_key = new_prefix + key[len(old_prefix) :]
                break
        migrated[new_key] = value
    return migrated


class TrainerRuntime:
    """Shared trainer setup and checkpoint lifecycle helpers."""

    def __init__(
        self,
        model: UCaGNN,
        loss_suite: LossSuite,
        data: Any,
        config: UCaGNNConfig,
        profiler: GPUProfiler | None = None,
        experiment_logger: Any = None,
        exp_id: int | None = None,
        mlflow_module: Any = None,
    ) -> None:
        """Initialize shared trainer state and optimizer-owned modules."""
        self.model = model
        self.loss_suite = loss_suite
        self.data = data
        self.config = config
        self.profiler = profiler
        self.experiment_logger = experiment_logger
        self.exp_id = exp_id
        self.mlflow_module = mlflow_module
        from ..training.evaluator import Evaluator

        self.evaluator = Evaluator(config)

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu",
        )
        configure_torch_runtime()
        self.use_amp = self.device.type == "cuda" and self.config.use_amp
        self.amp_dtype = torch.bfloat16
        self.model.to(self.device)
        self.loss_suite.to(self.device)
        self.popularity = move_tensor_to_device(data.popularity, self.device)
        branch_popularity_source = getattr(data, "popularity_count", data.popularity)
        self.branch_popularity = move_tensor_to_device(branch_popularity_source, self.device)
        self.propensity_targets = move_optional_tensor_to_device(
            getattr(data, "propensity_targets", None), self.device
        )
        self.trainable_parameters = list(model.parameters()) + list(
            loss_suite.parameters(),
        )

        # Separate sign-aware scalar parameters (alpha_pos / alpha_neg) from the
        # main parameter group.  On datasets with no signed interactions those
        # parameters receive zero task gradient, so Adam's epsilon denominator
        # amplifies the L2 weight-decay term enough to drive them to zero.  Keeping
        # them in a zero-decay group makes their values stable.
        gcn = getattr(model, "gcn", None)
        sign_params = [
            p
            for attr in ("alpha_pos", "alpha_neg")
            for p in ([getattr(gcn, attr)] if gcn is not None and hasattr(gcn, attr) else [])
        ]
        sign_param_ids = {id(p) for p in sign_params}
        main_params = [p for p in self.trainable_parameters if id(p) not in sign_param_ids]

        self.optimizer = self._build_optimizer(main_params, sign_params)

        self.scheduler = self._build_scheduler()

        sampler_train_users, sampler_train_items = self._get_train_interactions()
        self.sampler = NegativeSampler(
            n_items=data.n_items,
            popularity=branch_popularity_source
            if config.negative_sampling_strategy == "dice"
            else data.popularity,
            n_negatives=config.n_negatives,
            hard_negative_ratio=config.hard_negative_ratio,
            positive_user_ids=sampler_train_users,
            positive_item_ids=sampler_train_items,
            strategy=config.negative_sampling_strategy,
            dice_margin=config.dice_sampler_margin,
            dice_pool=config.dice_sampler_pool,
            dice_margin_decay=config.dice_margin_decay if config.dice_adaptive_decay else 1.0,
            exact_dice_pool_counts=config.baseline_family == "dice_paper",
        )

        # Optional EMA model for smoother generalization
        self.ema_model: torch.optim.swa_utils.AveragedModel | None = None
        if config.use_ema:
            ema_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(config.ema_decay)
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                model,
                multi_avg_fn=ema_fn,
            )

        self.best_ndcg = 0.0
        self.patience_counter = 0
        self.best_state = None
        self.training_peak_vram_mb: float | None = None
        self.completed_epoch = -1
        self.training_identity: dict[str, Any] | None = None
        self.training_hash: str | None = None
        self.evaluation_identity: dict[str, Any] | None = None
        self.evaluation_hash: str | None = None
        self.resume_history: dict[str, list] = {
            "train_loss": [],
            "val_metrics": [],
        }

    def _build_optimizer(
        self,
        main_params: list[torch.nn.Parameter],
        sign_params: list[torch.nn.Parameter],
    ) -> optim.Optimizer:
        """Build the optimizer required by the active model family."""
        if self.config.baseline_family == "lightgcn_paper":
            return optim.Adam(
                [{"params": main_params, "weight_decay": 0.0}],
                lr=self.config.lr,
            )
        if self.config.baseline_family == "dice_paper":
            return optim.Adam(
                [{"params": main_params, "weight_decay": self.config.weight_decay}],
                lr=self.config.lr,
                betas=(0.5, 0.99),
                amsgrad=True,
            )

        param_groups: list[dict] = [
            {"params": main_params, "weight_decay": self.config.weight_decay},
        ]
        if sign_params:
            param_groups.append({"params": sign_params, "weight_decay": 0.0})
        return optim.AdamW(
            param_groups,
            lr=self.config.lr,
            fused=self.device.type == "cuda",
        )

    def _build_scheduler(self) -> Any:
        """Build the configured learning-rate scheduler for the optimizer."""
        scheduler_name = self.config.lr_scheduler
        if scheduler_name == "none":
            return None
        if scheduler_name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience,
            )
        if scheduler_name == "step":
            return StepLR(
                self.optimizer,
                step_size=max(1, self.config.epochs // 10),
                gamma=self.config.lr_scheduler_factor,
            )
        if scheduler_name == "multi_step":
            return MultiStepLR(
                self.optimizer,
                milestones=[
                    max(1, int(self.config.epochs * 0.3)),
                    max(1, int(self.config.epochs * 0.8)),
                ],
                gamma=self.config.lr_scheduler_factor,
            )
        if scheduler_name == "exponential":
            return ExponentialLR(
                self.optimizer,
                gamma=max(0.01, self.config.lr_scheduler_factor),
            )
        if scheduler_name == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.config.epochs),
            )
        if scheduler_name == "cosine_restart":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(1, self.config.epochs // 10),
                T_mult=2,
            )
        if scheduler_name == "polynomial":
            exponent = max(0.1, self.config.lr_scheduler_factor)
            return LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: max(
                    0.0,
                    (1.0 - float(epoch) / float(self.config.epochs)) ** exponent,
                ),
            )
        if scheduler_name == "linear":
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=max(1, self.config.epochs),
            )
        raise ValueError(f"Unsupported lr_scheduler {scheduler_name!r}.")

    def _get_train_interactions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return training user/item tensors in the shared index space."""
        train_mask = getattr(self.data, "train_positive_mask", None)
        if train_mask is None:
            train_mask = positive_interaction_mask(
                self.data.train_mask,
                getattr(self.data, "labels", None),
            )
        train_users = self.data.user_nodes[train_mask]
        train_items = self.data.item_nodes[train_mask] - self.data.n_users
        return train_users, train_items

    def _apply_optimization_step(
        self,
        loss: torch.Tensor,
        retain_graph: bool = False,
    ) -> None:
        """Apply the shared backward, clipping, and optimizer step sequence."""
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=retain_graph)
        torch.nn.utils.clip_grad_norm_(
            self.trainable_parameters,
            max_norm=self.config.grad_clip_norm,
        )
        self.optimizer.step()
        if self.ema_model is not None:
            self.ema_model.update_parameters(self.model)

    def _move_optimizer_state(self, device: torch.device) -> None:
        """Move optimizer state tensors to ``device`` in place."""
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = move_tensor_to_device(value, device)

    @staticmethod
    def _invalidate_eval_feature_cache(eval_model: Any) -> None:
        """Drop cached full-catalog feature tensors before an evaluation retry."""
        embedding = getattr(eval_model, "embedding", None)
        invalidate = getattr(embedding, "invalidate_feature_cache", None)
        if callable(invalidate):
            invalidate()

    def _evaluate_split_metrics_on_cpu(
        self,
        eval_model: Any,
        mask: torch.Tensor,
        include_refined_diagnostics: bool = False,
    ) -> dict[str, float]:
        """Evaluate one split on CPU, restoring the model device afterward.

        Args:
            eval_model: Model used for evaluation.
            mask: Split mask to evaluate.
            include_refined_diagnostics: Whether to append refined scorer
                diagnostics.

        Returns:
            dict[str, float]: Evaluation metrics.

        """
        self._invalidate_eval_feature_cache(eval_model)
        eval_model.to(torch.device("cpu"))
        try:
            return self._evaluate_model(
                eval_model,
                mask.cpu(),
                use_amp=False,
                include_refined_diagnostics=include_refined_diagnostics,
            )
        finally:
            self._invalidate_eval_feature_cache(eval_model)
            eval_model.to(self.device)

    def _evaluate_model(
        self,
        eval_model: Any,
        mask: torch.Tensor,
        *,
        use_amp: bool,
        include_refined_diagnostics: bool = False,
    ) -> dict[str, float]:
        """Run evaluator metrics for one model/split pair under a chosen AMP policy.

        Args:
            eval_model: Model used for evaluation.
            mask: Split mask to evaluate.
            use_amp: Whether to enable AMP for the evaluation pass.
            include_refined_diagnostics: Whether to append refined scorer
                diagnostics.

        Returns:
            dict[str, float]: Evaluation metrics.

        """
        with autocast_context(use_amp=use_amp, amp_dtype=self.amp_dtype):
            return self.evaluator.evaluate(
                eval_model,
                self.data,
                mask,
                include_refined_diagnostics=include_refined_diagnostics,
            )

    def _prepare_cuda_evaluation_retry(self, eval_model: Any) -> None:
        """Free CUDA memory before retrying evaluation on the GPU.

        Args:
            eval_model: Model used for evaluation.

        Returns:
            None.

        """
        logger.warning(
            "Evaluation hit CUDA OOM; retrying after temporarily offloading "
            "optimizer state to CPU.",
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._invalidate_eval_feature_cache(eval_model)
        empty_cuda_cache(self.device)
        self._move_optimizer_state(torch.device("cpu"))
        empty_cuda_cache(self.device)

    def _restore_cuda_evaluation_retry(self, eval_model: Any) -> None:
        """Restore optimizer and cached state after a CUDA evaluation retry.

        Args:
            eval_model: Model used for evaluation.

        Returns:
            None.

        """
        empty_cuda_cache(self.device)
        self._move_optimizer_state(self.device)
        self._invalidate_eval_feature_cache(eval_model)
        empty_cuda_cache(self.device)

    def _retry_evaluation_after_optimizer_offload(
        self,
        eval_model: Any,
        mask: torch.Tensor,
        include_refined_diagnostics: bool = False,
    ) -> dict[str, float]:
        """Retry evaluation after optimizer offload, then fall back to CPU.

        Args:
            eval_model: Model used for evaluation.
            mask: Split mask to evaluate.
            include_refined_diagnostics: Whether to include refined diagnostics.

        Returns:
            dict[str, float]: Evaluation metrics.

        """
        try:
            return self._evaluate_model(
                eval_model,
                mask,
                use_amp=self.use_amp,
                include_refined_diagnostics=include_refined_diagnostics,
            )
        except torch.OutOfMemoryError:
            logger.warning(
                "Evaluation still hit CUDA OOM after optimizer offload; retrying on CPU.",
            )
            self._invalidate_eval_feature_cache(eval_model)
            empty_cuda_cache(self.device)
            return self._evaluate_split_metrics_on_cpu(
                eval_model,
                mask,
                include_refined_diagnostics=include_refined_diagnostics,
            )

    def _evaluate_split_metrics_after_cuda_oom(
        self,
        eval_model: Any,
        mask: torch.Tensor,
        include_refined_diagnostics: bool = False,
    ) -> dict[str, float]:
        """Handle the shared CUDA-OOM evaluation retry policy.

        Args:
            eval_model: Model used for evaluation.
            mask: Split mask to evaluate.
            include_refined_diagnostics: Whether to include refined diagnostics.

        Returns:
            dict[str, float]: Evaluation metrics.

        """
        self._prepare_cuda_evaluation_retry(eval_model)
        try:
            return self._retry_evaluation_after_optimizer_offload(
                eval_model,
                mask,
                include_refined_diagnostics=include_refined_diagnostics,
            )
        finally:
            self._restore_cuda_evaluation_retry(eval_model)

    def _evaluate_split_metrics(
        self,
        mask: torch.Tensor,
        *,
        include_refined_diagnostics: bool = False,
    ) -> dict[str, float]:
        """Evaluate one split with progressive CUDA-OOM fallbacks.

        Args:
            mask: Split mask to evaluate.
            include_refined_diagnostics: Whether to append refined scorer
                diagnostics. This is intentionally opt-in because validation
                runs every epoch; final test evaluation enables it explicitly.

        Returns:
            dict[str, float]: Evaluation metrics.

        """
        eval_model = self.ema_model if self.ema_model is not None else self.model
        try:
            return self._evaluate_model(
                eval_model,
                mask,
                use_amp=self.use_amp,
                include_refined_diagnostics=include_refined_diagnostics,
            )
        except torch.OutOfMemoryError:
            if self.device.type != "cuda":
                raise
        return self._evaluate_split_metrics_after_cuda_oom(
            eval_model,
            mask,
            include_refined_diagnostics=include_refined_diagnostics,
        )

    def _evaluate_validation_metrics(self) -> dict[str, float]:
        """Run the shared validation pass, using EMA weights when available.

        Returns:
            dict[str, float]: Validation metrics without refined diagnostics.

        """
        return self._evaluate_split_metrics(
            self.data.val_mask,
            include_refined_diagnostics=False,
        )

    def _log_epoch_summary(
        self,
        epoch: int,
        avg_loss: float,
        current_ndcg: float,
        primary_metric: str,
        skipped_batches: int = 0,
        resource_stats: TrainingResourceStats | None = None,
    ) -> None:
        """Emit the per-epoch training summary."""
        peak_vram_mb = (
            resource_stats.peak_vram_mb
            if resource_stats is not None
            else GPUProfiler.peak_vram_mb()
        )
        vram_str = f" | VRAM: {peak_vram_mb:.0f} MB" if peak_vram_mb is not None else ""
        logger.info(
            "Epoch %3d/%d | Loss: %.4f | %s: %.4f%s",
            epoch + 1,
            self.config.epochs,
            avg_loss,
            primary_metric,
            current_ndcg,
            vram_str,
        )
        if skipped_batches > 0:
            logger.warning(
                "Epoch %d skipped %d non-finite training batches.",
                epoch + 1,
                skipped_batches,
            )

    def _reset_epoch_vram_stats(self) -> None:
        """Reset CUDA peak memory stats at epoch start for per-epoch VRAM tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _step_scheduler(self, metric_value: float, epoch: int) -> None:
        """Step the LR scheduler after the curriculum warmup has completed."""
        if self.scheduler is None:
            return

        if epoch < self._curriculum_warmup_end:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metric_value)
            return

        self.scheduler.step()

    def _primary_metric_name(self) -> str:
        """Return the validation metric used for early stopping."""
        from ..training.evaluator import THESIS_EVAL_KS

        return f"NDCG@{THESIS_EVAL_KS[-1]}"

    @property
    def _curriculum_warmup_end(self) -> int:
        """Return the epoch index at which the full curriculum becomes active."""
        return max(
            self.config.auxiliary_losses_start_epoch,
            self.config.popularity_supervision_start_epoch,
        )

    def _log_epoch_to_sqlite(
        self,
        epoch: int,
        avg_loss: float,
        epoch_time_s: float,
        val_metrics: dict[str, float],
        resource_stats: TrainingResourceStats | None = None,
    ) -> None:
        """Persist epoch metrics, GPU utilization, and peak VRAM through the experiment logger."""
        resource_stats = resource_stats or TrainingResourceStats.from_current_cuda_peaks(
            self.device,
        )
        peak_vram_mb = resource_stats.peak_vram_mb
        if peak_vram_mb is not None:
            current_training_peak = getattr(self, "training_peak_vram_mb", None)
            if current_training_peak is None:
                self.training_peak_vram_mb = peak_vram_mb
            else:
                self.training_peak_vram_mb = max(current_training_peak, peak_vram_mb)
        if self.experiment_logger and self.exp_id is not None:
            self.experiment_logger.log_epoch(
                self.exp_id,
                epoch,
                avg_loss,
                epoch_time_s,
                val_metrics,
                [],
                self.model,
            )
            if resource_stats.avg_gpu_utilization_pct is not None:
                self.experiment_logger.log_metric(
                    self.exp_id,
                    "gpu_utilization_pct",
                    resource_stats.avg_gpu_utilization_pct,
                    epoch=epoch,
                    split="train",
                )
                self.experiment_logger.log_metric(
                    self.exp_id,
                    "train_avg_gpu_utilization_pct",
                    resource_stats.avg_gpu_utilization_pct,
                    epoch=epoch,
                    split="train",
                )
            if resource_stats.max_gpu_utilization_pct is not None:
                self.experiment_logger.log_metric(
                    self.exp_id,
                    "max_gpu_utilization_pct",
                    resource_stats.max_gpu_utilization_pct,
                    epoch=epoch,
                    split="train",
                )
                self.experiment_logger.log_metric(
                    self.exp_id,
                    "train_max_gpu_utilization_pct",
                    resource_stats.max_gpu_utilization_pct,
                    epoch=epoch,
                    split="train",
                )
            if resource_stats.pytorch_peak_allocated_mb is not None:
                self.experiment_logger.log_metric(
                    self.exp_id,
                    "train_peak_vram_allocated_mb",
                    resource_stats.pytorch_peak_allocated_mb,
                    epoch=epoch,
                    split="train",
                )
            if resource_stats.pytorch_peak_reserved_mb is not None:
                self.experiment_logger.log_metric(
                    self.exp_id,
                    "train_peak_vram_reserved_mb",
                    resource_stats.pytorch_peak_reserved_mb,
                    epoch=epoch,
                    split="train",
                )
            if resource_stats.nvidia_peak_memory_used_mb is not None:
                self.experiment_logger.log_metric(
                    self.exp_id,
                    "train_peak_gpu_memory_used_mb",
                    resource_stats.nvidia_peak_memory_used_mb,
                    epoch=epoch,
                    split="train",
                )
            if peak_vram_mb is not None:
                self.experiment_logger.log_metric(
                    self.exp_id,
                    "peak_vram_mb",
                    peak_vram_mb,
                    epoch=epoch,
                    split="train",
                )
        if self.mlflow_module is not None:
            mlflow_metrics = {
                f"val_{m}".replace("@", "_at_"): float(v) for m, v in val_metrics.items()
            }
            mlflow_metrics["train_loss"] = avg_loss
            if resource_stats.avg_gpu_utilization_pct is not None:
                mlflow_metrics["gpu_utilization_pct"] = resource_stats.avg_gpu_utilization_pct
                mlflow_metrics["train_avg_gpu_utilization_pct"] = (
                    resource_stats.avg_gpu_utilization_pct
                )
            if resource_stats.max_gpu_utilization_pct is not None:
                mlflow_metrics["train_max_gpu_utilization_pct"] = (
                    resource_stats.max_gpu_utilization_pct
                )
            if resource_stats.pytorch_peak_allocated_mb is not None:
                mlflow_metrics["train_peak_vram_allocated_mb"] = (
                    resource_stats.pytorch_peak_allocated_mb
                )
            if resource_stats.pytorch_peak_reserved_mb is not None:
                mlflow_metrics["train_peak_vram_reserved_mb"] = (
                    resource_stats.pytorch_peak_reserved_mb
                )
            if resource_stats.nvidia_peak_memory_used_mb is not None:
                mlflow_metrics["train_peak_gpu_memory_used_mb"] = (
                    resource_stats.nvidia_peak_memory_used_mb
                )
            if peak_vram_mb is not None:
                mlflow_metrics["peak_vram_mb"] = peak_vram_mb
            self.mlflow_module.log_metrics(mlflow_metrics, step=epoch)

    def _update_early_stopping(
        self,
        current_ndcg: float,
        primary_metric: str,
        epoch: int,
        history: dict[str, list],
        checkpoint_path: str | Path | None,
        checkpoint_every: int | None,
    ) -> bool:
        """Update early-stopping state and return whether training should stop.

        Patience counting is deferred until all curriculum phases are active
        so that auxiliary losses have time to contribute before early stopping
        can fire.
        """
        if current_ndcg > self.best_ndcg:
            self.best_ndcg = current_ndcg
            self.patience_counter = 0
            # Capture EMA weights as best state when available.
            # AveragedModel prefixes keys with "module." — strip it so
            # load_state_dict() into the base model works directly.
            if self.ema_model is not None:
                prefix = "module."
                self.best_state = {
                    (k.removeprefix(prefix)): v.cpu().clone()
                    for k, v in self.ema_model.state_dict().items()
                }
            else:
                self.best_state = {
                    key: value.cpu().clone() for key, value in self.model.state_dict().items()
                }
            return False

        if not self.config.use_early_stopping:
            self.patience_counter = 0
            return False

        # Defer patience counting until all curriculum phases are active
        if epoch < self._curriculum_warmup_end:
            return False

        self.patience_counter += 1
        if self.patience_counter < self.config.patience:
            return False

        logger.info(
            "Early stopping at epoch %d (best %s: %.4f)",
            epoch + 1,
            primary_metric,
            self.best_ndcg,
        )
        if checkpoint_path is not None and checkpoint_every is not None and checkpoint_every > 0:
            self.save_checkpoint(checkpoint_path, history=history)
        return True

    def _maybe_save_checkpoint(
        self,
        epoch: int,
        checkpoint_path: str | Path | None,
        checkpoint_every: int | None,
        history: dict[str, list],
    ) -> None:
        """Persist a periodic checkpoint when checkpointing is enabled."""
        if (
            checkpoint_path is not None
            and checkpoint_every is not None
            and checkpoint_every > 0
            and (epoch + 1) % checkpoint_every == 0
        ):
            self.save_checkpoint(checkpoint_path, history=history)

    def save_checkpoint(
        self,
        path: str | Path,
        history: dict[str, list] | None = None,
        is_complete: bool = False,
        test_metrics: dict[str, float] | None = None,
        exp_id: int | None = None,
        canonical_name: str | None = None,
    ) -> None:
        """Save training state in the shared checkpoint schema."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "loss_suite_state": self.loss_suite.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "ema_state": (self.ema_model.state_dict() if self.ema_model is not None else None),
            "config": self.config,
            "best_ndcg": self.best_ndcg,
            "patience_counter": self.patience_counter,
            "best_state": self.best_state,
            "completed_epoch": self.completed_epoch,
            "history": history if history is not None else self.resume_history,
            "rng_state": torch.get_rng_state(),
            "exp_id": exp_id if exp_id is not None else self.exp_id,
            "canonical_name": canonical_name,
            "training_identity": self.training_identity,
            "training_hash": self.training_hash,
            "evaluation_identity": self.evaluation_identity,
            "evaluation_hash": self.evaluation_hash,
            "is_complete": is_complete,
            "test_metrics": test_metrics,
        }
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load training state from the shared checkpoint schema."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model_state = _migrate_model_state(ckpt["model_state"])
        self.model.load_state_dict(model_state)
        self.loss_suite.load_state_dict(ckpt["loss_suite_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler_state = ckpt.get("scheduler_state")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)
        ema_state = ckpt.get("ema_state")
        if ema_state is not None and self.ema_model is not None:
            self.ema_model.load_state_dict(ema_state)
        self.best_ndcg = ckpt.get("best_ndcg", 0.0)
        self.patience_counter = ckpt.get("patience_counter", 0)
        self.best_state = ckpt.get("best_state")
        self.completed_epoch = ckpt.get("completed_epoch", -1)
        self.resume_history = ckpt.get(
            "history",
            {"train_loss": [], "val_metrics": []},
        )
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state.cpu())
        cuda_rng_state_all = ckpt.get("cuda_rng_state_all")
        if cuda_rng_state_all is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in cuda_rng_state_all])
            except RuntimeError:
                logger.warning(
                    "Failed to restore CUDA RNG state from checkpoint %s",
                    path,
                )
        logger.info("Checkpoint loaded from %s", path)
        return ckpt
