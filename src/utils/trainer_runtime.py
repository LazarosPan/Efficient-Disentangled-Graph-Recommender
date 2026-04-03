"""Shared trainer runtime support for setup, checkpointing, and early stopping."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..data.negative_sampler import NegativeSampler
from ..losses.loss_suite import LossSuite
from ..models.ucagnn import UCaGNN
from ..profiling.gpu_profiler import GPUProfiler
from .config import UCaGNNConfig

logger = logging.getLogger(__name__)


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
    ) -> None:
        """Initialize shared trainer state and optimizer-owned modules."""
        self.model = model
        self.loss_suite = loss_suite
        self.data = data
        self.config = config
        self.profiler = profiler
        self.experiment_logger = experiment_logger
        self.exp_id = exp_id
        from ..training.evaluator import Evaluator

        self.evaluator = Evaluator(config)

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = self.device.type == "cuda" and self.config.use_amp
        self.amp_dtype = torch.bfloat16
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("medium")
        self.model.to(self.device)
        self.loss_suite.to(self.device)
        self.popularity = data.popularity.to(self.device)
        self.trainable_parameters = list(model.parameters()) + list(
            loss_suite.parameters()
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
            for p in (
                [getattr(gcn, attr)] if gcn is not None and hasattr(gcn, attr) else []
            )
        ]
        sign_param_ids = {id(p) for p in sign_params}
        main_params = [
            p for p in self.trainable_parameters if id(p) not in sign_param_ids
        ]

        use_fused = self.device.type == "cuda"
        param_groups: list[dict] = [
            {"params": main_params, "weight_decay": config.weight_decay}
        ]
        if sign_params:
            param_groups.append({"params": sign_params, "weight_decay": 0.0})
        self.optimizer = optim.AdamW(
            param_groups,
            lr=config.lr,
            fused=use_fused,
        )

        self.scheduler: ReduceLROnPlateau | None = None
        if config.lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=config.lr_scheduler_factor,
                patience=config.lr_scheduler_patience,
            )

        self.sampler = NegativeSampler(
            n_items=data.n_items,
            popularity=data.popularity,
            n_negatives=config.n_negatives,
            hard_negative_ratio=config.hard_negative_ratio,
        )

        # Optional EMA model for smoother generalization
        self.ema_model: torch.optim.swa_utils.AveragedModel | None = None
        if config.use_ema:
            ema_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(config.ema_decay)
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                model, multi_avg_fn=ema_fn
            )

        self.best_ndcg = 0.0
        self.patience_counter = 0
        self.best_state = None
        self.completed_epoch = -1
        self.resume_history: dict[str, list] = {
            "train_loss": [],
            "val_metrics": [],
        }

    def _get_train_interactions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return training user/item tensors in the shared index space."""
        train_mask = self.data.train_mask
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

    def _evaluate_validation_metrics(self) -> dict[str, float]:
        """Run the shared validation pass, using EMA weights when available."""
        eval_model = self.ema_model if self.ema_model is not None else self.model
        with self._autocast_context():
            return self.evaluator.evaluate(
                eval_model,
                self.data,
                self.data.val_mask,
            )

    def _log_epoch_summary(
        self,
        epoch: int,
        avg_loss: float,
        current_ndcg: float,
        primary_metric: str,
        skipped_batches: int = 0,
    ) -> None:
        """Emit the per-epoch training summary."""
        logger.info(
            "Epoch %3d/%d | Loss: %.4f | %s: %.4f",
            epoch + 1,
            self.config.epochs,
            avg_loss,
            primary_metric,
            current_ndcg,
        )
        if skipped_batches > 0:
            logger.warning(
                "Epoch %d skipped %d non-finite training batches.",
                epoch + 1,
                skipped_batches,
            )

    def _autocast_context(self):
        """Return the CUDA bf16 autocast context when AMP is enabled."""
        if self.use_amp:
            return torch.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    def _set_epoch_profiling(self, epoch: int) -> bool:
        """Enable or disable profiling for the current epoch based on config."""
        should_profile = (
            self.profiler is not None
            and self.config.enable_profiling
            and (epoch + 1) % self.config.profiling_cadence == 0
        )
        if self.profiler:
            self.profiler.set_enabled(should_profile)
            self.profiler.reset()
        return should_profile

    def _step_scheduler(self, metric_value: float, epoch: int) -> None:
        """Step the LR scheduler after the curriculum warmup has completed."""
        if self.scheduler is None:
            return

        warmup_end = max(
            self.config.curriculum_phase1_end,
            self.config.curriculum_phase2_end,
        )
        if epoch < warmup_end:
            return

        self.scheduler.step(metric_value)

    def _primary_metric_name(self) -> str:
        """Return the validation metric used for early stopping."""
        from ..training.evaluator import THESIS_EVAL_KS

        return f"NDCG@{THESIS_EVAL_KS[-1]}"

    def _log_epoch_to_sqlite(
        self,
        epoch: int,
        avg_loss: float,
        epoch_time_s: float,
        val_metrics: dict[str, float],
        should_profile: bool,
    ) -> None:
        """Persist epoch metrics and profiling output through the experiment logger."""
        if self.experiment_logger and self.exp_id is not None:
            self.experiment_logger.log_epoch(
                self.exp_id,
                epoch,
                avg_loss,
                epoch_time_s,
                val_metrics,
                self.profiler.stages if should_profile and self.profiler else [],
                self.model,
            )

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
                    (k[len(prefix) :] if k.startswith(prefix) else k): v.cpu().clone()
                    for k, v in self.ema_model.state_dict().items()
                }
            else:
                self.best_state = {
                    key: value.cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            return False

        if not self.config.use_early_stopping:
            self.patience_counter = 0
            return False

        # Defer patience counting until all curriculum phases are active
        warmup_end = max(
            self.config.curriculum_phase1_end,
            self.config.curriculum_phase2_end,
        )
        if epoch < warmup_end:
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
        if (
            checkpoint_path is not None
            and checkpoint_every is not None
            and checkpoint_every > 0
        ):
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
            "ema_state": (
                self.ema_model.state_dict() if self.ema_model is not None else None
            ),
            "config": self.config,
            "best_ndcg": self.best_ndcg,
            "patience_counter": self.patience_counter,
            "best_state": self.best_state,
            "completed_epoch": self.completed_epoch,
            "history": history if history is not None else self.resume_history,
            "rng_state": torch.get_rng_state(),
            "exp_id": exp_id if exp_id is not None else self.exp_id,
            "canonical_name": canonical_name,
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
        self.model.load_state_dict(ckpt["model_state"])
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
                    "Failed to restore CUDA RNG state from checkpoint %s", path
                )
        logger.info("Checkpoint loaded from %s", path)
        return ckpt
