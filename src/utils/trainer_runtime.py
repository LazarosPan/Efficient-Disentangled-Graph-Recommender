"""Shared trainer runtime support for setup, checkpointing, and early stopping."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim

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
        self.model.to(self.device)
        self.loss_suite.to(self.device)

        self.optimizer = optim.Adam(
            list(model.parameters()) + list(loss_suite.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.sampler = NegativeSampler(
            n_items=data.n_items,
            popularity=data.popularity,
            n_negatives=config.n_negatives,
            hard_negative_ratio=config.hard_negative_ratio,
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
        train_users = self.data.user_nodes[train_mask].to(self.device)
        train_items = (self.data.item_nodes[train_mask] - self.data.n_users).to(
            self.device
        )
        return train_users, train_items

    def _shuffle_train_interactions(
        self,
        train_users: torch.Tensor,
        train_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shuffled training users and items for the current epoch."""
        perm = torch.randperm(train_users.size(0), device=self.device)
        return train_users[perm], train_items[perm]

    def _apply_optimization_step(
        self,
        loss: torch.Tensor,
        retain_graph: bool = False,
    ) -> None:
        """Apply the shared backward, clipping, and optimizer step sequence."""
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.loss_suite.parameters()),
            max_norm=self.config.grad_clip_norm,
        )
        self.optimizer.step()

    def _evaluate_validation_metrics(self) -> dict[str, float]:
        """Run the shared validation pass on the current model state."""
        return self.evaluator.evaluate(
            self.model,
            self.data,
            self.data.val_mask,
        )

    def _log_epoch_summary(
        self,
        epoch: int,
        avg_loss: float,
        current_ndcg: float,
        primary_metric: str,
        mode_suffix: str = "",
        skipped_batches: int = 0,
    ) -> None:
        """Emit the per-epoch training summary for the current trainer mode."""
        suffix = f" {mode_suffix}" if mode_suffix else ""
        logger.info(
            "Epoch %3d/%d | Loss: %.4f | %s: %.4f%s",
            epoch + 1,
            self.config.epochs,
            avg_loss,
            primary_metric,
            current_ndcg,
            suffix,
        )
        if skipped_batches > 0:
            logger.warning(
                "Epoch %d skipped %d non-finite training batches.",
                epoch + 1,
                skipped_batches,
            )

    def _prepare_history(self, history: dict[str, list] | None) -> dict[str, list]:
        """Normalize the training history structure used across trainer modes."""
        prepared_history = history or {
            "train_loss": [],
            "val_metrics": [],
        }
        prepared_history.setdefault("train_loss", [])
        prepared_history.setdefault("val_metrics", [])
        return prepared_history

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

    def _primary_metric_name(self) -> str:
        """Return the validation metric used for early stopping."""
        return f"NDCG@{self.config.eval_ks[-1]}"

    def _log_profiler_summary(self, should_profile: bool) -> None:
        """Emit the profiler summary when profiling is active and has samples."""
        if should_profile and self.profiler and self.profiler.stages:
            logger.info(self.profiler.summary())

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

    def _update_shared_training_state(
        self,
        epoch: int,
        history: dict[str, list],
    ) -> None:
        """Store the latest completed epoch and resume history."""
        self.completed_epoch = epoch
        self.resume_history = history

    def _update_early_stopping(
        self,
        current_ndcg: float,
        primary_metric: str,
        epoch: int,
        history: dict[str, list],
        checkpoint_path: str | Path | None,
        checkpoint_every: int | None,
    ) -> bool:
        """Update early-stopping state and return whether training should stop."""
        if current_ndcg > self.best_ndcg:
            self.best_ndcg = current_ndcg
            self.patience_counter = 0
            self.best_state = {
                key: value.cpu().clone()
                for key, value in self.model.state_dict().items()
            }
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

    def _restore_best_model(self) -> None:
        """Restore the best model parameters captured during training."""
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

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
            torch.set_rng_state(rng_state)
        cuda_rng_state_all = ckpt.get("cuda_rng_state_all")
        if cuda_rng_state_all is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_rng_state_all)
            except RuntimeError:
                logger.warning(
                    "Failed to restore CUDA RNG state from checkpoint %s", path
                )
        logger.info("Checkpoint loaded from %s", path)
        return ckpt
