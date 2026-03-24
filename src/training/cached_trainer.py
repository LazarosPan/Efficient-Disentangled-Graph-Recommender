"""Cached-propagation training: propagate once per epoch, reuse for all batches.

Since LGConv has NO learnable conv parameters, the propagation result is
identical within an epoch.  This trainer caches the propagation graph once and
scores each batch by indexing into the cached embeddings.

All backward() calls except the last use ``retain_graph=True`` to keep the
computation graph alive so that gradients can flow back to the embedding
tables through the shared propagation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from ..utils.config import UCaGNNConfig
from ..models.ucagnn import UCaGNN
from ..losses.loss_suite import LossSuite
from ..data.negative_sampler import NegativeSampler
from ..profiling.gpu_profiler import GPUProfiler, profile_stage
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class CachedPropagationTrainer:
    """Train U-CaGNN with once-per-epoch GCN propagation."""

    def __init__(
        self,
        model: UCaGNN,
        loss_suite: LossSuite,
        data,
        config: UCaGNNConfig,
        profiler: GPUProfiler | None = None,
        experiment_logger=None,
        exp_id: int | None = None,
    ) -> None:
        self.model = model
        self.loss_suite = loss_suite
        self.data = data
        self.config = config
        self.profiler = profiler
        self.experiment_logger = experiment_logger
        self.exp_id = exp_id
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

    def train(
        self,
        start_epoch: int = 0,
        history: dict[str, list] | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 1,
    ) -> dict[str, list]:
        """Run full training loop with cached propagation.

        Returns:
            History dict with per-epoch losses and metrics.
        """
        history = history or {
            "train_loss": [],
            "val_metrics": [],
        }
        history.setdefault("train_loss", [])
        history.setdefault("val_metrics", [])

        data = self.data
        edge_index = data.edge_index.to(self.device)
        edge_sign = (
            data.edge_sign.to(self.device) if hasattr(data, "edge_sign") else None
        )
        popularity = data.popularity.to(self.device)

        train_mask = data.train_mask
        train_users = data.user_nodes[train_mask].to(self.device)
        train_items = (data.item_nodes[train_mask] - data.n_users).to(self.device)

        n_train = train_users.size(0)

        for epoch in range(start_epoch, self.config.epochs):
            should_profile = (
                self.profiler is not None
                and self.config.enable_profiling
                and (epoch + 1) % self.config.profiling_cadence == 0
            )
            if self.profiler:
                self.profiler.set_enabled(should_profile)
                self.profiler.reset()

            epoch_start = time.perf_counter()

            # Shuffle training data
            perm = torch.randperm(n_train, device=self.device)
            train_users_shuffled = train_users[perm]
            train_items_shuffled = train_items[perm]

            epoch_losses: list[float] = []
            self.model.train()
            self.loss_suite.train()

            # ONCE per epoch: full propagation (retains computation graph)
            with profile_stage("propagate", self.profiler):
                init_embs = self.model.embedding.get_all_embeddings()
                propagated = self.model.gcn(
                    init_embs,
                    edge_index,
                    edge_sign,
                    n_users=self.model.n_users,
                    n_items=self.model.n_items,
                )

            n_batches = (n_train + self.config.batch_size - 1) // self.config.batch_size

            for batch_idx, start in enumerate(
                range(0, n_train, self.config.batch_size)
            ):
                end = min(start + self.config.batch_size, n_train)
                batch_users = train_users_shuffled[start:end]
                batch_pos_items = train_items_shuffled[start:end]
                batch_size = batch_users.size(0)

                # Negative sampling
                with profile_stage("neg_sample", self.profiler):
                    batch_neg_items = self.sampler.sample(
                        batch_size,
                        batch_pos_items,
                        self.device,
                    ).squeeze(-1)

                # Score using cached propagated embeddings
                with profile_stage("forward", self.profiler):
                    pos_scores = self.model.scoring(
                        propagated, batch_users, batch_pos_items
                    )
                    neg_scores = self.model.scoring(
                        propagated, batch_users, batch_neg_items
                    )

                    output = self._build_output(
                        propagated,
                        init_embs,
                        pos_scores,
                        neg_scores,
                        batch_users,
                        batch_pos_items,
                    )

                # Loss computation
                with profile_stage("loss", self.profiler):
                    losses = self.loss_suite(
                        output,
                        popularity,
                        batch_pos_items,
                        epoch,
                    )

                # Backward + step
                with profile_stage("backward", self.profiler):
                    self.optimizer.zero_grad()
                    is_last_batch = batch_idx == n_batches - 1
                    losses["total"].backward(retain_graph=not is_last_batch)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters())
                        + list(self.loss_suite.parameters()),
                        max_norm=self.config.grad_clip_norm,
                    )
                    self.optimizer.step()

                epoch_losses.append(losses["total"].item())

            avg_loss = float(np.mean(epoch_losses))
            history["train_loss"].append(avg_loss)

            # Validation
            with profile_stage("eval", self.profiler):
                val_metrics = self.evaluator.evaluate(
                    self.model,
                    data,
                    data.val_mask,
                )
            history["val_metrics"].append(val_metrics)
            epoch_time_s = time.perf_counter() - epoch_start

            # Logging
            primary_metric = f"NDCG@{self.config.eval_ks[-1]}"
            current_ndcg = val_metrics.get(primary_metric, 0.0)
            logger.info(
                f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"{primary_metric}: {current_ndcg:.4f} [cached]"
            )

            if should_profile and self.profiler and self.profiler.stages:
                logger.info(self.profiler.summary())

            # SQLite experiment logging
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

            self.completed_epoch = epoch
            self.resume_history = history

            # Early stopping
            if current_ndcg > self.best_ndcg:
                self.best_ndcg = current_ndcg
                self.patience_counter = 0
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best {primary_metric}: {self.best_ndcg:.4f})"
                    )
                    if checkpoint_path is not None and checkpoint_every > 0:
                        self.save_checkpoint(checkpoint_path, history=history)
                    break

            if (
                checkpoint_path is not None
                and checkpoint_every > 0
                and (epoch + 1) % checkpoint_every == 0
            ):
                self.save_checkpoint(checkpoint_path, history=history)

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return history

    def _build_output(
        self,
        propagated: dict[str, torch.Tensor],
        embeddings: dict[str, torch.Tensor],
        pos_scores: dict[str, torch.Tensor],
        neg_scores: dict[str, torch.Tensor],
        batch_users: torch.Tensor,
        batch_pos_items: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict]:
        """Build the output dict expected by LossSuite."""
        result: dict[str, torch.Tensor | dict] = {
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "embeddings": embeddings,
            "propagated": propagated,
        }

        # IPW weights
        if self.config.use_ipw:
            if self.config.use_dual_branch:
                item_emb = propagated["item_interest"][batch_pos_items]
            else:
                item_emb = propagated["item"][batch_pos_items]
            result["ipw_weights"] = self.model.propensity.get_ipw_weights(item_emb)
        else:
            result["ipw_weights"] = torch.ones(
                batch_users.size(0),
                device=batch_users.device,
            )

        return result

    def save_checkpoint(
        self,
        path: str | Path,
        history: dict[str, list] | None = None,
        is_complete: bool = False,
        test_metrics: dict[str, float] | None = None,
        exp_id: int | None = None,
        canonical_name: str | None = None,
    ) -> None:
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
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
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
        logger.info(f"Checkpoint loaded from {path}")
        return ckpt
