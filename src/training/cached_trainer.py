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

import numpy as np
import torch

from ..utils.config import UCaGNNConfig
from ..utils.trainer_runtime import TrainerRuntime
from ..models.ucagnn import UCaGNN
from ..losses.loss_suite import LossSuite
from ..profiling.gpu_profiler import GPUProfiler, profile_stage

logger = logging.getLogger(__name__)


class CachedPropagationTrainer(TrainerRuntime):
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
        super().__init__(
            model=model,
            loss_suite=loss_suite,
            data=data,
            config=config,
            profiler=profiler,
            experiment_logger=experiment_logger,
            exp_id=exp_id,
        )

    def train(
        self,
        start_epoch: int = 0,
        history: dict[str, list] | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int | None = 1,
    ) -> dict[str, list]:
        """Run full training loop with cached propagation.

        Returns:
            History dict with per-epoch losses and metrics.
        """
        history = self._prepare_history(history)

        data = self.data
        edge_index = data.edge_index.to(self.device)
        edge_sign = (
            data.edge_sign.to(self.device) if hasattr(data, "edge_sign") else None
        )
        popularity = data.popularity.to(self.device)

        train_users, train_items = self._get_train_interactions()
        n_train = train_users.size(0)

        for epoch in range(start_epoch, self.config.epochs):
            should_profile = self._set_epoch_profiling(epoch)

            epoch_start = time.perf_counter()

            train_users_shuffled, train_items_shuffled = (
                self._shuffle_train_interactions(
                    train_users,
                    train_items,
                )
            )

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
                    is_last_batch = batch_idx == n_batches - 1
                    self._apply_optimization_step(
                        losses["total"],
                        retain_graph=not is_last_batch,
                    )

                epoch_losses.append(losses["total"].item())

            avg_loss = float(np.mean(epoch_losses))
            history["train_loss"].append(avg_loss)

            with profile_stage("eval", self.profiler):
                val_metrics = self._evaluate_validation_metrics()
            history["val_metrics"].append(val_metrics)
            epoch_time_s = time.perf_counter() - epoch_start

            primary_metric = self._primary_metric_name()
            current_ndcg = val_metrics.get(primary_metric, 0.0)
            self._log_epoch_summary(
                epoch,
                avg_loss,
                current_ndcg,
                primary_metric,
                mode_suffix="[cached]",
            )

            self._log_profiler_summary(should_profile)

            # SQLite experiment logging
            self._log_epoch_to_sqlite(
                epoch,
                avg_loss,
                epoch_time_s,
                val_metrics,
                should_profile,
            )

            self._update_shared_training_state(epoch, history)

            # Early stopping
            if self._update_early_stopping(
                current_ndcg,
                primary_metric,
                epoch,
                history,
                checkpoint_path,
                checkpoint_every,
            ):
                break

            self._maybe_save_checkpoint(
                epoch,
                checkpoint_path,
                checkpoint_every,
                history,
            )

        # Restore best model
        self._restore_best_model()

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
