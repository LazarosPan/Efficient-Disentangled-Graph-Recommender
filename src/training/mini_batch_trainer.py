"""Mini-batch GNN training: extract subgraphs per batch for low-VRAM training.

Uses SubgraphSampler to extract a k-hop subgraph around each batch's seed
nodes (users + positive items + negative items), then runs GCN propagation
on the subgraph only.  This keeps VRAM usage proportional to the batch
neighbourhood size rather than the full graph.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ..utils.config import UCaGNNConfig
from ..utils.trainer_runtime import TrainerRuntime
from ..models.ucagnn import UCaGNN
from ..losses.loss_suite import LossSuite
from ..data.subgraph_sampler import SubgraphSampler
from ..profiling.gpu_profiler import GPUProfiler, profile_stage

logger = logging.getLogger(__name__)


class MiniBatchTrainer(TrainerRuntime):
    """Train U-CaGNN with mini-batch subgraph sampling."""

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

        self.subgraph_sampler = SubgraphSampler(
            edge_index=data.edge_index.to(self.device),
            edge_sign=(
                data.edge_sign.to(self.device)
                if hasattr(data, "edge_sign") and data.edge_sign is not None
                else None
            ),
            n_users=data.n_users,
            n_items=data.n_items,
            num_hops=config.max_gnn_layers,
            max_neighbors_per_hop=config.num_neighbors,
        )

    def train(
        self,
        start_epoch: int = 0,
        history: dict[str, list] | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int | None = 1,
    ) -> dict[str, list]:
        """Run full training loop with mini-batch subgraph sampling.

        Returns:
            History dict with per-epoch losses and metrics.
        """
        history = self._prepare_history(history)

        data = self.data
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

            for start in range(0, n_train, self.config.batch_size):
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

                # Extract subgraph
                with profile_stage("subgraph", self.profiler):
                    sub_batch = self.subgraph_sampler.sample(
                        batch_users,
                        batch_pos_items,
                        batch_neg_items,
                    )

                # Forward pass on subgraph
                with profile_stage("forward", self.profiler):
                    output = self.model.forward_subgraph(sub_batch)

                # Loss computation with local indices and local popularity
                with profile_stage("loss", self.profiler):
                    local_popularity = popularity[sub_batch.item_global_ids]
                    losses = self.loss_suite(
                        output,
                        local_popularity,
                        sub_batch.batch_pos_local,
                        epoch,
                    )

                # Backward + step
                with profile_stage("backward", self.profiler):
                    self._apply_optimization_step(losses["total"])

                epoch_losses.append(losses["total"].item())

            avg_loss = float(np.mean(epoch_losses))
            history["train_loss"].append(avg_loss)

            with profile_stage("eval", self.profiler):
                val_metrics = self._evaluate_validation_metrics()
            history["val_metrics"].append(val_metrics)
            epoch_time_s = time.perf_counter() - epoch_start

            primary_metric = self._primary_metric_name()
            current_ndcg = val_metrics.get(primary_metric, 0.0)
            self._step_scheduler(current_ndcg)
            self._log_epoch_summary(
                epoch,
                avg_loss,
                current_ndcg,
                primary_metric,
                mode_suffix="[mini-batch]",
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
