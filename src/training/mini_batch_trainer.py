"""Mini-batch GNN training: extract subgraphs per batch for low-VRAM training.

Uses SubgraphSampler to extract a k-hop subgraph around each batch's seed
nodes (users + positive items + negative items), then runs GCN propagation
on the subgraph only.  This keeps VRAM usage proportional to the batch
neighbourhood size rather than the full graph.

On CUDA the runtime now stages the full graph once on the accelerator and
keeps negative sampling plus sampled-BFS subgraph extraction on-device so the
GPU is no longer starved by a CPU-resident sampler.  When staging the full
graph would exceed VRAM, the trainer falls back to the original pinned-memory
CPU prefetch path.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import logging

import torch
from tqdm.auto import tqdm

from ..utils.config import UCaGNNConfig
from ..utils.trainer_runtime import TrainerRuntime
from ..models.ucagnn import UCaGNN
from ..losses.loss_suite import LossSuite
from ..data.subgraph_sampler import SubgraphBatch, SubgraphSampler
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

        self.subgraph_sampler, self.sampler_device = self._build_subgraph_sampler(data)
        self.train_users, self.train_items = self._get_train_interactions()

    def _build_subgraph_sampler(self, data) -> tuple[SubgraphSampler, torch.device]:
        """Stage the full graph on CUDA when possible, else keep the CPU path."""
        cpu_device = torch.device("cpu")
        target_device = self.device if self.device.type == "cuda" else cpu_device

        def _sampler_for(device: torch.device) -> SubgraphSampler:
            non_blocking = device.type == "cuda"
            return SubgraphSampler(
                edge_index=data.edge_index.to(device, non_blocking=non_blocking),
                edge_sign=(
                    data.edge_sign.to(device, non_blocking=non_blocking)
                    if hasattr(data, "edge_sign") and data.edge_sign is not None
                    else None
                ),
                edge_norm=(
                    data.edge_norm.to(device, non_blocking=non_blocking)
                    if hasattr(data, "edge_norm") and data.edge_norm is not None
                    else None
                ),
                n_users=data.n_users,
                n_items=data.n_items,
                num_hops=self.config.max_gnn_layers,
                max_neighbors_per_hop=self.config.num_neighbors,
            )

        if target_device.type != "cuda":
            return _sampler_for(cpu_device), cpu_device

        try:
            sampler = _sampler_for(target_device)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            logger.warning(
                "Falling back to CPU subgraph sampling after CUDA graph staging failed: %s",
                exc,
            )
            return _sampler_for(cpu_device), cpu_device

        logger.info("Mini-batch sampler staged on %s.", target_device)
        return sampler, target_device

    def _ensure_subgraph_sampler(self) -> None:
        """Rebuild the sampler after validation if its CUDA copy was released."""
        if self.subgraph_sampler is None:
            self.subgraph_sampler, self.sampler_device = self._build_subgraph_sampler(
                self.data
            )

    def _release_cuda_sampler_for_eval(self) -> bool:
        """Drop the CUDA sampler before full-graph validation to free VRAM."""
        if self.sampler_device.type != "cuda" or self.subgraph_sampler is None:
            return False
        self.subgraph_sampler = None
        self.sampler_device = torch.device("cpu")
        torch.cuda.empty_cache()
        return True

    def _prepare_batch(
        self,
        batch_users: torch.Tensor,
        batch_pos_items: torch.Tensor,
        random_seed: int,
    ) -> SubgraphBatch:
        """Sample negatives and extract a subgraph for one batch.

        CPU-resident samplers use a background-thread prefetch path and pinned
        host memory. CUDA-resident samplers prepare the full batch in device
        memory directly so subgraph extraction no longer waits on CPU copies.
        """
        self._ensure_subgraph_sampler()
        assert self.subgraph_sampler is not None
        if batch_users.device != self.sampler_device:
            non_blocking = self.sampler_device.type == "cuda"
            batch_users = batch_users.to(self.sampler_device, non_blocking=non_blocking)
            batch_pos_items = batch_pos_items.to(
                self.sampler_device,
                non_blocking=non_blocking,
            )

        generator = torch.Generator(device=self.sampler_device.type)
        generator.manual_seed(random_seed)
        batch_size = batch_users.size(0)
        batch_neg_items = self.sampler.sample(
            batch_size,
            batch_pos_items,
            self.sampler_device,
            generator=generator,
        ).squeeze(-1)
        prepared = self.subgraph_sampler.sample(
            batch_users,
            batch_pos_items,
            batch_neg_items,
            generator=generator,
        )
        if self.sampler_device.type == "cpu" and self.device.type == "cuda":
            return prepared.pin_memory()
        return prepared

    def _run_training_batch(
        self,
        sub_batch: SubgraphBatch,
        popularity: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run forward/loss for one already-prepared subgraph batch."""
        if sub_batch.sub_edge_index.device != self.device:
            sub_batch = sub_batch.to(self.device, non_blocking=True)

        with self._autocast_context():
            with profile_stage("forward", self.profiler):
                output = self.model.forward_subgraph(sub_batch)

            with profile_stage("loss", self.profiler):
                local_popularity = popularity[sub_batch.item_global_ids]
                losses = self.loss_suite(
                    output,
                    local_popularity,
                    sub_batch.batch_pos_local,
                    epoch,
                )

        return losses["total"].detach(), losses

    def _dispatch_batch(
        self,
        sub_batch: SubgraphBatch,
        popularity: torch.Tensor,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        progress_bar: tqdm | None,
        n_skipped: int,
    ) -> torch.Tensor | None:
        """Run one training step; return total_loss or None if the batch was skipped.

        Args:
            sub_batch: Prepared subgraph batch.
            popularity: Full-graph item popularity tensor.
            epoch: Current epoch index.
            batch_idx: Batch index within the current epoch.
            n_batches: Total number of batches per epoch.
            progress_bar: Optional tqdm bar.
            n_skipped: Skipped-batch count before this call (used for display).

        Returns:
            total_loss if the batch was processed, None if skipped due to non-finite loss.
        """
        total_loss, losses = self._run_training_batch(sub_batch, popularity, epoch)
        if not torch.isfinite(total_loss).all():
            self.optimizer.zero_grad(set_to_none=True)
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(skipped=n_skipped + 1)
            return None
        with profile_stage("backward", self.profiler):
            self._apply_optimization_step(losses["total"])
        if progress_bar is not None:
            progress_bar.update(1)
            if (
                batch_idx + 1 == n_batches
                or batch_idx % self.config.progress_bar_loss_cadence == 0
            ):
                progress_bar.set_postfix(loss=f"{float(total_loss.item()):.4f}")
        return total_loss

    def train(
        self,
        start_epoch: int = 0,
        history: dict[str, list] | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int | None = 1,
    ) -> dict[str, list]:
        """Run full training loop with mini-batch subgraph sampling.

        Uses a fixed background-worker prefetch pipeline: while the current
        batch runs forward/backward on GPU, multiple upcoming batches prepare
        their negatives and sampled subgraphs on CPU.

        Returns:
            History dict with per-epoch losses and metrics.
        """
        history = history or {"train_loss": [], "val_metrics": []}
        history.setdefault("train_loss", [])
        history.setdefault("val_metrics", [])

        popularity = self.popularity
        train_users = self.train_users
        train_items = self.train_items
        n_train = train_users.size(0)
        batch_sz = self.config.batch_size
        n_batches = (n_train + batch_sz - 1) // batch_sz

        for epoch in range(start_epoch, self.config.epochs):
            self._ensure_subgraph_sampler()
            should_profile = self._set_epoch_profiling(epoch)

            epoch_start = time.perf_counter()

            perm = torch.randperm(n_train)
            train_users_shuffled = train_users[perm]
            train_items_shuffled = train_items[perm]

            epoch_loss_total = torch.zeros((), device=self.device, dtype=torch.bfloat16)
            completed_batches = 0
            skipped_batches = 0
            self.model.train()
            self.loss_suite.train()
            progress_bar = (
                tqdm(
                    total=n_batches,
                    desc=f"Epoch {epoch + 1}/{self.config.epochs}",
                    unit="batch",
                    leave=False,
                    dynamic_ncols=True,
                )
                if self.config.show_progress_bar
                else None
            )

            # Build batch slices for the epoch
            starts = list(range(0, n_train, batch_sz))
            batches_per_epoch = max(1, len(starts))

            sub_batch: SubgraphBatch | None = None
            total_loss: torch.Tensor | None = None
            try:
                if self.sampler_device.type == "cpu":
                    worker_count = 4
                    prefetch_depth = max(2, worker_count)
                    with ThreadPoolExecutor(max_workers=worker_count) as pool:
                        pending: dict[int, Future[SubgraphBatch]] = {}

                        for batch_index in range(min(prefetch_depth, len(starts))):
                            start = starts[batch_index]
                            end = min(start + batch_sz, n_train)
                            pending[batch_index] = pool.submit(
                                self._prepare_batch,
                                train_users_shuffled[start:end],
                                train_items_shuffled[start:end],
                                int(
                                    self.config.seed
                                    + epoch * batches_per_epoch
                                    + batch_index
                                ),
                            )

                        for batch_idx in range(len(starts)):
                            sub_batch = pending.pop(batch_idx).result()
                            next_batch_idx = batch_idx + prefetch_depth
                            if next_batch_idx < len(starts):
                                start = starts[next_batch_idx]
                                end = min(start + batch_sz, n_train)
                                pending[next_batch_idx] = pool.submit(
                                    self._prepare_batch,
                                    train_users_shuffled[start:end],
                                    train_items_shuffled[start:end],
                                    int(
                                        self.config.seed
                                        + epoch * batches_per_epoch
                                        + next_batch_idx
                                    ),
                                )

                            total_loss = self._dispatch_batch(
                                sub_batch,
                                popularity,
                                epoch,
                                batch_idx,
                                n_batches,
                                progress_bar,
                                skipped_batches,
                            )
                            if total_loss is None:
                                skipped_batches += 1
                                continue
                            epoch_loss_total = epoch_loss_total + total_loss.to(
                                torch.bfloat16
                            )
                            completed_batches += 1
                else:
                    for batch_idx, start in enumerate(starts):
                        end = min(start + batch_sz, n_train)
                        sub_batch = self._prepare_batch(
                            train_users_shuffled[start:end],
                            train_items_shuffled[start:end],
                            int(
                                self.config.seed + epoch * batches_per_epoch + batch_idx
                            ),
                        )
                        total_loss = self._dispatch_batch(
                            sub_batch,
                            popularity,
                            epoch,
                            batch_idx,
                            n_batches,
                            progress_bar,
                            skipped_batches,
                        )
                        if total_loss is None:
                            skipped_batches += 1
                            continue
                        epoch_loss_total = epoch_loss_total + total_loss.to(
                            torch.bfloat16
                        )
                        completed_batches += 1
            finally:
                if progress_bar is not None:
                    progress_bar.close()

            # Release the epoch-local shuffled interactions and the last batch's
            # graph/loss tensors before full-graph validation.
            del perm, train_users_shuffled, train_items_shuffled
            sub_batch = None
            total_loss = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if completed_batches == 0:
                raise RuntimeError(
                    f"All training batches produced non-finite losses in epoch {epoch + 1}."
                )

            avg_loss = float((epoch_loss_total / completed_batches).item())

            history["train_loss"].append(avg_loss)
            released_cuda_sampler = self._release_cuda_sampler_for_eval()
            try:
                with profile_stage("eval", self.profiler):
                    val_metrics = self._evaluate_validation_metrics()
            finally:
                if released_cuda_sampler:
                    self._ensure_subgraph_sampler()
            history["val_metrics"].append(val_metrics)
            epoch_time_s = time.perf_counter() - epoch_start
            primary_metric = self._primary_metric_name()
            current_ndcg = val_metrics.get(primary_metric, 0.0)
            self._step_scheduler(current_ndcg, epoch)
            self._log_epoch_summary(
                epoch,
                avg_loss,
                current_ndcg,
                primary_metric,
                skipped_batches=skipped_batches,
            )
            if should_profile and self.profiler and self.profiler.stages:
                logger.info(self.profiler.summary())
            self._log_epoch_to_sqlite(
                epoch, avg_loss, epoch_time_s, val_metrics, should_profile
            )
            self.completed_epoch = epoch
            self.resume_history = history
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
                epoch, checkpoint_path, checkpoint_every, history
            )

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return history
