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

import logging
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import cast

import torch
from tqdm.auto import tqdm

from ..data.subgraph_sampler import SubgraphBatch, SubgraphSampler
from ..losses.loss_suite import LossSuite
from ..models.ucagnn import UCaGNN
from ..profiling.gpu_profiler import GPUProfiler, TrainingResourceMonitor
from ..utils.config import UCaGNNConfig
from ..utils.reproducibility import build_torch_generator
from ..utils.trainer_runtime import (
    TrainerRuntime,
    autocast_context,
    empty_cuda_cache,
    is_cuda_oom_error,
    move_tensor_to_device,
    stage_graph_tensors_for_device,
)

logger = logging.getLogger(__name__)

_EPOCH_LOSS_METRIC_NAMES = frozenset(
    {
        "raw_rec_loss",
        "raw_interest_bpr",
        "raw_conformity_bpr",
        "raw_independence",
        "raw_contrastive",
        "raw_popularity",
        "raw_align",
        "raw_uniform",
        "raw_propensity_calibration",
        "dice_high_mask_rate",
        "dice_low_mask_rate",
        "score_mix_interest_mean",
        "score_mix_conformity_mean",
        "score_mix_context_mean",
    },
)
_EPOCH_LOSS_METRIC_PREFIXES = ("normalized_", "weighted_")


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
        mlflow_module=None,
    ) -> None:
        super().__init__(
            model=model,
            loss_suite=loss_suite,
            data=data,
            config=config,
            profiler=profiler,
            experiment_logger=experiment_logger,
            exp_id=exp_id,
            mlflow_module=mlflow_module,
        )

        self._force_cpu_sampler = False
        self._full_graph_tensors: (
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None
        ) = None
        self._full_graph_tensor_device: torch.device | None = None
        if self.config.training_graph_mode == "sampled":
            self.subgraph_sampler, self.sampler_device = self._build_subgraph_sampler(data)
        else:
            self.subgraph_sampler = None
            self.sampler_device = self.device
        self.train_users, self.train_items = self._get_train_interactions()

    @staticmethod
    def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
        """Return min/mean/median/max summary statistics for one tensor."""
        values = tensor.detach().float().reshape(-1)
        return {
            "min": float(values.min().item()),
            "mean": float(values.mean().item()),
            "median": float(values.median().item()),
            "max": float(values.max().item()),
        }

    @staticmethod
    def _should_log_epoch_loss_metric(metric_name: str) -> bool:
        """Return whether a loss scalar should be aggregated per epoch."""
        return metric_name in _EPOCH_LOSS_METRIC_NAMES or metric_name.startswith(
            _EPOCH_LOSS_METRIC_PREFIXES,
        )

    @classmethod
    def _accumulate_epoch_loss_metrics(
        cls,
        accumulator: dict[str, torch.Tensor],
        losses: Mapping[str, torch.Tensor],
    ) -> None:
        """Accumulate scalar loss diagnostics without per-batch host sync."""
        for metric_name, value in losses.items():
            if not cls._should_log_epoch_loss_metric(metric_name):
                continue
            if not torch.is_tensor(value) or value.numel() != 1:
                continue
            scalar = value.detach().float()
            scalar = torch.where(torch.isfinite(scalar), scalar, torch.zeros_like(scalar))
            accumulator[metric_name] = (
                accumulator.get(metric_name, torch.zeros_like(scalar)) + scalar
            )

    @staticmethod
    def _finalize_epoch_loss_metrics(
        accumulator: Mapping[str, torch.Tensor],
        completed_batches: int,
    ) -> dict[str, float]:
        """Return host-side mean loss diagnostics after one epoch."""
        if completed_batches <= 0:
            return {}
        return {
            metric_name: float((value / completed_batches).item())
            for metric_name, value in sorted(accumulator.items())
        }

    def _build_subgraph_sampler(self, data) -> tuple[SubgraphSampler, torch.device]:
        """Stage the full graph on CUDA when possible, else keep the CPU path."""
        cpu_device = torch.device("cpu")
        target_device = (
            cpu_device if self._force_cpu_sampler or self.device.type != "cuda" else self.device
        )

        def _sampler_for(device: torch.device) -> SubgraphSampler:
            edge_index, edge_sign, edge_norm = stage_graph_tensors_for_device(
                data,
                device,
            )
            return SubgraphSampler(
                edge_index=edge_index,
                edge_sign=edge_sign,
                edge_norm=edge_norm,
                n_users=data.n_users,
                n_items=data.n_items,
                num_hops=self.config.max_gnn_layers,
                max_neighbors_per_hop=self.config.num_neighbors,
            )

        if target_device.type != "cuda":
            return _sampler_for(cpu_device), cpu_device

        try:
            sampler = _sampler_for(target_device)
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            empty_cuda_cache(target_device)
            logger.warning(
                "Falling back to CPU subgraph sampling after CUDA graph staging failed: %s",
                exc,
            )
            return _sampler_for(cpu_device), cpu_device

        logger.info("Mini-batch sampler staged on %s.", target_device)
        return sampler, target_device

    def _fallback_subgraph_sampler_to_cpu(self, exc: BaseException) -> None:
        """Rebuild the sampler on CPU after a CUDA batch-preparation OOM."""
        if self.sampler_device.type != "cuda":
            raise exc
        empty_cuda_cache(self.sampler_device)
        logger.warning(
            "Falling back to CPU subgraph sampling after CUDA batch preparation failed: %s",
            exc,
        )
        self._force_cpu_sampler = True
        self.subgraph_sampler, self.sampler_device = self._build_subgraph_sampler(
            self.data,
        )

    def _ensure_subgraph_sampler(self) -> None:
        """Rebuild the sampler after validation if its CUDA copy was released."""
        if self.config.training_graph_mode != "sampled":
            return
        if self.subgraph_sampler is None:
            self.subgraph_sampler, self.sampler_device = self._build_subgraph_sampler(
                self.data,
            )

    def _release_cuda_sampler_for_eval(self) -> bool:
        """Drop the CUDA sampler before full-graph validation to free VRAM."""
        if self.sampler_device.type != "cuda" or self.subgraph_sampler is None:
            return False
        self.subgraph_sampler = None
        self.sampler_device = torch.device("cpu")
        empty_cuda_cache(self.device)
        return True

    def _get_full_graph_training_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return cached full-graph tensors for full-graph optimizer steps."""
        if self._full_graph_tensors is None or self._full_graph_tensor_device != self.device:
            self._full_graph_tensors = stage_graph_tensors_for_device(self.data, self.device)
            self._full_graph_tensor_device = self.device
        return self._full_graph_tensors

    def _release_full_graph_cache_for_eval(self) -> bool:
        """Release cached CUDA graph tensors before validation/test scoring."""
        if self._full_graph_tensor_device is None or self._full_graph_tensor_device.type != "cuda":
            return False
        self._full_graph_tensors = None
        self._full_graph_tensor_device = None
        empty_cuda_cache(self.device)
        return True

    def _prepare_batch_on_sampler_device(
        self,
        batch_users: torch.Tensor,
        batch_pos_items: torch.Tensor,
        random_seed: int,
        epoch: int,
    ) -> SubgraphBatch:
        """Sample negatives and extract a subgraph for one batch.

        CPU-resident samplers use a background-thread prefetch path and pinned
        host memory. CUDA-resident samplers prepare the full batch in device
        memory directly so subgraph extraction no longer waits on CPU copies.
        """
        self._ensure_subgraph_sampler()
        assert self.subgraph_sampler is not None
        if batch_users.device != self.sampler_device:
            batch_users = move_tensor_to_device(batch_users, self.sampler_device)
            batch_pos_items = move_tensor_to_device(
                batch_pos_items,
                self.sampler_device,
            )

        generator = build_torch_generator(random_seed, self.sampler_device)
        batch_size = batch_users.size(0)
        batch_neg_items, dice_negative_mask = self.sampler.sample_with_metadata(
            batch_size,
            batch_pos_items,
            user_ids=batch_users,
            device=self.sampler_device,
            generator=generator,
            epoch=epoch,
        )
        batch_users, batch_pos_items, batch_neg_items, dice_negative_mask = (
            self._expand_sampled_negatives(
                batch_users,
                batch_pos_items,
                batch_neg_items,
                dice_negative_mask,
                batch_size,
            )
        )
        prepared = self.subgraph_sampler.sample(
            batch_users,
            batch_pos_items,
            batch_neg_items,
            generator=generator,
            dice_negative_mask=dice_negative_mask,
        )
        if self.sampler_device.type == "cpu" and self.device.type == "cuda":
            return prepared.pin_memory()
        return prepared

    def _prepare_batch(
        self,
        batch_users: torch.Tensor,
        batch_pos_items: torch.Tensor,
        random_seed: int,
        epoch: int,
    ) -> SubgraphBatch:
        """Prepare one batch, retrying on the CPU sampler after a CUDA OOM."""
        try:
            return self._prepare_batch_on_sampler_device(
                batch_users,
                batch_pos_items,
                random_seed,
                epoch,
            )
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            self._fallback_subgraph_sampler_to_cpu(exc)
            return self._prepare_batch_on_sampler_device(
                batch_users,
                batch_pos_items,
                random_seed,
                epoch,
            )

    def _run_training_batch(
        self,
        sub_batch: SubgraphBatch,
        popularity: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run forward/loss for one already-prepared subgraph batch."""
        if sub_batch.sub_edge_index.device != self.device:
            sub_batch = sub_batch.to(self.device, non_blocking=True)

        local_prop_targets = (
            self.propensity_targets[sub_batch.item_global_ids]
            if self.propensity_targets is not None
            else None
        )
        local_branch_popularity = self.branch_popularity[sub_batch.item_global_ids]
        with autocast_context(use_amp=self.use_amp, amp_dtype=self.amp_dtype):
            output = self.model.forward_subgraph(sub_batch)
            local_popularity = popularity[sub_batch.item_global_ids]
            losses = self.loss_suite(
                output,
                local_popularity,
                sub_batch.batch_pos_local,
                epoch,
                propensity_targets=local_prop_targets,
                branch_item_popularity=local_branch_popularity,
            )

        if logger.isEnabledFor(logging.DEBUG):
            debug_losses = {
                key: float(value.detach().float().cpu())
                for key, value in losses.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            logger.debug("Batch loss components: %s", debug_losses)

            ipw_weights = output.get("ipw_weights")
            if isinstance(ipw_weights, torch.Tensor):
                logger.debug("IPW stats: %s", self._tensor_stats(ipw_weights))

            if local_prop_targets is not None:
                prop_targets = local_prop_targets[sub_batch.batch_pos_local]
                logger.debug("Propensity target stats: %s", self._tensor_stats(prop_targets))

            pos_scores = cast(dict[str, torch.Tensor], output["pos_scores"])
            neg_scores = cast(dict[str, torch.Tensor], output["neg_scores"])
            for score_name in (
                "final_score",
                "interest_score",
                "conformity_score",
                "context_score",
            ):
                pos_score = pos_scores.get(score_name)
                neg_score = neg_scores.get(score_name)
                if isinstance(pos_score, torch.Tensor) and isinstance(neg_score, torch.Tensor):
                    logger.debug(
                        "%s stats: pos=%s neg=%s",
                        score_name,
                        self._tensor_stats(pos_score),
                        self._tensor_stats(neg_score),
                    )

        return losses["total"].detach(), losses

    def _expand_sampled_negatives(
        self,
        batch_users: torch.Tensor,
        batch_pos_items: torch.Tensor,
        batch_neg_items: torch.Tensor,
        dice_negative_mask: torch.Tensor | None,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Flatten sampled negatives and align users, positives, and DICE masks."""
        if self.config.n_negatives > 1:
            batch_neg_matrix = batch_neg_items.reshape(batch_size, self.config.n_negatives)
            batch_users = batch_users.repeat_interleave(self.config.n_negatives)
            batch_pos_items = batch_pos_items.repeat_interleave(self.config.n_negatives)
            batch_neg_items = batch_neg_matrix.reshape(-1)
            if dice_negative_mask is not None:
                dice_negative_mask = dice_negative_mask.reshape(-1)
            return batch_users, batch_pos_items, batch_neg_items, dice_negative_mask

        batch_neg_items = batch_neg_items.squeeze(-1)
        if dice_negative_mask is not None:
            dice_negative_mask = dice_negative_mask.squeeze(-1)
        return batch_users, batch_pos_items, batch_neg_items, dice_negative_mask

    def _run_full_graph_training_batch(
        self,
        batch_users: torch.Tensor,
        batch_pos_items: torch.Tensor,
        popularity: torch.Tensor,
        epoch: int,
        random_seed: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one paper-baseline batch using full-graph propagation."""
        batch_users = move_tensor_to_device(batch_users, self.device)
        batch_pos_items = move_tensor_to_device(batch_pos_items, self.device)
        generator = build_torch_generator(random_seed, self.device)
        batch_size = batch_users.size(0)
        batch_neg_items, dice_negative_mask = self.sampler.sample_with_metadata(
            batch_size,
            batch_pos_items,
            user_ids=batch_users,
            device=self.device,
            generator=generator,
            epoch=epoch,
        )
        batch_users, batch_pos_items, batch_neg_items, dice_negative_mask = (
            self._expand_sampled_negatives(
                batch_users,
                batch_pos_items,
                batch_neg_items,
                dice_negative_mask,
                batch_size,
            )
        )

        edge_index, edge_sign, edge_norm = self._get_full_graph_training_tensors()
        with autocast_context(use_amp=self.use_amp, amp_dtype=self.amp_dtype):
            forward_kwargs = (
                {"dice_negative_mask": dice_negative_mask} if dice_negative_mask is not None else {}
            )
            output = self.model(
                edge_index,
                batch_users,
                batch_pos_items,
                batch_neg_items,
                edge_sign=edge_sign,
                edge_norm=edge_norm,
                **forward_kwargs,
            )
            losses = self.loss_suite(
                output,
                popularity,
                batch_pos_items,
                epoch,
                propensity_targets=self.propensity_targets,
                branch_item_popularity=self.branch_popularity,
            )
        return losses["total"].detach(), losses

    def _dispatch_full_graph_batch(
        self,
        batch_users: torch.Tensor,
        batch_pos_items: torch.Tensor,
        popularity: torch.Tensor,
        epoch: int,
        random_seed: int,
        batch_idx: int,
        n_batches: int,
        progress_bar: tqdm | None,
        n_skipped: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        """Run one full-graph training step."""
        total_loss, losses = self._run_full_graph_training_batch(
            batch_users,
            batch_pos_items,
            popularity,
            epoch,
            random_seed,
        )
        if not torch.isfinite(total_loss).all():
            self.optimizer.zero_grad(set_to_none=True)
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(skipped=n_skipped + 1)
            return None
        self._apply_optimization_step(losses["total"])
        if progress_bar is not None:
            progress_bar.update(1)
            if batch_idx + 1 == n_batches or batch_idx % self.config.progress_bar_loss_cadence == 0:
                progress_bar.set_postfix(loss=f"{float(total_loss.item()):.4f}")
        return total_loss, losses

    def _dispatch_batch(
        self,
        sub_batch: SubgraphBatch,
        popularity: torch.Tensor,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        progress_bar: tqdm | None,
        n_skipped: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
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
            ``(total_loss, losses)`` if processed, else None for non-finite loss.

        """
        total_loss, losses = self._run_training_batch(sub_batch, popularity, epoch)
        if not torch.isfinite(total_loss).all():
            self.optimizer.zero_grad(set_to_none=True)
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(skipped=n_skipped + 1)
            return None
        self._apply_optimization_step(losses["total"])
        if progress_bar is not None:
            progress_bar.update(1)
            if batch_idx + 1 == n_batches or batch_idx % self.config.progress_bar_loss_cadence == 0:
                progress_bar.set_postfix(loss=f"{float(total_loss.item()):.4f}")
        return total_loss, losses

    def _iter_epoch_batches(
        self,
        starts: list[int],
        train_users_shuffled: torch.Tensor,
        train_items_shuffled: torch.Tensor,
        n_train: int,
        batch_sz: int,
        epoch: int,
        batches_per_epoch: int,
    ):
        """Yield (batch_idx, SubgraphBatch) for each batch in an epoch.

        Uses a prefetch thread pool on CPU and direct synchronous preparation on CUDA.

        Args:
            starts: List of start indices for each batch slice.
            train_users_shuffled: Shuffled user IDs for this epoch.
            train_items_shuffled: Shuffled item IDs for this epoch.
            n_train: Total training interactions.
            batch_sz: Batch size.
            epoch: Current epoch index.
            batches_per_epoch: Total batches per epoch (for seed derivation).

        Yields:
            (batch_idx, SubgraphBatch) tuples.

        """
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
                        int(self.config.seed + epoch * batches_per_epoch + batch_index),
                        epoch,
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
                                self.config.seed + epoch * batches_per_epoch + next_batch_idx,
                            ),
                            epoch,
                        )
                    yield batch_idx, sub_batch
        else:
            for batch_idx, start in enumerate(starts):
                end = min(start + batch_sz, n_train)
                sub_batch = self._prepare_batch(
                    train_users_shuffled[start:end],
                    train_items_shuffled[start:end],
                    int(self.config.seed + epoch * batches_per_epoch + batch_idx),
                    epoch,
                )
                yield batch_idx, sub_batch

    def train(
        self,
        start_epoch: int = 0,
        history: dict[str, list] | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int | None = 1,
        epoch_callback: Callable[[int, Mapping[str, float], float], None] | None = None,
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
            if self.config.training_graph_mode == "sampled":
                self._ensure_subgraph_sampler()
            self._reset_epoch_vram_stats()

            epoch_start = time.perf_counter()
            perm = torch.randperm(
                n_train,
                generator=build_torch_generator(
                    self.config.seed + epoch,
                    train_users.device,
                ),
                device=train_users.device,
            )
            train_users_shuffled = train_users[perm]
            train_items_shuffled = train_items[perm]

            epoch_loss_total = torch.zeros((), device=self.device, dtype=torch.bfloat16)
            epoch_loss_metric_sums: dict[str, torch.Tensor] = {}
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
            resource_monitor = TrainingResourceMonitor(self.device).start()
            resource_stats = None
            try:
                if self.config.training_graph_mode == "full":
                    for batch_idx, start in enumerate(starts):
                        end = min(start + batch_sz, n_train)
                        batch_result = self._dispatch_full_graph_batch(
                            train_users_shuffled[start:end],
                            train_items_shuffled[start:end],
                            popularity,
                            epoch,
                            int(self.config.seed + epoch * batches_per_epoch + batch_idx),
                            batch_idx,
                            n_batches,
                            progress_bar,
                            skipped_batches,
                        )
                        if batch_result is None:
                            skipped_batches += 1
                            continue
                        total_loss, losses = batch_result
                        epoch_loss_total = epoch_loss_total + total_loss.to(torch.bfloat16)
                        self._accumulate_epoch_loss_metrics(epoch_loss_metric_sums, losses)
                        completed_batches += 1
                else:
                    for batch_idx, sub_batch in self._iter_epoch_batches(
                        starts,
                        train_users_shuffled,
                        train_items_shuffled,
                        n_train,
                        batch_sz,
                        epoch,
                        batches_per_epoch,
                    ):
                        batch_result = self._dispatch_batch(
                            sub_batch,
                            popularity,
                            epoch,
                            batch_idx,
                            n_batches,
                            progress_bar,
                            skipped_batches,
                        )
                        if batch_result is None:
                            skipped_batches += 1
                            continue
                        total_loss, losses = batch_result
                        epoch_loss_total = epoch_loss_total + total_loss.to(torch.bfloat16)
                        self._accumulate_epoch_loss_metrics(epoch_loss_metric_sums, losses)
                        completed_batches += 1
            finally:
                resource_stats = resource_monitor.stop()
                if progress_bar is not None:
                    progress_bar.close()

            # Release the epoch-local shuffled interactions and the last batch's
            # graph/loss tensors before full-graph validation.
            del perm, train_users_shuffled, train_items_shuffled
            sub_batch = None
            total_loss = None
            losses = None
            empty_cuda_cache(self.device)

            if completed_batches == 0:
                raise RuntimeError(
                    f"All training batches produced non-finite losses in epoch {epoch + 1}.",
                )

            avg_loss = float((epoch_loss_total / completed_batches).item())
            train_metrics = self._finalize_epoch_loss_metrics(
                epoch_loss_metric_sums,
                completed_batches,
            )

            history["train_loss"].append(avg_loss)
            history.setdefault("train_metrics", []).append(train_metrics)
            released_cuda_sampler = False
            if self.config.training_graph_mode == "sampled":
                released_cuda_sampler = self._release_cuda_sampler_for_eval()
            elif self.config.training_graph_mode == "full":
                self._release_full_graph_cache_for_eval()
            try:
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
                resource_stats=resource_stats,
            )
            self._log_epoch_to_sqlite(
                epoch,
                avg_loss,
                epoch_time_s,
                val_metrics,
                resource_stats=resource_stats,
                train_metrics=train_metrics,
            )
            self.completed_epoch = epoch
            self.resume_history = history
            if epoch_callback is not None:
                epoch_callback(epoch, val_metrics, epoch_time_s)
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

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return history
