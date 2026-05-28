"""GPU-vectorized link-prediction evaluation using PyG metrics."""

from __future__ import annotations

from typing import Final

import torch
from torch_geometric.metrics import (
    LinkPredAveragePopularity,
    LinkPredHitRatio,
    LinkPredMetricCollection,
    LinkPredNDCG,
    LinkPredPersonalization,
    LinkPredRecall,
)

from ..utils.config import UCaGNNConfig
from ..utils.trainer_runtime import (
    autocast_context,
    model_device,
    move_optional_tensor_to_device,
    move_tensor_to_device,
    stage_graph_tensors_for_device,
)

THESIS_PRIMARY_METRICS: Final[tuple[str, ...]] = (
    "NDCG@20",
    "Recall@20",
    "AveragePopularity@20",
    "HitRatio@20",
    "Personalization@20",
    "NDCG@40",
    "Recall@40",
    "AveragePopularity@40",
    "HitRatio@40",
    "Personalization@40",
)

LOWER_IS_BETTER_METRICS: Final[frozenset[str]] = frozenset(
    {"AveragePopularity@20", "AveragePopularity@40"},
)
THESIS_EVAL_KS: Final[tuple[int, ...]] = (20, 40)


class _SafeLinkPredPersonalization(LinkPredPersonalization):
    """Return a finite personalization score for degenerate tiny-user splits.

    PyG's base implementation computes the average dissimilarity across user
    pairs. When fewer than two users are evaluated, the pair count is zero and
    the upstream metric returns ``0 / 0 -> nan``. For tiny smoke-validation
    runs, treating that case as ``0.0`` keeps the metric defined without
    changing the behavior on real multi-user evaluation splits.

    """

    def compute(self) -> torch.Tensor:
        """Compute personalization, returning ``0.0`` when no user pairs exist."""
        if int(self.total) < 2:
            return torch.zeros((), device=self.total.device, dtype=torch.get_default_dtype())
        return super().compute()


class Evaluator:
    """Batched GPU evaluation for the PyG link-prediction metric suite.

    Runtime evaluation is intentionally restricted to the thesis-primary
    metric set so reporting, validation, and saved results stay aligned.

    Ground-truth and exclusion dicts are built on first access per split mask
    and cached for subsequent epochs, avoiding repeated O(N) Python loops and
    D2H transfers.
    """

    def __init__(self, config: UCaGNNConfig) -> None:
        self.config = config
        # The evaluator reads its score view from the config so a caller can
        # reuse the same trained checkpoint with an alternate eval mode.
        self.eval_scoring_mode = config.eval_scoring_mode
        # Cache keyed by mask tensor identity (id()) to avoid rebuilding per epoch.
        self._split_cache: dict[int, dict] = {}

    def _build_metrics(
        self,
        n_items: int,
        popularity: torch.Tensor,
    ) -> LinkPredMetricCollection:
        """Build the thesis-primary PyG metric bundle.

        ``LinkPredMetricCollection`` still needs one metric instance per metric
        family and cutoff, but runtime updates happen through a single shared
        collection call.
        """
        metrics: dict[str, object] = {}
        for k in THESIS_EVAL_KS:
            metrics[f"NDCG@{k}"] = LinkPredNDCG(k=k)
            metrics[f"Recall@{k}"] = LinkPredRecall(k=k)
            metrics[f"AveragePopularity@{k}"] = LinkPredAveragePopularity(
                k=k,
                popularity=popularity,
            )
            metrics[f"HitRatio@{k}"] = LinkPredHitRatio(k=k)
            metrics[f"Personalization@{k}"] = _SafeLinkPredPersonalization(k=k)
        return LinkPredMetricCollection(metrics)

    @staticmethod
    def _observed_non_target_mask(data, target_mask: torch.Tensor) -> torch.Tensor:
        """Return interactions to exclude from the recommendation scoring pool.

        Applies split-aware exclusion so that test-set labels never influence
        model selection through the validation metric used for early stopping:

        - Val evaluation: only training interactions are excluded.  Test items
          remain in the scoring pool so no test-set knowledge reaches early
          stopping or best-model selection.
        - Test evaluation: training + validation interactions are excluded,
          matching the standard temporal evaluation protocol.
        """
        exclude_mask = torch.zeros_like(target_mask, dtype=torch.bool)

        train_mask = getattr(data, "train_mask", None)
        if train_mask is not None:
            exclude_mask |= move_tensor_to_device(
                train_mask.bool(),
                exclude_mask.device,
            )

        # When the caller passes exactly data.test_mask (identity check), also
        # exclude validation interactions so the test pool contains only items
        # the model has never encountered during training or validation.
        test_mask = getattr(data, "test_mask", None)
        if test_mask is not None and target_mask is test_mask:
            val_mask = getattr(data, "val_mask", None)
            if val_mask is not None:
                exclude_mask |= move_tensor_to_device(
                    val_mask.bool(),
                    exclude_mask.device,
                )

        return exclude_mask

    @staticmethod
    def _group_items_by_user(
        user_nodes: torch.Tensor,
        item_nodes: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Group CPU item IDs by CPU user ID with one pass over unique users.

        Args:
            user_nodes: CPU user IDs for one interaction slice.
            item_nodes: CPU item IDs aligned to ``user_nodes``.

        Returns:
            Mapping from user ID to a CPU ``torch.long`` tensor of item IDs.

        """
        if user_nodes.numel() == 0:
            return {}

        sort_order = user_nodes.argsort(stable=True)
        sorted_users = user_nodes[sort_order]
        sorted_items = item_nodes[sort_order].long()
        unique_users, counts = torch.unique_consecutive(sorted_users, return_counts=True)
        item_groups = sorted_items.split(counts.tolist())
        return {
            int(user_id.item()): items
            for user_id, items in zip(unique_users, item_groups, strict=True)
        }

    def _get_or_build_split_cache(
        self,
        data,
        mask: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
        """Return (user_gt, user_seen_items, unique_users) for the split mask.

        Built once on first call and stored; subsequent epoch calls for the
        same split (same mask object) return instantly with no Python loops or
        D2H transfers.
        """
        key = id(mask)
        if key not in self._split_cache:
            target_mask = mask.bool()
            if target_mask.device.type != "cpu":
                target_mask = target_mask.cpu()
            exclude_mask = self._observed_non_target_mask(data, mask)
            if exclude_mask.device.type != "cpu":
                exclude_mask = exclude_mask.cpu()

            user_nodes_cpu = data.user_nodes[target_mask]
            item_nodes_cpu = data.item_nodes[target_mask] - data.n_users
            unique_users = user_nodes_cpu.unique()
            user_gt = self._group_items_by_user(user_nodes_cpu, item_nodes_cpu)

            exclude_user_nodes = data.user_nodes[exclude_mask]
            exclude_item_nodes = data.item_nodes[exclude_mask] - data.n_users
            user_seen_items = self._group_items_by_user(
                exclude_user_nodes,
                exclude_item_nodes,
            )

            self._split_cache[key] = {
                "user_gt": user_gt,
                "user_seen_items": user_seen_items,
                "unique_users": unique_users,
            }
        entry = self._split_cache[key]
        return entry["user_gt"], entry["user_seen_items"], entry["unique_users"]

    @torch.no_grad()
    def evaluate(
        self,
        model,
        data,
        mask: torch.Tensor,
        batch_size: int = 512,
    ) -> dict[str, float]:
        """Evaluate model on users present in mask."""
        model.eval()
        device = model_device(model)

        user_gt, user_seen_items, unique_users = self._get_or_build_split_cache(
            data,
            mask,
        )
        n_items = data.n_items
        if unique_users.numel() == 0:
            return {}
        unique_users_cpu = unique_users
        unique_users = move_tensor_to_device(unique_users_cpu, device)
        use_eval_amp = device.type == "cuda" and self.config.use_amp

        edge_index, edge_sign, edge_norm = stage_graph_tensors_for_device(
            data,
            device,
        )
        popularity = move_optional_tensor_to_device(
            data.popularity,
            device,
            dtype=torch.bfloat16,
        )
        assert popularity is not None
        metrics = self._build_metrics(n_items=n_items, popularity=popularity)
        metrics = metrics.to(device)

        # Cap score matrix at ~512 MB regardless of catalogue size.
        # score_matrix bytes = users * n_items * 4; solve for users:
        score_matrix_budget_mb = 512
        safe_batch = max(1, int(score_matrix_budget_mb * 1024**2 / (n_items * 4)))
        effective_batch = min(batch_size, safe_batch)

        # Propagate once over the full graph; reuse across all user batches.
        with autocast_context(use_amp=use_eval_amp):
            propagated = model.get_propagated_for_eval(
                edge_index,
                edge_sign,
                edge_norm,
                embedding_dtype=torch.bfloat16 if use_eval_amp else None,
            )

        # CAGRA candidate pre-filtering (opt-in, GPU-native ANN via cuVS).
        # When cagra_candidate_k > 0 we restrict per-user scoring to the top-K
        # nearest-neighbor items, drastically reducing eval VRAM.
        cagra_candidate_k = self.config.cagra_candidate_k
        cagra_index = None
        if cagra_candidate_k > 0 and cagra_candidate_k < n_items and device.type == "cuda":
            try:
                import cupy as cp
                from cuvs.neighbors import cagra as cuvs_cagra

                item_key = "item_interest" if "item_interest" in propagated else "item"
                item_embs = propagated[item_key].float().contiguous()
                item_cp = cp.asarray(item_embs.detach())
                index_params = cuvs_cagra.IndexParams(
                    metric="inner_product",
                    graph_degree=self.config.cagra_out_degree,
                    intermediate_graph_degree=self.config.cagra_initial_degree,
                )
                cagra_index = cuvs_cagra.build(index_params, item_cp)
            except Exception as exc:
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "CAGRA candidate pre-filtering disabled: %s", exc
                )
                cagra_index = None

        max_k = max(THESIS_EVAL_KS)
        for start in range(0, unique_users.size(0), effective_batch):
            batch_users = unique_users[start : start + effective_batch]
            batch_user_ids = unique_users_cpu[start : start + effective_batch].tolist()
            with autocast_context(use_amp=use_eval_amp):
                scores = model.score_users_from_propagated(
                    propagated,
                    batch_users,
                    scoring_mode=self.eval_scoring_mode,
                )
            scores = scores.float()

            # Mask out non-candidate items from CAGRA index (if built).
            if cagra_index is not None:
                try:
                    import cupy as cp
                    from cuvs.neighbors import cagra as cuvs_cagra

                    user_key = "user_interest" if "user_interest" in propagated else "user"
                    user_embs = propagated[user_key][batch_users].float().contiguous()
                    user_cp = cp.asarray(user_embs.detach())
                    search_params = cuvs_cagra.SearchParams(
                        itopk_size=max(cagra_candidate_k, self.config.cagra_itopk_size),
                    )
                    _, neighbors_cp = cuvs_cagra.search(
                        search_params, cagra_index, user_cp, cagra_candidate_k
                    )
                    # neighbors_cp: (B, cagra_candidate_k) cupy int64
                    neighbors_t = torch.as_tensor(neighbors_cp, device=device)
                    candidate_mask = torch.zeros(
                        scores.size(0), n_items, dtype=torch.bool, device=device
                    )
                    candidate_mask.scatter_(1, neighbors_t, True)
                    scores[~candidate_mask] = float("-inf")
                except Exception as exc:
                    import logging as _logging

                    _logging.getLogger(__name__).warning(
                        "CAGRA search failed, using full-catalog scores: %s", exc
                    )

            seen_row_parts: list[torch.Tensor] = []
            seen_col_parts: list[torch.Tensor] = []
            for row_index, user_id in enumerate(batch_user_ids):
                seen_items = user_seen_items.get(user_id)
                if seen_items is None or seen_items.numel() == 0:
                    continue
                seen_row_parts.append(
                    torch.full((seen_items.numel(),), row_index, dtype=torch.long),
                )
                seen_col_parts.append(seen_items)
            if seen_row_parts:
                seen_rows = move_tensor_to_device(torch.cat(seen_row_parts), device)
                seen_cols = move_tensor_to_device(torch.cat(seen_col_parts), device)
                scores[seen_rows, seen_cols] = float("-inf")

            # Build batch ground truth on-the-fly (small: batch_size * n_items)
            gt_rows: list[torch.Tensor] = []
            keep_rows: list[int] = []
            for row_index, uid in enumerate(batch_user_ids):
                items = user_gt.get(uid)
                if items is None or items.numel() == 0:
                    continue
                keep_rows.append(row_index)
                gt_rows.append(items)
            if not gt_rows:
                continue

            keep_indices = torch.tensor(keep_rows, device=device, dtype=torch.long)
            scores = scores.index_select(0, keep_indices)
            if scores.size(0) == 0:
                continue

            _, pred_index_mat = torch.topk(scores, max_k, dim=-1)

            # Build edge_label_index from sparse ground truth
            gt_counts = torch.tensor([items.numel() for items in gt_rows], device=device)
            gt_user_index = torch.repeat_interleave(
                torch.arange(len(gt_rows), device=device),
                gt_counts,
            )
            gt_item_index = move_tensor_to_device(torch.cat(gt_rows), device)
            edge_label_index = (
                gt_user_index,
                gt_item_index,
            )
            metrics.update(pred_index_mat, edge_label_index)

        return {name: value.item() for name, value in metrics.compute().items()}
