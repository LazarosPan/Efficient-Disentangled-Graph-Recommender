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

    def _get_or_build_split_cache(
        self,
        data,
        mask: torch.Tensor,
    ) -> tuple[dict[int, list[int]], dict[int, list[int]], torch.Tensor]:
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

            user_gt: dict[int, list[int]] = {}
            for u, i in zip(user_nodes_cpu.tolist(), item_nodes_cpu.tolist(), strict=False):
                user_gt.setdefault(u, []).append(i)

            exclude_user_nodes = data.user_nodes[exclude_mask]
            exclude_item_nodes = data.item_nodes[exclude_mask] - data.n_users
            user_seen_items: dict[int, list[int]] = {}
            for u, i in zip(exclude_user_nodes.tolist(), exclude_item_nodes.tolist(), strict=False):
                user_seen_items.setdefault(u, []).append(i)

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

            seen_rows: list[int] = []
            seen_cols: list[int] = []
            for row_index, user_id in enumerate(batch_user_ids):
                seen_items = user_seen_items.get(user_id)
                if seen_items:
                    seen_rows.extend([row_index] * len(seen_items))
                    seen_cols.extend(seen_items)
            if seen_rows:
                scores[seen_rows, seen_cols] = float("-inf")

            # Build batch ground truth on-the-fly (small: batch_size * n_items)
            gt_rows = []
            keep_mask = []
            for uid in batch_user_ids:
                items = user_gt.get(uid)
                if items:
                    gt_rows.append(items)
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            keep = torch.tensor(keep_mask, device=device)
            if not keep.any():
                continue

            scores = scores[keep]
            if scores.size(0) == 0:
                continue

            _, pred_index_mat = torch.topk(scores, max_k, dim=-1)

            # Build edge_label_index from sparse ground truth
            gt_user_list: list[int] = []
            gt_item_list: list[int] = []
            for local_idx, items in enumerate(gt_rows):
                for item_id in items:
                    gt_user_list.append(local_idx)
                    gt_item_list.append(item_id)
            edge_label_index = (
                torch.tensor(gt_user_list, device=device),
                torch.tensor(gt_item_list, device=device),
            )
            metrics.update(pred_index_mat, edge_label_index)

        return {name: value.item() for name, value in metrics.compute().items()}
