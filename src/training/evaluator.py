"""GPU-vectorized link-prediction evaluation using PyG metrics."""

from __future__ import annotations

import torch
from typing import Final
from torch_geometric.metrics import (
    LinkPredAveragePopularity,
    LinkPredMetricCollection,
    LinkPredNDCG,
    LinkPredRecall,
)

from ..utils.config import UCaGNNConfig

THESIS_PRIMARY_METRICS: Final[tuple[str, ...]] = (
    "NDCG@20",
    "Recall@20",
    "AveragePopularity@20",
    "NDCG@50",
    "Recall@50",
    "AveragePopularity@50",
)

LOWER_IS_BETTER_METRICS: Final[frozenset[str]] = frozenset(
    {"AveragePopularity@20", "AveragePopularity@50"}
)
THESIS_EVAL_KS: Final[tuple[int, ...]] = (20, 50)


class Evaluator:
    """Batched GPU evaluation for the PyG link-prediction metric suite.

    Runtime evaluation is intentionally restricted to the thesis-primary
    metric set so reporting, validation, and saved results stay aligned.
    """

    def __init__(self, config: UCaGNNConfig) -> None:
        self.config = config
        self.eval_scoring_mode = config.eval_scoring_mode

    def _build_metrics(
        self, n_items: int, popularity: torch.Tensor
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
        return LinkPredMetricCollection(metrics)

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
        device = next(model.parameters()).device

        user_nodes = data.user_nodes[mask].to(device)
        item_nodes = (data.item_nodes[mask] - data.n_users).to(device)
        unique_users = user_nodes.unique()
        n_items = data.n_items
        if unique_users.numel() == 0:
            return {}

        gt_matrix = torch.zeros(unique_users.max().item() + 1, n_items, device=device)
        gt_matrix[user_nodes, item_nodes] = 1.0

        edge_index = data.edge_index.to(device)
        edge_sign = data.edge_sign.to(device) if hasattr(data, "edge_sign") else None
        popularity = data.popularity.to(device).float()
        metrics = self._build_metrics(n_items=n_items, popularity=popularity)
        metrics = metrics.to(device)

        max_k = max(THESIS_EVAL_KS)
        for start in range(0, unique_users.size(0), batch_size):
            batch_users = unique_users[start : start + batch_size]
            scores = model.get_all_scores(
                edge_index,
                batch_users,
                edge_sign,
                scoring_mode=self.eval_scoring_mode,
            )
            gt_batch = gt_matrix[batch_users]

            has_gt = gt_batch.sum(dim=-1) > 0
            if not has_gt.any():
                continue

            scores = scores[has_gt]
            gt_batch = gt_batch[has_gt]
            if scores.size(0) == 0:
                continue

            _, pred_index_mat = torch.topk(scores, max_k, dim=-1)
            batch_gt_users, batch_gt_items = gt_batch.nonzero(as_tuple=True)
            edge_label_index = (batch_gt_users, batch_gt_items)
            metrics.update(pred_index_mat, edge_label_index)

        return {name: value.item() for name, value in metrics.compute().items()}
