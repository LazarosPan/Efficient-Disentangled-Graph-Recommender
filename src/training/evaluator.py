"""GPU-vectorized Recall@K and NDCG@K evaluation using PyG metrics."""

from __future__ import annotations

import torch
import numpy as np
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG

from ..utils.config import UCaGNNConfig


class Evaluator:
    """Batched GPU evaluation for Recall@K and NDCG@K.

    Uses PyG's ``LinkPredRecall`` and ``LinkPredNDCG`` for metric computation.
    """

    def __init__(self, config: UCaGNNConfig) -> None:
        self.ks = config.eval_ks
        self._metrics: dict[str, LinkPredRecall | LinkPredNDCG] = {}
        for k in self.ks:
            self._metrics[f"Recall@{k}"] = LinkPredRecall(k=k)
            self._metrics[f"NDCG@{k}"] = LinkPredNDCG(k=k)

    def _reset(self) -> None:
        for m in self._metrics.values():
            m.reset()

    @torch.no_grad()
    def evaluate(
        self,
        model,
        data,
        mask: torch.Tensor,
        batch_size: int = 512,
    ) -> dict[str, float]:
        """Evaluate model on users present in mask.

        Args:
            model: UCaGNN model.
            data: PyG Data object.
            mask: Boolean mask selecting interactions for evaluation.
            batch_size: Users per eval batch.

        Returns:
            Dict like {"Recall@10": 0.15, "NDCG@20": 0.12, ...}
        """
        model.eval()
        device = next(model.parameters()).device

        # Move metrics to the correct device
        for m in self._metrics.values():
            m.to(device)
        self._reset()

        user_nodes = data.user_nodes[mask].to(device)
        item_nodes = (data.item_nodes[mask] - data.n_users).to(device)

        unique_users = user_nodes.unique()
        n_items = data.n_items

        # Ground-truth matrix for masked interactions
        gt_matrix = torch.zeros(
            unique_users.max().item() + 1, n_items, device=device
        )
        gt_matrix[user_nodes, item_nodes] = 1.0

        edge_index = data.edge_index.to(device)
        edge_sign = (
            data.edge_sign.to(device) if hasattr(data, "edge_sign") else None
        )

        max_k = max(self.ks)

        for start in range(0, unique_users.size(0), batch_size):
            batch_users = unique_users[start : start + batch_size]
            scores = model.get_all_scores(edge_index, batch_users, edge_sign)
            gt_batch = gt_matrix[batch_users]

            # Skip users with no ground-truth
            has_gt = gt_batch.sum(dim=-1) > 0
            if not has_gt.any():
                continue
            scores = scores[has_gt]
            gt_batch = gt_batch[has_gt]

            # Top-k predicted item indices
            _, pred_index_mat = torch.topk(scores, max_k, dim=-1)

            # Ground-truth edge_label_index: (user_idx_in_batch, item_id) pairs
            batch_gt_users, batch_gt_items = gt_batch.nonzero(as_tuple=True)

            for name, metric in self._metrics.items():
                metric.update(pred_index_mat, (batch_gt_users, batch_gt_items))

        results = {}
        for name, metric in self._metrics.items():
            results[name] = metric.compute().item()

        return results
