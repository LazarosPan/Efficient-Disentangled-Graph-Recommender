"""Shared canonical baseline helpers.

The paper-baseline modules use these helpers to expose the same trainer and
evaluator surface as U-CaGNN while keeping their architecture code separate.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import coalesce, degree

from ...utils.config import UCaGNNConfig
from ..lightgcn import DualBranchGCN


def normalized_edge_weight(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    edge_norm: torch.Tensor | None,
    add_self_loops: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return edge index and symmetric normalization for sparse propagation."""
    if add_self_loops:
        node_ids = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
        loop_index = torch.stack([node_ids, node_ids], dim=0)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_index = coalesce(edge_index)
        edge_norm = None

    if edge_norm is not None:
        return edge_index, edge_norm.to(device=edge_index.device, dtype=dtype)

    deg = degree(edge_index[0], num_nodes=num_nodes)
    inv_sqrt_deg = deg.pow(-0.5)
    inv_sqrt_deg[inv_sqrt_deg == float("inf")] = 0.0
    weights = inv_sqrt_deg[edge_index[0]] * inv_sqrt_deg[edge_index[1]]
    return edge_index, weights.to(dtype=dtype)


def build_sparse_adjacency(
    edge_index: torch.Tensor,
    edge_norm: torch.Tensor | None,
    *,
    num_nodes: int,
    dtype: torch.dtype,
    add_self_loops: bool = False,
) -> torch.Tensor:
    """Build a coalesced normalized sparse adjacency matrix."""
    resolved_edge_index, weights = normalized_edge_weight(
        edge_index,
        num_nodes,
        edge_norm=edge_norm,
        add_self_loops=add_self_loops,
        dtype=dtype,
    )
    return DualBranchGCN._build_sparse_adjacency(
        resolved_edge_index,
        weights,
        num_nodes=num_nodes,
        dtype=dtype,
    )


def score_pairwise(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
) -> torch.Tensor:
    """Return pairwise dot-product scores."""
    return (user_emb[user_ids] * item_emb[item_ids]).sum(dim=-1)


def fixed_score_mix_weights(
    user_ids: torch.Tensor,
    *,
    interest_weight: float,
    conformity_weight: float,
) -> torch.Tensor:
    """Return full three-component score-mix weights for diagnostics."""
    context_weight = 0.0
    weights = torch.tensor(
        [[interest_weight, conformity_weight, context_weight]],
        device=user_ids.device,
        dtype=torch.float32,
    )
    return weights.expand(user_ids.size(0), -1)


def score_dict(
    *,
    final_score: torch.Tensor,
    interest_score: torch.Tensor,
    conformity_score: torch.Tensor | None,
    user_ids: torch.Tensor,
    interest_weight: float,
    conformity_weight: float,
) -> dict[str, torch.Tensor]:
    """Build the shared baseline score dictionary."""
    if conformity_score is None:
        conformity_score = torch.zeros_like(interest_score)
    return {
        "final_score": final_score,
        "interest_score": interest_score,
        "conformity_score": conformity_score,
        "context_score": torch.zeros_like(final_score),
        "score_mix_weights": fixed_score_mix_weights(
            user_ids,
            interest_weight=interest_weight,
            conformity_weight=conformity_weight,
        ).to(device=final_score.device, dtype=final_score.dtype),
    }


class CanonicalBaselineRecommender(nn.Module):
    """Common metadata and output helpers for canonical baseline models."""

    def __init__(self, n_users: int, n_items: int, config: UCaGNNConfig) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.config = config

    def _training_output(
        self,
        *,
        embeddings: dict[str, torch.Tensor],
        propagated: dict[str, torch.Tensor],
        pos_scores: dict[str, torch.Tensor],
        neg_scores: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        dice_negative_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Return the common training payload consumed by ``LossSuite``."""
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "embeddings": embeddings,
            "propagated": propagated,
            "ipw_weights": torch.ones(user_ids.size(0), device=user_ids.device),
            "loss_user_ids": user_ids,
            "loss_neg_item_ids": neg_item_ids,
        }
        if dice_negative_mask is not None:
            output["dice_negative_mask"] = dice_negative_mask
        return output

    @torch.no_grad()
    def get_all_score_components(
        self,
        edge_index: torch.Tensor,
        user_ids: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
        edge_norm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return full-catalog score components for diagnostics."""
        propagated = self.get_propagated_for_eval(edge_index, edge_sign, edge_norm)
        return self.get_score_components_from_propagated(propagated, user_ids)
