"""Paper LightGCN baseline over canonical interaction tensors."""

from __future__ import annotations

import torch
from torch import nn

from ...utils.config import UCaGNNConfig
from ..lightgcn import LightGCNBranch
from .common import (
    CanonicalBaselineRecommender,
    propagate_user_item_channels,
    score_dict,
    score_propagated_matrix,
    score_propagated_pair,
)


class PaperLightGCN(CanonicalBaselineRecommender):
    """LightGCN as the paper equation: embeddings + normalized propagation + BPR.

    This class intentionally omits feature transforms, activations, learned
    score mixing, dropout, and CAGRA-specific logic. It accepts the same
    canonical graph and batch tensors as U-CaGNN so comparisons share data,
    splits, negative sampling, and evaluation.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        config: UCaGNNConfig,
    ) -> None:
        super().__init__(n_users, n_items, config)
        if config.dropout != 0.0:
            raise ValueError("PaperLightGCN requires dropout=0.0.")
        self.user_embedding = nn.Embedding(n_users, config.embed_dim)
        self.item_embedding = nn.Embedding(n_items, config.embed_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.propagation = LightGCNBranch(config.single_branch_gnn_layers, dropout=0.0)

    def _initial_embeddings(
        self,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        user = self.user_embedding.weight
        item = self.item_embedding.weight
        if dtype is not None:
            user = user.to(dtype=dtype)
            item = item.to(dtype=dtype)
        return {"user": user, "item": item}

    def get_propagated_for_eval(
        self,
        edge_index: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
        edge_norm: torch.Tensor | None = None,
        embedding_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Propagate full-graph LightGCN embeddings once for evaluation."""
        del edge_sign
        embeddings = self._initial_embeddings(dtype=embedding_dtype)
        return propagate_user_item_channels(
            edge_index,
            edge_norm,
            n_users=self.n_users,
            n_items=self.n_items,
            channel_specs=(
                (
                    "user",
                    "item",
                    embeddings["user"],
                    embeddings["item"],
                    self.propagation,
                ),
            ),
        )

    def _score_pairs(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        final_score = score_propagated_pair(
            propagated,
            user_key="user",
            item_key="item",
            user_ids=user_ids,
            item_ids=item_ids,
        )
        return score_dict(
            final_score=final_score,
            interest_score=final_score,
            conformity_score=None,
            user_ids=user_ids,
            interest_weight=1.0,
            conformity_weight=0.0,
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
        edge_norm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Return the shared training payload for one BPR batch."""
        del edge_sign
        embeddings = self._initial_embeddings()
        propagated = self.get_propagated_for_eval(edge_index, edge_norm=edge_norm)
        return self._training_output(
            embeddings=embeddings,
            propagated=propagated,
            pos_scores=self._score_pairs(propagated, user_ids, pos_item_ids),
            neg_scores=self._score_pairs(propagated, user_ids, neg_item_ids),
            user_ids=user_ids,
            neg_item_ids=neg_item_ids,
        )

    @torch.no_grad()
    def score_users_from_propagated(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return full-catalog LightGCN dot-product scores."""
        return score_propagated_matrix(
            propagated,
            user_key="user",
            item_key="item",
            user_ids=user_ids,
        )

    @torch.no_grad()
    def get_score_components_from_propagated(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return LightGCN scores in the shared diagnostics shape."""
        final_score = self.score_users_from_propagated(propagated, user_ids)
        return score_dict(
            final_score=final_score,
            interest_score=final_score,
            conformity_score=torch.zeros_like(final_score),
            user_ids=user_ids,
            interest_weight=1.0,
            conformity_weight=0.0,
        )
