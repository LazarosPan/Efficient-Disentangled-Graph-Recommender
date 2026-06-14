"""DICE-family paper baselines over canonical interaction tensors."""

from __future__ import annotations

import math

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


class PaperGCNDICE(CanonicalBaselineRecommender):
    """DICE paper GCN variant over canonical data and shared evaluation.

    The paper-facing name is GCN-DICE: separate interest/conformity embedding
    tables, LightGCN-style propagation for each channel, dropout after each
    propagation layer, summed interest+conformity click score, and self-looped
    graph propagation.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        config: UCaGNNConfig,
    ) -> None:
        super().__init__(n_users, n_items, config)
        self.embeddings_int = nn.Parameter(
            torch.empty(n_users + n_items, config.embed_dim),
        )
        self.embeddings_pop = nn.Parameter(
            torch.empty(n_users + n_items, config.embed_dim),
        )
        self.interest_propagation = LightGCNBranch(
            config.interest_gnn_layers,
            dropout=config.dropout,
        )
        self.conformity_propagation = LightGCNBranch(
            config.conformity_gnn_layers,
            dropout=config.dropout,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize DICE channel embeddings with the original uniform rule."""
        stdv = 1.0 / math.sqrt(self.config.embed_dim)
        nn.init.uniform_(self.embeddings_int, -stdv, stdv)
        nn.init.uniform_(self.embeddings_pop, -stdv, stdv)

    def _initial_embeddings(
        self,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        int_embeddings = self.embeddings_int
        pop_embeddings = self.embeddings_pop
        if dtype is not None:
            int_embeddings = int_embeddings.to(dtype=dtype)
            pop_embeddings = pop_embeddings.to(dtype=dtype)
        return {
            "user_interest": int_embeddings[: self.n_users],
            "item_interest": int_embeddings[self.n_users :],
            "user_conformity": pop_embeddings[: self.n_users],
            "item_conformity": pop_embeddings[self.n_users :],
        }

    def get_propagated_for_eval(
        self,
        edge_index: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
        edge_norm: torch.Tensor | None = None,
        embedding_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Propagate the two GCN-DICE channels over a self-looped graph."""
        del edge_sign
        embeddings = self._initial_embeddings(dtype=embedding_dtype)
        return propagate_user_item_channels(
            edge_index,
            edge_norm=None,
            n_users=self.n_users,
            n_items=self.n_items,
            add_self_loops=True,
            channel_specs=(
                (
                    "user_interest",
                    "item_interest",
                    embeddings["user_interest"],
                    embeddings["item_interest"],
                    self.interest_propagation,
                ),
                (
                    "user_conformity",
                    "item_conformity",
                    embeddings["user_conformity"],
                    embeddings["item_conformity"],
                    self.conformity_propagation,
                ),
            ),
        )

    def _score_pairs(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        interest_score = score_propagated_pair(
            propagated,
            user_key="user_interest",
            item_key="item_interest",
            user_ids=user_ids,
            item_ids=item_ids,
        )
        conformity_score = score_propagated_pair(
            propagated,
            user_key="user_conformity",
            item_key="item_conformity",
            user_ids=user_ids,
            item_ids=item_ids,
        )
        return score_dict(
            final_score=interest_score + conformity_score,
            interest_score=interest_score,
            conformity_score=conformity_score,
            user_ids=user_ids,
            interest_weight=1.0,
            conformity_weight=1.0,
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
        edge_norm: torch.Tensor | None = None,
        dice_negative_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Return the shared training payload for one GCN-DICE batch."""
        del edge_sign, edge_norm
        embeddings = self._initial_embeddings()
        propagated = self.get_propagated_for_eval(edge_index)
        return self._training_output(
            embeddings=embeddings,
            propagated=propagated,
            pos_scores=self._score_pairs(propagated, user_ids, pos_item_ids),
            neg_scores=self._score_pairs(propagated, user_ids, neg_item_ids),
            user_ids=user_ids,
            neg_item_ids=neg_item_ids,
            dice_negative_mask=dice_negative_mask,
        )

    @torch.no_grad()
    def score_users_from_propagated(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return full-catalog DICE interest+conformity scores."""
        interest = score_propagated_matrix(
            propagated,
            user_key="user_interest",
            item_key="item_interest",
            user_ids=user_ids,
        )
        conformity = score_propagated_matrix(
            propagated,
            user_key="user_conformity",
            item_key="item_conformity",
            user_ids=user_ids,
        )
        return interest + conformity

    @torch.no_grad()
    def get_score_components_from_propagated(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return full-catalog DICE score components for diagnostics."""
        interest = score_propagated_matrix(
            propagated,
            user_key="user_interest",
            item_key="item_interest",
            user_ids=user_ids,
        )
        conformity = score_propagated_matrix(
            propagated,
            user_key="user_conformity",
            item_key="item_conformity",
            user_ids=user_ids,
        )
        scores = score_dict(
            final_score=interest + conformity,
            interest_score=interest,
            conformity_score=conformity,
            user_ids=user_ids,
            interest_weight=1.0,
            conformity_weight=1.0,
        )
        scores.update(
            {
                "user_interest_emb": propagated["user_interest"][user_ids],
                "user_conformity_emb": propagated["user_conformity"][user_ids],
            },
        )
        return scores
