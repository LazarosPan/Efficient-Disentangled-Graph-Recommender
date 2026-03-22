"""U-CaGNN: Unified Causal Graph Neural Network for recommendations.

Orchestrates Modules A (embeddings), B (LightGCN), C (scoring), F (propensity).
Every component is toggleable via UCaGNNConfig.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.config import UCaGNNConfig
from ..data.subgraph_sampler import SubgraphBatch
from .embeddings import EmbeddingModule
from .lightgcn import DualBranchGCN
from .scoring import ScoringModule
from .propensity import PropensityEstimator


class UCaGNN(nn.Module):
    """Main model: forward = embed → propagate → score.

    Different configs produce different model variants:
    - Non-causal LightGCN: ``use_dual_branch=False``
    - DICE-like: ``use_dual_branch=True``, only L_rec + L_ortho
    - Full U-CaGNN: all toggles enabled
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        config: UCaGNNConfig,
        item_features: torch.Tensor | None = None,
        item_popularity: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_users = n_users
        self.n_items = n_items

        # Module A: Embeddings
        self.embedding = EmbeddingModule(
            n_users,
            n_items,
            config,
            item_features=item_features,
            item_popularity=item_popularity,
        )

        # Module B: GCN propagation
        self.gcn = DualBranchGCN(config)

        # Module C: Scoring
        self.scoring = ScoringModule(config)

        # Module F: Propensity (optional)
        if config.use_ipw:
            self.propensity = PropensityEstimator(config)

    def forward(
        self,
        edge_index: torch.Tensor,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            edge_index: (2, E) graph edges.
            user_ids: (B,) user indices.
            pos_item_ids: (B,) positive item indices.
            neg_item_ids: (B,) negative item indices.
            edge_sign: (E_interactions,) sign weights for edges.

        Returns:
            Dict containing:
            - pos_scores: dict with interest/conformity/cf/final scores for positives
            - neg_scores: dict with scores for negatives
            - embeddings: dict of initial embeddings (pre-GNN)
            - propagated: dict of propagated embeddings (post-GNN)
            - ipw_weights: (B,) inverse propensity weights (if use_ipw)
        """
        # Module A: Get initial embeddings
        init_embs = self.embedding.get_all_embeddings()

        # Module B: GCN propagation
        propagated = self.gcn(
            init_embs, edge_index, edge_sign,
            n_users=self.n_users, n_items=self.n_items,
        )

        # Module C: Score positive and negative pairs
        pos_scores = self.scoring(propagated, user_ids, pos_item_ids)
        neg_scores = self.scoring(propagated, user_ids, neg_item_ids)

        result = {
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "embeddings": init_embs,
            "propagated": propagated,
        }

        # Module F: IPW weights
        if self.config.use_ipw:
            if self.config.use_dual_branch:
                item_emb = propagated["item_interest"][pos_item_ids]
            else:
                item_emb = propagated["item"][pos_item_ids]
            result["ipw_weights"] = self.propensity.get_ipw_weights(item_emb)
        else:
            result["ipw_weights"] = torch.ones(user_ids.size(0), device=user_ids.device)

        return result

    def forward_subgraph(self, batch: SubgraphBatch) -> dict[str, torch.Tensor]:
        """Mini-batch forward pass on a subgraph.

        Indexes into embedding tables for subgraph nodes only, runs GCN
        on the subgraph, and scores using local indices.

        Args:
            batch: SubgraphBatch from SubgraphSampler.

        Returns:
            Dict matching the output format of ``forward()``.
        """
        # Index into embedding tables for subgraph nodes only
        sub_embs = self.embedding.get_subgraph_embeddings(
            batch.user_global_ids,
            batch.item_global_ids,
        )

        # GCN on subgraph
        propagated = self.gcn(
            sub_embs, batch.sub_edge_index, batch.sub_edge_sign,
            n_users=batch.n_sub_users, n_items=batch.n_sub_items,
        )

        # Score using local indices
        pos_scores = self.scoring(propagated, batch.batch_user_local, batch.batch_pos_local)
        neg_scores = self.scoring(propagated, batch.batch_user_local, batch.batch_neg_local)

        result: dict[str, torch.Tensor | dict] = {
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "embeddings": sub_embs,
            "propagated": propagated,
        }

        # IPW weights
        if self.config.use_ipw:
            if self.config.use_dual_branch:
                item_emb = propagated["item_interest"][batch.batch_pos_local]
            else:
                item_emb = propagated["item"][batch.batch_pos_local]
            result["ipw_weights"] = self.propensity.get_ipw_weights(item_emb)
        else:
            result["ipw_weights"] = torch.ones(
                batch.batch_user_local.size(0),
                device=batch.batch_user_local.device,
            )

        return result

    def get_all_scores(
        self,
        edge_index: torch.Tensor,
        user_ids: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
        scoring_mode: str | None = None,
    ) -> torch.Tensor:
        """Score all items for given users (for evaluation).

        Args:
            edge_index: (2, E) graph edges.
            user_ids: (B,) user indices.
            edge_sign: optional edge signs.

        Returns:
            (B, n_items) score matrix.
        """
        return self.get_all_score_components(
            edge_index=edge_index,
            user_ids=user_ids,
            edge_sign=edge_sign,
            scoring_mode=scoring_mode,
        )["final_score"]

    def get_all_score_components(
        self,
        edge_index: torch.Tensor,
        user_ids: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
        scoring_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return full-catalog score components for evaluation and diagnostics."""
        init_embs = self.embedding.get_all_embeddings()
        propagated = self.gcn(
            init_embs, edge_index, edge_sign,
            n_users=self.n_users, n_items=self.n_items,
        )
        resolved_scoring_mode = scoring_mode or self.config.eval_scoring_mode

        if self.config.use_dual_branch:
            user_interest = propagated["user_interest"][user_ids]
            item_interest = propagated["item_interest"]
            interest_scores = user_interest @ item_interest.t()

            user_conformity = propagated["user_conformity"][user_ids]
            item_conformity = propagated["item_conformity"]
            conformity_scores = user_conformity @ item_conformity.t()

            if self.config.use_counterfactual:
                cf_scores = interest_scores - conformity_scores
            else:
                cf_scores = torch.zeros_like(interest_scores)

            final_scores = self.scoring.combine_scores(
                interest_scores,
                conformity_scores,
                cf_scores,
                scoring_mode=resolved_scoring_mode,
            )
            return {
                "interest_score": interest_scores,
                "conformity_score": conformity_scores,
                "cf_score": cf_scores,
                "final_score": final_scores,
                "user_interest_emb": user_interest,
                "user_conformity_emb": user_conformity,
            }

        user_embedding = propagated["user"][user_ids]
        item_embedding = propagated["item"]
        interest_scores = user_embedding @ item_embedding.t()
        zeros = torch.zeros_like(interest_scores)
        return {
            "interest_score": interest_scores,
            "conformity_score": zeros,
            "cf_score": zeros,
            "final_score": interest_scores,
        }

    def get_score_weight_summary(self, scoring_mode: str = "default") -> dict[str, float]:
        return self.scoring.get_score_weight_summary(scoring_mode=scoring_mode)
