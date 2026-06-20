"""EDGRec: efficient disentangled graph recommender.

Orchestrates the embedding, propagation, scoring, and optional propensity layers.
Every component is toggleable via EDGRecConfig.
"""

from __future__ import annotations

import torch
from torch import nn

from ..data.subgraph_sampler import SubgraphBatch
from ..utils.config import EDGRecConfig
from .common import training_output_payload
from .embeddings import EmbeddingModule
from .lightgcn import DualBranchGCN
from .propensity import PropensityEstimator
from .scoring import ScoringModule


class EDGRec(nn.Module):
    """Main model: forward = embed → propagate → score.

    Different configs produce different model variants:
    - Sampled LightGCN: ``use_dual_branch=False``
    - DICE-like: ``use_dual_branch=True``, only L_rec + L_ortho
    - Full EDGRec: all toggles enabled
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        config: EDGRecConfig,
        item_features: torch.Tensor | None = None,
        item_popularity: torch.Tensor | None = None,
        item_recency: torch.Tensor | None = None,
        item_propensity_targets: torch.Tensor | None = None,
        item_age: torch.Tensor | None = None,
        recent_train_items: torch.Tensor | None = None,
        recent_train_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_users = n_users
        self.n_items = n_items
        context_feature_dim = 4
        if config.use_features and item_features is not None and item_features.numel() > 0:
            context_feature_dim += int(item_features.size(-1))

        # Embedding layer
        self.embedding = EmbeddingModule(
            n_users,
            n_items,
            config,
            item_features=item_features,
            item_popularity=item_popularity,
            item_recency=item_recency,
            item_propensity_targets=item_propensity_targets,
            item_age=item_age,
            recent_train_items=recent_train_items,
            recent_train_mask=recent_train_mask,
        )

        # Propagation layer
        self.gcn = DualBranchGCN(config)

        # Scoring layer
        self.scoring = ScoringModule(config, context_feature_dim=context_feature_dim)

        # Optional propensity layer
        if config.use_ipw:
            self._propensity_mlp = PropensityEstimator(config)

        # Optional: compile GCN for speedup on static graph structures
        if config.use_torch_compile:
            self.gcn = torch.compile(self.gcn, dynamic=True)  # type: ignore[assignment]

    def propagate_embeddings(
        self,
        embeddings: dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_sign: torch.Tensor | None,
        *,
        n_users: int,
        n_items: int,
        edge_norm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run graph propagation for a prepared embedding bundle."""
        propagated = self.gcn(
            embeddings,
            edge_index,
            edge_sign,
            n_users=n_users,
            n_items=n_items,
            edge_norm=edge_norm,
        )
        for key in (
            "item_pop",
            "item_popularity",
            "item_recency",
            "item_propensity_targets",
            "item_age",
            "item_safe_features",
            "recent_train_items",
            "recent_train_mask",
        ):
            if key in embeddings:
                propagated[key] = embeddings[key]
        return propagated

    def _with_static_metadata(
        self,
        propagated: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Merge embedding-owned static metadata into a propagated bundle.

        Args:
            propagated: Propagated embedding dictionary.

        Returns:
            Propagated dictionary with missing static metadata filled in.
        """
        if "recent_train_items" in propagated and "recent_train_mask" in propagated:
            return propagated
        merged = dict(self.embedding.get_embeddings())
        merged.update(propagated)
        return merged

    def build_training_output(
        self,
        embeddings: dict[str, torch.Tensor],
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        dice_negative_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Build the shared training payload from propagated embeddings."""
        scoring_inputs = propagated
        recent_train_item_interest = embeddings.get("recent_train_item_interest")
        if recent_train_item_interest is not None:
            scoring_inputs = dict(propagated)
            scoring_inputs["recent_train_item_interest"] = recent_train_item_interest
        pos_scores = self.scoring(
            scoring_inputs,
            user_ids,
            pos_item_ids,
        )
        neg_scores = self.scoring(
            scoring_inputs,
            user_ids,
            neg_item_ids,
        )

        if self.config.use_ipw:
            item_key = "item_interest" if self.config.use_dual_branch else "item"
            propensity = self._propensity_mlp(propagated[item_key][pos_item_ids])
            ipw_weights = 1.0 / propensity
        else:
            propensity = None
            ipw_weights = None

        return training_output_payload(
            embeddings=embeddings,
            propagated=propagated,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            user_ids=user_ids,
            neg_item_ids=neg_item_ids,
            ipw_weights=ipw_weights,
            dice_negative_mask=dice_negative_mask,
            propensity_scores=propensity,
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
        """Full forward pass.

        Args:
            edge_index: (2, E) graph edges.
            user_ids: (B,) user indices.
            pos_item_ids: (B,) positive item indices.
            neg_item_ids: (B,) negative item indices.
            edge_sign: (E_interactions,) sign weights for edges.

        Returns:
            Dict containing:
            - pos_scores: dict with interest/conformity/branch-contrast/final scores
              for positives
            - neg_scores: dict with scores for negatives
            - embeddings: dict of initial embeddings (pre-GNN)
            - propagated: dict of propagated embeddings (post-GNN)
            - ipw_weights: (B,) inverse propensity weights (if use_ipw)

        """
        # Get initial embeddings from the embedding layer
        init_embs = self.embedding.get_embeddings()

        # Run propagation on the full graph
        propagated = self.propagate_embeddings(
            init_embs,
            edge_index,
            edge_sign,
            n_users=self.n_users,
            n_items=self.n_items,
            edge_norm=edge_norm,
        )

        return self.build_training_output(
            init_embs,
            propagated,
            user_ids,
            pos_item_ids,
            neg_item_ids,
            dice_negative_mask=dice_negative_mask,
        )

    def forward_subgraph(
        self,
        batch: SubgraphBatch,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Mini-batch forward pass on a subgraph.

        Indexes into embedding tables for subgraph nodes only, runs GCN
        on the subgraph, and scores using local indices.

        Args:
            batch: SubgraphBatch from SubgraphSampler.

        Returns:
            Dict matching the output format of ``forward()``.

        """
        # Index into embedding tables for subgraph nodes only
        sub_embs = self.embedding.get_embeddings(
            batch.user_global_ids,
            batch.item_global_ids,
        )
        sub_embs["recent_train_item_interest"] = self.embedding.get_recent_train_item_interest(
            batch.user_global_ids,
        )

        # GCN on subgraph
        propagated = self.propagate_embeddings(
            sub_embs,
            batch.sub_edge_index,
            batch.sub_edge_sign,
            n_users=batch.n_sub_users,
            n_items=batch.n_sub_items,
            edge_norm=batch.sub_edge_norm,
        )

        return self.build_training_output(
            sub_embs,
            propagated,
            batch.batch_user_local,
            batch.batch_pos_local,
            batch.batch_neg_local,
            dice_negative_mask=batch.dice_negative_mask,
        )

    def get_propagated_for_eval(
        self,
        edge_index: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
        edge_norm: torch.Tensor | None = None,
        embedding_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Propagate full-graph embeddings once for evaluation.

        Returns the propagated embedding dict that can be reused across
        multiple scoring batches, avoiding repeated GCN forward passes.
        """
        return self.propagate_embeddings(
            self.embedding.get_embeddings(dtype=embedding_dtype),
            edge_index,
            edge_sign,
            n_users=self.n_users,
            n_items=self.n_items,
            edge_norm=edge_norm,
        )

    @torch.no_grad()
    def score_users_from_propagated(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score all items for given users using pre-propagated embeddings.

        Args:
            propagated: Output of ``get_propagated_for_eval``.
            user_ids: (B,) user indices.

        Returns:
            (B, n_items) score matrix.

        """
        return self.scoring.score_all_items(
            self._with_static_metadata(propagated),
            user_ids,
        )["final_score"]

    @torch.no_grad()
    def get_score_components_from_propagated(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return full-catalog score components from reused propagated embeddings.

        Args:
            propagated: Output of ``get_propagated_for_eval``.
            user_ids: (B,) user indices.

        Returns:
            Refined scorer outputs aligned with ``score_all_items``.

        """
        propagated_with_metadata = self._with_static_metadata(propagated)
        scores = self.scoring.score_all_items(
            propagated_with_metadata,
            user_ids,
        )
        if self.config.use_dual_branch:
            scores["user_interest_emb"] = propagated_with_metadata["user_interest"][user_ids]
            scores["user_conformity_emb"] = propagated_with_metadata["user_conformity"][user_ids]
        return scores
