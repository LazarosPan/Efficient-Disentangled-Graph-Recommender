"""Module A: Dual user + item + popularity embedding tables."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.config import UCaGNNConfig


class EmbeddingModule(nn.Module):
    """Embedding tables for users and items.

    When ``use_dual_branch=True``: separate interest and conformity embeddings for users.
    When ``use_popularity_emb=True``: additional popularity embedding for items.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        config: UCaGNNConfig,
        item_features: torch.Tensor | None = None,
        item_popularity: torch.Tensor | None = None,
        item_recency: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_users = n_users
        self.n_items = n_items
        d = config.embed_dim
        resolved_item_popularity = (
            item_popularity.float()
            if item_popularity is not None
            else torch.zeros(n_items, dtype=torch.bfloat16)
        )
        self.register_buffer(
            "item_popularity",
            resolved_item_popularity,
            persistent=False,
        )
        resolved_item_recency = (
            item_recency.float()
            if item_recency is not None
            else torch.zeros(n_items, dtype=torch.bfloat16)
        )
        self.register_buffer(
            "item_recency",
            resolved_item_recency,
            persistent=False,
        )
        self.has_item_features = bool(
            config.use_features
            and item_features is not None
            and item_features.numel() > 0
        )

        # User embeddings (xavier_uniform_ per PyG LightGCN convention)
        if config.use_dual_branch:
            self.user_interest = nn.Embedding(n_users, d)
            self.user_conformity = nn.Embedding(n_users, d)
            nn.init.xavier_uniform_(self.user_interest.weight)
            nn.init.xavier_uniform_(self.user_conformity.weight)
        else:
            self.user_embed = nn.Embedding(n_users, d)
            nn.init.xavier_uniform_(self.user_embed.weight)

        # Item embedding (always single)
        self.item_embed = nn.Embedding(n_items, d)
        nn.init.xavier_uniform_(self.item_embed.weight)

        # Optional popularity embedding
        if config.use_popularity_emb:
            self.item_pop = nn.Embedding(n_items, config.pop_embed_dim)
            nn.init.xavier_uniform_(self.item_pop.weight)

        if self.has_item_features:
            assert item_features is not None
            self.register_buffer(
                "item_feature_matrix",
                item_features.float(),
                persistent=False,
            )
            self.item_feature_proj = nn.Linear(item_features.size(-1), d)
            self.item_feature_norm = nn.LayerNorm(d)
            self.item_interest_gate = nn.Parameter(torch.tensor(0.0))
            self.item_conformity_gate = nn.Parameter(torch.tensor(0.0))
            self.popularity_modulator = nn.Sequential(
                nn.Linear(1, d),
                nn.Sigmoid(),
            )
            # Cached projected features (static — only invalidated on reset)
            self._cached_projected_features: torch.Tensor | None = None
            self._cached_popularity_gate: torch.Tensor | None = None

    @staticmethod
    def _select_embedding_rows(
        embedding: nn.Embedding,
        ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return all embedding rows or an indexed subset."""
        return embedding.weight if ids is None else embedding.weight[ids]

    @staticmethod
    def _cast_floating_tensors(
        tensors: dict[str, torch.Tensor],
        dtype: torch.dtype | None,
    ) -> dict[str, torch.Tensor]:
        """Cast only floating-point tensors, preserving integer index tensors."""
        if dtype is None:
            return tensors
        return {
            key: value.to(dtype=dtype) if torch.is_floating_point(value) else value
            for key, value in tensors.items()
        }

    def _build_user_embeddings(
        self,
        user_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the user embedding tensors needed by the active branch setup."""
        if self.config.use_dual_branch:
            return {
                "user_interest": self._select_embedding_rows(
                    self.user_interest,
                    user_ids,
                ),
                "user_conformity": self._select_embedding_rows(
                    self.user_conformity,
                    user_ids,
                ),
            }
        return {"user": self._select_embedding_rows(self.user_embed, user_ids)}

    def _build_popularity_embeddings(
        self,
        item_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return optional popularity embeddings when enabled."""
        if not self.config.use_popularity_emb:
            return {}
        return {"item_pop": self._select_embedding_rows(self.item_pop, item_ids)}

    def _build_item_metadata(
        self,
        item_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return non-trainable item metadata needed by scoring/loss layers."""
        popularity = (
            self.item_popularity if item_ids is None else self.item_popularity[item_ids]
        )
        recency = self.item_recency if item_ids is None else self.item_recency[item_ids]
        return {
            "item_popularity": popularity,
            "item_recency": recency,
        }

    def _ensure_feature_cache(self) -> None:
        """Compute projected features + popularity gate, caching in eval mode.

        During training the projection weights change, so we recompute every
        call.  In eval mode the result is cached for repeated full-catalog
        scoring calls.
        """
        if self._cached_projected_features is not None and not self.training:
            return  # eval-mode cache hit
        assert hasattr(self, "item_feature_matrix")
        assert hasattr(self, "item_popularity")
        self._cached_projected_features = self.item_feature_norm(
            self.item_feature_proj(self.item_feature_matrix)
        )
        self._cached_popularity_gate = self.popularity_modulator(
            self.item_popularity.unsqueeze(-1)
        )

    def invalidate_feature_cache(self) -> None:
        """Clear cached feature projections (call if features change)."""
        if self.has_item_features:
            self._cached_projected_features = None
            self._cached_popularity_gate = None

    def _get_item_base_embeddings(
        self,
        item_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        item_embed = self._select_embedding_rows(self.item_embed, item_ids)

        if not self.has_item_features:
            return item_embed, None, None

        if item_ids is not None and self.training:
            # Training-time subgraph path: project only the needed item subset
            # to avoid O(n_items × feature_dim × D) full-catalog linear map
            # every batch.  Eval uses the full-catalog cache below.
            subset = self.item_feature_matrix[item_ids]
            projected = self.item_feature_norm(self.item_feature_proj(subset))
            pop_gate = self.popularity_modulator(
                self.item_popularity[item_ids].unsqueeze(-1)
            )
            return item_embed, projected, pop_gate

        # Full-catalog path (eval) or get_all_embeddings (item_ids is None):
        # populate the cache once per eval call and slice as needed.
        self._ensure_feature_cache()
        assert self._cached_projected_features is not None
        assert self._cached_popularity_gate is not None
        if item_ids is None:
            return (
                item_embed,
                self._cached_projected_features,
                self._cached_popularity_gate,
            )
        return (
            item_embed,
            self._cached_projected_features[item_ids],
            self._cached_popularity_gate[item_ids],
        )

    def _build_item_embeddings(
        self,
        item_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        item_embed, projected_features, pop_gate = self._get_item_base_embeddings(
            item_ids
        )
        if projected_features is None or pop_gate is None:
            return {"item": item_embed}

        interest_gate = torch.sigmoid(self.item_interest_gate)
        conformity_gate = torch.sigmoid(self.item_conformity_gate)

        item_interest = item_embed + interest_gate * projected_features
        item_conformity = item_embed + conformity_gate * (projected_features * pop_gate)
        return {
            "item": item_embed,
            "item_interest": item_interest,
            "item_conformity": item_conformity,
        }

    def get_all_embeddings(
        self,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a dict of all initial (pre-GNN) embeddings."""
        out = self._build_user_embeddings()
        out.update(self._build_item_embeddings())
        out.update(self._build_item_metadata())
        out.update(self._build_popularity_embeddings())
        return self._cast_floating_tensors(out, dtype)

    def get_subgraph_embeddings(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return initial embeddings restricted to a subgraph node set."""
        out = self._build_user_embeddings(user_ids)
        out.update(self._build_item_embeddings(item_ids))
        out.update(self._build_item_metadata(item_ids))
        out.update(self._build_popularity_embeddings(item_ids))
        return out

    def get_stacked_embeddings(self) -> torch.Tensor:
        """Return (n_users + n_items, D) combined node embeddings for graph building.

        Uses interest embeddings for users when dual_branch is active.
        """
        user_embeddings = self._build_user_embeddings()
        if self.config.use_dual_branch:
            user_emb = user_embeddings["user_interest"]
        else:
            user_emb = user_embeddings["user"]
        item_embs = self._build_item_embeddings()
        item_emb = item_embs.get("item_interest", item_embs["item"])
        return torch.cat([user_emb, item_emb], dim=0)
