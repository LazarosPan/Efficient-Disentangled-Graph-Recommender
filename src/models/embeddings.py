"""Module A: Dual user + item + popularity embedding tables."""

from __future__ import annotations

import torch
from torch import nn

from ..utils.config import UCaGNNConfig


def _register_bf16_buffer(
    module: nn.Module,
    name: str,
    tensor: torch.Tensor | None,
    n: int,
) -> None:
    """Register a 1-D bfloat16 non-persistent buffer, using zeros when absent.

    Args:
        module: The ``nn.Module`` on which to call ``register_buffer``.
        name: Buffer attribute name.
        tensor: Source tensor to cast, or ``None`` to create an all-zero fallback.
        n: Size of the fallback zero tensor (used only when *tensor* is ``None``).

    """
    resolved = (
        tensor.to(torch.bfloat16) if tensor is not None else torch.zeros(n, dtype=torch.bfloat16)
    )
    module.register_buffer(name, resolved, persistent=False)


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
        _register_bf16_buffer(self, "item_popularity", item_popularity, n_items)
        _register_bf16_buffer(self, "item_recency", item_recency, n_items)
        self.has_item_features = bool(
            config.use_features and item_features is not None and item_features.numel() > 0,
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
                item_features.to(torch.bfloat16),
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

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        """Return the dtype of the first parameter owned by ``module``."""
        return next(module.parameters()).dtype

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
        popularity = self.item_popularity if item_ids is None else self.item_popularity[item_ids]
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
        feature_dtype = self._module_dtype(self.item_feature_proj)
        pop_dtype = self._module_dtype(self.popularity_modulator)
        self._cached_projected_features = self.item_feature_norm(
            self.item_feature_proj(self.item_feature_matrix.to(dtype=feature_dtype)),
        )
        self._cached_popularity_gate = self.popularity_modulator(
            self.item_popularity.unsqueeze(-1).to(dtype=pop_dtype),
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
            # to avoid O(n_items * feature_dim * D) full-catalog linear map
            # every batch.  Eval uses the full-catalog cache below.
            feature_dtype = self._module_dtype(self.item_feature_proj)
            pop_dtype = self._module_dtype(self.popularity_modulator)
            subset = self.item_feature_matrix[item_ids].to(dtype=feature_dtype)
            projected = self.item_feature_norm(self.item_feature_proj(subset))
            pop_gate = self.popularity_modulator(
                self.item_popularity[item_ids].unsqueeze(-1).to(dtype=pop_dtype),
            )
            return item_embed, projected, pop_gate

        # Full-catalog path (eval) or full-catalog get_embeddings (item_ids is None):
        # populate the cache once per eval call and slice as needed.
        self._ensure_feature_cache()
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
            item_ids,
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

    def get_embeddings(
        self,
        user_ids: torch.Tensor | None = None,
        item_ids: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a dict of initial (pre-GNN) embeddings, optionally filtered.

        Args:
            user_ids: Optional 1-D tensor of user global indices; ``None`` returns all.
            item_ids: Optional 1-D tensor of item global indices; ``None`` returns all.
            dtype: Optional floating-point dtype to cast output tensors to.

        Returns:
            Dict of named embedding tensors for the requested node subset.

        """
        out = self._build_user_embeddings(user_ids)
        out.update(self._build_item_embeddings(item_ids))
        out.update(self._build_item_metadata(item_ids))
        out.update(self._build_popularity_embeddings(item_ids))
        return self._cast_floating_tensors(out, dtype)

    def get_stacked_embeddings(self) -> torch.Tensor:
        """Return (n_users + n_items, D) combined node embeddings for graph building.

        Uses interest embeddings for users when dual_branch is active.
        """
        user_embeddings = self._build_user_embeddings()
        user_emb = (
            user_embeddings["user_interest"]
            if self.config.use_dual_branch
            else user_embeddings["user"]
        )
        item_embs = self._build_item_embeddings()
        item_emb = item_embs.get("item_interest", item_embs["item"])
        return torch.cat([user_emb, item_emb], dim=0)
