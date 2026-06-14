"""Embedding layer: user, item, and optional popularity embedding tables."""

from __future__ import annotations

import torch
from torch import nn

from ..utils.config import UCaGNNConfig
from .common import module_parameter_dtype


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


def _scale_feature_matrix_to_unit(features: torch.Tensor) -> torch.Tensor:
    """Scale each feature column to ``[0, 1]`` and map non-finite values to zero.

    Args:
        features: Item feature matrix with shape ``(n_items, n_features)``.

    Returns:
        Float32 feature matrix with the same shape and bounded values.

    """
    values = features.float()
    finite_mask = torch.isfinite(values)
    safe_values = torch.where(finite_mask, values, torch.zeros_like(values))
    has_valid = finite_mask.any(dim=0)

    inf = torch.full_like(safe_values, float("inf"))
    neg_inf = torch.full_like(safe_values, float("-inf"))
    min_values = torch.where(finite_mask, safe_values, inf).amin(dim=0)
    max_values = torch.where(finite_mask, safe_values, neg_inf).amax(dim=0)
    min_values = torch.where(has_valid, min_values, torch.zeros_like(min_values))
    max_values = torch.where(has_valid, max_values, torch.zeros_like(max_values))

    ranges = max_values - min_values
    varying = ranges > 0
    scaled = torch.where(
        varying.unsqueeze(0),
        (safe_values - min_values.unsqueeze(0)) / ranges.clamp_min(1e-12).unsqueeze(0),
        torch.zeros_like(safe_values),
    )
    constant_nonzero = (~varying) & has_valid & (max_values != 0)
    scaled = torch.where(
        constant_nonzero.unsqueeze(0) & finite_mask,
        torch.ones_like(scaled),
        scaled,
    )
    scaled = torch.where(finite_mask, scaled, torch.zeros_like(scaled))
    return scaled.clamp(0.0, 1.0)


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
        item_propensity_targets: torch.Tensor | None = None,
        item_age: torch.Tensor | None = None,
        recent_train_items: torch.Tensor | None = None,
        recent_train_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_users = n_users
        self.n_items = n_items
        d = config.embed_dim
        _register_bf16_buffer(self, "item_popularity", item_popularity, n_items)
        _register_bf16_buffer(self, "item_recency", item_recency, n_items)
        _register_bf16_buffer(
            self,
            "item_propensity_targets",
            item_propensity_targets,
            n_items,
        )
        _register_bf16_buffer(self, "item_age", item_age, n_items)
        self.register_buffer(
            "recent_train_items",
            (
                recent_train_items.to(dtype=torch.long)
                if recent_train_items is not None
                else torch.zeros((n_users, 10), dtype=torch.long)
            ),
            persistent=False,
        )
        self.register_buffer(
            "recent_train_mask",
            (
                recent_train_mask.to(dtype=torch.bool)
                if recent_train_mask is not None
                else torch.zeros((n_users, 10), dtype=torch.bool)
            ),
            persistent=False,
        )
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
            self.item_pop = nn.Embedding(n_items, config.popularity_embedding_dimensions)
            nn.init.xavier_uniform_(self.item_pop.weight)

        if self.has_item_features:
            assert item_features is not None
            self.register_buffer(
                "item_feature_matrix",
                _scale_feature_matrix_to_unit(item_features).to(torch.bfloat16),
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
        popularity = self.item_popularity if item_ids is None else self.item_popularity[item_ids]
        recency = self.item_recency if item_ids is None else self.item_recency[item_ids]
        exposure = (
            self.item_propensity_targets
            if item_ids is None
            else self.item_propensity_targets[item_ids]
        )
        age = self.item_age if item_ids is None else self.item_age[item_ids]
        metadata = {
            "item_popularity": popularity,
            "item_recency": recency,
            "item_propensity_targets": exposure,
            "item_age": age,
        }
        if self.has_item_features:
            feature_matrix = (
                self.item_feature_matrix if item_ids is None else self.item_feature_matrix[item_ids]
            )
            metadata["item_safe_features"] = feature_matrix
        return metadata

    def _build_user_metadata(
        self,
        user_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return non-trainable user metadata needed by scoring layers."""
        recent_items = (
            self.recent_train_items if user_ids is None else self.recent_train_items[user_ids]
        )
        recent_mask = (
            self.recent_train_mask if user_ids is None else self.recent_train_mask[user_ids]
        )
        return {
            "recent_train_items": recent_items,
            "recent_train_mask": recent_mask,
        }

    def _ensure_feature_cache(self) -> None:
        """Compute projected features + popularity gate, caching in eval mode.

        During training the projection weights change, so we recompute every
        call.  In eval mode the result is cached for repeated full-catalog
        scoring calls.
        """
        if self._cached_projected_features is not None and not self.training:
            return  # eval-mode cache hit
        feature_dtype = module_parameter_dtype(self.item_feature_proj)
        pop_dtype = module_parameter_dtype(self.popularity_modulator)
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
            feature_dtype = module_parameter_dtype(self.item_feature_proj)
            pop_dtype = module_parameter_dtype(self.popularity_modulator)
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
        out.update(self._build_user_metadata(user_ids))
        out.update(self._build_item_embeddings(item_ids))
        out.update(self._build_item_metadata(item_ids))
        out.update(self._build_popularity_embeddings(item_ids))
        return self._cast_floating_tensors(out, dtype)

    def get_recent_train_item_interest(
        self,
        user_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return item-interest embeddings for each user's recent-train history.

        Args:
            user_ids: Optional user ids whose history embeddings should be returned.

        Returns:
            Tensor with shape ``(n_users_or_batch, history_size, embed_dim)``.

        """
        recent_items = (
            self.recent_train_items if user_ids is None else self.recent_train_items[user_ids]
        )
        flat_recent_items = recent_items.reshape(-1)
        if flat_recent_items.numel() == 0:
            return torch.zeros(
                (*recent_items.shape, self.config.embed_dim),
                device=recent_items.device,
                dtype=self.item_embed.weight.dtype,
            )

        unique_items, inverse = flat_recent_items.unique(sorted=False, return_inverse=True)
        item_embeddings = self._build_item_embeddings(unique_items)
        interest_embeddings = item_embeddings.get("item_interest", item_embeddings["item"])
        return interest_embeddings[inverse].reshape(*recent_items.shape, -1)

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
