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
    ) -> None:
        super().__init__()
        self.config = config
        self.n_users = n_users
        self.n_items = n_items
        d = config.embed_dim
        self.has_item_features = bool(
            config.use_features and item_features is not None and item_features.numel() > 0
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
            assert item_popularity is not None
            self.register_buffer(
                "item_feature_matrix",
                item_features.float(),
                persistent=False,
            )
            self.register_buffer(
                "item_popularity",
                item_popularity.float(),
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

    def _get_item_base_embeddings(
        self,
        item_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if item_ids is None:
            item_embed = self.item_embed.weight
        else:
            item_embed = self.item_embed.weight[item_ids]

        if not self.has_item_features:
            return item_embed, None, None

        assert hasattr(self, "item_feature_matrix")
        assert hasattr(self, "item_popularity")
        if item_ids is None:
            item_features = self.item_feature_matrix
            popularity = self.item_popularity
        else:
            item_features = self.item_feature_matrix[item_ids]
            popularity = self.item_popularity[item_ids]

        projected = self.item_feature_norm(self.item_feature_proj(item_features))
        return item_embed, projected, popularity

    def _build_item_embeddings(
        self,
        item_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        item_embed, projected_features, popularity = self._get_item_base_embeddings(item_ids)
        if projected_features is None or popularity is None:
            return {"item": item_embed}

        interest_gate = torch.sigmoid(self.item_interest_gate)
        conformity_gate = torch.sigmoid(self.item_conformity_gate)
        popularity_gate = self.popularity_modulator(popularity.unsqueeze(-1))

        item_interest = item_embed + interest_gate * projected_features
        item_conformity = item_embed + conformity_gate * (projected_features * popularity_gate)
        return {
            "item": item_embed,
            "item_interest": item_interest,
            "item_conformity": item_conformity,
        }

    def get_all_embeddings(self) -> dict[str, torch.Tensor]:
        """Return a dict of all initial (pre-GNN) embeddings."""
        out: dict[str, torch.Tensor] = {}
        if self.config.use_dual_branch:
            out["user_interest"] = self.user_interest.weight
            out["user_conformity"] = self.user_conformity.weight
        else:
            out["user"] = self.user_embed.weight
        out.update(self._build_item_embeddings())
        if self.config.use_popularity_emb:
            out["item_pop"] = self.item_pop.weight
        return out

    def get_subgraph_embeddings(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return initial embeddings restricted to a subgraph node set."""
        out: dict[str, torch.Tensor] = {}
        if self.config.use_dual_branch:
            out["user_interest"] = self.user_interest.weight[user_ids]
            out["user_conformity"] = self.user_conformity.weight[user_ids]
        else:
            out["user"] = self.user_embed.weight[user_ids]
        out.update(self._build_item_embeddings(item_ids))
        if self.config.use_popularity_emb:
            out["item_pop"] = self.item_pop.weight[item_ids]
        return out

    def get_stacked_embeddings(self) -> torch.Tensor:
        """Return (n_users + n_items, D) combined node embeddings for graph building.

        Uses interest embeddings for users when dual_branch is active.
        """
        if self.config.use_dual_branch:
            user_emb = self.user_interest.weight
        else:
            user_emb = self.user_embed.weight
        item_embs = self._build_item_embeddings()
        item_emb = item_embs.get("item_interest", item_embs["item"])
        return torch.cat([user_emb, item_emb], dim=0)
