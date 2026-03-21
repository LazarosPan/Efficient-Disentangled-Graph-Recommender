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

    def __init__(self, n_users: int, n_items: int, config: UCaGNNConfig) -> None:
        super().__init__()
        self.config = config
        self.n_users = n_users
        self.n_items = n_items
        d = config.embed_dim

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

    def get_all_embeddings(self) -> dict[str, torch.Tensor]:
        """Return a dict of all initial (pre-GNN) embeddings."""
        out: dict[str, torch.Tensor] = {}
        if self.config.use_dual_branch:
            out["user_interest"] = self.user_interest.weight
            out["user_conformity"] = self.user_conformity.weight
        else:
            out["user"] = self.user_embed.weight
        out["item"] = self.item_embed.weight
        if self.config.use_popularity_emb:
            out["item_pop"] = self.item_pop.weight
        return out

    def get_stacked_embeddings(self) -> torch.Tensor:
        """Return (n_users + n_items, D) combined node embeddings for graph building.

        Uses interest embeddings for users when dual_branch is active.
        """
        if self.config.use_dual_branch:
            user_emb = self.user_interest.weight
        else:
            user_emb = self.user_embed.weight
        return torch.cat([user_emb, self.item_embed.weight], dim=0)
