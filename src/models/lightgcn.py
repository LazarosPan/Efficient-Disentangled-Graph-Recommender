"""Module B: LightGCN propagation with optional dual-branch and sign-aware weighting."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn.conv import LGConv

from ..utils.config import UCaGNNConfig


class LightGCNBranch(nn.Module):
    """Multi-layer LightGCN with alpha-weighted layer combination (He et al., 2020).

    Uses PyG's ``LGConv`` for message passing — symmetric normalization,
    no learnable weights, optional edge_weight support.
    """

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        alpha = 1.0 / (n_layers + 1)
        self.register_buffer("alpha", torch.full((n_layers + 1,), alpha))
        self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = x * self.alpha[0]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]
        return out


class DualBranchGCN(nn.Module):
    """Module B: LightGCN with optional dual-branch and sign-aware edge weights.

    - ``use_dual_branch=True``: two separate GCN branches (interest, conformity)
    - ``use_sign_aware=True``: learnable alpha_pos/alpha_neg scalars for edge weighting
    """

    def __init__(self, config: UCaGNNConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_dual_branch:
            self.interest_branch = LightGCNBranch(config.resolved_interest_gnn_layers)
            self.conformity_branch = LightGCNBranch(
                config.resolved_conformity_gnn_layers
            )
        else:
            self.single_branch = LightGCNBranch(config.n_gnn_layers)

        if config.use_sign_aware:
            self.alpha_pos = nn.Parameter(torch.tensor(0.7))
            self.alpha_neg = nn.Parameter(torch.tensor(0.3))

    def reset_parameters(self) -> None:
        if self.config.use_dual_branch:
            self.interest_branch.reset_parameters()
            self.conformity_branch.reset_parameters()
        else:
            self.single_branch.reset_parameters()
        if self.config.use_sign_aware:
            self.alpha_pos.data.fill_(0.7)
            self.alpha_neg.data.fill_(0.3)

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
        n_users: int = 0,
        n_items: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Propagate embeddings through GCN branches.

        Returns dict with propagated user/item embeddings per branch.
        """
        edge_weight = self._compute_edge_weights(edge_sign, edge_index.size(1))

        out: dict[str, torch.Tensor] = {}

        if self.config.use_dual_branch:
            item_interest = embeddings.get("item_interest", embeddings["item"])
            item_conformity = embeddings.get("item_conformity", embeddings["item"])
            x_int = torch.cat([embeddings["user_interest"], item_interest], dim=0)
            h_int = self.interest_branch(x_int, edge_index, edge_weight)
            out["user_interest"] = h_int[:n_users]
            out["item_interest"] = h_int[n_users:]

            x_conf = torch.cat([embeddings["user_conformity"], item_conformity], dim=0)
            h_conf = self.conformity_branch(x_conf, edge_index, edge_weight)
            out["user_conformity"] = h_conf[:n_users]
            out["item_conformity"] = h_conf[n_users:]
        else:
            x = torch.cat([embeddings["user"], embeddings["item"]], dim=0)
            h = self.single_branch(x, edge_index, edge_weight)
            out["user"] = h[:n_users]
            out["item"] = h[n_users:]

        return out

    def _compute_edge_weights(
        self, edge_sign: torch.Tensor | None, n_edges: int
    ) -> torch.Tensor | None:
        if not self.config.use_sign_aware or edge_sign is None:
            return None

        assert edge_sign.size(0) == n_edges, (
            f"edge_sign length ({edge_sign.size(0)}) != n_edges ({n_edges}). "
            "This indicates a bug in edge construction upstream."
        )

        has_positive = bool((edge_sign > 0).any().item())
        has_negative = bool((edge_sign < 0).any().item())
        if not (has_positive and has_negative):
            return None

        pos_mask = (edge_sign > 0).float()
        neg_mask = (edge_sign < 0).float()
        neutral_mask = (edge_sign == 0).float()
        return self.alpha_pos * pos_mask + self.alpha_neg * neg_mask + neutral_mask
