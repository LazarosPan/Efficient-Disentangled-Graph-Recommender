"""Module B: LightGCN propagation with optional dual-branch and sign-aware weighting."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn.conv import LGConv

from ..utils.config import UCaGNNConfig


class LightGCNBranch(nn.Module):
    """Multi-layer LightGCN with alpha-weighted layer combination (He et al., 2020).

    Uses PyG's ``LGConv`` with ``normalize=False``; degree normalization is
    handled externally via pre-computed full-graph ``edge_weight`` so training
    (subgraph) and evaluation (full graph) use identical normalization factors.
    """

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        alpha = 1.0 / (n_layers + 1)
        self.register_buffer("alpha", torch.full((n_layers + 1,), alpha))
        self.convs = nn.ModuleList([LGConv(normalize=False) for _ in range(n_layers)])

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
        edge_norm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Propagate embeddings through GCN branches.

        Args:
            embeddings: Dict of initial user/item embeddings.
            edge_index: (2, E) graph edges.
            edge_sign: (E,) optional sign weights for sign-aware weighting.
            n_users: Number of user nodes in the graph.
            n_items: Number of item nodes in the graph.
            edge_norm: (E,) precomputed full-graph symmetric normalization
                ``1/sqrt(deg_u * deg_v)``.  When provided, LGConv operates
                with ``normalize=False`` and these weights serve as the sole
                normalization factor (optionally multiplied by sign weights).

        Returns:
            Dict with propagated user/item embeddings per branch.
        """
        edge_weight = self._compute_edge_weights(edge_sign, edge_norm)

        out: dict[str, torch.Tensor] = {}

        if self.config.use_dual_branch:
            item_interest = embeddings.get("item_interest", embeddings["item"])
            item_conformity = embeddings.get("item_conformity", embeddings["item"])
            x_int = torch.cat([embeddings["user_interest"], item_interest], dim=0)
            # Cast edge_weight to match input dtype (prevents AMP scatter mismatch)
            ew = edge_weight.to(dtype=x_int.dtype) if edge_weight is not None else None
            h_int = self.interest_branch(x_int, edge_index, ew)
            out["user_interest"] = h_int[:n_users]
            out["item_interest"] = h_int[n_users:]

            x_conf = torch.cat([embeddings["user_conformity"], item_conformity], dim=0)
            ew = edge_weight.to(dtype=x_conf.dtype) if edge_weight is not None else None
            h_conf = self.conformity_branch(x_conf, edge_index, ew)
            out["user_conformity"] = h_conf[:n_users]
            out["item_conformity"] = h_conf[n_users:]
        else:
            x = torch.cat([embeddings["user"], embeddings["item"]], dim=0)
            ew = edge_weight.to(dtype=x.dtype) if edge_weight is not None else None
            h = self.single_branch(x, edge_index, ew)
            out["user"] = h[:n_users]
            out["item"] = h[n_users:]

        return out

    def _compute_edge_weights(
        self,
        edge_sign: torch.Tensor | None,
        edge_norm: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return the combined sign-aware and degree-normalization weights."""
        return self._combine_weights(
            self._compute_edge_weights_impl(edge_sign), edge_norm
        )

    @staticmethod
    def _combine_weights(
        sign_weight: torch.Tensor | None,
        edge_norm: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Combine optional sign weight and degree normalization into one tensor.

        Returns:
            ``sign_weight * edge_norm`` if both are present,
            whichever is non-None if only one is given,
            or None if both are None.
        """
        if sign_weight is not None and edge_norm is not None:
            return sign_weight * edge_norm.to(dtype=sign_weight.dtype)
        return sign_weight if sign_weight is not None else edge_norm

    def _compute_edge_weights_impl(
        self, edge_sign: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Compute compile-safe sign-aware edge weights around a unit baseline."""
        if not self.config.use_sign_aware:
            return None
        if edge_sign is None:
            return None

        pos_mask = edge_sign > 0
        neg_mask = edge_sign < 0
        neutral_mask = ~(pos_mask | neg_mask)

        pos_weight = pos_mask.to(edge_sign.dtype)
        neutral_weight = neutral_mask.to(edge_sign.dtype)
        negative_ratio = (self.alpha_neg / self.alpha_pos.clamp_min(1e-6)).clamp(
            min=0.0,
            max=1.0,
        )
        signed_weight = (
            pos_weight + negative_ratio * neg_mask.to(edge_sign.dtype) + neutral_weight
        )

        # When a graph contains only one interaction sign, keep the standard
        # LightGCN weighting path instead of injecting a synthetic asymmetry.
        has_mixed_sign = (pos_mask.any() & neg_mask.any()).to(edge_sign.dtype)
        baseline = torch.ones_like(edge_sign)
        return baseline + has_mixed_sign * (signed_weight - baseline)
