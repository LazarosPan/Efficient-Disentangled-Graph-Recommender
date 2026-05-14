"""Module B: LightGCN propagation with optional dual-branch and sign-aware weighting."""

from __future__ import annotations

import torch
from torch import nn

from ..utils.config import UCaGNNConfig


class LightGCNBranch(nn.Module):
    """Multi-layer LightGCN with alpha-weighted layer combination (He et al., 2020).

    Uses repeated sparse adjacency matmuls; degree normalization is handled
    externally via pre-computed full-graph ``edge_weight`` so training
    (subgraph) and evaluation (full graph) use identical normalization factors
    without materializing edge-wise gather-scatter messages.
    """

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        alpha = 1.0 / (n_layers + 1)
        self.register_buffer("alpha", torch.full((n_layers + 1,), alpha))

    def reset_parameters(self) -> None:
        """Reset module state.

        Returns:
            None. The branch has no learnable propagation parameters.

        """
        return None

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate one embedding table through repeated sparse matmuls.

        Args:
            x: Node embeddings with shape ``(num_nodes, embed_dim)``.
            adj: Sparse normalized adjacency matrix with shape
                ``(num_nodes, num_nodes)``.

        Returns:
            torch.Tensor: Alpha-averaged LightGCN embeddings.

        """
        compute_dtype = torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype
        x_work = x.to(dtype=compute_dtype)
        adj_work = self._cast_sparse_values(adj, compute_dtype)

        with torch.autocast(device_type=x.device.type, enabled=False):
            out = x_work * self.alpha[0].to(dtype=compute_dtype)
            for i in range(self.n_layers):
                x_work = torch.sparse.mm(adj_work, x_work)
                out = out + x_work * self.alpha[i + 1].to(dtype=compute_dtype)

        return out.to(dtype=x.dtype) if out.dtype != x.dtype else out

    @staticmethod
    def _cast_sparse_values(
        adj: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``adj`` with sparse values cast to ``dtype`` when needed.

        Args:
            adj: COO sparse adjacency matrix.
            dtype: Target value dtype for sparse matmul.

        Returns:
            torch.Tensor: Sparse adjacency with values in ``dtype``.

        """
        if adj.dtype == dtype:
            return adj
        return torch.sparse_coo_tensor(
            adj.indices(),
            adj.values().to(dtype=dtype),
            size=adj.size(),
            device=adj.device,
            dtype=dtype,
        ).coalesce()


class DualBranchGCN(nn.Module):
    """Module B: LightGCN with optional dual-branch and sign-aware edge weights.

    - ``use_dual_branch=True``: two separate GCN branches (interest, conformity)
    - ``use_sign_aware=True``: learnable alpha_pos/alpha_neg scalars for edge weighting
    """

    def __init__(self, config: UCaGNNConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_dual_branch:
            self.interest_branch = LightGCNBranch(config.interest_gnn_layers)
            self.conformity_branch = LightGCNBranch(config.conformity_gnn_layers)
        else:
            self.single_branch = LightGCNBranch(config.single_branch_gnn_layers)

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
        edge_weight = self._combine_weights(
            self._compute_edge_weights_impl(edge_sign),
            edge_norm,
        )

        # Cast edge_weight once to match embedding dtype (prevents AMP scatter mismatch).
        # All branches use the same dtype since all embedding tables share initialization.
        x_dtype = next(iter(embeddings.values())).dtype
        ew = edge_weight.to(dtype=x_dtype) if edge_weight is not None else None
        adj = self._build_sparse_adjacency(
            edge_index,
            ew,
            num_nodes=n_users + n_items,
            dtype=x_dtype,
        )

        out: dict[str, torch.Tensor] = {}

        if self.config.use_dual_branch:
            item_interest = embeddings.get("item_interest", embeddings["item"])
            item_conformity = embeddings.get("item_conformity", embeddings["item"])
            x_int = torch.cat([embeddings["user_interest"], item_interest], dim=0)
            h_int = self.interest_branch(x_int, adj)
            out["user_interest"] = h_int[:n_users]
            out["item_interest"] = h_int[n_users:]

            x_conf = torch.cat([embeddings["user_conformity"], item_conformity], dim=0)
            h_conf = self.conformity_branch(x_conf, adj)
            out["user_conformity"] = h_conf[:n_users]
            out["item_conformity"] = h_conf[n_users:]
        else:
            x = torch.cat([embeddings["user"], embeddings["item"]], dim=0)
            h = self.single_branch(x, adj)
            out["user"] = h[:n_users]
            out["item"] = h[n_users:]

        return out

    @staticmethod
    def _build_sparse_adjacency(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        *,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a sparse adjacency tensor for LightGCN propagation.

        Args:
            edge_index: Graph connectivity with shape ``(2, E)``.
            edge_weight: Optional normalized edge weights aligned to
                ``edge_index``.
            num_nodes: Total number of nodes in the bipartite graph.
            dtype: Propagation dtype for adjacency values.

        Returns:
            torch.Tensor: Coalesced COO sparse adjacency matrix.

        """
        values = (
            edge_weight
            if edge_weight is not None
            else torch.ones(edge_index.size(1), device=edge_index.device, dtype=dtype)
        )
        return torch.sparse_coo_tensor(
            edge_index,
            values,
            size=(num_nodes, num_nodes),
            device=edge_index.device,
            dtype=dtype,
        ).coalesce()

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
        self,
        edge_sign: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Compute compile-safe sign-aware edge weights around a unit baseline."""
        if not self.config.use_sign_aware:
            return None
        if edge_sign is None:
            return None

        pos_mask = edge_sign > 0
        neg_mask = edge_sign < 0
        neutral_mask = ~(pos_mask | neg_mask)

        if not bool(pos_mask.any() and neg_mask.any()):
            return torch.ones_like(edge_sign)

        pos_weight = pos_mask.to(edge_sign.dtype)
        neutral_weight = neutral_mask.to(edge_sign.dtype)
        negative_ratio = (self.alpha_neg / self.alpha_pos.clamp_min(1e-6)).clamp(
            min=0.0,
            max=1.0,
        )
        signed_weight = pos_weight + negative_ratio * neg_mask.to(edge_sign.dtype) + neutral_weight

        return signed_weight
