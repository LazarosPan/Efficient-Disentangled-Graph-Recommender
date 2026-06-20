"""Propagation layer: LightGCN with optional dual-branch and sign-aware weighting."""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.nn import functional

from ..utils.config import EDGRecConfig

_EDGE_PROPAGATION_CHUNK_BYTES = 64 * 1024 * 1024


class LightGCNBranch(nn.Module):
    """Multi-layer LightGCN with alpha-weighted layer combination (He et al., 2020).

    Uses repeated graph propagation with alpha-weighted layer averaging.
    The legacy ``forward`` path accepts a prebuilt sparse adjacency for paper
    baselines, while ``forward_edges`` propagates directly from edge lists in
    bounded chunks to avoid per-batch sparse COO coalescing workspaces during
    sampled EDGRec training.
    """

    def __init__(self, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
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
        sparse_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate one embedding table through repeated sparse matmuls.

        Args:
            x: Node embeddings with shape ``(num_nodes, embed_dim)``.
            sparse_adjacency: Sparse normalized adjacency matrix with shape
                ``(num_nodes, num_nodes)``.

        Returns:
            torch.Tensor: Alpha-averaged LightGCN embeddings.

        """
        compute_dtype = torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype
        x_work = x.to(dtype=compute_dtype)
        sparse_adjacency_work = self._cast_sparse_values(sparse_adjacency, compute_dtype)

        with torch.autocast(device_type=x.device.type, enabled=False):
            out = x_work * self.alpha[0].to(dtype=compute_dtype)
            for i in range(self.n_layers):
                x_work = torch.sparse.mm(sparse_adjacency_work, x_work)
                if self.dropout > 0:
                    x_work = functional.dropout(x_work, p=self.dropout, training=self.training)
                out = out + x_work * self.alpha[i + 1].to(dtype=compute_dtype)

        return out.to(dtype=x.dtype) if out.dtype != x.dtype else out

    def forward_edges(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        *,
        num_nodes: int,
    ) -> torch.Tensor:
        """Propagate from COO edge lists without constructing sparse tensors.

        Args:
            x: Node embeddings with shape ``(num_nodes, embed_dim)``.
            edge_index: Directed graph connectivity with shape ``(2, E)``.
            edge_weight: Optional edge weights aligned to ``edge_index``.
            num_nodes: Number of graph nodes.

        Returns:
            torch.Tensor: Alpha-averaged LightGCN embeddings.

        """
        compute_dtype = torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype
        x_work = x.to(dtype=compute_dtype)
        weight_work = edge_weight.to(dtype=compute_dtype) if edge_weight is not None else None

        with torch.autocast(device_type=x.device.type, enabled=False):
            out = x_work * self.alpha[0].to(dtype=compute_dtype)
            for i in range(self.n_layers):
                x_work = self._edge_index_matmul(
                    x_work,
                    edge_index,
                    weight_work,
                    num_nodes=num_nodes,
                )
                if self.dropout > 0:
                    x_work = functional.dropout(x_work, p=self.dropout, training=self.training)
                out = out + x_work * self.alpha[i + 1].to(dtype=compute_dtype)

        return out.to(dtype=x.dtype) if out.dtype != x.dtype else out

    @staticmethod
    def _edge_index_matmul(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        *,
        num_nodes: int,
    ) -> torch.Tensor:
        """Return sparse adjacency propagation using bounded edge chunks."""
        out = x.new_zeros((num_nodes, x.size(-1)))
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return out

        element_size = max(1, x.element_size())
        chunk_size = max(1, _EDGE_PROPAGATION_CHUNK_BYTES // (x.size(-1) * element_size))
        row, col = edge_index
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            src_emb = x[col[start:end]]
            if edge_weight is not None:
                src_emb = src_emb * edge_weight[start:end].unsqueeze(-1)
            out.index_add_(0, row[start:end], src_emb)
        return out

    @staticmethod
    def _cast_sparse_values(
        sparse_adjacency: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return sparse adjacency values cast to ``dtype`` when needed.

        Args:
            sparse_adjacency: Sparse adjacency matrix.
            dtype: Target value dtype for sparse matmul.

        Returns:
            torch.Tensor: Sparse adjacency with values in ``dtype``.

        """
        if sparse_adjacency.dtype == dtype:
            return sparse_adjacency
        is_coalesced = sparse_adjacency.is_coalesced()
        indices = sparse_adjacency.indices() if is_coalesced else sparse_adjacency._indices()
        values = sparse_adjacency.values() if is_coalesced else sparse_adjacency._values()
        cast_sparse_adjacency = torch.sparse_coo_tensor(
            indices,
            values.to(dtype=dtype),
            size=sparse_adjacency.size(),
            device=sparse_adjacency.device,
            dtype=dtype,
        )
        return cast_sparse_adjacency.coalesce() if is_coalesced else cast_sparse_adjacency


class DualBranchGCN(nn.Module):
    """Propagation layer: LightGCN with optional dual-branch and sign-aware edge weights.

    - ``use_dual_branch=True``: two separate GCN branches (interest, conformity)
    - ``use_sign_aware=True``: learnable alpha_pos/alpha_neg scalars for edge weighting
    """

    def __init__(self, config: EDGRecConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_dual_branch:
            self.interest_branch = LightGCNBranch(config.interest_gnn_layers, config.dropout)
            self.conformity_branch = LightGCNBranch(config.conformity_gnn_layers, config.dropout)
        else:
            self.single_branch = LightGCNBranch(config.single_branch_gnn_layers, config.dropout)

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
        if self.config.profile_training_stages:
            self.last_stage_times_s = {}

        # Cast edge_weight once to match embedding dtype (prevents AMP scatter mismatch).
        # All branches use the same dtype since all embedding tables share initialization.
        embedding_dtype = next(iter(embeddings.values())).dtype
        edge_weight_for_propagation = (
            edge_weight.to(dtype=embedding_dtype) if edge_weight is not None else None
        )
        num_nodes = n_users + n_items
        use_cuda_sparse_adjacency = (
            edge_index.device.type == "cuda"
            and self.config.propagation_backend
            in {
                "auto",
                "cuda_sparse_adjacency",
            }
        )
        sparse_adjacency = None
        if use_cuda_sparse_adjacency:
            adjacency_start = time.perf_counter()
            sparse_adjacency = self._build_sparse_adjacency(
                edge_index,
                edge_weight_for_propagation,
                num_nodes=num_nodes,
                dtype=embedding_dtype,
                coalesce=False,
            )
            if self.config.profile_training_stages:
                self.last_stage_times_s["adjacency"] = time.perf_counter() - adjacency_start

        out: dict[str, torch.Tensor] = {}
        propagation_start = time.perf_counter() if self.config.profile_training_stages else None

        if self.config.use_dual_branch:
            item_interest = embeddings.get("item_interest", embeddings["item"])
            item_conformity = embeddings.get("item_conformity", embeddings["item"])
            x_int = torch.cat([embeddings["user_interest"], item_interest], dim=0)
            h_int = (
                self.interest_branch(x_int, sparse_adjacency)
                if sparse_adjacency is not None
                else self.interest_branch.forward_edges(
                    x_int,
                    edge_index,
                    edge_weight_for_propagation,
                    num_nodes=num_nodes,
                )
            )
            out["user_interest"] = h_int[:n_users]
            out["item_interest"] = h_int[n_users:]

            x_conf = torch.cat([embeddings["user_conformity"], item_conformity], dim=0)
            h_conf = (
                self.conformity_branch(x_conf, sparse_adjacency)
                if sparse_adjacency is not None
                else self.conformity_branch.forward_edges(
                    x_conf,
                    edge_index,
                    edge_weight_for_propagation,
                    num_nodes=num_nodes,
                )
            )
            out["user_conformity"] = h_conf[:n_users]
            out["item_conformity"] = h_conf[n_users:]
            out["item"] = 0.5 * (out["item_interest"] + out["item_conformity"])
        else:
            x = torch.cat([embeddings["user"], embeddings["item"]], dim=0)
            h = (
                self.single_branch(x, sparse_adjacency)
                if sparse_adjacency is not None
                else self.single_branch.forward_edges(
                    x,
                    edge_index,
                    edge_weight_for_propagation,
                    num_nodes=num_nodes,
                )
            )
            out["user"] = h[:n_users]
            out["item"] = h[n_users:]

        if propagation_start is not None:
            self.last_stage_times_s["propagation"] = time.perf_counter() - propagation_start
        return out

    @staticmethod
    def _build_sparse_adjacency(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        *,
        num_nodes: int,
        dtype: torch.dtype,
        coalesce: bool = True,
    ) -> torch.Tensor:
        """Build a sparse adjacency tensor for LightGCN propagation.

        Args:
            edge_index: Graph connectivity with shape ``(2, E)``.
            edge_weight: Optional normalized edge weights aligned to
                ``edge_index``.
            num_nodes: Total number of nodes in the bipartite graph.
            dtype: Propagation dtype for adjacency values.
            coalesce: Whether to coalesce the sparse tensor. EDGRec's
                CUDA sampled-subgraph path leaves this False to avoid a large
                per-batch sparse coalescing workspace.

        Returns:
            torch.Tensor: PyTorch sparse adjacency matrix.

        """
        values = (
            edge_weight
            if edge_weight is not None
            else torch.ones(edge_index.size(1), device=edge_index.device, dtype=dtype)
        )
        sparse_adjacency = torch.sparse_coo_tensor(
            edge_index,
            values,
            size=(num_nodes, num_nodes),
            device=edge_index.device,
            dtype=dtype,
        )
        return sparse_adjacency.coalesce() if coalesce else sparse_adjacency

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
        if not bool(neg_mask.any().item()):
            return (pos_mask | neutral_mask).to(edge_sign.dtype)

        pos_weight = pos_mask.to(edge_sign.dtype)
        neutral_weight = neutral_mask.to(edge_sign.dtype)
        negative_ratio = (self.alpha_neg / self.alpha_pos.clamp_min(1e-6)).clamp(
            min=0.0,
            max=1.0,
        )
        signed_weight = pos_weight + negative_ratio * neg_mask.to(edge_sign.dtype) + neutral_weight

        return signed_weight
