"""Propagation layer: LightGCN with optional dual-branch and sign-aware weighting."""

from __future__ import annotations

import time
from collections.abc import Callable

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
    bounded chunks to avoid per-batch sparse-adjacency coalescing workspaces during
    sampled EDGRec training.
    """

    def __init__(self, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        alpha = 1.0 / (n_layers + 1)
        self.register_buffer("alpha", torch.full((n_layers + 1,), alpha))

    def forward(
        self,
        node_embeddings: torch.Tensor,
        sparse_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate one embedding table through repeated sparse matmuls.

        Args:
            node_embeddings: Node embeddings with shape ``(num_nodes, embed_dim)``.
            sparse_adjacency: Sparse normalized adjacency matrix with shape
                ``(num_nodes, num_nodes)``.

        Returns:
            torch.Tensor: Alpha-averaged LightGCN embeddings.

        """
        compute_dtype = self._propagation_compute_dtype(node_embeddings)
        sparse_adjacency_for_propagation = self._cast_sparse_adjacency_values(
            sparse_adjacency,
            compute_dtype,
        )
        return self._propagate_layers(
            node_embeddings,
            compute_dtype=compute_dtype,
            propagate_next_layer=lambda current_embeddings: torch.sparse.mm(
                sparse_adjacency_for_propagation,
                current_embeddings,
            ),
        )

    def forward_edges(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        *,
        num_nodes: int,
    ) -> torch.Tensor:
        """Propagate from edge-index lists without constructing sparse tensors.

        Args:
            node_embeddings: Node embeddings with shape ``(num_nodes, embed_dim)``.
            edge_index: Directed graph connectivity with shape ``(2, E)``.
            edge_weight: Optional edge weights aligned to ``edge_index``.
            num_nodes: Number of graph nodes.

        Returns:
            torch.Tensor: Alpha-averaged LightGCN embeddings.

        """
        compute_dtype = self._propagation_compute_dtype(node_embeddings)
        edge_weight_for_propagation = (
            edge_weight.to(dtype=compute_dtype) if edge_weight is not None else None
        )
        return self._propagate_layers(
            node_embeddings,
            compute_dtype=compute_dtype,
            propagate_next_layer=lambda current_embeddings: self._propagate_edge_index_chunks(
                current_embeddings,
                edge_index,
                edge_weight_for_propagation,
                num_nodes=num_nodes,
            ),
        )

    @staticmethod
    def _propagation_compute_dtype(node_embeddings: torch.Tensor) -> torch.dtype:
        """Return the dtype used for numerically stable propagation work."""
        return (
            torch.float32
            if node_embeddings.dtype in {torch.float16, torch.bfloat16}
            else node_embeddings.dtype
        )

    def _propagate_layers(
        self,
        node_embeddings: torch.Tensor,
        *,
        compute_dtype: torch.dtype,
        propagate_next_layer: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Return alpha-weighted LightGCN propagation for one backend."""
        propagated_embeddings = node_embeddings.to(dtype=compute_dtype)
        with torch.autocast(device_type=node_embeddings.device.type, enabled=False):
            weighted_layer_sum = propagated_embeddings * self.alpha[0].to(dtype=compute_dtype)
            for layer_index in range(self.n_layers):
                propagated_embeddings = propagate_next_layer(propagated_embeddings)
                if self.dropout > 0:
                    propagated_embeddings = functional.dropout(
                        propagated_embeddings,
                        p=self.dropout,
                        training=self.training,
                    )
                weighted_layer_sum = weighted_layer_sum + propagated_embeddings * self.alpha[
                    layer_index + 1
                ].to(dtype=compute_dtype)

        return (
            weighted_layer_sum.to(dtype=node_embeddings.dtype)
            if weighted_layer_sum.dtype != node_embeddings.dtype
            else weighted_layer_sum
        )

    @staticmethod
    def _propagate_edge_index_chunks(
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        *,
        num_nodes: int,
    ) -> torch.Tensor:
        """Return sparse adjacency propagation using bounded edge chunks."""
        next_layer_embeddings = node_embeddings.new_zeros(
            (num_nodes, node_embeddings.size(-1)),
        )
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return next_layer_embeddings

        element_size = max(1, node_embeddings.element_size())
        chunk_size = max(
            1,
            _EDGE_PROPAGATION_CHUNK_BYTES // (node_embeddings.size(-1) * element_size),
        )
        target_node_ids, source_node_ids = edge_index
        for chunk_start in range(0, num_edges, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_edges)
            source_node_embeddings = node_embeddings[source_node_ids[chunk_start:chunk_end]]
            if edge_weight is not None:
                source_node_embeddings = source_node_embeddings * edge_weight[
                    chunk_start:chunk_end
                ].unsqueeze(-1)
            next_layer_embeddings.index_add_(
                0,
                target_node_ids[chunk_start:chunk_end],
                source_node_embeddings,
            )
        return next_layer_embeddings

    @staticmethod
    def _cast_sparse_adjacency_values(
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
        adjacency_indices = (
            sparse_adjacency.indices() if is_coalesced else sparse_adjacency._indices()
        )
        adjacency_values = sparse_adjacency.values() if is_coalesced else sparse_adjacency._values()
        sparse_adjacency_with_cast_values = torch.sparse_coo_tensor(
            adjacency_indices,
            adjacency_values.to(dtype=dtype),
            size=sparse_adjacency.size(),
            device=sparse_adjacency.device,
            dtype=dtype,
        )
        return (
            sparse_adjacency_with_cast_values.coalesce()
            if is_coalesced
            else sparse_adjacency_with_cast_values
        )


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
        total_nodes = n_users + n_items
        use_cuda_sparse_adjacency = (
            edge_index.device.type == "cuda"
            and self.config.propagation_backend
            in {
                "auto",
                "cuda_sparse_adjacency",
            }
        )
        propagation_adjacency = None
        if use_cuda_sparse_adjacency:
            adjacency_start = time.perf_counter()
            propagation_adjacency = self._build_sparse_adjacency_tensor(
                edge_index,
                edge_weight_for_propagation,
                num_nodes=total_nodes,
                dtype=embedding_dtype,
                coalesce=False,
            )
            if self.config.profile_training_stages:
                self.last_stage_times_s["adjacency"] = time.perf_counter() - adjacency_start

        propagated_output: dict[str, torch.Tensor] = {}
        propagation_start = time.perf_counter() if self.config.profile_training_stages else None

        def propagate_branch_embeddings(
            branch: LightGCNBranch,
            branch_node_embeddings: torch.Tensor,
        ) -> torch.Tensor:
            if propagation_adjacency is not None:
                return branch(branch_node_embeddings, propagation_adjacency)
            return branch.forward_edges(
                branch_node_embeddings,
                edge_index,
                edge_weight_for_propagation,
                num_nodes=total_nodes,
            )

        if self.config.use_dual_branch:
            item_interest_embeddings = embeddings.get("item_interest", embeddings["item"])
            item_conformity_embeddings = embeddings.get("item_conformity", embeddings["item"])
            interest_node_embeddings = torch.cat(
                [embeddings["user_interest"], item_interest_embeddings],
                dim=0,
            )
            propagated_interest_embeddings = propagate_branch_embeddings(
                self.interest_branch,
                interest_node_embeddings,
            )
            propagated_output["user_interest"] = propagated_interest_embeddings[:n_users]
            propagated_output["item_interest"] = propagated_interest_embeddings[n_users:]

            conformity_node_embeddings = torch.cat(
                [embeddings["user_conformity"], item_conformity_embeddings],
                dim=0,
            )
            propagated_conformity_embeddings = propagate_branch_embeddings(
                self.conformity_branch,
                conformity_node_embeddings,
            )
            propagated_output["user_conformity"] = propagated_conformity_embeddings[:n_users]
            propagated_output["item_conformity"] = propagated_conformity_embeddings[n_users:]
            propagated_output["item"] = 0.5 * (
                propagated_output["item_interest"] + propagated_output["item_conformity"]
            )
        else:
            single_branch_node_embeddings = torch.cat(
                [embeddings["user"], embeddings["item"]],
                dim=0,
            )
            propagated_single_branch_embeddings = propagate_branch_embeddings(
                self.single_branch,
                single_branch_node_embeddings,
            )
            propagated_output["user"] = propagated_single_branch_embeddings[:n_users]
            propagated_output["item"] = propagated_single_branch_embeddings[n_users:]

        if propagation_start is not None:
            self.last_stage_times_s["propagation"] = time.perf_counter() - propagation_start
        return propagated_output

    @staticmethod
    def _build_sparse_adjacency_tensor(
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
        """Compute sign-aware edge weights around a unit baseline."""
        if not self.config.use_sign_aware:
            return None
        if edge_sign is None:
            return None

        positive_edge_mask = edge_sign > 0
        negative_edge_mask = edge_sign < 0
        neutral_edge_mask = ~(positive_edge_mask | negative_edge_mask)
        non_negative_edge_weight = (positive_edge_mask | neutral_edge_mask).to(edge_sign.dtype)
        negative_ratio = (self.alpha_neg / self.alpha_pos.clamp_min(1e-6)).clamp(
            min=0.0,
            max=1.0,
        )
        if torch.compiler.is_compiling() or edge_sign.device.type == "cuda":
            return non_negative_edge_weight + negative_ratio * negative_edge_mask.to(
                edge_sign.dtype,
            )

        if not bool(negative_edge_mask.any().item()):
            return non_negative_edge_weight

        signed_edge_weight = non_negative_edge_weight + negative_ratio * negative_edge_mask.to(
            edge_sign.dtype
        )

        return signed_edge_weight
