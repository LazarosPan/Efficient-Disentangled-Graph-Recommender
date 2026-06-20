"""Subgraph sampler for mini-batch GNN training.

Extracts a k-hop subgraph around seed nodes (batch users + items + negatives),
then rearranges nodes into a users-first layout compatible with DualBranchGCN.

When ``max_neighbors_per_hop`` is set, neighbours are sampled during the BFS
expansion rather than post-hoc.  This keeps subgraph sizes proportional to
the fan-out budget regardless of dataset density, which is critical for large
datasets such as MovieLens-20M where a full k-hop extraction from a batch of
4096 seeds spans essentially the entire 20M-edge graph.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch_geometric.utils import k_hop_subgraph


@dataclass
class SubgraphBatch:
    """Pre-processed subgraph ready for DualBranchGCN."""

    sub_edge_index: torch.Tensor  # (2, E_sub) — users-first layout
    sub_edge_sign: torch.Tensor | None  # (E_sub,) aligned signs, or None
    sub_edge_norm: torch.Tensor | None  # (E_sub,) full-graph degree norms, or None
    user_global_ids: torch.Tensor  # global user IDs in subgraph
    item_global_ids: torch.Tensor  # global item IDs (0-indexed, not offset)
    n_sub_users: int
    n_sub_items: int
    batch_user_local: torch.Tensor  # local indices for batch users
    batch_pos_local: torch.Tensor  # local indices for batch pos items
    batch_neg_local: torch.Tensor  # local indices for batch neg items
    dice_negative_mask: torch.Tensor | None = None  # (B,) high-pop DICE mask, if used
    stage_times_s: dict[str, float] | None = None

    def _map_tensors(
        self,
        fn: Callable[[torch.Tensor | None], torch.Tensor | None],
    ) -> SubgraphBatch:
        """Return a new SubgraphBatch with *fn* applied to every tensor field.

        Args:
            fn: Callable applied to each ``Tensor | None`` field; scalar fields
                are forwarded unchanged.

        Returns:
            New SubgraphBatch with transformed tensors.

        """
        return SubgraphBatch(
            sub_edge_index=fn(self.sub_edge_index),  # type: ignore[arg-type]
            sub_edge_sign=fn(self.sub_edge_sign),
            sub_edge_norm=fn(self.sub_edge_norm),
            user_global_ids=fn(self.user_global_ids),  # type: ignore[arg-type]
            item_global_ids=fn(self.item_global_ids),  # type: ignore[arg-type]
            n_sub_users=self.n_sub_users,
            n_sub_items=self.n_sub_items,
            batch_user_local=fn(self.batch_user_local),  # type: ignore[arg-type]
            batch_pos_local=fn(self.batch_pos_local),  # type: ignore[arg-type]
            batch_neg_local=fn(self.batch_neg_local),  # type: ignore[arg-type]
            dice_negative_mask=fn(self.dice_negative_mask),
            stage_times_s=self.stage_times_s,
        )

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> SubgraphBatch:
        """Return a new SubgraphBatch with all tensors moved to *device*.

        Args:
            device: Target device (e.g. ``torch.device("cuda:0")``).
            non_blocking: When True the host→device DMA transfer is issued
                asynchronously so the caller can overlap it with GPU work.

        Returns:
            New SubgraphBatch on *device*.

        """
        return self._map_tensors(
            lambda x: x.to(device, non_blocking=non_blocking) if x is not None else None,
        )

    def pin_memory(self) -> SubgraphBatch:
        """Return a new SubgraphBatch with all CPU tensors in pinned memory.

        Pinned host memory enables asynchronous DMA transfers when combined
        with ``to(device, non_blocking=True)``, allowing the host→device copy
        to overlap with GPU compute on the previous batch.

        Returns:
            New SubgraphBatch with pinned tensors.

        """
        return self._map_tensors(lambda x: x.pin_memory() if x is not None else None)


class SubgraphSampler:
    """Extract k-hop subgraphs for mini-batch training.

    When ``max_neighbors_per_hop`` is provided the sampler uses a vectorised
    sampled-BFS that applies per-hop fan-out *during* expansion.  For each hop
    a compressed sparse row adjacency precomputed at construction time lets
    frontier nodes be processed with bounded random row offsets, so memory
    scales with the fan-out budget rather than the frontier's total incident
    degree.

    Without ``max_neighbors_per_hop`` the original PyG ``k_hop_subgraph`` path
    is used unchanged (full k-hop neighbourhood, no sampling).

    Args:
        edge_index: (2, E) full graph edges (both directions already present).
        edge_sign: (E,) aligned edge signs, or None.
        n_users: Number of user nodes (IDs in [0, n_users)).
        n_items: Number of item nodes (IDs in [n_users, n_users + n_items)).
        num_hops: Number of GNN layers (k-hop neighbourhood).
        max_neighbors_per_hop: Per-hop fan-out limits (list of length num_hops).
            None means no fan-out limit (use full k-hop subgraph).

    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_sign: torch.Tensor | None,
        n_users: int,
        n_items: int,
        num_hops: int,
        max_neighbors_per_hop: list[int] | None = None,
        edge_norm: torch.Tensor | None = None,
    ) -> None:
        self.edge_index = edge_index
        self.edge_sign = edge_sign
        self.edge_norm = edge_norm
        self.n_users = n_users
        self.n_items = n_items
        self.num_hops = num_hops
        self.max_neighbors_per_hop = max_neighbors_per_hop
        self._num_nodes = n_users + n_items

        if max_neighbors_per_hop is not None:
            self._precompute_compressed_sparse_row_adjacency()

    # ------------------------------------------------------------------
    # Compressed sparse row adjacency (used by sampled-BFS path)
    # ------------------------------------------------------------------

    def _precompute_compressed_sparse_row_adjacency(self) -> None:
        """Build a compressed sparse row adjacency indexed by source node.

        ``_compressed_sparse_row_offsets[u] : _compressed_sparse_row_offsets[u+1]``
        is the range of positions in ``_compressed_sparse_neighbor_nodes`` /
        ``_compressed_sparse_original_edge_ids`` that describe the neighbours of
        node u (outgoing edges from u in the undirected bipartite graph).
        """
        device = self.edge_index.device
        num_edges = self.edge_index.size(1)
        src = self.edge_index[0]

        sort_order = src.argsort()
        self._compressed_sparse_neighbor_nodes = self.edge_index[1][
            sort_order
        ]  # (num_edges,) destination nodes
        self._compressed_sparse_original_edge_ids = (
            sort_order  # (num_edges,) original edge indices
        )

        row_offsets = torch.zeros(self._num_nodes + 1, dtype=torch.long, device=device)
        row_offsets.scatter_add_(
            0,
            src + 1,
            torch.ones(num_edges, dtype=torch.long, device=device),
        )
        row_offsets.cumsum_(0)
        self._compressed_sparse_row_offsets = row_offsets  # (N+1,)

    # ------------------------------------------------------------------
    # Sampled-BFS path
    # ------------------------------------------------------------------

    def _sample_one_hop(
        self,
        frontier: torch.Tensor,
        max_neighbors: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample up to ``max_neighbors`` neighbours for every node in ``frontier``.

        All frontier nodes are processed simultaneously with bounded compressed
        sparse row offset gathers; there is no Python loop over nodes and no
        sort over the full incident edge set.

        Args:
            frontier: (F,) global node IDs to expand from.
            max_neighbors: Maximum neighbours per frontier node.

        Returns:
            Tuple of ``(sampled_edge_orig_ids, sampled_neighbor_nodes)``
            where both tensors are 1-D and contain the kept edges / the
            neighbour endpoints (with duplicates; deduplicated by the caller).

        """
        device = frontier.device
        if frontier.numel() == 0:
            empty = frontier.new_empty(0)
            return empty, empty

        row_start_offsets = self._compressed_sparse_row_offsets[frontier]  # (F,)
        row_end_offsets = self._compressed_sparse_row_offsets[frontier + 1]  # (F,)
        degrees = row_end_offsets - row_start_offsets  # (F,)

        has_neighbors = degrees > 0
        if not has_neighbors.any():
            empty = frontier.new_empty(0)
            return empty, empty

        active_starts = row_start_offsets[has_neighbors]
        active_degrees = degrees[has_neighbors]
        active_node_count = active_starts.size(0)
        if active_degrees.device.type == "cuda":
            fanout_width = max_neighbors
        else:
            fanout_width = min(max_neighbors, int(active_degrees.max().item()))

        offsets = torch.arange(fanout_width, device=device).expand(
            active_node_count,
            fanout_width,
        )
        keep = offsets < active_degrees.unsqueeze(1)
        oversubscribed_nodes = active_degrees > fanout_width
        if oversubscribed_nodes.any():
            random_offsets = torch.floor(
                torch.rand(
                    (active_node_count, fanout_width),
                    device=device,
                    generator=generator,
                )
                * active_degrees.unsqueeze(1),
            ).to(dtype=torch.long)
            offsets = torch.where(
                oversubscribed_nodes.unsqueeze(1),
                random_offsets,
                offsets,
            )

        flattened_sparse_positions = (active_starts.unsqueeze(1) + offsets)[keep]
        sampled_edge_ids = self._compressed_sparse_original_edge_ids[
            flattened_sparse_positions
        ]
        sampled_neighbors = self._compressed_sparse_neighbor_nodes[
            flattened_sparse_positions
        ]

        return sampled_edge_ids, sampled_neighbors

    def _sampled_k_hop(
        self,
        seed_nodes: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sampled BFS from ``seed_nodes`` with per-hop fan-out limits.

        Returns:
            ``(all_nodes, subgraph_edge_index, all_edge_orig_ids)`` where
            ``all_nodes`` are global node IDs in discovery order,
            ``subgraph_edge_index`` is the (2, E) edge index with local node
            indices, and
            ``all_edge_orig_ids`` are the original edge indices (for sign lookup).

        """
        assert self.max_neighbors_per_hop is not None
        device = seed_nodes.device

        in_subgraph = torch.zeros(self._num_nodes, dtype=torch.bool, device=device)
        in_subgraph[seed_nodes] = True
        frontier = seed_nodes.unique()

        all_edge_orig_ids: list[torch.Tensor] = []

        for max_neighbors_for_hop in self.max_neighbors_per_hop:
            edge_ids, neighbors = self._sample_one_hop(
                frontier,
                max_neighbors_for_hop,
                generator=generator,
            )
            if edge_ids.numel() > 0:
                all_edge_orig_ids.append(edge_ids)
            # New nodes to expand from in the next hop
            if neighbors.numel() > 0:
                is_new = ~in_subgraph[neighbors]
                new_nodes = neighbors[is_new]
                if new_nodes.numel() > 0:
                    new_nodes = new_nodes.unique()
                    in_subgraph[new_nodes] = True
                    frontier = new_nodes
                else:
                    frontier = frontier.new_empty(0)
            else:
                frontier = frontier.new_empty(0)

        all_nodes = in_subgraph.nonzero(as_tuple=True)[0]

        if all_edge_orig_ids:
            combined_edge_ids = torch.cat(all_edge_orig_ids)
            all_edge_ids = (
                combined_edge_ids.unique()
                if combined_edge_ids.numel() > 0
                else combined_edge_ids
            )
        else:
            all_edge_ids = seed_nodes.new_empty(0)

        # Relabel nodes to local indices
        node_to_local = torch.full(
            (self._num_nodes,),
            -1,
            dtype=torch.long,
            device=device,
        )
        node_to_local[all_nodes] = torch.arange(all_nodes.size(0), device=device)

        if all_edge_ids.numel() > 0:
            original_source_nodes = self.edge_index[0][all_edge_ids]
            original_destination_nodes = self.edge_index[1][all_edge_ids]
            subgraph_edge_index = torch.stack(
                [
                    node_to_local[original_source_nodes],
                    node_to_local[original_destination_nodes],
                ],
                dim=0,
            )
        else:
            subgraph_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        return all_nodes, subgraph_edge_index, all_edge_ids

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_users: torch.Tensor,
        batch_pos_items: torch.Tensor,
        batch_neg_items: torch.Tensor,
        generator: torch.Generator | None = None,
        dice_negative_mask: torch.Tensor | None = None,
    ) -> SubgraphBatch:
        """Extract a subgraph around batch seed nodes.

        Args:
            batch_users: (B,) user IDs (global, in [0, n_users)).
            batch_pos_items: (B,) positive item IDs (0-indexed, NOT offset).
            batch_neg_items: (B,) negative item IDs (0-indexed, NOT offset).
            generator: Optional deterministic RNG for sampled fan-out.
            dice_negative_mask: Optional DICE high-popularity mask aligned to
                ``batch_neg_items``.

        Returns:
            SubgraphBatch with users-first layout and local indices.

        """
        positive_item_global_nodes = batch_pos_items + self.n_users
        negative_item_global_nodes = batch_neg_items + self.n_users
        seed_nodes = torch.cat(
            [batch_users, positive_item_global_nodes, negative_item_global_nodes],
        ).unique()

        if self.max_neighbors_per_hop is not None:
            # Sampled BFS: O(fan-out budget) regardless of dataset density.
            subset, subgraph_edge_index, all_edge_ids = self._sampled_k_hop(
                seed_nodes,
                generator=generator,
            )
            subgraph_edge_sign = (
                self.edge_sign[all_edge_ids]
                if self.edge_sign is not None and all_edge_ids.numel() > 0
                else None
            )
            subgraph_edge_norm = (
                self.edge_norm[all_edge_ids]
                if self.edge_norm is not None and all_edge_ids.numel() > 0
                else None
            )
        else:
            # Full k-hop subgraph (original path, no sampling).
            subset, subgraph_edge_index, _mapping, edge_mask = k_hop_subgraph(
                seed_nodes.to(self.edge_index.device),
                self.num_hops,
                self.edge_index,
                relabel_nodes=True,
                num_nodes=self._num_nodes,
            )
            subgraph_edge_sign = (
                self.edge_sign[edge_mask] if self.edge_sign is not None else None
            )
            subgraph_edge_norm = (
                self.edge_norm[edge_mask] if self.edge_norm is not None else None
            )

        # Separate users and items; rearrange to users-first layout.
        is_user = subset < self.n_users
        user_positions = is_user.nonzero(as_tuple=True)[0]
        item_positions = (~is_user).nonzero(as_tuple=True)[0]

        perm = torch.cat([user_positions, item_positions])
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm), device=perm.device)
        reordered_subgraph_edge_index = inv_perm[subgraph_edge_index]

        user_global_ids = subset[user_positions]
        item_global_ids = subset[item_positions] - self.n_users
        sorted_user_ids, user_sort_idx = self._sorted_local_index(user_global_ids)
        sorted_item_ids, item_sort_idx = self._sorted_local_index(item_global_ids)

        return SubgraphBatch(
            sub_edge_index=reordered_subgraph_edge_index,
            sub_edge_sign=subgraph_edge_sign,
            sub_edge_norm=subgraph_edge_norm,
            user_global_ids=user_global_ids,
            item_global_ids=item_global_ids,
            n_sub_users=user_positions.size(0),
            n_sub_items=item_positions.size(0),
            batch_user_local=self._map_to_local(
                batch_users,
                sorted_ids=sorted_user_ids,
                sort_idx=user_sort_idx,
            ),
            batch_pos_local=self._map_to_local(
                batch_pos_items,
                sorted_ids=sorted_item_ids,
                sort_idx=item_sort_idx,
            ),
            batch_neg_local=self._map_to_local(
                batch_neg_items,
                sorted_ids=sorted_item_ids,
                sort_idx=item_sort_idx,
            ),
            dice_negative_mask=dice_negative_mask,
        )

    @staticmethod
    def _sorted_local_index(
        subgraph_global_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one reusable sorted index for a subgraph-local ID space.

        Args:
            subgraph_global_ids: Global IDs present in one local node partition.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Sorted IDs and the permutation back
            to local-order indices.

        """
        return subgraph_global_ids.sort()

    @staticmethod
    def _map_to_local(
        global_ids: torch.Tensor,
        *,
        sorted_ids: torch.Tensor,
        sort_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Map global IDs to local indices within the subgraph.

        Uses a searchsorted-based approach for efficiency.

        Args:
            global_ids: Global IDs to remap.
            sorted_ids: Sorted subgraph IDs from ``_sorted_local_index``.
            sort_idx: Permutation from sorted order back to local order.

        Returns:
            torch.Tensor: Local indices for ``global_ids``.

        """
        positions = torch.searchsorted(sorted_ids, global_ids)
        positions = positions.clamp(max=sort_idx.size(0) - 1)
        return sort_idx[positions]
