"""Subgraph sampler for mini-batch GNN training.

Extracts a k-hop subgraph around seed nodes (batch users + items + negatives),
then rearranges nodes into a users-first layout compatible with DualBranchGCN.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.utils import k_hop_subgraph


@dataclass
class SubgraphBatch:
    """Pre-processed subgraph ready for DualBranchGCN."""

    sub_edge_index: torch.Tensor  # (2, E_sub) — users-first layout
    sub_edge_sign: torch.Tensor | None  # (E_sub,) aligned signs, or None
    user_global_ids: torch.Tensor  # global user IDs in subgraph
    item_global_ids: torch.Tensor  # global item IDs (0-indexed, not offset)
    n_sub_users: int
    n_sub_items: int
    batch_user_local: torch.Tensor  # local indices for batch users
    batch_pos_local: torch.Tensor  # local indices for batch pos items
    batch_neg_local: torch.Tensor  # local indices for batch neg items


class SubgraphSampler:
    """Extract k-hop subgraphs for mini-batch training.

    Uses PyG's ``k_hop_subgraph`` to extract a subgraph around seed nodes,
    then rearranges node ordering into users-first layout for DualBranchGCN.

    Optionally applies per-hop fan-out limits to control subgraph size.

    Args:
        edge_index: (2, E) full graph edges.
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
    ) -> None:
        self.edge_index = edge_index
        self.edge_sign = edge_sign
        self.n_users = n_users
        self.n_items = n_items
        self.num_hops = num_hops
        self.max_neighbors_per_hop = max_neighbors_per_hop
        self._num_nodes = n_users + n_items

    def sample(
        self,
        batch_users: torch.Tensor,
        batch_pos_items: torch.Tensor,
        batch_neg_items: torch.Tensor,
    ) -> SubgraphBatch:
        """Extract a subgraph around batch seed nodes.

        Args:
            batch_users: (B,) user IDs (global, in [0, n_users)).
            batch_pos_items: (B,) positive item IDs (0-indexed, NOT offset).
            batch_neg_items: (B,) negative item IDs (0-indexed, NOT offset).

        Returns:
            SubgraphBatch with users-first layout and local indices.
        """
        # Offset items to graph-global IDs
        pos_global = batch_pos_items + self.n_users
        neg_global = batch_neg_items + self.n_users

        # Collect unique seed nodes
        seed = torch.cat([batch_users, pos_global, neg_global]).unique()

        # Extract k-hop subgraph
        subset, sub_ei, mapping, edge_mask = k_hop_subgraph(
            seed.to(self.edge_index.device),
            self.num_hops,
            self.edge_index,
            relabel_nodes=True,
            num_nodes=self._num_nodes,
        )

        # Apply fan-out limits if configured
        if self.max_neighbors_per_hop is not None:
            sub_ei, edge_mask = self._apply_fanout(
                subset,
                sub_ei,
                edge_mask,
            )

        # Separate users and items in the subgraph
        is_user = subset < self.n_users
        user_positions = is_user.nonzero(as_tuple=True)[0]
        item_positions = (~is_user).nonzero(as_tuple=True)[0]

        # Build permutation: users first, then items
        perm = torch.cat([user_positions, item_positions])
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm), device=perm.device)

        # Remap edge_index to users-first layout
        sub_ei_reordered = inv_perm[sub_ei]

        # Extract aligned edge_sign
        sub_sign = self.edge_sign[edge_mask] if self.edge_sign is not None else None

        # Global IDs for users and items in the subgraph
        user_global_ids = subset[user_positions]
        item_global_ids = subset[item_positions] - self.n_users  # back to 0-indexed

        n_sub_users = user_positions.size(0)
        n_sub_items = item_positions.size(0)

        # Map batch user/item IDs to local indices in the users-first layout
        batch_user_local = self._map_to_local(
            batch_users,
            user_global_ids,
        )
        batch_pos_local = self._map_to_local(
            batch_pos_items,
            item_global_ids,
        )
        batch_neg_local = self._map_to_local(
            batch_neg_items,
            item_global_ids,
        )

        return SubgraphBatch(
            sub_edge_index=sub_ei_reordered,
            sub_edge_sign=sub_sign,
            user_global_ids=user_global_ids,
            item_global_ids=item_global_ids,
            n_sub_users=n_sub_users,
            n_sub_items=n_sub_items,
            batch_user_local=batch_user_local,
            batch_pos_local=batch_pos_local,
            batch_neg_local=batch_neg_local,
        )

    def _apply_fanout(
        self,
        subset: torch.Tensor,
        sub_ei: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-filter subgraph edges to respect per-hop fan-out limits.

        For each target node, randomly subsample incoming edges up to the
        fan-out limit for that hop.  Uses the last (outermost) hop's limit
        since k_hop_subgraph returns all neighbours at once.
        """
        max_nb = self.max_neighbors_per_hop[-1]  # use outermost-hop limit
        n_sub = subset.size(0)
        device = sub_ei.device

        targets = sub_ei[1]
        counts = torch.bincount(targets, minlength=n_sub)
        overcrowded = (counts > max_nb).nonzero(as_tuple=True)[0]

        if overcrowded.numel() == 0:
            return sub_ei, edge_mask

        keep_mask = torch.ones(sub_ei.size(1), dtype=torch.bool, device=device)
        for node_local in overcrowded:
            incoming = (targets == node_local).nonzero(as_tuple=True)[0]
            drop_idx = incoming[
                torch.randperm(incoming.size(0), device=device)[max_nb:]
            ]
            keep_mask[drop_idx] = False

        sub_ei = sub_ei[:, keep_mask]

        # Update edge_mask: only keep edges that survived fan-out filtering
        original_edge_indices = edge_mask.nonzero(as_tuple=True)[0]
        new_edge_mask = torch.zeros_like(edge_mask)
        new_edge_mask[original_edge_indices[keep_mask]] = True

        return sub_ei, new_edge_mask

    @staticmethod
    def _map_to_local(
        global_ids: torch.Tensor,
        subgraph_global_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Map global IDs to local indices within the subgraph.

        Uses a searchsorted-based approach for efficiency.
        """
        sorted_ids, sort_idx = subgraph_global_ids.sort()
        positions = torch.searchsorted(sorted_ids, global_ids)
        positions = positions.clamp(max=sort_idx.size(0) - 1)
        return sort_idx[positions]
