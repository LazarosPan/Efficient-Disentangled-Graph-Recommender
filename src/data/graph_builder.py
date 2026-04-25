"""Graph builder: CanonicalInteractions → PyG Data with edge_index, masks, and sign weights.

Edge signs are aligned to edge_index so that edge_sign[i] corresponds to
edge_index[:, i]. Edges without interaction semantics (kNN / CAGRA
neighbors) receive a neutral sign of 0.0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import coalesce

from torch_geometric.utils import degree

from .canonical import CanonicalInteractions
from ..utils.config import UCaGNNConfig
from ..utils.interaction_indexing import (
    compute_normalized_popularity,
    compute_time_windowed_popularity,
)


def build_graph(
    canonical: CanonicalInteractions,
    config: UCaGNNConfig,
    embeddings: torch.Tensor | None = None,
) -> Data:
    """Convert canonical interactions into a PyG Data object.

    Constructs a bipartite user-item graph where item node IDs are offset
    by ``n_users`` so that all node IDs are unique.

    Args:
        canonical: The loaded dataset.
        config: Model config (controls graph_method, split ratios, etc.).
        embeddings: Optional (n_users + n_items, D) node embeddings used to add
            kNN/CAGRA similarity edges. When absent, the graph reduces to the
            train-interaction bipartite edges only.

    Returns:
        PyG Data with ``edge_index``, ``edge_sign``, ``train/val/test_mask``,
        ``popularity``, and ``n_users`` / ``n_items`` attributes.
    """
    n_users = canonical.n_users
    n_items = canonical.n_items

    # Bipartite edges: item IDs offset by n_users
    user_nodes = torch.from_numpy(canonical.user_id).long()
    item_nodes = torch.from_numpy(canonical.item_id).long() + n_users

    # Per-interaction signs (N,) — will be sliced to match train edges
    all_signs = torch.from_numpy(canonical.sign).float()

    # Split masks (predefined if available, else temporal)
    train_mask, val_mask, test_mask = canonical.get_splits(
        config.train_ratio,
        config.val_ratio,
        derived_split_mode=config.derived_split_mode,
    )
    train_mask_t = torch.from_numpy(train_mask)
    val_mask_t = torch.from_numpy(val_mask)
    test_mask_t = torch.from_numpy(test_mask)

    # Build edge_index + aligned edge_sign depending on method.
    if config.graph_method == "knn":
        edge_index, edge_sign = _build_knn(
            user_nodes,
            item_nodes,
            train_mask_t,
            all_signs,
            embeddings,
            config.knn_k,
        )
    elif config.graph_method == "cagra":
        edge_index, edge_sign = _build_cagra(
            user_nodes,
            item_nodes,
            train_mask_t,
            all_signs,
            embeddings,
            config,
        )
    else:
        raise ValueError(f"Unknown graph_method: {config.graph_method}")

    # Popularity must be derived from the final training split only so held-out
    # validation/test interactions never leak into training or evaluation.
    if config.popularity_window_seconds is None:
        popularity_array = compute_normalized_popularity(
            canonical.item_id[train_mask],
            n_items,
        )
    else:
        popularity_array = compute_time_windowed_popularity(
            canonical.item_id[train_mask],
            n_items,
            canonical.timestamp[train_mask],
            config.popularity_window_seconds,
        )
    popularity = torch.from_numpy(popularity_array).float()

    labels = torch.from_numpy(canonical.label).float()

    # Precompute full-graph symmetric degree normalization: 1/sqrt(deg_u * deg_v).
    # Stored on the Data object so that both training (subgraph) and evaluation
    # (full graph) use identical normalization factors, eliminating the train/eval
    # degree-normalization inconsistency that arises with mini-batch LGConv.
    num_nodes = n_users + n_items
    dg = degree(edge_index[0], num_nodes=num_nodes)
    inv_sqrt_dg = dg.pow(-0.5)
    inv_sqrt_dg[inv_sqrt_dg == float("inf")] = 0.0
    edge_norm = inv_sqrt_dg[edge_index[0]] * inv_sqrt_dg[edge_index[1]]

    data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    data.edge_norm = edge_norm
    data.edge_sign = edge_sign
    data.train_mask = train_mask_t
    data.val_mask = val_mask_t
    data.test_mask = test_mask_t
    data.popularity = popularity
    data.labels = labels
    data.user_nodes = user_nodes
    data.item_nodes = item_nodes
    data.n_users = n_users
    data.n_items = n_items
    if canonical.user_features is not None:
        data.user_features = torch.from_numpy(canonical.user_features).float()
    if canonical.item_features is not None:
        data.item_features = torch.from_numpy(canonical.item_features).float()
    if canonical.raw_target is not None:
        data.raw_target = torch.from_numpy(canonical.raw_target).float()
    if canonical.exposure_flag is not None:
        data.exposure_flag = torch.from_numpy(canonical.exposure_flag.astype(np.bool_))
    if canonical.behavior_type is not None:
        data.behavior_type = canonical.behavior_type.copy()
    if canonical.source_domain is not None:
        data.source_domain = canonical.source_domain.copy()
    if canonical.feedback_type is not None:
        data.feedback_type = canonical.feedback_type
    if canonical.preprocessing_preset is not None:
        data.preprocessing_preset = canonical.preprocessing_preset
    if canonical.metadata is not None:
        data.metadata = canonical.metadata
    data.derived_split_mode = config.derived_split_mode

    return data


def _build_interaction_graph(
    user_nodes: torch.Tensor,
    item_nodes: torch.Tensor,
    train_mask: torch.Tensor,
    all_signs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct bipartite edges from training interactions (undirected).

    Returns ``(edge_index, edge_sign)`` where both tensors are aligned:
    undirected edges duplicate the train signs for both directions.
    """
    src = user_nodes[train_mask]
    dst = item_nodes[train_mask]
    train_signs = all_signs[train_mask]

    # Undirected: add both directions
    edge_index = torch.stack(
        [
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ],
        dim=0,
    )

    # Duplicate signs for both directions (forward + backward)
    edge_sign = torch.cat([train_signs, train_signs])

    return edge_index, edge_sign


def _build_knn(
    user_nodes: torch.Tensor,
    item_nodes: torch.Tensor,
    train_mask: torch.Tensor,
    all_signs: torch.Tensor,
    embeddings: torch.Tensor | None,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bipartite edges + kNN edges in embedding space.

    Falls back to the train-interaction graph if no embeddings are provided.
    kNN edges receive a neutral sign of 0.0.
    """
    # Always include bipartite interaction edges
    bipartite_ei, bipartite_sign = _build_interaction_graph(
        user_nodes,
        item_nodes,
        train_mask,
        all_signs,
    )

    if embeddings is None:
        return bipartite_ei, bipartite_sign

    # kNN on the embedding space
    knn_edges = knn_graph(embeddings, k=k, loop=False)
    knn_sign = torch.zeros(knn_edges.size(1))

    # Combine and coalesce (keep signs aligned via attr parameter)
    combined_ei = torch.cat([bipartite_ei, knn_edges], dim=1)
    combined_sign = torch.cat([bipartite_sign, knn_sign])
    edge_index, edge_sign = coalesce(combined_ei, combined_sign)
    return edge_index, edge_sign


def _build_cagra(
    user_nodes: torch.Tensor,
    item_nodes: torch.Tensor,
    train_mask: torch.Tensor,
    all_signs: torch.Tensor,
    embeddings: torch.Tensor | None,
    config: UCaGNNConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bipartite edges + CAGRA GPU-accelerated ANN graph.

    Falls back to kNN if cuvs is not available.
    CAGRA edges receive a neutral sign of 0.0.
    """
    bipartite_ei, bipartite_sign = _build_interaction_graph(
        user_nodes,
        item_nodes,
        train_mask,
        all_signs,
    )

    if embeddings is None:
        return bipartite_ei, bipartite_sign

    try:
        from cuvs.neighbors import cagra

        emb_np = embeddings.detach().cpu().numpy().astype("float32")
        index_params = cagra.IndexParams(
            intermediate_graph_degree=config.cagra_initial_degree,
            graph_degree=config.cagra_out_degree,
        )
        index = cagra.build(index_params, emb_np)

        search_params = cagra.SearchParams(team_size=config.cagra_team_size)
        _, neighbors = cagra.search(
            search_params,
            index,
            emb_np,
            k=config.knn_k,
        )

        neighbors_np = neighbors.copy_to_host()
        n_nodes = embeddings.shape[0]
        k = neighbors_np.shape[1]
        src = np.repeat(np.arange(n_nodes), k)
        dst = neighbors_np.ravel()
        valid = (src != dst) & (dst >= 0) & (dst < n_nodes)
        cagra_edges = torch.tensor(np.stack([src[valid], dst[valid]]), dtype=torch.long)
        cagra_sign = torch.zeros(cagra_edges.size(1))

        combined_ei = torch.cat([bipartite_ei, cagra_edges], dim=1)
        combined_sign = torch.cat([bipartite_sign, cagra_sign])
        edge_index, edge_sign = coalesce(combined_ei, combined_sign)
        return edge_index, edge_sign

    except ImportError:
        # Fallback to kNN if cuvs not available
        return _build_knn(
            user_nodes,
            item_nodes,
            train_mask,
            all_signs,
            embeddings,
            config.knn_k,
        )
