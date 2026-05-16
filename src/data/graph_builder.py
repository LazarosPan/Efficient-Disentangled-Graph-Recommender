"""Graph builder: CanonicalInteractions -> PyG Data with edge_index, masks, and sign weights.

Edge signs are aligned to edge_index so that edge_sign[i] corresponds to
edge_index[:, i]. Edges without interaction semantics (CAGRA neighbors)
receive a neutral sign of 0.0.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, degree

from ..utils.config import UCaGNNConfig
from ..utils.interaction_indexing import compute_normalized_popularity
from .canonical import CanonicalInteractions

_FLOAT_CANONICAL_FIELDS = frozenset(
    {
        "user_features",
        "item_features",
        "raw_target",
        "repeat_mean_target",
        "repeat_max_target",
        "repeat_latest_target",
    },
)
_LONG_CANONICAL_FIELDS = frozenset(
    {
        "repeat_count",
        "repeat_first_timestamp",
        "repeat_last_timestamp",
        "repeat_behavior_counts",
    },
)
_BOOL_CANONICAL_FIELDS = frozenset({"exposure_flag"})


def _ensure_writable_numpy(value: np.ndarray) -> np.ndarray:
    """Return a writable NumPy array for safe Torch conversion.

    Args:
        value: Source NumPy array.

    Returns:
        np.ndarray: The original array when writable, otherwise a copy.

    """
    return value if value.flags.writeable else value.copy()


def _canonical_numpy_to_tensor(
    field_name: str,
    value: np.ndarray,
) -> torch.Tensor:
    """Convert one canonical NumPy field to its graph-boundary tensor form.

    Args:
        field_name: Canonical field name being transferred to ``Data``.
        value: NumPy payload for the field.

    Returns:
        torch.Tensor: Converted tensor with the expected dtype.

    """
    writable_value = _ensure_writable_numpy(value)
    if field_name in _FLOAT_CANONICAL_FIELDS:
        return torch.from_numpy(writable_value).float()
    if field_name in _LONG_CANONICAL_FIELDS:
        return torch.from_numpy(writable_value).long()
    if field_name in _BOOL_CANONICAL_FIELDS:
        return torch.from_numpy(_ensure_writable_numpy(writable_value.astype(np.bool_, copy=False)))
    raise KeyError(f"Unsupported canonical tensor field '{field_name}'.")


def _resolve_training_popularity(
    canonical: CanonicalInteractions,
    train_mask: np.ndarray,
    n_items: int,
) -> torch.Tensor:
    """Compute train-only item popularity and return it as a float tensor.

    Args:
        canonical: Canonical dataset whose training rows define the popularity.
        train_mask: Boolean mask selecting the training interactions.
        n_items: Number of catalog items.
    Returns:
        torch.Tensor: Float tensor of shape ``(n_items,)`` with train-only
        popularity values.

    """
    popularity_array = compute_normalized_popularity(canonical.item_id[train_mask], n_items)
    return torch.from_numpy(popularity_array).float()


def _attach_optional_canonical_fields(
    data: Data,
    canonical: CanonicalInteractions,
) -> None:
    """Copy optional canonical payloads onto the PyG ``Data`` object.

    Args:
        data: Graph payload being assembled.
        canonical: Canonical dataset that may carry optional side information.

    Returns:
        None. ``data`` is updated in place.

    """
    tensor_field_names = (
        "user_features",
        "item_features",
        "raw_target",
        "exposure_flag",
        "repeat_count",
        "repeat_mean_target",
        "repeat_max_target",
        "repeat_latest_target",
        "repeat_first_timestamp",
        "repeat_last_timestamp",
        "repeat_behavior_counts",
    )
    array_field_names = ("behavior_type", "source_domain", "repeat_behavior_labels")
    passthrough_field_names = ("feedback_type", "preprocessing_preset", "metadata")

    for field_name in tensor_field_names:
        value = getattr(canonical, field_name)
        if value is not None:
            setattr(data, field_name, _canonical_numpy_to_tensor(field_name, value))

    for field_name in array_field_names:
        value = getattr(canonical, field_name)
        if value is not None:
            setattr(data, field_name, value.copy())

    for field_name in passthrough_field_names:
        value = getattr(canonical, field_name)
        if value is not None:
            setattr(data, field_name, value)


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
        config: Model config (controls split ratios, CAGRA settings, etc.).
        embeddings: Optional (n_users + n_items, D) node embeddings used to add
            CAGRA similarity edges. When absent, the graph reduces to the
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
    edge_index, edge_sign = _build_cagra(
        user_nodes,
        item_nodes,
        train_mask_t,
        all_signs,
        embeddings,
        config,
    )

    # Popularity must be derived from the final training split only so held-out
    # validation/test interactions never leak into training or evaluation.
    # Both the count-based and time-windowed paths receive item_id[train_mask]
    # and timestamp[train_mask] respectively — no val/test rows contribute.
    popularity = _resolve_training_popularity(
        canonical,
        train_mask,
        n_items,
    )

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
    _attach_optional_canonical_fields(data, canonical)
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
    edge_sign = torch.cat([train_signs, train_signs])

    return edge_index, edge_sign


def _to_cagra_array(embeddings: torch.Tensor) -> object:
    """Convert torch embeddings to a cuVS-compatible array.

    Args:
        embeddings: Node embeddings used to build or query the ANN graph.

    Returns:
        A CUDA-array-interface object suitable for `cuvs.neighbors.cagra`.

    Raises:
        RuntimeError: If CuPy is unavailable for the conversion.

    """
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("cupy is required for cagra graph construction") from exc

    contiguous = embeddings.detach().contiguous()
    if contiguous.is_cuda:
        from_dlpack = getattr(cp, "from_dlpack", None)
        if from_dlpack is not None:
            return from_dlpack(contiguous)

        from torch.utils import dlpack as torch_dlpack

        return cp.fromDlpack(torch_dlpack.to_dlpack(contiguous))

    return cp.asarray(contiguous.float().cpu().numpy().astype("float32", copy=False))


def _cagra_neighbors_to_tensor(neighbors: object) -> torch.Tensor:
    """Materialize CAGRA neighbor results as a CPU ``torch.long`` tensor.

    Args:
        neighbors: cuVS neighbor output.

    Returns:
        torch.Tensor: CPU tensor with shape ``(n_nodes, k)``.

    """
    if isinstance(neighbors, torch.Tensor):
        return neighbors.to(device="cpu", dtype=torch.long)
    if isinstance(neighbors, np.ndarray):
        return torch.from_numpy(neighbors).long()
    if hasattr(neighbors, "copy_to_host"):
        return torch.from_numpy(np.asarray(neighbors.copy_to_host())).long()

    import cupy as cp

    return torch.from_numpy(np.asarray(cp.asnumpy(neighbors))).long()


def _build_cagra_edge_index(
    neighbors: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Build directed ANN edges from a dense neighbor table.

    Args:
        neighbors: Neighbor IDs with shape ``(n_nodes, k)`` on CPU.
        n_nodes: Number of graph nodes represented by ``neighbors``.

    Returns:
        torch.Tensor: Directed edge index with invalid/self edges removed.

    """
    if neighbors.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    k = neighbors.size(1)
    src = torch.arange(n_nodes, dtype=torch.long).repeat_interleave(k)
    dst = neighbors.reshape(-1)
    valid = (src != dst) & (dst >= 0) & (dst < n_nodes)
    return torch.stack((src[valid], dst[valid]), dim=0)


def _make_undirected_zero_signed_edges(
    directed_edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror neutral ANN edges so the graph stays undirected.

    Args:
        directed_edges: Directed ANN edges with shape ``(2, E)``.

    Returns:
        Tuple of undirected edge index and aligned zero signs.

    """
    reverse_edges = directed_edges.flip(0)
    undirected_edges = torch.cat([directed_edges, reverse_edges], dim=1)
    edge_sign = torch.zeros(undirected_edges.size(1), dtype=torch.float32)
    return undirected_edges, edge_sign


def _build_cagra(
    user_nodes: torch.Tensor,
    item_nodes: torch.Tensor,
    train_mask: torch.Tensor,
    all_signs: torch.Tensor,
    embeddings: torch.Tensor | None,
    config: UCaGNNConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bipartite edges + CAGRA GPU-accelerated ANN graph.

    When ``config.graph_policy == "cagra_augmented"``, CAGRA build/search errors
    are treated as hard failures so thesis runs never silently degrade to the
    observed graph. Other callers keep the legacy warning-and-fallback behavior.
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

        cagra_embeddings = _to_cagra_array(embeddings)
        index_params = cagra.IndexParams(
            intermediate_graph_degree=config.cagra_initial_degree,
            graph_degree=config.cagra_out_degree,
            metric=config.cagra_metric,
        )
        index = cagra.build(index_params, cagra_embeddings)

        search_params = cagra.SearchParams(
            team_size=config.cagra_team_size,
            rand_xor_mask=int(config.seed),
            itopk_size=config.cagra_itopk_size,
        )
        _, neighbors = cagra.search(
            search_params,
            index,
            cagra_embeddings,
            k=config.cagra_k,
        )

        neighbors_t = _cagra_neighbors_to_tensor(neighbors)
        n_nodes = embeddings.shape[0]
        cagra_edges = _build_cagra_edge_index(neighbors_t, n_nodes)
        cagra_edges, cagra_sign = _make_undirected_zero_signed_edges(cagra_edges)

        combined_ei = torch.cat([bipartite_ei, cagra_edges], dim=1)
        combined_sign = torch.cat([bipartite_sign, cagra_sign])
        edge_index, edge_sign = coalesce(combined_ei, combined_sign)
        return edge_index, edge_sign

    except (ImportError, Exception) as exc:
        msg = (
            f"CAGRA graph construction unavailable; using train-interaction graph: {exc}"
            if isinstance(exc, ImportError)
            else f"CAGRA graph construction failed; using train-interaction graph: {exc}"
        )
        if config.graph_policy == "cagra_augmented":
            raise RuntimeError(
                msg.replace(
                    "using train-interaction graph",
                    "graph_policy='cagra_augmented' requires a working CAGRA build",
                ),
            ) from exc
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return bipartite_ei, bipartite_sign
