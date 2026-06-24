"""Graph builder: CanonicalInteractions -> PyG Data with edge_index, masks, and sign weights."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from ..utils.config import EDGRecConfig
from ..utils.interaction_indexing import compute_normalized_popularity, compute_popularity_counts
from .canonical import CanonicalInteractions
from .interaction_masks import positive_interaction_mask

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
    positive_train_mask: np.ndarray,
    n_items: int,
) -> torch.Tensor:
    """Compute raw positive-train item popularity counts for PyG ARP.

    Args:
        canonical: Canonical dataset whose training rows define popularity.
        positive_train_mask: Boolean mask selecting positive training rows.
        n_items: Number of catalog items.
    Returns:
        torch.Tensor: Float tensor of shape ``(n_items,)`` with train-only raw
        interaction counts. This tensor is passed directly to PyG
        ``LinkPredAveragePopularity`` so the logged ARP metric keeps PyG
        semantics.

    """
    popularity_array = compute_popularity_counts(
        canonical.item_id[positive_train_mask],
        n_items,
    )
    return torch.from_numpy(popularity_array).float()


def _resolve_training_popularity_counts(
    canonical: CanonicalInteractions,
    positive_train_mask: np.ndarray,
    n_items: int,
) -> torch.Tensor:
    """Compute raw positive-train item popularity counts."""
    counts = compute_popularity_counts(canonical.item_id[positive_train_mask], n_items)
    return torch.from_numpy(counts).float()


def _resolve_normalized_training_popularity(
    canonical: CanonicalInteractions,
    positive_train_mask: np.ndarray,
    n_items: int,
) -> torch.Tensor:
    """Compute log-normalized train popularity for model/loss targets."""
    popularity_array = compute_normalized_popularity(
        canonical.item_id[positive_train_mask],
        n_items,
    )
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
    passthrough_field_names = (
        "feedback_type",
        "preprocessing_preset",
        "metadata",
        "user_feature_names",
        "item_feature_names",
        "user_feature_sources",
        "item_feature_sources",
        "user_feature_raw_columns",
        "item_feature_raw_columns",
        "user_feature_roles",
        "item_feature_roles",
        "user_feature_groups",
        "item_feature_groups",
    )

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
    config: EDGRecConfig,
) -> Data:
    """Convert canonical interactions into a PyG Data object.

    Constructs a bipartite user-item graph where item node IDs are offset
    by ``n_users`` so that all node IDs are unique.

    Args:
        canonical: The loaded dataset.
        config: Model config controlling split ratios and graph construction.

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
    labels = torch.from_numpy(canonical.label).float()
    train_positive_mask = positive_interaction_mask(train_mask, canonical.label)
    val_positive_mask = positive_interaction_mask(val_mask, canonical.label)
    test_positive_mask = positive_interaction_mask(test_mask, canonical.label)
    train_positive_mask_t = torch.from_numpy(train_positive_mask)
    val_positive_mask_t = torch.from_numpy(val_positive_mask)
    test_positive_mask_t = torch.from_numpy(test_positive_mask)
    edge_index, edge_sign = _build_interaction_graph(
        user_nodes,
        item_nodes,
        _apply_train_edge_dropout(
            train_positive_mask_t,
            keep_prob=config.train_edge_keep_prob,
            seed=config.seed,
        ),
        all_signs,
    )

    # Popularity for PyG AveragePopularity is raw positive-train item counts.
    # Held-out validation/test interactions never leak into this tensor. CRRU
    # normalizes the logged raw ARP post hoc with the largest train item count.
    popularity = _resolve_training_popularity(
        canonical,
        train_positive_mask,
        n_items,
    )
    popularity_count = _resolve_training_popularity_counts(
        canonical,
        train_positive_mask,
        n_items,
    )
    normalized_popularity = _resolve_normalized_training_popularity(
        canonical,
        train_positive_mask,
        n_items,
    )

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
    data.train_positive_mask = train_positive_mask_t
    data.val_positive_mask = val_positive_mask_t
    data.test_positive_mask = test_positive_mask_t
    data.popularity = popularity
    data.popularity_count = popularity_count
    data.normalized_popularity = normalized_popularity
    data.largest_training_item_interaction_count = float(popularity_count.max().item()) if (
        popularity_count.numel()
    ) else 0.0
    data.labels = labels
    data.train_edge_keep_prob = float(config.train_edge_keep_prob)

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
    interaction_signs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct bipartite edges from training interactions (undirected).

    Returns ``(edge_index, edge_sign)`` where both tensors are aligned:
    undirected edges duplicate the train signs for both directions.
    """
    train_user_nodes = user_nodes[train_mask]
    train_item_nodes = item_nodes[train_mask]
    train_edge_signs = interaction_signs[train_mask]
    num_train_edges = train_user_nodes.numel()

    edge_index = torch.empty(
        (2, num_train_edges * 2),
        dtype=user_nodes.dtype,
        device=user_nodes.device,
    )
    edge_index[0, :num_train_edges] = train_user_nodes
    edge_index[1, :num_train_edges] = train_item_nodes
    edge_index[0, num_train_edges:] = train_item_nodes
    edge_index[1, num_train_edges:] = train_user_nodes

    edge_sign = torch.empty(
        num_train_edges * 2,
        dtype=train_edge_signs.dtype,
        device=train_edge_signs.device,
    )
    edge_sign[:num_train_edges] = train_edge_signs
    edge_sign[num_train_edges:] = train_edge_signs

    return edge_index, edge_sign


def _apply_train_edge_dropout(
    train_mask: torch.Tensor,
    *,
    keep_prob: float,
    seed: int,
) -> torch.Tensor:
    """Return a train-edge mask after split-safe observed-edge dropout."""
    if keep_prob >= 1.0:
        return train_mask

    train_indices = torch.nonzero(train_mask, as_tuple=False).flatten()
    if train_indices.numel() <= 1:
        return train_mask

    generator = torch.Generator(device=train_indices.device)
    generator.manual_seed(int(seed))
    kept = torch.rand(train_indices.numel(), generator=generator) < keep_prob
    if not bool(kept.any()):
        kept[torch.randint(train_indices.numel(), (1,), generator=generator)] = True

    dropped_mask = torch.zeros_like(train_mask)
    dropped_mask[train_indices[kept]] = True
    return dropped_mask
