"""Shared helpers for contiguous interaction indexing and popularity scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset_loader_utils import downcast_numeric_array


@dataclass(frozen=True, slots=True)
class InteractionIndex:
    """Contiguous interaction indexing outputs.

    Args:
        user_id: Contiguously reindexed user IDs aligned to interactions.
        item_id: Contiguously reindexed item IDs aligned to interactions.
        n_users: Number of unique users present in the interactions.
        n_items: Number of unique items present in the interactions.
        user_map: Mapping from raw user IDs to contiguous user IDs.
        item_map: Mapping from raw item IDs to contiguous item IDs.

    Returns:
        InteractionIndex: Immutable container for the reindexing outputs.
    """

    user_id: np.ndarray
    item_id: np.ndarray
    n_users: int
    n_items: int
    user_map: dict[int, int]
    item_map: dict[int, int]


def remap_interaction_ids(
    raw_user_ids: np.ndarray,
    raw_item_ids: np.ndarray,
) -> InteractionIndex:
    """Remap raw user and item IDs to contiguous integer ranges.

    Args:
        raw_user_ids: Raw user IDs aligned to interactions.
        raw_item_ids: Raw item IDs aligned to interactions.

    Returns:
        InteractionIndex: Contiguous IDs, counts, and raw-to-contiguous maps.
        The returned ID arrays are narrowed to the smallest safe integer dtype.
    """

    unique_users, user_inverse = np.unique(raw_user_ids, return_inverse=True)
    unique_items, item_inverse = np.unique(raw_item_ids, return_inverse=True)

    user_map = {int(user_id): index for index, user_id in enumerate(unique_users)}
    item_map = {int(item_id): index for index, item_id in enumerate(unique_items)}

    return InteractionIndex(
        user_id=downcast_numeric_array(user_inverse),
        item_id=downcast_numeric_array(item_inverse),
        n_users=int(unique_users.size),
        n_items=int(unique_items.size),
        user_map=user_map,
        item_map=item_map,
    )


def compute_normalized_popularity(item_id: np.ndarray, n_items: int) -> np.ndarray:
    """Compute max-normalized item popularity from reindexed item IDs.

    Args:
        item_id: Contiguous item IDs aligned to interactions.
        n_items: Number of unique items represented in item_id.

    Returns:
        np.ndarray: Float32 popularity counts normalized to [0, 1] by max count.
    """

    pop_counts = np.bincount(item_id, minlength=n_items).astype(np.float32)
    if pop_counts.size == 0:
        return pop_counts

    max_count = float(pop_counts.max())
    if max_count <= 0.0:
        return pop_counts
    return pop_counts / max_count


def compute_time_windowed_popularity(
    item_id: np.ndarray,
    n_items: int,
    timestamp: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    """Compute popularity using only interactions inside a trailing time window.

    Args:
        item_id: Contiguous item IDs aligned to interactions.
        n_items: Number of unique items represented in item_id.
        timestamp: Timestamps aligned to interactions.
        window_seconds: Width of the trailing window in seconds.

    Returns:
        np.ndarray: Max-normalized popularity restricted to the selected window.
        Falls back to all valid timestamps when the requested window is empty.
    """
    if item_id.size == 0:
        return np.zeros(n_items, dtype=np.float32)

    valid_timestamp_mask = timestamp > 0
    if not np.any(valid_timestamp_mask):
        return compute_normalized_popularity(item_id, n_items)

    latest_timestamp = int(timestamp[valid_timestamp_mask].max())
    earliest_allowed = latest_timestamp - window_seconds
    window_mask = valid_timestamp_mask & (timestamp >= earliest_allowed)
    if not np.any(window_mask):
        return compute_normalized_popularity(item_id[valid_timestamp_mask], n_items)
    return compute_normalized_popularity(item_id[window_mask], n_items)
