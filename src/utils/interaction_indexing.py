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


@dataclass(frozen=True, slots=True)
class PairwisePriorityCollapse:
    """Repeat-aware summary for collapsed raw user-item pairs.

    Args:
        keep_idx: Representative row indices in original encounter order.
        repeat_count: Number of raw rows aggregated into each kept pair.
        priority_mean: Mean priority value across each repeated pair.
        priority_max: Maximum priority value for each repeated pair.
        first_timestamp: Earliest timestamp per pair when timestamps are provided.
        last_timestamp: Latest timestamp per pair when timestamps are provided.

    Returns:
        PairwisePriorityCollapse: Immutable summary aligned to ``keep_idx``.

    """

    keep_idx: np.ndarray
    repeat_count: np.ndarray
    priority_mean: np.ndarray
    priority_max: np.ndarray
    first_timestamp: np.ndarray | None = None
    last_timestamp: np.ndarray | None = None


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


def compute_explicit_rating_signals(
    ratings: np.ndarray,
    threshold: float = 4.0,
    neutral_rating: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute binary label and graded sign from explicit numeric ratings.

    Args:
        ratings: Raw rating values (e.g. 1-5 for MovieLens).
        threshold: Ratings at or above this value are labelled positive (1.0).
        neutral_rating: Rating mapped to sign 0; higher ratings give positive sign,
            lower give negative.  The scale is half the range above neutral (2.0).

    Returns:
        Tuple ``(label, sign)`` both as ``float32`` arrays aligned to *ratings*.

    """
    label = (ratings >= threshold).astype(np.float32)
    sign = ((ratings - neutral_rating) / 2.0).astype(np.float32)
    return label, sign


def _validate_pairwise_collapse_inputs(
    raw_user_ids: np.ndarray,
    raw_item_ids: np.ndarray,
    priority: np.ndarray,
    aligned_arrays: tuple[np.ndarray, ...] = (),
    timestamp: np.ndarray | None = None,
) -> int:
    """Validate pairwise-collapse inputs and return the shared interaction length."""
    n_rows = int(raw_user_ids.size)
    if raw_item_ids.size != n_rows or priority.size != n_rows:
        raise ValueError("Pairwise collapse inputs must share the same length.")
    if timestamp is not None and timestamp.size != n_rows:
        raise ValueError("Timestamp tie-breakers must align with the interaction rows.")
    for values in aligned_arrays:
        if values.size != n_rows:
            raise ValueError("Aligned arrays must share the interaction-row length.")
    return n_rows


def _collapse_aligned_arrays(
    keep_idx: np.ndarray,
    raw_user_ids: np.ndarray,
    raw_item_ids: np.ndarray,
    aligned_arrays: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    """Slice the aligned interaction arrays to the retained pairwise rows."""
    return (
        raw_user_ids[keep_idx],
        raw_item_ids[keep_idx],
        *(values[keep_idx] for values in aligned_arrays),
    )


def pairwise_collapse_summary_to_canonical_fields(
    summary: PairwisePriorityCollapse | None,
) -> dict[str, np.ndarray | None]:
    """Convert a pairwise-collapse summary into canonical repeated-pair fields.

    Args:
        summary: Repeat-aware pairwise summary aligned to collapsed rows.

    Returns:
        dict[str, np.ndarray | None]: Keyword arguments for
        ``build_indexed_canonical_interactions(...)`` covering repeated-pair
        aggregates. Returns ``None`` for each field when no summary is provided.

    """
    if summary is None:
        return {
            "repeat_count": None,
            "repeat_priority_mean": None,
            "repeat_priority_max": None,
            "repeat_first_timestamp": None,
            "repeat_last_timestamp": None,
        }

    return {
        "repeat_count": downcast_numeric_array(summary.repeat_count),
        "repeat_priority_mean": summary.priority_mean,
        "repeat_priority_max": summary.priority_max,
        "repeat_first_timestamp": (downcast_numeric_array(summary.first_timestamp) if summary.first_timestamp is not None else None),
        "repeat_last_timestamp": (downcast_numeric_array(summary.last_timestamp) if summary.last_timestamp is not None else None),
    }


def build_repeat_collapse_metadata(
    *,
    applied: bool,
    dropped_rows: int,
    reason: str,
) -> dict[str, str | int | bool]:
    """Build the shared repeat-collapse metadata payload for loaders.

    Args:
        applied: Whether repeated-pair collapse ran for the current preset.
        dropped_rows: Number of raw rows removed by the collapse.
        reason: Human-readable explanation of the collapse policy.

    Returns:
        dict[str, str | int | bool]: Canonical repeat-collapse metadata.

    """
    return {
        "applied": applied,
        "dropped_rows": dropped_rows,
        "stage": "pre_split",
        "reason": reason,
        "preserves_repeat_stats": applied,
    }


def build_repeat_collapse_canonical_payload(
    summary: PairwisePriorityCollapse | None,
    *,
    applied: bool,
    dropped_rows: int,
    reason: str,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build canonical repeat fields plus merged repeat-collapse metadata.

    Args:
        summary: Repeat-aware pairwise summary aligned to collapsed rows.
        applied: Whether repeated-pair collapse ran for the current preset.
        dropped_rows: Number of raw rows removed by the collapse.
        reason: Human-readable explanation of the collapse policy.
        metadata: Optional existing metadata to preserve and extend.

    Returns:
        dict[str, object]: Keyword arguments for canonical construction containing
        repeat-aware fields plus merged ``metadata`` with a ``repeat_collapse`` entry.

    """
    merged_metadata = dict(metadata or {})
    merged_metadata["repeat_collapse"] = build_repeat_collapse_metadata(
        applied=applied,
        dropped_rows=dropped_rows,
        reason=reason,
    )
    return {
        **pairwise_collapse_summary_to_canonical_fields(summary),
        "metadata": merged_metadata,
    }


def select_pairwise_max_priority_indices(
    raw_user_ids: np.ndarray,
    raw_item_ids: np.ndarray,
    priority: np.ndarray,
    timestamp: np.ndarray | None = None,
) -> np.ndarray:
    """Select one representative row per raw user-item pair.

    Args:
        raw_user_ids: Raw user IDs aligned to interactions.
        raw_item_ids: Raw item IDs aligned to interactions.
        priority: Scalar priority per interaction; larger values win.
        timestamp: Optional timestamps used as a secondary tie-breaker.

    Returns:
        np.ndarray: Original row indices for the selected representative rows,
        sorted in encounter order.

    """
    n_rows = _validate_pairwise_collapse_inputs(
        raw_user_ids,
        raw_item_ids,
        priority,
        timestamp=timestamp,
    )
    if n_rows == 0:
        return np.zeros(0, dtype=np.int64)

    return summarize_pairwise_max_priority_collapse(
        raw_user_ids,
        raw_item_ids,
        priority,
        timestamp=timestamp,
    ).keep_idx


def summarize_pairwise_max_priority_collapse(
    raw_user_ids: np.ndarray,
    raw_item_ids: np.ndarray,
    priority: np.ndarray,
    timestamp: np.ndarray | None = None,
) -> PairwisePriorityCollapse:
    """Summarize repeated raw user-item pairs under max-priority collapse.

    Args:
        raw_user_ids: Raw user IDs aligned to interactions.
        raw_item_ids: Raw item IDs aligned to interactions.
        priority: Scalar priority per interaction; larger values win.
        timestamp: Optional timestamps used as a secondary tie-breaker.

    Returns:
        PairwisePriorityCollapse: Representative row indices plus repeat-aware
        aggregate statistics aligned to the retained encounter order.

    """
    n_rows = _validate_pairwise_collapse_inputs(
        raw_user_ids,
        raw_item_ids,
        priority,
        timestamp=timestamp,
    )
    if n_rows == 0:
        empty_int = np.zeros(0, dtype=np.int64)
        empty_float = np.zeros(0, dtype=np.float32)
        return PairwisePriorityCollapse(
            keep_idx=empty_int,
            repeat_count=empty_int,
            priority_mean=empty_float,
            priority_max=empty_float,
            first_timestamp=(empty_int.copy() if timestamp is not None else None),
            last_timestamp=(empty_int.copy() if timestamp is not None else None),
        )

    timestamp_values = timestamp.astype(np.int64, copy=False) if timestamp is not None else np.zeros(n_rows, dtype=np.int64)
    priority_values = priority.astype(np.float64, copy=False)
    row_index = np.arange(n_rows, dtype=np.int64)
    order = np.lexsort(
        (
            row_index,
            timestamp_values,
            priority_values,
            raw_item_ids,
            raw_user_ids,
        ),
    )
    ordered_users = raw_user_ids[order]
    ordered_items = raw_item_ids[order]
    pair_start = np.ones(order.size, dtype=bool)
    if order.size > 1:
        pair_start[1:] = (ordered_users[1:] != ordered_users[:-1]) | (ordered_items[1:] != ordered_items[:-1])
    pair_starts = np.flatnonzero(pair_start)
    pair_ends = np.concatenate(
        [pair_starts[1:], np.array([order.size], dtype=np.int64)],
    )
    selected = order[pair_ends - 1]
    repeat_count = pair_ends - pair_starts
    priority_sum = np.add.reduceat(priority_values[order], pair_starts)
    priority_mean = (priority_sum / repeat_count).astype(np.float32, copy=False)
    priority_max = priority_values[order][pair_ends - 1].astype(np.float32, copy=False)
    first_timestamp = None
    last_timestamp = None
    if timestamp is not None:
        ordered_timestamps = timestamp_values[order]
        first_timestamp = np.minimum.reduceat(ordered_timestamps, pair_starts)
        last_timestamp = np.maximum.reduceat(ordered_timestamps, pair_starts)

    encounter_order = np.argsort(selected)
    return PairwisePriorityCollapse(
        keep_idx=selected[encounter_order],
        repeat_count=repeat_count[encounter_order],
        priority_mean=priority_mean[encounter_order],
        priority_max=priority_max[encounter_order],
        first_timestamp=(first_timestamp[encounter_order] if first_timestamp is not None else None),
        last_timestamp=(last_timestamp[encounter_order] if last_timestamp is not None else None),
    )


def collapse_pairwise_max_priority_rows(
    raw_user_ids: np.ndarray,
    raw_item_ids: np.ndarray,
    priority: np.ndarray,
    *aligned_arrays: np.ndarray,
    timestamp: np.ndarray | None = None,
) -> tuple[tuple[np.ndarray, ...], int]:
    """Collapse aligned interaction rows to one representative row per pair.

    Args:
        raw_user_ids: Raw user IDs aligned to interactions.
        raw_item_ids: Raw item IDs aligned to interactions.
        priority: Scalar priority per interaction; larger values win.
        aligned_arrays: Additional arrays aligned to the interaction rows that
            should be sliced with the same selected indices.
        timestamp: Optional timestamps used as a secondary tie-breaker.

    Returns:
        Tuple ``(collapsed_arrays, dropped_rows)``. ``collapsed_arrays`` contains
        ``raw_user_ids``, ``raw_item_ids``, and every array from ``aligned_arrays``
        in the same order after pairwise collapse.

    Raises:
        ValueError: If any aligned array does not match the interaction length.

    """
    n_rows = _validate_pairwise_collapse_inputs(
        raw_user_ids,
        raw_item_ids,
        priority,
        aligned_arrays=aligned_arrays,
        timestamp=timestamp,
    )
    keep_idx = summarize_pairwise_max_priority_collapse(
        raw_user_ids,
        raw_item_ids,
        priority,
        timestamp=timestamp,
    ).keep_idx
    collapsed_arrays = _collapse_aligned_arrays(
        keep_idx,
        raw_user_ids,
        raw_item_ids,
        aligned_arrays,
    )
    return collapsed_arrays, int(n_rows - keep_idx.size)


def collapse_pairwise_max_priority_rows_with_stats(
    raw_user_ids: np.ndarray,
    raw_item_ids: np.ndarray,
    priority: np.ndarray,
    *aligned_arrays: np.ndarray,
    timestamp: np.ndarray | None = None,
) -> tuple[tuple[np.ndarray, ...], PairwisePriorityCollapse, int]:
    """Collapse rows and preserve repeat-aware aggregate statistics.

    Args:
        raw_user_ids: Raw user IDs aligned to interactions.
        raw_item_ids: Raw item IDs aligned to interactions.
        priority: Scalar priority per interaction; larger values win.
        aligned_arrays: Additional arrays aligned to the interaction rows that
            should be sliced with the retained indices.
        timestamp: Optional timestamps used as a secondary tie-breaker.

    Returns:
        Tuple ``(collapsed_arrays, summary, dropped_rows)`` where ``summary``
        holds repeat-count and priority aggregates aligned to the collapsed rows.

    Raises:
        ValueError: If any aligned array does not match the interaction length.

    """
    n_rows = _validate_pairwise_collapse_inputs(
        raw_user_ids,
        raw_item_ids,
        priority,
        aligned_arrays=aligned_arrays,
        timestamp=timestamp,
    )

    summary = summarize_pairwise_max_priority_collapse(
        raw_user_ids,
        raw_item_ids,
        priority,
        timestamp=timestamp,
    )
    collapsed_arrays = _collapse_aligned_arrays(
        summary.keep_idx,
        raw_user_ids,
        raw_item_ids,
        aligned_arrays,
    )
    return collapsed_arrays, summary, int(n_rows - summary.keep_idx.size)
