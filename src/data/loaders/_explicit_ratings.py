"""Shared helpers for explicit-rating dataset loaders.

Both the MovieLens-1M and MovieLens-20M loaders use
``prepare_explicit_rating_feedback`` and ``build_explicit_rating_canonical``
to share identical reindexing, label/sign derivation, and canonical assembly
while each loader keeps its own file-format-specific parsing path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from ...utils.csv_features import build_multi_hot_features
from ...utils.interaction_indexing import (
    InteractionIndex,
    compute_explicit_rating_signals,
    compute_normalized_popularity,
    remap_interaction_ids,
)
from ..canonical import (
    CanonicalInteractions,
    build_indexed_canonical_interactions,
)


@dataclass(frozen=True, slots=True)
class PreparedExplicitRatings:
    """Store canonicalized explicit-rating arrays before feature assembly.

    Args:
        indexed: Reindexed user/item ids and lookup maps.
        label: Binary labels derived from explicit ratings.
        sign: Graded signs derived from explicit ratings.
        popularity: Dataset-level max-normalized per-item interaction counts.
            The runtime train-only popularity used by the model is recomputed
            later from the final training split in ``build_graph()``.
        raw_target: Original explicit ratings as float32.

    Returns:
        PreparedExplicitRatings: Shared MovieLens canonicalization payload.

    """

    indexed: InteractionIndex
    label: np.ndarray
    sign: np.ndarray
    popularity: np.ndarray
    raw_target: np.ndarray


def build_movie_genre_features(
    genre_rows: Iterable[tuple[int, str]],
    item_map: Mapping[int, int],
    n_items: int,
) -> tuple[np.ndarray | None, int]:
    """Build a MovieLens multi-hot genre matrix from raw item-id/genre rows.

    Args:
        genre_rows: Iterable of ``(raw_item_id, raw_genre_string)`` rows.
        item_map: Raw-to-contiguous item-id mapping for the current dataset.
        n_items: Total number of reindexed items.

    Returns:
        tuple[np.ndarray | None, int]: The multi-hot genre matrix and the number
        of items that contributed at least one retained genre row.

    """
    mapped_genres: dict[int, list[str]] = {}
    for raw_item_id, raw_genres in genre_rows:
        mapped_item_id = item_map.get(raw_item_id)
        if mapped_item_id is None:
            continue
        mapped_genres[mapped_item_id] = [genre.strip() for genre in raw_genres.split("|") if genre.strip()]
    return build_multi_hot_features(mapped_genres, n_items), len(mapped_genres)


def prepare_explicit_rating_feedback(
    raw_users: np.ndarray,
    raw_items: np.ndarray,
    ratings: np.ndarray,
) -> PreparedExplicitRatings:
    """Prepare shared canonical fields for explicit-rating datasets.

    Args:
        raw_users: Raw user ids aligned to explicit interactions.
        raw_items: Raw item ids aligned to explicit interactions.
        ratings: Explicit rating values aligned to the interactions.

    Returns:
        PreparedExplicitRatings: Reindexed ids plus shared label/sign/popularity
        fields used by the MovieLens loaders. The returned popularity vector is a
        dataset-level summary for canonical inspection rather than the runtime
        train-only popularity signal.

    """

    indexed = remap_interaction_ids(raw_users, raw_items)
    label, sign = compute_explicit_rating_signals(ratings)
    popularity = compute_normalized_popularity(indexed.item_id, indexed.n_items)
    return PreparedExplicitRatings(
        indexed=indexed,
        label=label,
        sign=sign,
        popularity=popularity,
        raw_target=ratings.astype(np.float32, copy=False),
    )


def build_explicit_rating_canonical(
    prepared: PreparedExplicitRatings,
    timestamps: np.ndarray,
    *,
    user_features: np.ndarray | None = None,
    item_features: np.ndarray | None = None,
    preprocessing_preset: str = "movielens_explicit",
) -> CanonicalInteractions:
    """Build a canonical dataset from prepared explicit-rating fields.

    Args:
        prepared: Shared explicit-rating canonicalization payload.
        timestamps: Rating timestamps aligned to the interactions.
        user_features: Optional user side-feature matrix.
        item_features: Optional item side-feature matrix.
        preprocessing_preset: Loader preset recorded on the canonical dataset.

    Returns:
        CanonicalInteractions: Canonical explicit-feedback dataset.

    """

    return build_indexed_canonical_interactions(
        prepared.indexed,
        label=prepared.label,
        timestamp=timestamps,
        sign=prepared.sign,
        popularity=prepared.popularity,
        user_features=user_features,
        item_features=item_features,
        raw_target=prepared.raw_target,
        feedback_type="explicit",
        preprocessing_preset=preprocessing_preset,
    )
