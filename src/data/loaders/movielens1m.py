"""MovieLens 1M loader — parses raw ratings/users/movies files into canonical form."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ...utils.dataset_loader_utils import (
    downcast_numeric_arrays,
    resolve_local_dataset_dir,
)
from ..canonical import CanonicalInteractions
from ..feature_policy import (
    DEFAULT_FEATURE_POLICY,
    FeaturePolicyName,
    resolve_feature_source,
)
from ._explicit_ratings import (
    build_explicit_rating_canonical,
    build_movie_genre_features,
    prepare_explicit_rating_feedback,
)

# Based on the README.md file `data/MovieLens1M/raw/README.md`:
# - Gender is denoted by a "M" for male and "F" for female
# - Age is chosen from the following ranges:

# 	*  1:  "Under 18"
# 	* 18:  "18-24"
# 	* 25:  "25-34"
# 	* 35:  "35-44"
# 	* 45:  "45-49"
# 	* 50:  "50-55"
# 	* 56:  "56+"

# ML-1M age buckets mapped to ordinal values
_AGE_MAP = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

logger = logging.getLogger(__name__)


def _parse_users_dat(
    raw_dir: Path,
    user_map: dict[int, int],
    n_users: int,
) -> np.ndarray | None:
    """Parse users.dat -> (n_users, 3) array: [gender, age_ordinal, occupation]."""
    path = raw_dir / "users.dat"
    if not path.exists():
        return None

    features = np.zeros((n_users, 3), dtype=np.uint8)
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue
            uid = int(parts[0])
            if uid not in user_map:
                continue
            idx = user_map[uid]
            features[idx, 0] = 0 if parts[1] == "F" else 1  # gender
            features[idx, 1] = _AGE_MAP.get(int(parts[2]), 3)  # age ordinal
            features[idx, 2] = int(parts[3])  # occupation (0-20)
    return features


def _parse_movies_dat(
    raw_dir: Path,
    item_map: dict[int, int],
    n_items: int,
) -> np.ndarray | None:
    """Parse movies.dat into a multi-hot genre matrix inferred from the file."""
    path = raw_dir / "movies.dat"
    if not path.exists():
        return None

    genre_rows: list[tuple[int, str]] = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            try:
                movie_id = int(parts[0])
            except ValueError:
                continue
            genre_rows.append((movie_id, parts[2]))

    features, mapped_item_count = build_movie_genre_features(
        genre_rows,
        item_map,
        n_items,
    )
    if features is not None:
        logger.info(
            "Loaded MovieLens1M genre features for %d / %d items (%d columns).",
            mapped_item_count,
            n_items,
            features.shape[1],
        )
    return features


def _resolve_raw_dir(data_dir: str) -> Path:
    """Resolve the raw ML-1M directory from local repository data only."""
    raw_base = Path(data_dir) / "MovieLens1M" / "raw"
    return resolve_local_dataset_dir(
        candidates=[
            raw_base,
            raw_base / "ml-1m",
            Path(data_dir) / "ml-1m",
        ],
        required_files=["ratings.dat", "users.dat", "movies.dat"],
        missing_message=(
            "MovieLens1M raw files not found in the local data directory. "
            f"Checked under {raw_base}."
        ),
    )


def _parse_ratings_dat(
    raw_dir: Path,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse ratings.dat into raw user IDs, item IDs, ratings, and timestamps."""
    path = raw_dir / "ratings.dat"
    user_ids: list[int] = []
    item_ids: list[int] = []
    ratings: list[float] = []
    timestamps: list[int] = []
    malformed_rows = 0

    row_count = 0
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 4:
                malformed_rows += 1
                continue
            try:
                user_id = int(parts[0])
                item_id = int(parts[1])
                rating = float(parts[2])
                timestamp = int(parts[3])
            except ValueError:
                malformed_rows += 1
                continue
            if not np.isfinite(rating):
                malformed_rows += 1
                continue
            user_ids.append(user_id)
            item_ids.append(item_id)
            ratings.append(rating)
            timestamps.append(timestamp)
            row_count += 1
            if max_rows is not None and row_count >= max_rows:
                break

    if malformed_rows > 0:
        logger.warning(
            "MovieLens1M loader skipped %d malformed ratings rows.",
            malformed_rows,
        )
    if not user_ids:
        raise ValueError("MovieLens1M loader found no valid rating rows.")

    raw_users, raw_items, timestamps_arr = downcast_numeric_arrays(
        np.asarray(user_ids, dtype=np.int64),
        np.asarray(item_ids, dtype=np.int64),
        np.asarray(timestamps, dtype=np.int64),
    )
    return raw_users, raw_items, np.asarray(ratings, dtype=np.float32), timestamps_arr


def load_movielens1m(
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
    preprocessing_preset: str | None = None,
) -> CanonicalInteractions:
    """Load ML-1M from local raw files.

    Raw parsing is more robust than depending on a processed HeteroData layout,
    which has changed across PyG releases, and keeps loading strictly local to
    the repository data directory.

    Label: rating >= 4 -> positive (1.0), else negative (0.0)
    Sign:  rating mapped to [-1, 1] via ``(rating - 3) / 2``
    """
    raw_dir = _resolve_raw_dir(data_dir)
    raw_users, raw_items, ratings, timestamps = _parse_ratings_dat(
        raw_dir,
        max_rows=max_rows,
    )

    prepared = prepare_explicit_rating_feedback(raw_users, raw_items, ratings)
    indexed = prepared.indexed
    n_users = indexed.n_users
    n_items = indexed.n_items
    user_map = indexed.user_map
    item_map = indexed.item_map

    user_features = None
    item_features = None
    if include_optional_features:
        load_user_features, _ = resolve_feature_source(
            feature_policy,
            "movielens1m",
            "user_features",
            "raw/users.dat",
        )
        if load_user_features:
            user_features = _parse_users_dat(raw_dir, user_map, n_users)
        load_item_features, _ = resolve_feature_source(
            feature_policy,
            "movielens1m",
            "item_features",
            "raw/movies.dat",
        )
        if load_item_features:
            item_features = _parse_movies_dat(raw_dir, item_map, n_items)

    effective_preset = preprocessing_preset or "movielens_explicit"

    return build_explicit_rating_canonical(
        prepared,
        timestamps,
        user_features=user_features,
        item_features=item_features,
        preprocessing_preset=effective_preset,
    )
