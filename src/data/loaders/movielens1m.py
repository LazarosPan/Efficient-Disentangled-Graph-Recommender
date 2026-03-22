"""MovieLens 1M loader — parses raw ratings/users/movies files into canonical form."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..canonical import CanonicalInteractions
from ..feature_policy import DEFAULT_FEATURE_POLICY, FeaturePolicyName


# ML-1M age buckets mapped to ordinal values
_AGE_MAP = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

# ML-1M genre list (18 genres)
_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
_GENRE_TO_IDX = {g: i for i, g in enumerate(_GENRES)}


def _parse_users_dat(raw_dir: Path, user_map: dict[int, int], n_users: int) -> np.ndarray | None:
    """Parse users.dat -> (n_users, 3) array: [gender, age_ordinal, occupation]."""
    path = raw_dir / "users.dat"
    if not path.exists():
        return None

    features = np.zeros((n_users, 3), dtype=np.float32)
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue
            uid = int(parts[0])
            if uid not in user_map:
                continue
            idx = user_map[uid]
            features[idx, 0] = 0.0 if parts[1] == "F" else 1.0  # gender
            features[idx, 1] = float(_AGE_MAP.get(int(parts[2]), 3))  # age ordinal
            features[idx, 2] = float(parts[3])  # occupation (0-20)
    return features


def _parse_movies_dat(raw_dir: Path, item_map: dict[int, int], n_items: int) -> np.ndarray | None:
    """Parse movies.dat -> (n_items, 18) multi-hot genre vector."""
    path = raw_dir / "movies.dat"
    if not path.exists():
        return None

    features = np.zeros((n_items, len(_GENRES)), dtype=np.float32)
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            mid = int(parts[0])
            if mid not in item_map:
                continue
            idx = item_map[mid]
            for genre in parts[2].split("|"):
                gi = _GENRE_TO_IDX.get(genre)
                if gi is not None:
                    features[idx, gi] = 1.0
    return features


def _resolve_raw_dir(data_dir: str) -> Path:
    """Resolve the raw ML-1M directory from local repository data only."""
    raw_base = Path(data_dir) / "MovieLens1M" / "raw"
    candidate_dirs = [
        raw_base,
        raw_base / "ml-1m",
        Path(data_dir) / "ml-1m",
    ]
    required_files = {"ratings.dat", "users.dat", "movies.dat"}

    for raw_dir in candidate_dirs:
        if all((raw_dir / name).exists() for name in required_files):
            return raw_dir

    raise FileNotFoundError(
        "MovieLens1M raw files not found in the local data directory. "
        f"Checked under {raw_base}."
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

    row_count = 0
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue
            user_ids.append(int(parts[0]))
            item_ids.append(int(parts[1]))
            ratings.append(float(parts[2]))
            timestamps.append(int(parts[3]))
            row_count += 1
            if max_rows is not None and row_count >= max_rows:
                break

    return (
        np.asarray(user_ids, dtype=np.int64),
        np.asarray(item_ids, dtype=np.int64),
        np.asarray(ratings, dtype=np.float32),
        np.asarray(timestamps, dtype=np.int64),
    )


def load_movielens1m(
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
) -> CanonicalInteractions:
    """Load ML-1M from local raw files.

    Raw parsing is more robust than depending on a processed HeteroData layout,
    which has changed across PyG releases, and keeps loading strictly local to
    the repository data directory.

    Label: rating >= 4 -> positive (1.0), else negative (0.0)
    Sign:  rating mapped to [-1, 1] via ``(rating - 3) / 2``
    """
    del feature_policy
    raw_dir = _resolve_raw_dir(data_dir)
    raw_users, raw_items, ratings, timestamps = _parse_ratings_dat(raw_dir, max_rows=max_rows)

    # Re-index to contiguous 0-based IDs
    unique_users = np.unique(raw_users)
    unique_items = np.unique(raw_items)
    user_map = {int(uid): idx for idx, uid in enumerate(unique_users)}
    item_map = {int(iid): idx for idx, iid in enumerate(unique_items)}

    user_id = np.array([user_map[int(u)] for u in raw_users], dtype=np.int64)
    item_id = np.array([item_map[int(i)] for i in raw_items], dtype=np.int64)

    # Labels and signs
    label = (ratings >= 4.0).astype(np.float32)
    sign = ((ratings - 3.0) / 2.0).astype(np.float32)

    # Popularity: per-item interaction count, normalized to [0, 1]
    n_users = len(unique_users)
    n_items = len(unique_items)
    pop_counts = np.bincount(item_id, minlength=n_items).astype(np.float32)
    popularity = pop_counts / pop_counts.max() if pop_counts.max() > 0 else pop_counts

    user_features = None
    item_features = None
    if include_optional_features:
        user_features = _parse_users_dat(raw_dir, user_map, n_users)
        item_features = _parse_movies_dat(raw_dir, item_map, n_items)

    return CanonicalInteractions(
        user_id=user_id,
        item_id=item_id,
        label=label,
        timestamp=timestamps,
        sign=sign,
        popularity=popularity,
        n_users=n_users,
        n_items=n_items,
        user_map=user_map,
        item_map=item_map,
        user_features=user_features,
        item_features=item_features,
    )
