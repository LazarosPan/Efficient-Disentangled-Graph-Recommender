"""MovieLens 20M loader — ratings.csv + optional genre and genome-tag features."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..canonical import CanonicalInteractions
from ..feature_policy import DEFAULT_FEATURE_POLICY, FeaturePolicyName

logger = logging.getLogger(__name__)

# ML-20M genre list (20 genres)
_GENRES_20M = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
    "Western", "(no genres listed)",
]
_GENRE_TO_IDX_20M = {g: i for i, g in enumerate(_GENRES_20M)}


def _load_movie_genres(
    path: Path, item_map: dict[int, int], n_items: int
) -> np.ndarray | None:
    """Parse movies.csv -> (n_items, 20) multi-hot genre vector.

    Format: movieId,title,genres (pipe-separated)
    """
    if not path.exists():
        return None

    features = np.zeros((n_items, len(_GENRES_20M)), dtype=np.float32)
    matched = 0
    with open(path, encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            # Handle commas in movie titles: split from right for genres
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                mid = int(parts[0])
            except ValueError:
                continue
            if mid not in item_map:
                continue
            idx = item_map[mid]
            genres_str = parts[-1]  # genres is always the last column
            for genre in genres_str.split("|"):
                gi = _GENRE_TO_IDX_20M.get(genre.strip())
                if gi is not None:
                    features[idx, gi] = 1.0
            matched += 1

    logger.info("Loaded genre features for %d / %d items", matched, n_items)
    return features if matched > 0 else None


def _load_genome_scores(
    path: Path, item_map: dict[int, int], n_items: int
) -> np.ndarray | None:
    """Parse genome-scores.csv -> (n_items, n_tags) dense tag relevance matrix.

    Format: movieId,tagId,relevance (11.7M rows for ~27K movies x 1128 tags)
    Pivots the long format into a (n_items, n_tags) matrix.
    """
    if not path.exists():
        return None

    # First pass: determine max tagId
    max_tag_id = 0
    tag_data: dict[int, list[tuple[int, float]]] = {}
    with open(path, encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                mid = int(parts[0])
                tid = int(parts[1])
                rel = float(parts[2])
            except ValueError:
                continue
            if mid not in item_map:
                continue
            max_tag_id = max(max_tag_id, tid)
            mapped = item_map[mid]
            if mapped not in tag_data:
                tag_data[mapped] = []
            tag_data[mapped].append((tid, rel))

    if not tag_data or max_tag_id == 0:
        return None

    n_tags = max_tag_id  # tagIds are 1-based, use as 0-indexed with -1
    features = np.zeros((n_items, n_tags), dtype=np.float32)
    for mapped_id, entries in tag_data.items():
        for tid, rel in entries:
            features[mapped_id, tid - 1] = rel

    logger.info(
        "Loaded genome-score features: %d items x %d tags",
        len(tag_data), n_tags,
    )
    return features


def load_movielens20m(
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
) -> CanonicalInteractions:
    """Load ML-20M from ``data_dir/MovieLens20M/raw/ratings.csv``.

    Format: userId,movieId,rating,timestamp (header row)
    Label: rating >= 4 -> positive
    Sign:  (rating - 3) / 2 -> [-1, 1]
    """
    path = Path(data_dir) / "MovieLens20M" / "raw" / "ratings.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"ML-20M ratings not found at {path}. "
            "Download from https://grouplens.org/datasets/movielens/20m/"
        )

    # Use numpy for speed on 20M rows
    data = np.loadtxt(
        path, delimiter=",", skiprows=1,
        dtype=np.float64, usecols=(0, 1, 2, 3), max_rows=max_rows,
    )

    raw_users = data[:, 0].astype(np.int64)
    raw_items = data[:, 1].astype(np.int64)
    ratings = data[:, 2].astype(np.float32)
    timestamps = data[:, 3].astype(np.int64)

    unique_users = np.unique(raw_users)
    unique_items = np.unique(raw_items)
    user_map = {int(uid): idx for idx, uid in enumerate(unique_users)}
    item_map = {int(iid): idx for idx, iid in enumerate(unique_items)}

    user_id = np.array([user_map[int(u)] for u in raw_users], dtype=np.int64)
    item_id = np.array([item_map[int(i)] for i in raw_items], dtype=np.int64)

    label = (ratings >= 4.0).astype(np.float32)
    sign = ((ratings - 3.0) / 2.0).astype(np.float32)

    n_users = len(unique_users)
    n_items = len(unique_items)
    pop_counts = np.bincount(item_id, minlength=n_items).astype(np.float32)
    popularity = pop_counts / pop_counts.max() if pop_counts.max() > 0 else pop_counts

    del feature_policy
    item_features = None
    if include_optional_features:
        raw_dir = Path(data_dir) / "MovieLens20M" / "raw"
        genre_feats = _load_movie_genres(raw_dir / "movies.csv", item_map, n_items)
        genome_feats = _load_genome_scores(raw_dir / "genome-scores.csv", item_map, n_items)
        item_parts = [f for f in [genre_feats, genome_feats] if f is not None]
        item_features = np.hstack(item_parts) if item_parts else None

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
        item_features=item_features,
    )
