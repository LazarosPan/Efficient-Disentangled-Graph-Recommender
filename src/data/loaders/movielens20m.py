"""MovieLens 20M loader — ratings.csv + optional genre and genome-tag features."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ...utils.dataset_loader_utils import downcast_numeric_array
from ..canonical import CanonicalInteractions
from ..feature_policy import (
    DEFAULT_FEATURE_POLICY,
    FeaturePolicyName,
    resolve_feature_source,
)
from ...utils.interaction_indexing import (
    compute_normalized_popularity,
    remap_interaction_ids,
)

logger = logging.getLogger(__name__)

# ML-20M genre list (20 genres)
_GENRES_20M = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "(no genres listed)",
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

    features = np.zeros((n_items, len(_GENRES_20M)), dtype=np.uint8)
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
        len(tag_data),
        n_tags,
    )
    return downcast_numeric_array(features, allow_float16=True)


def load_movielens20m(
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
    preprocessing_preset: str | None = None,
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
        path,
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
        usecols=(0, 1, 2, 3),
        max_rows=max_rows,
    )

    raw_users = downcast_numeric_array(data[:, 0].astype(np.int64))
    raw_items = downcast_numeric_array(data[:, 1].astype(np.int64))
    ratings = data[:, 2].astype(np.float32)
    timestamps = downcast_numeric_array(data[:, 3].astype(np.int64))

    indexed = remap_interaction_ids(raw_users, raw_items)
    user_id = indexed.user_id
    item_id = indexed.item_id
    n_users = indexed.n_users
    n_items = indexed.n_items
    user_map = indexed.user_map
    item_map = indexed.item_map

    label = (ratings >= 4.0).astype(np.float32)
    sign = ((ratings - 3.0) / 2.0).astype(np.float32)

    popularity = compute_normalized_popularity(item_id, n_items)

    item_features = None
    if include_optional_features:
        raw_dir = Path(data_dir) / "MovieLens20M" / "raw"
        genre_feats = None
        genome_feats = None
        load_genres, _ = resolve_feature_source(
            feature_policy,
            "movielens20m",
            "item_features",
            "raw/movies.csv",
        )
        if load_genres:
            genre_feats = _load_movie_genres(raw_dir / "movies.csv", item_map, n_items)
        load_genome_scores, _ = resolve_feature_source(
            feature_policy,
            "movielens20m",
            "item_features",
            "raw/genome-scores.csv",
        )
        if load_genome_scores:
            genome_feats = _load_genome_scores(
                raw_dir / "genome-scores.csv",
                item_map,
                n_items,
            )
        item_parts = [f for f in [genre_feats, genome_feats] if f is not None]
        item_features = np.hstack(item_parts) if item_parts else None

    effective_preset = preprocessing_preset or "movielens_explicit"

    return CanonicalInteractions(
        user_id=user_id,
        item_id=item_id,
        label=label,
        timestamp=timestamps,
        sign=sign,
        raw_target=ratings.astype(np.float32, copy=False),
        popularity=popularity,
        n_users=n_users,
        n_items=n_items,
        user_map=user_map,
        item_map=item_map,
        item_features=item_features,
        feedback_type="explicit",
        preprocessing_preset=effective_preset,
    )
