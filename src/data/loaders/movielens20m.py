"""MovieLens 20M loader — ratings.csv + optional genre and genome-tag features."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from ...utils.csv_features import (
    FeatureBlock,
    make_feature_block,
    stack_feature_metadata_blocks,
)
from ...utils.dataset_loader_utils import downcast_numeric_array, downcast_numeric_arrays
from ..canonical import CanonicalInteractions
from ..feature_policy import (
    DEFAULT_FEATURE_POLICY,
    FeaturePolicyName,
    resolve_feature_source,
)
from ._explicit_ratings import (
    build_explicit_rating_canonical,
    build_movie_genre_feature_block,
    prepare_explicit_rating_feedback,
)

logger = logging.getLogger(__name__)


def _load_movie_genres(
    path: Path,
    item_map: dict[int, int],
    n_items: int,
) -> FeatureBlock | None:
    """Parse movies.csv into a multi-hot genre matrix inferred from the file.

    Format: movieId,title,genres (pipe-separated)
    """
    if not path.exists():
        return None

    genre_rows: list[tuple[int, str]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_movie_id = row.get("movieId")
            if raw_movie_id is None:
                continue
            try:
                movie_id = int(raw_movie_id)
            except ValueError:
                continue
            genre_rows.append((movie_id, row.get("genres", "")))

    block, mapped_item_count = build_movie_genre_feature_block(
        genre_rows,
        item_map,
        n_items,
        dataset_name="movielens20m",
        relative_path="raw/movies.csv",
    )
    if block is not None:
        logger.info(
            "Loaded MovieLens20M genre features for %d / %d items (%d columns).",
            mapped_item_count,
            n_items,
            block.matrix.shape[1],
        )
    return block


def _load_genome_scores(
    path: Path,
    item_map: dict[int, int],
    n_items: int,
) -> FeatureBlock | None:
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
    matrix = downcast_numeric_array(features, allow_float16=True)
    return make_feature_block(
        matrix,
        dataset_name="movielens20m",
        aspect="item_features",
        relative_path="raw/genome-scores.csv",
        raw_columns=tuple("relevance" for _ in range(n_tags)),
        encoded_columns=tuple(f"relevance={tag_id}" for tag_id in range(1, n_tags + 1)),
    )


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
            f"ML-20M ratings not found at {path}. Download from https://grouplens.org/datasets/movielens/20m/",
        )

    # Use numpy for speed on 20M rows
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
        usecols=(0, 1, 2, 3),
        max_rows=max_rows,
        ndmin=2,
    )
    if data.size == 0:
        raise ValueError("MovieLens20M loader found no valid rating rows.")

    raw_users, raw_items, timestamps = downcast_numeric_arrays(
        data[:, 0].astype(np.int64),
        data[:, 1].astype(np.int64),
        data[:, 3].astype(np.int64),
    )
    ratings = data[:, 2].astype(np.float32)

    prepared = prepare_explicit_rating_feedback(raw_users, raw_items, ratings)
    indexed = prepared.indexed
    n_items = indexed.n_items
    item_map = indexed.item_map

    item_feature_block = None
    if include_optional_features:
        raw_dir = Path(data_dir) / "MovieLens20M" / "raw"
        genre_block = None
        genome_block = None
        load_genres, _ = resolve_feature_source(
            feature_policy,
            "movielens20m",
            "item_features",
            "raw/movies.csv",
        )
        if load_genres:
            genre_block = _load_movie_genres(raw_dir / "movies.csv", item_map, n_items)
        load_genome_scores, _ = resolve_feature_source(
            feature_policy,
            "movielens20m",
            "item_features",
            "raw/genome-scores.csv",
        )
        if load_genome_scores:
            genome_block = _load_genome_scores(
                raw_dir / "genome-scores.csv",
                item_map,
                n_items,
            )
        item_feature_block = stack_feature_metadata_blocks(genre_block, genome_block)

    effective_preset = preprocessing_preset or "movielens_explicit"

    return build_explicit_rating_canonical(
        prepared,
        timestamps,
        item_features=None if item_feature_block is None else item_feature_block.matrix,
        item_feature_names=None if item_feature_block is None else item_feature_block.names,
        item_feature_sources=None if item_feature_block is None else item_feature_block.sources,
        item_feature_raw_columns=(
            None if item_feature_block is None else item_feature_block.raw_features
        ),
        item_feature_roles=None if item_feature_block is None else item_feature_block.roles,
        item_feature_groups=None if item_feature_block is None else item_feature_block.groups,
        preprocessing_preset=effective_preset,
    )
