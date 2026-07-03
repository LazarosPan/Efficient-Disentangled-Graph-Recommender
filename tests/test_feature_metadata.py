"""Feature metadata propagation tests."""

from __future__ import annotations

import numpy as np
from src.data.canonical import CanonicalInteractions
from src.data.graph_builder import build_graph
from src.utils.config import EDGRecConfig
from src.utils.csv_features import (
    FeatureBlock,
    load_csv_feature_block,
    stack_feature_metadata_blocks,
)


def _canonical_with_item_features() -> CanonicalInteractions:
    """Return a tiny canonical dataset with item feature metadata."""
    return CanonicalInteractions(
        user_id=np.array([0, 0, 1], dtype=np.int64),
        item_id=np.array([0, 1, 1], dtype=np.int64),
        label=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        timestamp=np.array([1, 2, 3], dtype=np.int64),
        sign=np.array([1.0, -1.0, 1.0], dtype=np.float32),
        popularity=np.array([0.5, 1.0], dtype=np.float32),
        n_users=2,
        n_items=2,
        user_map={10: 0, 11: 1},
        item_map={20: 0, 21: 1},
        item_features=np.ones((2, 2), dtype=np.float32),
        item_feature_names=("movies::genres=Action", "movies::genres=Drama"),
        item_feature_sources=("raw/movies.dat", "raw/movies.dat"),
        item_feature_raw_columns=("genres", "genres"),
        item_feature_roles=("safe_pre_treatment", "safe_pre_treatment"),
        item_feature_groups=("item_genre", "item_genre"),
    )


def test_feature_metadata_length_equals_feature_matrix_width() -> None:
    """Canonical feature metadata must match matrix width."""
    canonical = _canonical_with_item_features()

    assert canonical.item_features is not None
    assert len(canonical.item_feature_names or ()) == canonical.item_features.shape[1]
    assert len(canonical.item_feature_sources or ()) == canonical.item_features.shape[1]
    assert len(canonical.item_feature_raw_columns or ()) == canonical.item_features.shape[1]
    assert len(canonical.item_feature_roles or ()) == canonical.item_features.shape[1]
    assert len(canonical.item_feature_groups or ()) == canonical.item_features.shape[1]


def test_feature_metadata_propagates_to_graph_data() -> None:
    """PyG Data carries aligned feature metadata for reporting and subset search."""
    canonical = _canonical_with_item_features()
    data = build_graph(canonical, EDGRecConfig(dataset="movielens1m", device="cpu"))

    assert data.item_feature_names == canonical.item_feature_names
    assert data.item_feature_sources == canonical.item_feature_sources
    assert data.item_feature_raw_columns == canonical.item_feature_raw_columns
    assert data.item_feature_groups == canonical.item_feature_groups


def test_csv_feature_block_names_preserve_column_order(tmp_path) -> None:
    """CSV metadata uses source-stem names in encoded column order."""
    path = tmp_path / "item_daily_features.csv"
    path.write_text(
        "video_id,author_id,music_id,video_type\n10,100,7,short\n11,101,8,long\n",
        encoding="utf-8",
    )

    block = load_csv_feature_block(
        path,
        "video_id",
        {10: 0, 11: 1},
        2,
        include_columns=("author_id", "music_id", "video_type"),
        dataset_name="kuairec_v2",
        aspect="item_features",
        relative_path="data/item_daily_features.csv",
    )

    assert block is not None
    assert block.names == (
        "item_daily_features::author_id",
        "item_daily_features::music_id",
        "item_daily_features::video_type",
    )
    assert block.raw_features == ("author_id", "music_id", "video_type")
    assert block.matrix.shape == (2, 3)


def test_csv_listlike_tag_feature_is_multihot_with_ordered_metadata(tmp_path) -> None:
    """List-like tag features expand to multi-hot columns with aligned metadata."""
    path = tmp_path / "video_features_basic_1k.csv"
    path.write_text(
        'video_id,video_duration,tag\n10,4.0,"2,1"\n11,8.0,3\n',
        encoding="utf-8",
    )

    block = load_csv_feature_block(
        path,
        "video_id",
        {10: 0, 11: 1},
        2,
        include_columns=("video_duration", "tag"),
        dataset_name="kuairand1k",
        aspect="item_features",
        relative_path="data/video_features_basic_1k.csv",
    )

    assert block is not None
    assert block.names == (
        "video_features_basic_1k::video_duration",
        "video_features_basic_1k::tag=1",
        "video_features_basic_1k::tag=2",
        "video_features_basic_1k::tag=3",
    )
    assert block.raw_features == ("video_duration", "tag", "tag", "tag")
    assert block.groups == (
        "item_video_metadata",
        "item_category",
        "item_category",
        "item_category",
    )
    assert block.matrix.shape == (2, 4)


def test_stacked_feature_metadata_preserves_column_order() -> None:
    """Stacking feature blocks keeps matrix and metadata column order aligned."""
    left = FeatureBlock(
        matrix=np.ones((2, 1), dtype=np.float32),
        names=("a::x",),
        sources=("a",),
        roles=("safe_pre_treatment",),
        groups=("other_safe_item_feature",),
        raw_features=("x",),
    )
    right = FeatureBlock(
        matrix=np.zeros((2, 2), dtype=np.float32),
        names=("b::y", "b::z"),
        sources=("b", "b"),
        roles=("safe_pre_treatment", "proxy_only"),
        groups=("item_category", "item_category"),
        raw_features=("y", "z"),
    )

    stacked = stack_feature_metadata_blocks(left, right)

    assert stacked is not None
    assert stacked.matrix.shape == (2, 3)
    assert stacked.names == ("a::x", "b::y", "b::z")
    assert stacked.raw_features == ("x", "y", "z")
