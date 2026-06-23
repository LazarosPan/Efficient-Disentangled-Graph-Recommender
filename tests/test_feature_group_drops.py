"""Feature-group perturbation tests."""

from __future__ import annotations

import numpy as np
import torch
from src.data.canonical import CanonicalInteractions
from src.data.feature_groups import (
    apply_graph_item_feature_subset,
    infer_feature_group,
    loaded_thesis_safe_item_feature_groups,
    required_feature_subset_profiles,
    zero_item_feature_subset_columns,
)
from src.data.graph_builder import build_graph
from src.utils.config import EDGRecConfig


def _tiny_canonical() -> CanonicalInteractions:
    """Return canonical interactions with two item feature groups."""
    return CanonicalInteractions(
        user_id=np.array([0, 0, 1, 1], dtype=np.int64),
        item_id=np.array([0, 1, 1, 2], dtype=np.int64),
        label=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        timestamp=np.array([1, 2, 3, 4], dtype=np.int64),
        sign=np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float32),
        popularity=np.array([0.5, 1.0, 0.5], dtype=np.float32),
        n_users=2,
        n_items=3,
        user_map={10: 0, 11: 1},
        item_map={20: 0, 21: 1, 22: 2},
        item_features=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        ),
        item_feature_names=("movies::genres=Action", "basic::author_id", "basic::server_width"),
        item_feature_sources=("movies", "basic", "basic"),
        item_feature_raw_columns=("genres", "author_id", "server_width"),
        item_feature_roles=(
            "safe_pre_treatment",
            "safe_pre_treatment",
            "safe_pre_treatment",
        ),
        item_feature_groups=("item_genre", "item_author_music", "item_resolution"),
    )


def test_feature_group_assignment_is_deterministic() -> None:
    """Known source/column pairs map to stable thesis feature groups."""
    cases = {
        ("movielens1m", "movies.dat", "genres", "safe_pre_treatment"): "item_genre",
        (
            "kuairec_v2",
            "item_daily_features.csv",
            "author_id",
            "safe_pre_treatment",
        ): "item_author_music",
        (
            "kuairec_v2",
            "item_daily_features.csv",
            "upload_dt",
            "safe_pre_treatment",
        ): "item_upload_time",
        (
            "kuairec_v2",
            "item_daily_features.csv",
            "server_width",
            "safe_pre_treatment",
        ): "item_resolution",
        (
            "kuairec_v2",
            "item_daily_features.csv",
            "video_duration",
            "safe_pre_treatment",
        ): "item_video_metadata",
        (
            "kuairand1k",
            "video_features_basic_1k.csv",
            "tag",
            "safe_pre_treatment",
        ): "item_category",
        (
            "taobao",
            "raw/UserBehavior.csv",
            "category_id",
            "safe_pre_treatment",
        ): "item_category",
        ("movielens1m", "users.dat", "gender", "safe_pre_treatment"): "user_demographic",
    }

    assert {case: infer_feature_group(*case) for case in cases} == cases


def test_feature_subset_exclude_preserves_matrix_shape() -> None:
    """Graph-level exclude_groups preserves item feature matrix shape."""
    canonical = _tiny_canonical()
    config = EDGRecConfig(
        dataset="movielens1m",
        use_features=True,
        use_popularity_head=False,
        feature_subset_mode="exclude_groups",
        feature_exclude_groups=["item_author_music"],
        device="cpu",
    )
    data = build_graph(canonical, config)
    before_shape = tuple(data.item_features.shape)

    apply_graph_item_feature_subset(data, config)

    assert tuple(data.item_features.shape) == before_shape
    assert torch.all(data.item_features[:, 1] == 0)
    assert torch.any(data.item_features[:, 0] != 0)
    assert torch.any(data.item_features[:, 2] != 0)


def test_feature_groups_are_derived_from_loaded_columns() -> None:
    """Loaded thesis-safe item groups come from actual item feature metadata."""
    canonical = _tiny_canonical()

    assert loaded_thesis_safe_item_feature_groups(canonical) == (
        "item_genre",
        "item_author_music",
        "item_resolution",
    )


def test_graph_only_dataset_is_not_applicable() -> None:
    """No loaded item features yields only the graph-only control profile."""
    canonical = CanonicalInteractions(
        user_id=np.array([0], dtype=np.int64),
        item_id=np.array([0], dtype=np.int64),
        label=np.array([1.0], dtype=np.float32),
        timestamp=np.array([1], dtype=np.int64),
        sign=np.array([1.0], dtype=np.float32),
        popularity=np.array([1.0], dtype=np.float32),
        n_users=1,
        n_items=1,
        user_map={1: 0},
        item_map={1: 0},
    )

    groups = loaded_thesis_safe_item_feature_groups(canonical)

    assert groups == ()
    assert required_feature_subset_profiles(groups) == ("graph_only",)


def test_include_groups_zeroes_non_selected_columns_and_preserves_shape() -> None:
    """include_groups keeps selected groups and zeros all others."""
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    subset = zero_item_feature_subset_columns(
        matrix,
        feature_groups=("item_genre", "item_author_music", "item_resolution"),
        mode="include_groups",
        include_groups=["item_genre"],
        exclude_groups=[],
    )

    assert tuple(subset.shape) == tuple(matrix.shape)
    assert torch.equal(subset, torch.tensor([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]))


def test_exclude_groups_zeroes_only_selected_columns_and_preserves_shape() -> None:
    """exclude_groups zeros only the named groups."""
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    subset = zero_item_feature_subset_columns(
        matrix,
        feature_groups=("item_genre", "item_author_music", "item_resolution"),
        mode="exclude_groups",
        include_groups=[],
        exclude_groups=["item_author_music"],
    )

    assert tuple(subset.shape) == tuple(matrix.shape)
    assert torch.equal(subset, torch.tensor([[1.0, 0.0, 3.0], [4.0, 0.0, 6.0]]))
