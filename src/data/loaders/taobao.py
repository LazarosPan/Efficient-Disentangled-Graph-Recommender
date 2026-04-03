"""Taobao (UserBehavior) loader — UserBehavior.csv."""

from __future__ import annotations

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

# Behavior -> sign mapping:  buy > cart > fav > pv
_BEHAVIOR_SIGN = {"buy": 1.0, "cart": 0.5, "fav": 0.25, "pv": -0.25}
_BEHAVIOR_LABEL = {"buy": 1.0, "cart": 1.0, "fav": 1.0, "pv": 0.0}
_BEHAVIOR_TARGET = {"buy": 3.0, "cart": 2.0, "fav": 1.0, "pv": 0.0}


def load_taobao(
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
    preprocessing_preset: str | None = None,
) -> CanonicalInteractions:
    """Load Taobao UserBehavior from ``data_dir/Taobao/raw/UserBehavior.csv``.

    Format: UserID,ItemID,CategoryID,BehaviorType,Timestamp (no header)
    """
    path = Path(data_dir) / "Taobao" / "raw" / "UserBehavior.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Taobao dataset not found at {path}. "
            "Download UserBehavior.csv from Tianchi."
        )

    raw_users, raw_items, categories, behaviors, timestamps = [], [], [], [], []
    row_count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            raw_users.append(int(parts[0]))
            raw_items.append(int(parts[1]))
            categories.append(int(parts[2]))
            behaviors.append(parts[3])
            timestamps.append(int(parts[4]))
            row_count += 1
            if max_rows is not None and row_count >= max_rows:
                break

    raw_users_arr = downcast_numeric_array(np.array(raw_users, dtype=np.int64))
    raw_items_arr = downcast_numeric_array(np.array(raw_items, dtype=np.int64))
    categories_arr = downcast_numeric_array(np.array(categories, dtype=np.int64))
    timestamps_arr = downcast_numeric_array(np.array(timestamps, dtype=np.int64))

    indexed = remap_interaction_ids(raw_users_arr, raw_items_arr)
    user_id = indexed.user_id
    item_id = indexed.item_id
    n_users = indexed.n_users
    n_items = indexed.n_items
    user_map = indexed.user_map
    item_map = indexed.item_map

    label = np.array([_BEHAVIOR_LABEL.get(b, 0.0) for b in behaviors], dtype=np.float32)
    sign = np.array([_BEHAVIOR_SIGN.get(b, 0.0) for b in behaviors], dtype=np.float32)
    raw_target = np.array(
        [_BEHAVIOR_TARGET.get(b, 0.0) for b in behaviors],
        dtype=np.float32,
    )
    behavior_type = np.asarray(behaviors, dtype="<U16")

    popularity = compute_normalized_popularity(item_id, n_items)

    item_cat_reindexed = None
    load_item_features, _ = resolve_feature_source(
        feature_policy,
        "taobao",
        "item_features",
        "raw/UserBehavior.csv",
    )
    if include_optional_features and load_item_features:
        # Item features: category_id as a single-column feature per item.
        item_categories = np.zeros(n_items, dtype=categories_arr.dtype)
        item_categories[item_id] = categories_arr

        unique_cats = np.unique(item_categories)
        cat_map = {int(c): idx for idx, c in enumerate(unique_cats)}
        item_cat_reindexed = downcast_numeric_array(
            np.array([cat_map[int(c)] for c in item_categories], dtype=np.int64)
        ).reshape(-1, 1)

    effective_preset = preprocessing_preset or "taobao_multibehavior"

    return CanonicalInteractions(
        user_id=user_id,
        item_id=item_id,
        label=label,
        timestamp=timestamps_arr,
        sign=sign,
        raw_target=raw_target,
        behavior_type=behavior_type,
        popularity=popularity,
        n_users=n_users,
        n_items=n_items,
        user_map=user_map,
        item_map=item_map,
        item_features=item_cat_reindexed,
        feedback_type="multi-behavior",
        preprocessing_preset=effective_preset,
    )
