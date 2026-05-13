"""Taobao (UserBehavior) loader — UserBehavior.csv."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from ...utils.dataset_loader_utils import downcast_numeric_array, downcast_numeric_arrays
from ...utils.interaction_indexing import (
    build_repeat_collapse_canonical_payload,
    collapse_pairwise_max_priority_rows_with_stats,
    remap_interaction_ids,
)
from ..canonical import (
    CanonicalInteractions,
    build_indexed_canonical_interactions,
)
from ..feature_policy import (
    DEFAULT_FEATURE_POLICY,
    FeaturePolicyName,
    resolve_feature_source,
)

# Behavior -> sign mapping:  buy > cart > fav > pv
_BEHAVIOR_SIGN = {"buy": 1.0, "cart": 0.5, "fav": 0.25, "pv": -0.25}
_BEHAVIOR_LABEL = {"buy": 1.0, "cart": 1.0, "fav": 1.0, "pv": 0.0}
_BEHAVIOR_TARGET = {"buy": 3.0, "cart": 2.0, "fav": 1.0, "pv": 0.0}

logger = logging.getLogger(__name__)

_TAOBAO_COLUMNS = ["user_id", "item_id", "category_id", "behavior", "timestamp"]
_TAOBAO_SCHEMA: pl.Schema = pl.Schema(
    {
        "user_id": pl.Int64,
        "item_id": pl.Int64,
        "category_id": pl.Int64,
        "behavior": pl.String,
        "timestamp": pl.Int64,
    }
)


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
            f"Taobao dataset not found at {path}. Download UserBehavior.csv from Tianchi.",
        )

    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=_TAOBAO_COLUMNS,
        schema=_TAOBAO_SCHEMA,
        n_rows=max_rows,
        ignore_errors=True,
    )

    n_loaded = len(df)
    df = df.filter(pl.col("behavior").is_in(list(_BEHAVIOR_LABEL.keys())))
    unknown_behavior_rows = n_loaded - len(df)
    df = df.drop_nulls(subset=["user_id", "item_id", "category_id", "timestamp"])
    malformed_core_rows = (n_loaded - unknown_behavior_rows) - len(df)

    if malformed_core_rows > 0:
        logger.warning(
            "Taobao loader skipped %d malformed interaction rows.",
            malformed_core_rows,
        )
    if unknown_behavior_rows > 0:
        logger.warning(
            "Taobao loader skipped %d rows with unknown behavior labels.",
            unknown_behavior_rows,
        )
    if len(df) == 0:
        raise ValueError("Taobao loader found no valid interactions to load.")

    raw_users_arr, raw_items_arr, categories_arr, timestamps_arr = downcast_numeric_arrays(
        df["user_id"].to_numpy(),
        df["item_id"].to_numpy(),
        df["category_id"].to_numpy(),
        df["timestamp"].to_numpy(),
    )

    effective_preset = preprocessing_preset or "taobao_multibehavior"
    df = df.with_columns(
        pl.col("behavior").replace(_BEHAVIOR_LABEL, default=0.0).cast(pl.Float32).alias("label"),
        pl.col("behavior").replace(_BEHAVIOR_SIGN, default=0.0).cast(pl.Float32).alias("sign"),
        pl.col("behavior").replace(_BEHAVIOR_TARGET, default=0.0).cast(pl.Float32).alias("raw_target"),
    )
    behavior_type = np.asarray(df["behavior"].to_list(), dtype="<U16")
    label = df["label"].to_numpy()
    sign = df["sign"].to_numpy()
    raw_target = df["raw_target"].to_numpy()
    collapsed_rows = 0
    repeat_summary = None
    if effective_preset == "taobao_multibehavior":
        # Collapse repeated user-item pairs before split construction so one
        # raw pair cannot leak across train/val/test boundaries. The strongest
        # observed behavior wins, with timestamp as the tie-breaker.
        (
            (
                raw_users_arr,
                raw_items_arr,
                categories_arr,
                timestamps_arr,
                label,
                sign,
                raw_target,
                behavior_type,
            ),
            repeat_summary,
            collapsed_rows,
        ) = collapse_pairwise_max_priority_rows_with_stats(
            raw_users_arr,
            raw_items_arr,
            raw_target,
            categories_arr,
            timestamps_arr,
            label,
            sign,
            raw_target,
            behavior_type,
            timestamp=timestamps_arr,
        )

    indexed = remap_interaction_ids(raw_users_arr, raw_items_arr)
    item_id = indexed.item_id
    n_items = indexed.n_items

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
            np.array([cat_map[int(c)] for c in item_categories], dtype=np.int64),
        ).reshape(-1, 1)

    return build_indexed_canonical_interactions(
        indexed,
        label=label,
        timestamp=timestamps_arr,
        sign=sign,
        raw_target=raw_target,
        behavior_type=behavior_type,
        item_features=item_cat_reindexed,
        **build_repeat_collapse_canonical_payload(
            repeat_summary,
            applied=effective_preset == "taobao_multibehavior",
            dropped_rows=collapsed_rows,
            reason=("Collapse repeated raw user-item pairs before splitting so the same pair cannot span train/val/test; keep the strongest behavior with timestamp tie-breaks."),
        ),
        feedback_type="multi-behavior",
        preprocessing_preset=effective_preset,
    )
