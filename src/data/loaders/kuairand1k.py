"""KuaiRand-1K loader — log_standard_4_08_to_4_21_*.csv + optional features."""

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
from ...utils.csv_features import load_csv_features
from ...utils.interaction_indexing import (
    compute_normalized_popularity,
    remap_interaction_ids,
)

logger = logging.getLogger(__name__)


def load_kuairand1k(
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
    preprocessing_preset: str | None = None,
) -> CanonicalInteractions:
    """Load KuaiRand-1K from ``data_dir/KuaiRand-1K/data/``.

    Scans for ``log_standard*.csv`` and ``log_random*.csv`` files.
    The random-exposure logs (is_rand=1) are critical for causal analysis.
    Uses expanded engagement signals for richer label/sign computation:
    - Label: click OR like OR follow -> positive
    - Sign (graded): like=1.0, follow=0.7, comment=0.5, click+long_view=0.3,
      neutral=0.0, hate=-1.0
    - is_rand flag preserved as metadata for causal analysis
    """
    base = Path(data_dir) / "KuaiRand-1K" / "data"
    if not base.exists():
        raise FileNotFoundError(
            f"KuaiRand-1K not found at {base}. Download from https://kuairand.com/"
        )

    csv_files = sorted(base.glob("log_standard*.csv"))
    csv_files += sorted(base.glob("log_random*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No log_standard*.csv or log_random*.csv files found in {base}"
        )

    # Collect all columns
    raw_users, raw_items, timestamps = [], [], []
    clicks, likes, hates = [], [], []
    follows, comments = [], []
    long_views = []
    play_times, durations = [], []
    is_rands = []
    malformed_counts = {
        "is_click": 0,
        "is_like": 0,
        "is_hate": 0,
        "is_follow": 0,
        "is_comment": 0,
        "long_view": 0,
        "play_time_ms": 0,
        "duration_ms": 0,
        "is_rand": 0,
        "time_ms|timestamp": 0,
    }

    row_count = 0
    for csv_path in csv_files:
        with open(csv_path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            col = {name: idx for idx, name in enumerate(header)}

            uid_c = col["user_id"]
            vid_c = col["video_id"]
            click_c = col.get("is_click", -1)
            like_c = col.get("is_like", -1)
            hate_c = col.get("is_hate", -1)
            follow_c = col.get("is_follow", -1)
            comment_c = col.get("is_comment", -1)
            long_view_c = col.get("long_view", -1)
            play_time_c = col.get("play_time_ms", -1)
            duration_c = col.get("duration_ms", -1)
            is_rand_c = col.get("is_rand", -1)
            ts_c = col.get("time_ms", col.get("timestamp", -1))
            optional_fields = (
                ("is_click", click_c, int, clicks, 0),
                ("is_like", like_c, int, likes, 0),
                ("is_hate", hate_c, int, hates, 0),
                ("is_follow", follow_c, int, follows, 0),
                ("is_comment", comment_c, int, comments, 0),
                ("long_view", long_view_c, int, long_views, 0),
                ("play_time_ms", play_time_c, float, play_times, 0.0),
                ("duration_ms", duration_c, float, durations, 0.0),
                ("is_rand", is_rand_c, int, is_rands, 0),
                ("time_ms|timestamp", ts_c, int, timestamps, 0),
            )

            for line in f:
                parts = line.strip().split(",")
                if len(parts) <= max(uid_c, vid_c):
                    continue
                raw_users.append(int(parts[uid_c]))
                raw_items.append(int(parts[vid_c]))
                for (
                    field_name,
                    column_idx,
                    parser,
                    target_values,
                    default_value,
                ) in optional_fields:
                    parsed_value = default_value
                    if 0 <= column_idx < len(parts):
                        try:
                            parsed_value = parser(parts[column_idx])
                        except (TypeError, ValueError):
                            malformed_counts[field_name] += 1
                    target_values.append(parsed_value)
                row_count += 1
                if max_rows is not None and row_count >= max_rows:
                    break
        if max_rows is not None and row_count >= max_rows:
            break

    malformed_counts = {
        name: count for name, count in malformed_counts.items() if count > 0
    }
    if malformed_counts:
        logger.warning(
            "KuaiRand-1K loader coerced malformed optional field values to neutral defaults: %s",
            malformed_counts,
        )

    raw_users_arr = downcast_numeric_array(np.array(raw_users, dtype=np.int64))
    raw_items_arr = downcast_numeric_array(np.array(raw_items, dtype=np.int64))
    clicks_arr = np.array(clicks, dtype=np.uint8)
    likes_arr = np.array(likes, dtype=np.uint8)
    hates_arr = np.array(hates, dtype=np.uint8)
    follows_arr = np.array(follows, dtype=np.uint8)
    comments_arr = np.array(comments, dtype=np.uint8)
    long_views_arr = np.array(long_views, dtype=np.uint8)
    play_times_arr = downcast_numeric_array(np.array(play_times, dtype=np.float32))
    durations_arr = downcast_numeric_array(np.array(durations, dtype=np.float32))
    is_rand_arr = np.array(is_rands, dtype=np.uint8)
    timestamps_arr = downcast_numeric_array(np.array(timestamps, dtype=np.int64))

    indexed = remap_interaction_ids(raw_users_arr, raw_items_arr)
    user_id = indexed.user_id
    item_id = indexed.item_id
    n_users = indexed.n_users
    n_items = indexed.n_items
    user_map = indexed.user_map
    item_map = indexed.item_map

    # Expanded label: click OR like OR follow
    label = ((clicks_arr > 0) | (likes_arr > 0) | (follows_arr > 0)).astype(np.float32)

    # Graded sign: like=1.0 > follow=0.7 > comment=0.5 > click+long_view=0.3 > neutral=0.0 > hate=-1.0
    sign = np.full(len(user_id), 0.0, dtype=np.float32)
    sign[hates_arr > 0] = -1.0
    sign[(clicks_arr > 0) & (long_views_arr > 0)] = 0.3
    sign[comments_arr > 0] = 0.5
    sign[follows_arr > 0] = 0.7
    sign[likes_arr > 0] = 1.0  # highest priority: last assignment wins

    behavior_type = np.full(len(user_id), "neutral", dtype="<U10")
    behavior_type[hates_arr > 0] = "hate"
    behavior_type[clicks_arr > 0] = "click"
    behavior_type[(clicks_arr > 0) & (long_views_arr > 0)] = "long_view"
    behavior_type[comments_arr > 0] = "comment"
    behavior_type[follows_arr > 0] = "follow"
    behavior_type[likes_arr > 0] = "like"

    raw_target = np.zeros(len(user_id), dtype=np.float32)
    valid_duration = durations_arr > 0
    raw_target[valid_duration] = np.clip(
        play_times_arr[valid_duration] / durations_arr[valid_duration],
        0.0,
        5.0,
    ).astype(np.float32)

    popularity = compute_normalized_popularity(item_id, n_items)

    user_features = None
    item_features = None
    if include_optional_features:
        feat_base = Path(data_dir) / "KuaiRand-1K" / "data"
        load_user_features, user_feature_columns = resolve_feature_source(
            feature_policy,
            "kuairand1k",
            "user_features",
            "data/user_features_1k.csv",
        )
        if load_user_features:
            user_features = load_csv_features(
                feat_base / "user_features_1k.csv",
                "user_id",
                user_map,
                n_users,
                include_columns=user_feature_columns,
            )

        item_features_basic = None
        item_features_stat = None
        load_basic_item_features, basic_item_columns = resolve_feature_source(
            feature_policy,
            "kuairand1k",
            "item_features",
            "data/video_features_basic_1k.csv",
        )
        if load_basic_item_features:
            item_features = load_csv_features(
                feat_base / "video_features_basic_1k.csv",
                "video_id",
                item_map,
                n_items,
                include_columns=basic_item_columns,
            )
            item_features_basic = item_features
        load_stat_item_features, stat_item_columns = resolve_feature_source(
            feature_policy,
            "kuairand1k",
            "item_features",
            "data/video_features_statistic_1k.csv",
        )
        if load_stat_item_features:
            item_features_stat = load_csv_features(
                feat_base / "video_features_statistic_1k.csv",
                "video_id",
                item_map,
                n_items,
                include_columns=stat_item_columns,
            )
        if item_features_basic is not None and item_features_stat is not None:
            item_features = np.hstack([item_features_basic, item_features_stat])
        elif item_features_basic is not None:
            item_features = item_features_basic
        elif item_features_stat is not None:
            item_features = item_features_stat

    # Store is_rand as metadata for causal analysis
    metadata = {"is_rand": is_rand_arr.astype(bool, copy=False)}
    source_domain = np.where(is_rand_arr > 0, "random", "standard").astype("<U8")
    effective_preset = preprocessing_preset or "kuairand_causal"

    return CanonicalInteractions(
        user_id=user_id,
        item_id=item_id,
        label=label,
        timestamp=timestamps_arr,
        sign=sign,
        raw_target=raw_target,
        behavior_type=behavior_type,
        exposure_flag=is_rand_arr.astype(bool),
        source_domain=source_domain,
        popularity=popularity,
        n_users=n_users,
        n_items=n_items,
        user_map=user_map,
        item_map=item_map,
        user_features=user_features,
        item_features=item_features,
        metadata=metadata,
        feedback_type="randomized-exposure",
        preprocessing_preset=effective_preset,
    )
