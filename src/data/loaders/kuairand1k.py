"""KuaiRand-1K loader — log_standard_4_08_to_4_21_*.csv + optional features."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from ...utils.csv_features import (
    PolicyCsvFeatureSpec,
    load_policy_csv_feature_blocks,
)
from ...utils.dataset_loader_utils import downcast_numeric_arrays
from ...utils.interaction_indexing import (
    build_repeat_collapse_canonical_payload,
    collapse_pairwise_max_priority_rows_with_stats,
    remap_interaction_ids,
    summarize_pairwise_aligned_counts,
)
from ..canonical import (
    CanonicalInteractions,
    build_indexed_canonical_interactions,
)
from ..feature_policy import (
    DEFAULT_FEATURE_POLICY,
    FeaturePolicyName,
)

logger = logging.getLogger(__name__)

# Optional behavior columns and their neutral fill values
_OPTIONAL_UINT_COLS = (
    "is_click",
    "is_like",
    "is_hate",
    "is_follow",
    "is_comment",
    "long_view",
    "is_rand",
)
_OPTIONAL_FLOAT_COLS = ("play_time_ms", "duration_ms")
_REPEAT_BEHAVIOR_LABELS = (
    "click",
    "like",
    "follow",
    "comment",
    "hate",
    "long_view",
)


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
    - Sign (graded): like=1.0, follow=0.7, comment=0.0, click+long_view=0.3,
      neutral=0.0, hate=-1.0
    - is_rand flag preserved as metadata for causal analysis
    """
    base = Path(data_dir) / "KuaiRand-1K" / "data"
    if not base.exists():
        raise FileNotFoundError(
            f"KuaiRand-1K not found at {base}. Download from https://kuairand.com/",
        )

    csv_files = sorted(base.glob("log_standard*.csv"))
    csv_files += sorted(base.glob("log_random*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No log_standard*.csv or log_random*.csv files found in {base}",
        )

    # Load all CSV files and concatenate; diagonal_relaxed fills missing columns with null
    dfs = [
        pl.read_csv(
            csv_path,
            schema_overrides={
                "user_id": pl.Int64,
                "video_id": pl.Int64,
                "time_ms": pl.Int64,
                "timestamp": pl.Int64,
                "play_time_ms": pl.Float32,
                "duration_ms": pl.Float32,
            },
            ignore_errors=True,
        )
        for csv_path in csv_files
    ]
    df = pl.concat(dfs, how="diagonal_relaxed")
    if max_rows is not None:
        df = df.head(max_rows)

    # Unify timestamp column: prefer time_ms, fall back to timestamp
    if "time_ms" in df.columns and "timestamp" in df.columns:
        df = (
            df.with_columns(
                pl.coalesce(["time_ms", "timestamp"]).alias("_ts"),
            )
            .drop(["time_ms", "timestamp"])
            .rename({"_ts": "time_ms"})
        )
    elif "timestamp" in df.columns:
        df = df.rename({"timestamp": "time_ms"})

    n_loaded = len(df)
    df = df.drop_nulls(subset=["user_id", "video_id"])
    malformed_core_rows = n_loaded - len(df)
    if malformed_core_rows > 0:
        logger.warning(
            "KuaiRand-1K loader skipped %d malformed core interaction rows.",
            malformed_core_rows,
        )
    if len(df) == 0:
        raise ValueError("KuaiRand-1K loader found no valid interaction rows.")

    # Fill missing optional columns with 0
    for col in _OPTIONAL_UINT_COLS:
        df = (
            df.with_columns(pl.lit(0, dtype=pl.UInt8).alias(col))
            if col not in df.columns
            else df.with_columns(pl.col(col).fill_null(0).cast(pl.UInt8))
        )
    for col in _OPTIONAL_FLOAT_COLS:
        df = (
            df.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(col))
            if col not in df.columns
            else df.with_columns(pl.col(col).fill_nan(0.0).fill_null(0.0).cast(pl.Float32))
        )
    df = (
        df.with_columns(pl.lit(0, dtype=pl.Int64).alias("time_ms"))
        if "time_ms" not in df.columns
        else df.with_columns(pl.col("time_ms").fill_null(0).cast(pl.Int64))
    )

    raw_users_arr, raw_items_arr, play_times_arr, durations_arr, timestamps_arr = (
        downcast_numeric_arrays(
            df["user_id"].to_numpy(),
            df["video_id"].to_numpy(),
            df["play_time_ms"].to_numpy(),
            df["duration_ms"].to_numpy(),
            df["time_ms"].to_numpy(),
        )
    )
    clicks_arr = df["is_click"].to_numpy()
    likes_arr = df["is_like"].to_numpy()
    hates_arr = df["is_hate"].to_numpy()
    follows_arr = df["is_follow"].to_numpy()
    comments_arr = df["is_comment"].to_numpy()
    long_views_arr = df["long_view"].to_numpy()
    is_rand_arr = df["is_rand"].to_numpy()

    effective_preset = preprocessing_preset or "kuairand_causal"
    if effective_preset == "kuairand_random_only":
        keep_mask = is_rand_arr > 0
        if not np.any(keep_mask):
            raise ValueError("KuaiRand-1K random-only preset found no randomized rows.")
        raw_users_arr = raw_users_arr[keep_mask]
        raw_items_arr = raw_items_arr[keep_mask]
        clicks_arr = clicks_arr[keep_mask]
        likes_arr = likes_arr[keep_mask]
        hates_arr = hates_arr[keep_mask]
        follows_arr = follows_arr[keep_mask]
        comments_arr = comments_arr[keep_mask]
        long_views_arr = long_views_arr[keep_mask]
        play_times_arr = play_times_arr[keep_mask]
        durations_arr = durations_arr[keep_mask]
        is_rand_arr = is_rand_arr[keep_mask]
        timestamps_arr = timestamps_arr[keep_mask]

    # Compute watch-ratio priority before collapsing so the best interaction per
    # user-item pair is retained consistently across both presets. The collapse
    # happens before split construction so one raw pair cannot leak across
    # train/val/test boundaries; timestamp breaks ties between equally strong
    # outcomes.
    raw_target = np.zeros(len(raw_users_arr), dtype=np.float32)
    valid_dur_mask = durations_arr > 0
    raw_target[valid_dur_mask] = np.clip(
        play_times_arr[valid_dur_mask] / durations_arr[valid_dur_mask],
        0.0,
        5.0,
    ).astype(np.float32)
    repeat_behavior_counts = summarize_pairwise_aligned_counts(
        raw_users_arr,
        raw_items_arr,
        raw_target,
        np.column_stack(
            (
                clicks_arr > 0,
                likes_arr > 0,
                follows_arr > 0,
                comments_arr > 0,
                hates_arr > 0,
                long_views_arr > 0,
            ),
        ).astype(np.uint8, copy=False),
        timestamp=timestamps_arr,
    )

    (
        (
            raw_users_arr,
            raw_items_arr,
            clicks_arr,
            likes_arr,
            hates_arr,
            follows_arr,
            comments_arr,
            long_views_arr,
            play_times_arr,
            durations_arr,
            is_rand_arr,
            timestamps_arr,
            raw_target,
        ),
        repeat_summary,
        collapsed_rows,
    ) = collapse_pairwise_max_priority_rows_with_stats(
        raw_users_arr,
        raw_items_arr,
        raw_target,
        clicks_arr,
        likes_arr,
        hates_arr,
        follows_arr,
        comments_arr,
        long_views_arr,
        play_times_arr,
        durations_arr,
        is_rand_arr,
        timestamps_arr,
        raw_target,
        timestamp=timestamps_arr,
    )
    if collapsed_rows > 0:
        logger.info(
            "KuaiRand-1K collapsed %d repeated user-item pairs (preset=%s).",
            collapsed_rows,
            effective_preset,
        )

    indexed = remap_interaction_ids(raw_users_arr, raw_items_arr)
    user_map = indexed.user_map
    item_map = indexed.item_map
    n_users = indexed.n_users
    n_items = indexed.n_items

    # Expanded label: click OR like OR follow
    label = ((clicks_arr > 0) | (likes_arr > 0) | (follows_arr > 0)).astype(np.float32)

    # Graded sign: like=1.0 > follow=0.7 > click+long_view=0.3 > neutral/comment=0.0 > hate=-1.0
    sign = np.full(len(indexed.user_id), 0.0, dtype=np.float32)
    sign[hates_arr > 0] = -1.0
    sign[(clicks_arr > 0) & (long_views_arr > 0)] = 0.3
    sign[follows_arr > 0] = 0.7
    sign[likes_arr > 0] = 1.0  # highest priority: last assignment wins

    behavior_type = np.full(len(indexed.user_id), "neutral", dtype="<U10")
    behavior_type[hates_arr > 0] = "hate"
    behavior_type[clicks_arr > 0] = "click"
    behavior_type[(clicks_arr > 0) & (long_views_arr > 0)] = "long_view"
    behavior_type[comments_arr > 0] = "comment"
    behavior_type[follows_arr > 0] = "follow"
    behavior_type[likes_arr > 0] = "like"

    user_features = None
    item_features = None
    if include_optional_features:
        user_features = load_policy_csv_feature_blocks(
            feature_policy=feature_policy,
            dataset_name="kuairand1k",
            aspect="user_features",
            id_map=user_map,
            n_entities=n_users,
            sources=(
                PolicyCsvFeatureSpec(
                    path=base / "user_features_1k.csv",
                    relative_path="data/user_features_1k.csv",
                    id_col="user_id",
                ),
            ),
        )
        item_features = load_policy_csv_feature_blocks(
            feature_policy=feature_policy,
            dataset_name="kuairand1k",
            aspect="item_features",
            id_map=item_map,
            n_entities=n_items,
            sources=(
                PolicyCsvFeatureSpec(
                    path=base / "video_features_basic_1k.csv",
                    relative_path="data/video_features_basic_1k.csv",
                    id_col="video_id",
                ),
                PolicyCsvFeatureSpec(
                    path=base / "video_features_statistic_1k.csv",
                    relative_path="data/video_features_statistic_1k.csv",
                    id_col="video_id",
                ),
            ),
        )

    # Store is_rand as metadata for causal analysis
    source_domain = np.where(is_rand_arr > 0, "random", "standard").astype("<U8")

    return build_indexed_canonical_interactions(
        indexed,
        label=label,
        timestamp=timestamps_arr,
        sign=sign,
        raw_target=raw_target,
        behavior_type=behavior_type,
        exposure_flag=is_rand_arr.astype(bool),
        source_domain=source_domain,
        **build_repeat_collapse_canonical_payload(
            repeat_summary,
            applied=True,
            dropped_rows=collapsed_rows,
            reason=(
                "Collapse repeated raw user-item pairs before splitting so the "
                "same pair cannot span train/val/test; keep the strongest "
                "watch-ratio outcome with timestamp tie-breaks."
            ),
            repeat_behavior_counts=repeat_behavior_counts,
            repeat_behavior_labels=np.asarray(_REPEAT_BEHAVIOR_LABELS, dtype="<U12"),
            metadata={
                "is_rand": is_rand_arr.astype(bool, copy=False),
                "exposure_summary": {
                    "randomized_count": int(np.sum(is_rand_arr > 0)),
                    "standard_count": int(np.sum(is_rand_arr == 0)),
                    "view": effective_preset,
                },
            },
        ),
        user_features=user_features,
        item_features=item_features,
        feedback_type="randomized-exposure",
        preprocessing_preset=effective_preset,
    )
