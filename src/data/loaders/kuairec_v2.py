"""KuaiRec v2 loader — big_matrix.csv + optional user/item features."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import polars as pl

from ...utils.csv_features import (
    PolicyCsvFeatureSpec,
    load_policy_csv_feature_blocks,
    stack_feature_blocks,
)
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

logger = logging.getLogger(__name__)

KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_5 = "kuairec_big_matrix_watch_ratio_threshold_0_5"
KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_75 = "kuairec_big_matrix_watch_ratio_threshold_0_75"
KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_1_0 = "kuairec_big_matrix_watch_ratio_threshold_1_0"
KUIREC_BIG_MATRIX_WATCH_RATIO_RAW = "kuairec_big_matrix_watch_ratio_raw"
KUIREC_SMALL_MATRIX_FULL_OBSERVATION = "kuairec_small_matrix_full_observation"

_KUIREC_LEGACY_PRESET_ALIASES = {
    "kuairec_watchratio": KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_5,
    "kuairec_watchratio_wr_0_75": KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_75,
    "kuairec_watchratio_wr_1_0": KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_1_0,
    "kuairec_watchratio_raw": KUIREC_BIG_MATRIX_WATCH_RATIO_RAW,
    "kuairec_fullobs": KUIREC_SMALL_MATRIX_FULL_OBSERVATION,
}
_KUIREC_PRESET_MATRIX_VARIANTS = {
    KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_5: "big_matrix",
    KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_75: "big_matrix",
    KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_1_0: "big_matrix",
    KUIREC_BIG_MATRIX_WATCH_RATIO_RAW: "big_matrix",
    KUIREC_SMALL_MATRIX_FULL_OBSERVATION: "small_matrix",
}
_KUIREC_MATRIX_DEFAULT_PRESETS = {
    "big_matrix": KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_5,
    "small_matrix": KUIREC_SMALL_MATRIX_FULL_OBSERVATION,
}
_KUIREC_CLIPPED_WATCHRATIO_PRESETS = frozenset(
    (
        KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_5,
        KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_75,
        KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_1_0,
    ),
)
_KUIREC_WATCH_RATIO_THRESHOLDS = {
    KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_5: 0.5,
    KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_75: 0.75,
    KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_1_0: 1.0,
}


def _parse_listlike_ints(raw_value: str) -> list[int]:
    """Parse a list-like CSV field such as ``[27, 9]`` into integers.

    Args:
        raw_value: Raw CSV field content.

    Returns:
        Parsed integer values in encounter order.

    """
    cleaned = raw_value.strip().strip('"').strip("'").strip()
    if not cleaned:
        return []
    cleaned = cleaned.strip("[]").strip()
    if not cleaned:
        return []

    values: list[int] = []
    for token in cleaned.replace(",", " ").split():
        try:
            values.append(int(float(token)))
        except ValueError:
            continue
    return values


def _load_item_categories(
    path: Path,
    id_map: dict[int, int],
    n_items: int,
) -> np.ndarray | None:
    """Load item_categories.csv -> (n_items, n_categories) multi-hot vector.

    Format: video_id, feat (where feat is a list-like string of category IDs).
    """
    if not path.exists():
        return None

    # First pass: collect all category IDs to determine dimensionality
    raw_categories: dict[int, list[int]] = {}
    all_cat_ids: set[int] = set()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row.get("video_id")
            if video_id is None:
                continue
            try:
                vid = int(float(video_id))
            except ValueError:
                continue
            if vid not in id_map:
                continue
            cats = _parse_listlike_ints(row.get("feat", ""))
            all_cat_ids.update(cats)
            raw_categories[id_map[vid]] = cats

    if not raw_categories or not all_cat_ids:
        return None

    cat_to_idx = {c: i for i, c in enumerate(sorted(all_cat_ids))}
    n_cats = len(cat_to_idx)
    features = np.zeros((n_items, n_cats), dtype=np.uint8)
    for mapped_id, cats in raw_categories.items():
        for c in cats:
            features[mapped_id, cat_to_idx[c]] = 1.0
    return features


def _encode_caption_category_features(features: np.ndarray) -> np.ndarray:
    """Encode dense caption category IDs as bounded categorical codes.

    Args:
        features: Integer category-ID matrix with ``0`` reserved for missing.

    Returns:
        Float32 matrix with non-missing IDs mapped to ``1/N..1`` per column.

    """
    encoded = np.zeros(features.shape, dtype=np.float32)
    for column_index in range(features.shape[1]):
        column = features[:, column_index]
        unique_values, inverse = np.unique(column, return_inverse=True)
        non_missing = unique_values != 0
        n_values = int(non_missing.sum())
        if n_values == 0:
            continue
        codes = np.zeros(unique_values.shape, dtype=np.float32)
        codes[non_missing] = np.arange(1, n_values + 1, dtype=np.float32) / float(n_values)
        encoded[:, column_index] = codes[inverse]
    return encoded


def _load_caption_categories(
    path: Path,
    id_map: dict[int, int],
    n_items: int,
    include_columns: tuple[str, ...] | None = None,
) -> np.ndarray | None:
    """Load hierarchical category IDs from kuairec_caption_category.csv.

    Uses only first/second/third_level_category_id columns (skips text fields).
    Returns (n_items, 3) ordinal-encoded array.
    """
    if not path.exists():
        return None

    target_cols = list(
        include_columns
        or (
            "first_level_category_id",
            "second_level_category_id",
            "third_level_category_id",
        ),
    )
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "video_id" not in reader.fieldnames:
            return None
        available_cols = {column for column in target_cols if column in reader.fieldnames}
        if not available_cols:
            return None
        features = np.zeros((n_items, len(target_cols)), dtype=np.int32)
        for row in reader:
            video_id = row.get("video_id")
            if video_id is None:
                continue
            try:
                vid = int(float(video_id))
            except ValueError:
                continue
            if vid not in id_map:
                continue
            mapped = id_map[vid]
            for column_index, column_name in enumerate(target_cols):
                if column_name not in available_cols:
                    continue
                raw_value = row.get(column_name, "")
                if raw_value in {"", None}:
                    continue
                try:
                    features[mapped, column_index] = int(float(raw_value))
                except (TypeError, ValueError):
                    continue
    return downcast_numeric_array(
        _encode_caption_category_features(features),
        allow_float16=True,
    )


def _resolve_kuairec_view(
    preprocessing_preset: str | None,
    matrix_variant: str | None,
) -> tuple[str, str]:
    """Resolve the KuaiRec preprocessing preset and backing matrix.

    ``preprocessing_preset`` is the semantic run contract. ``matrix_variant`` is
    kept for direct loader compatibility and must agree when both are supplied.
    """
    if preprocessing_preset is not None:
        preprocessing_preset = _KUIREC_LEGACY_PRESET_ALIASES.get(
            preprocessing_preset,
            preprocessing_preset,
        )
        if preprocessing_preset not in _KUIREC_PRESET_MATRIX_VARIANTS:
            available = ", ".join(sorted(_KUIREC_PRESET_MATRIX_VARIANTS))
            raise ValueError(
                (
                    f"Unknown KuaiRec preprocessing_preset '{preprocessing_preset}'. "
                    f"Available presets: {available}"
                ),
            )
        expected_matrix = _KUIREC_PRESET_MATRIX_VARIANTS[preprocessing_preset]
        if matrix_variant is not None and matrix_variant != expected_matrix:
            raise ValueError(
                (
                    "KuaiRec preprocessing_preset and matrix_variant conflict: "
                    f"preprocessing_preset='{preprocessing_preset}' requires "
                    f"matrix_variant='{expected_matrix}', got '{matrix_variant}'."
                ),
            )
        return preprocessing_preset, expected_matrix

    if matrix_variant is not None:
        if matrix_variant not in _KUIREC_MATRIX_DEFAULT_PRESETS:
            raise ValueError("matrix_variant must be 'big_matrix' or 'small_matrix'")
        return _KUIREC_MATRIX_DEFAULT_PRESETS[matrix_variant], matrix_variant

    return KUIREC_BIG_MATRIX_WATCH_RATIO_THRESHOLD_0_5, "big_matrix"


def load_kuairec_v2(
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
    preprocessing_preset: str | None = None,
    matrix_variant: str | None = None,
) -> CanonicalInteractions:
    """Load KuaiRec v2 from ``data_dir/KuaiRec_v2/data/<matrix_variant>.csv``.

    Expected columns: user_id, video_id, watch_ratio, timestamp
    The default view uses ``big_matrix`` /
    ``kuairec_big_matrix_watch_ratio_threshold_0_5`` so the repository uses the
    sparse, realistic ranking benchmark by default. The ``small_matrix`` /
    ``kuairec_small_matrix_full_observation`` path remains available via an
    explicit ``preprocessing_preset`` when the near-fully-observed variant is
    needed for comparison.
    Label: watch_ratio >= preset threshold -> positive
    Sign:  (watch_ratio clipped to [0,2] - 1) -> [-1, 1]
    """
    effective_preset, matrix_variant = _resolve_kuairec_view(
        preprocessing_preset,
        matrix_variant,
    )
    base = Path(data_dir) / "KuaiRec_v2" / "data"
    path = base / f"{matrix_variant}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"KuaiRec v2 not found at {path}. Download from https://kuairec.com/",
        )

    df = pl.read_csv(
        path,
        schema_overrides={
            "user_id": pl.Int64,
            "video_id": pl.Int64,
            "watch_ratio": pl.Float32,
            "timestamp": pl.Int64,
        },
        n_rows=max_rows,
        ignore_errors=True,
    )
    n_loaded = len(df)
    df = df.drop_nulls(subset=["user_id", "video_id", "watch_ratio"])
    df = df.filter(pl.col("watch_ratio").is_finite())
    malformed_core_rows = n_loaded - len(df)

    if malformed_core_rows > 0:
        logger.warning(
            "KuaiRec v2 loader skipped %d malformed or non-finite interaction rows.",
            malformed_core_rows,
        )
    if len(df) == 0:
        raise ValueError("KuaiRec v2 loader found no valid interaction rows.")
    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp").fill_null(0).cast(pl.Int64))

    raw_users_arr, raw_items_arr = downcast_numeric_arrays(
        df["user_id"].to_numpy(),
        df["video_id"].to_numpy(),
    )
    raw_watch_ratio = df["watch_ratio"].to_numpy()
    timestamps_arr = (
        downcast_numeric_arrays(df["timestamp"].to_numpy())[0]
        if "timestamp" in df.columns
        else np.zeros(len(df), dtype=np.int32)
    )

    stabilized_watch_ratio = (
        np.clip(raw_watch_ratio, 0.0, 5.0)
        if effective_preset in _KUIREC_CLIPPED_WATCHRATIO_PRESETS
        else raw_watch_ratio
    )
    collapsed_rows = 0
    repeat_summary = None
    if effective_preset in _KUIREC_CLIPPED_WATCHRATIO_PRESETS:
        # Collapse repeated user-item pairs before split construction so one
        # watch-ratio trajectory cannot leak across train/val/test boundaries.
        # The retained row is the strongest stabilized watch-ratio observation,
        # with timestamp as the secondary tie-breaker.
        (
            (
                raw_users_arr,
                raw_items_arr,
                timestamps_arr,
                stabilized_watch_ratio,
            ),
            repeat_summary,
            collapsed_rows,
        ) = collapse_pairwise_max_priority_rows_with_stats(
            raw_users_arr,
            raw_items_arr,
            stabilized_watch_ratio,
            timestamps_arr,
            stabilized_watch_ratio,
            timestamp=timestamps_arr,
        )

    indexed = remap_interaction_ids(raw_users_arr, raw_items_arr)
    user_map = indexed.user_map
    item_map = indexed.item_map
    n_users = indexed.n_users
    n_items = indexed.n_items

    watch_ratio_threshold = _KUIREC_WATCH_RATIO_THRESHOLDS.get(effective_preset, 0.5)
    label = (stabilized_watch_ratio >= watch_ratio_threshold).astype(np.float32)
    sign = (np.clip(stabilized_watch_ratio, 0.0, 2.0) - 1.0).astype(np.float32)

    user_features = None
    item_features = None
    if include_optional_features:
        user_features = load_policy_csv_feature_blocks(
            feature_policy=feature_policy,
            dataset_name="kuairec_v2",
            aspect="user_features",
            id_map=user_map,
            n_entities=n_users,
            sources=(
                PolicyCsvFeatureSpec(
                    path=base / "user_features.csv",
                    relative_path="data/user_features.csv",
                    id_col="user_id",
                ),
                PolicyCsvFeatureSpec(
                    path=base / "user_features_raw.csv",
                    relative_path="data/user_features_raw.csv",
                    id_col="user_id",
                ),
            ),
        )

        item_caption_cats = None
        item_cat_feats = None
        item_features_daily = load_policy_csv_feature_blocks(
            feature_policy=feature_policy,
            dataset_name="kuairec_v2",
            aspect="item_features",
            id_map=item_map,
            n_entities=n_items,
            sources=(
                PolicyCsvFeatureSpec(
                    path=base / "item_daily_features.csv",
                    relative_path="data/item_daily_features.csv",
                    id_col="video_id",
                ),
            ),
        )
        load_caption_categories, caption_columns = resolve_feature_source(
            feature_policy,
            "kuairec_v2",
            "item_features",
            "data/kuairec_caption_category.csv",
        )
        if load_caption_categories:
            item_caption_cats = _load_caption_categories(
                base / "kuairec_caption_category.csv",
                item_map,
                n_items,
                include_columns=caption_columns,
            )
        load_item_categories, _ = resolve_feature_source(
            feature_policy,
            "kuairec_v2",
            "item_features",
            "data/item_categories.csv",
        )
        if load_item_categories:
            item_cat_feats = _load_item_categories(
                base / "item_categories.csv",
                item_map,
                n_items,
            )
        item_features = stack_feature_blocks(
            item_features_daily,
            item_cat_feats,
            item_caption_cats,
        )

    source_domain = np.full(len(indexed.user_id), matrix_variant, dtype="<U16")
    return build_indexed_canonical_interactions(
        indexed,
        label=label,
        timestamp=timestamps_arr,
        sign=sign,
        raw_target=stabilized_watch_ratio,
        source_domain=source_domain,
        user_features=user_features,
        item_features=item_features,
        **build_repeat_collapse_canonical_payload(
            repeat_summary,
            applied=effective_preset in _KUIREC_CLIPPED_WATCHRATIO_PRESETS,
            dropped_rows=collapsed_rows,
            reason=(
                "Collapse repeated raw user-item pairs before splitting so the "
                "same pair cannot span train/val/test; keep the strongest "
                "watch-ratio observation with timestamp tie-breaks."
            ),
            metadata={
                "matrix_variant": matrix_variant,
                "watch_ratio_policy": (
                    "clipped_to_5"
                    if effective_preset in _KUIREC_CLIPPED_WATCHRATIO_PRESETS
                    else "raw"
                ),
                "watch_ratio_threshold": watch_ratio_threshold,
            },
        ),
        feedback_type="implicit",
        preprocessing_preset=effective_preset,
    )
