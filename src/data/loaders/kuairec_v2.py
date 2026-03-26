"""KuaiRec v2 loader — big_matrix.csv + optional user/item features."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ...utils.dataset_loader_utils import try_parse_timestamp_seconds
from ..canonical import CanonicalInteractions
from ..feature_policy import DEFAULT_FEATURE_POLICY, FeaturePolicyName
from ...utils.csv_features import load_csv_features
from ...utils.interaction_indexing import (
    compute_normalized_popularity,
    remap_interaction_ids,
)

logger = logging.getLogger(__name__)


def _load_item_categories(
    path: Path, id_map: dict[int, int], n_items: int
) -> np.ndarray | None:
    """Load item_categories.csv -> (n_items, n_categories) multi-hot vector.

    Format: video_id, feat (where feat is a list-like string of category IDs).
    """
    if not path.exists():
        return None

    # First pass: collect all category IDs to determine dimensionality
    raw_categories: dict[int, list[int]] = {}
    all_cat_ids: set[int] = set()
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        vid_idx = header.index("video_id")
        feat_idx = header.index("feat")
        for line in f:
            parts = line.strip().split(",", len(header) - 1)
            if len(parts) <= max(vid_idx, feat_idx):
                continue
            vid = int(parts[vid_idx])
            if vid not in id_map:
                continue
            feat_str = parts[feat_idx].strip().strip("[]")
            cats = []
            for tok in feat_str.split():
                tok = tok.strip().rstrip(",")
                if tok:
                    try:
                        cats.append(int(float(tok)))
                        all_cat_ids.add(cats[-1])
                    except ValueError:
                        continue
            raw_categories[id_map[vid]] = cats

    if not raw_categories or not all_cat_ids:
        return None

    cat_to_idx = {c: i for i, c in enumerate(sorted(all_cat_ids))}
    n_cats = len(cat_to_idx)
    features = np.zeros((n_items, n_cats), dtype=np.float32)
    for mapped_id, cats in raw_categories.items():
        for c in cats:
            features[mapped_id, cat_to_idx[c]] = 1.0
    return features


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
        )
    )
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        vid_idx = header.index("video_id") if "video_id" in header else -1
        if vid_idx < 0:
            return None
        col_indices = []
        for tc in target_cols:
            if tc in header:
                col_indices.append(header.index(tc))
            else:
                col_indices.append(-1)

        if all(ci < 0 for ci in col_indices):
            return None

        features = np.zeros((n_items, len(target_cols)), dtype=np.float32)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= vid_idx:
                continue
            try:
                vid = int(parts[vid_idx])
            except ValueError:
                continue
            if vid not in id_map:
                continue
            mapped = id_map[vid]
            for j, ci in enumerate(col_indices):
                if ci >= 0 and ci < len(parts):
                    try:
                        features[mapped, j] = float(parts[ci])
                    except ValueError:
                        pass
    return features


def load_kuairec_v2(
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
) -> CanonicalInteractions:
    """Load KuaiRec v2 from ``data_dir/KuaiRec_v2/data/big_matrix.csv``.

    Expected columns: user_id, video_id, watch_ratio, timestamp
    Label: watch_ratio >= 0.5 -> positive
    Sign:  (watch_ratio clipped to [0,2] - 1) -> [-1, 1]
    """
    base = Path(data_dir) / "KuaiRec_v2" / "data"
    path = base / "big_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"KuaiRec v2 not found at {path}. Download from https://kuairec.com/"
        )

    raw_users, raw_items, watch_ratios, timestamps = [], [], [], []
    malformed_timestamp_count = 0
    row_count = 0
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        uid_col = header.index("user_id")
        vid_col = header.index("video_id")
        wr_col = header.index("watch_ratio")
        ts_col = header.index("timestamp") if "timestamp" in header else -1

        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(uid_col, vid_col, wr_col):
                continue
            raw_users.append(int(parts[uid_col]))
            raw_items.append(int(parts[vid_col]))
            watch_ratios.append(float(parts[wr_col]))
            raw_timestamp = parts[ts_col] if ts_col >= 0 else None
            parsed_timestamp = try_parse_timestamp_seconds(raw_timestamp)
            if raw_timestamp is not None and parsed_timestamp is None:
                malformed_timestamp_count += 1
            timestamps.append(parsed_timestamp if parsed_timestamp is not None else 0)
            row_count += 1
            if max_rows is not None and row_count >= max_rows:
                break

    if malformed_timestamp_count > 0:
        logger.warning(
            "KuaiRec v2 loader coerced %d malformed timestamp values to 0.",
            malformed_timestamp_count,
        )

    raw_users_arr = np.array(raw_users, dtype=np.int64)
    raw_items_arr = np.array(raw_items, dtype=np.int64)
    wr_arr = np.array(watch_ratios, dtype=np.float32)
    timestamps_arr = np.array(timestamps, dtype=np.int64)

    indexed = remap_interaction_ids(raw_users_arr, raw_items_arr)
    user_id = indexed.user_id
    item_id = indexed.item_id
    n_users = indexed.n_users
    n_items = indexed.n_items
    user_map = indexed.user_map
    item_map = indexed.item_map

    # Clamp outliers before computing sign (watch_ratio can exceed 100)
    wr_arr = np.clip(wr_arr, 0.0, 5.0)

    label = (wr_arr >= 0.5).astype(np.float32)
    sign = (np.clip(wr_arr, 0.0, 2.0) - 1.0).astype(np.float32)

    popularity = compute_normalized_popularity(item_id, n_items)

    user_features = None
    item_features = None
    if include_optional_features:
        if feature_policy == "all_optional":
            user_features = load_csv_features(
                base / "user_features.csv", "user_id", user_map, n_users
            )
            user_features_raw = load_csv_features(
                base / "user_features_raw.csv", "user_id", user_map, n_users
            )
            if user_features is not None and user_features_raw is not None:
                user_features = np.hstack([user_features, user_features_raw])
            elif user_features_raw is not None:
                user_features = user_features_raw

            item_features_daily = load_csv_features(
                base / "item_daily_features.csv", "video_id", item_map, n_items
            )
            item_caption_cats = _load_caption_categories(
                base / "kuairec_caption_category.csv", item_map, n_items
            )
        else:
            item_features_daily = load_csv_features(
                base / "item_daily_features.csv",
                "video_id",
                item_map,
                n_items,
                include_columns=(
                    "author_id",
                    "video_type",
                    "upload_dt",
                    "upload_type",
                    "visible_status",
                    "music_id",
                ),
            )
            item_caption_cats = _load_caption_categories(
                base / "kuairec_caption_category.csv",
                item_map,
                n_items,
                include_columns=(
                    "first_level_category_id",
                    "second_level_category_id",
                    "third_level_category_id",
                ),
            )

        item_cat_feats = _load_item_categories(
            base / "item_categories.csv", item_map, n_items
        )
        item_parts = [
            f
            for f in [item_features_daily, item_cat_feats, item_caption_cats]
            if f is not None
        ]
        item_features = np.hstack(item_parts) if item_parts else None

    return CanonicalInteractions(
        user_id=user_id,
        item_id=item_id,
        label=label,
        timestamp=timestamps_arr,
        sign=sign,
        popularity=popularity,
        n_users=n_users,
        n_items=n_items,
        user_map=user_map,
        item_map=item_map,
        user_features=user_features,
        item_features=item_features,
    )
