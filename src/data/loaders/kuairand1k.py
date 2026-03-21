"""KuaiRand-1K loader — log_standard_4_08_to_4_21_*.csv + optional features."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..canonical import CanonicalInteractions


def _load_csv_features(path: Path, id_col: str, id_map: dict[int, int], n_entities: int) -> np.ndarray | None:
    """Load a feature CSV, re-index rows by id_map, return (n_entities, F) float32 array."""
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        id_idx = header.index(id_col)
        feat_indices = [i for i in range(len(header)) if i != id_idx]

        rows: dict[int, list[float]] = {}
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < len(header):
                continue
            entity_id = int(parts[id_idx])
            if entity_id not in id_map:
                continue
            mapped_id = id_map[entity_id]
            feats = []
            for fi in feat_indices:
                try:
                    feats.append(float(parts[fi]))
                except (ValueError, IndexError):
                    feats.append(0.0)
            rows[mapped_id] = feats

    if not rows:
        return None

    n_feats = len(feat_indices)
    features = np.zeros((n_entities, n_feats), dtype=np.float32)
    for mapped_id, feats in rows.items():
        features[mapped_id] = feats

    return features


def _safe_col(parts: list[str], idx: int, default: str = "0") -> str:
    """Safely access a column by index, returning default if out of range."""
    if idx < 0 or idx >= len(parts):
        return default
    return parts[idx]


def _safe_int(value: str, default: int = 0) -> int:
    """Parse int from CSV field, returning default on failure."""
    try:
        return int(value)
    except (ValueError, IndexError):
        return default


def _safe_float(value: str, default: float = 0.0) -> float:
    """Parse float from CSV field, returning default on failure."""
    try:
        return float(value)
    except (ValueError, IndexError):
        return default


def load_kuairand1k(data_dir: str = "data") -> CanonicalInteractions:
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
            f"KuaiRand-1K not found at {base}. "
            "Download from https://kuairand.com/"
        )

    csv_files = sorted(base.glob("log_standard*.csv"))
    csv_files += sorted(base.glob("log_random*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No log_standard*.csv or log_random*.csv files found in {base}")

    # Collect all columns
    raw_users, raw_items, timestamps = [], [], []
    clicks, likes, hates = [], [], []
    follows, comments, forwards = [], [], []
    long_views = []
    play_times, durations = [], []
    is_rands = []

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
            forward_c = col.get("is_forward", -1)
            long_view_c = col.get("long_view", -1)
            play_time_c = col.get("play_time_ms", -1)
            duration_c = col.get("duration_ms", -1)
            is_rand_c = col.get("is_rand", -1)
            ts_c = col.get("time_ms", col.get("timestamp", -1))

            for line in f:
                parts = line.strip().split(",")
                if len(parts) <= max(uid_c, vid_c):
                    continue
                raw_users.append(int(parts[uid_c]))
                raw_items.append(int(parts[vid_c]))
                clicks.append(_safe_int(_safe_col(parts, click_c)))
                likes.append(_safe_int(_safe_col(parts, like_c)))
                hates.append(_safe_int(_safe_col(parts, hate_c)))
                follows.append(_safe_int(_safe_col(parts, follow_c)))
                comments.append(_safe_int(_safe_col(parts, comment_c)))
                forwards.append(_safe_int(_safe_col(parts, forward_c)))
                long_views.append(_safe_int(_safe_col(parts, long_view_c)))
                play_times.append(_safe_float(_safe_col(parts, play_time_c, "0.0")))
                durations.append(_safe_float(_safe_col(parts, duration_c, "0.0")))
                is_rands.append(_safe_int(_safe_col(parts, is_rand_c)))
                timestamps.append(_safe_int(_safe_col(parts, ts_c)))

    raw_users_arr = np.array(raw_users, dtype=np.int64)
    raw_items_arr = np.array(raw_items, dtype=np.int64)
    clicks_arr = np.array(clicks, dtype=np.int32)
    likes_arr = np.array(likes, dtype=np.int32)
    hates_arr = np.array(hates, dtype=np.int32)
    follows_arr = np.array(follows, dtype=np.int32)
    comments_arr = np.array(comments, dtype=np.int32)
    long_views_arr = np.array(long_views, dtype=np.int32)
    is_rand_arr = np.array(is_rands, dtype=np.int32)
    timestamps_arr = np.array(timestamps, dtype=np.int64)

    unique_users = np.unique(raw_users_arr)
    unique_items = np.unique(raw_items_arr)
    user_map = {int(uid): idx for idx, uid in enumerate(unique_users)}
    item_map = {int(iid): idx for idx, iid in enumerate(unique_items)}

    user_id = np.array([user_map[int(u)] for u in raw_users_arr], dtype=np.int64)
    item_id = np.array([item_map[int(i)] for i in raw_items_arr], dtype=np.int64)

    # Expanded label: click OR like OR follow
    label = ((clicks_arr > 0) | (likes_arr > 0) | (follows_arr > 0)).astype(np.float32)

    # Graded sign: like=1.0 > follow=0.7 > comment=0.5 > click+long_view=0.3 > neutral=0.0 > hate=-1.0
    sign = np.full(len(user_id), 0.0, dtype=np.float32)
    sign[hates_arr > 0] = -1.0
    sign[(clicks_arr > 0) & (long_views_arr > 0)] = 0.3
    sign[comments_arr > 0] = 0.5
    sign[follows_arr > 0] = 0.7
    sign[likes_arr > 0] = 1.0  # highest priority: last assignment wins

    n_users = len(unique_users)
    n_items = len(unique_items)
    pop_counts = np.bincount(item_id, minlength=n_items).astype(np.float32)
    popularity = pop_counts / pop_counts.max() if pop_counts.max() > 0 else pop_counts

    # Side features (optional)
    feat_base = Path(data_dir) / "KuaiRand-1K" / "data"
    user_features = _load_csv_features(
        feat_base / "user_features_1k.csv", "user_id", user_map, n_users
    )
    item_features_basic = _load_csv_features(
        feat_base / "video_features_basic_1k.csv", "video_id", item_map, n_items
    )
    item_features_stat = _load_csv_features(
        feat_base / "video_features_statistic_1k.csv", "video_id", item_map, n_items
    )
    # Concatenate basic + statistic features if both present
    if item_features_basic is not None and item_features_stat is not None:
        item_features = np.hstack([item_features_basic, item_features_stat])
    elif item_features_basic is not None:
        item_features = item_features_basic
    elif item_features_stat is not None:
        item_features = item_features_stat
    else:
        item_features = None

    # Store is_rand as metadata for causal analysis
    metadata = {"is_rand": is_rand_arr}

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
        metadata=metadata,
    )
