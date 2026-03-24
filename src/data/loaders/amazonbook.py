"""Amazon-Book loader from local LightGCN-format raw files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..canonical import CanonicalInteractions
from ..feature_policy import DEFAULT_FEATURE_POLICY, FeaturePolicyName


def _parse_interaction_file(
    path: Path,
    max_rows: int | None = None,
) -> list[tuple[int, int]]:
    """Parse LightGCN-format interaction file.

    Each line: ``user_id item1 item2 ...`` (space-separated).
    Returns list of (user_id, item_id) pairs.
    """
    pairs: list[tuple[int, int]] = []
    row_count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            uid = int(tokens[0])
            for iid_str in tokens[1:]:
                pairs.append((uid, int(iid_str)))
                row_count += 1
                if max_rows is not None and row_count >= max_rows:
                    return pairs
    return pairs


def _resolve_raw_dir(data_dir: str) -> Path:
    """Resolve the local Amazon-Book raw directory without triggering downloads."""
    candidates = [
        Path(data_dir) / "AmazonBook" / "raw",
        Path(data_dir) / "AmazonBook" / "raw" / "amazon-book",
    ]
    required_files = {"train.txt", "test.txt", "user_list.txt", "item_list.txt"}
    for raw_dir in candidates:
        if all((raw_dir / name).exists() for name in required_files):
            return raw_dir
    raise FileNotFoundError(
        "AmazonBook raw files not found under data/AmazonBook/raw. "
        "Expected train.txt, test.txt, user_list.txt, and item_list.txt."
    )


def _build_arrays(
    train_pairs: list[tuple[int, int]],
    test_pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate train/test pairs and create split masks."""
    all_pairs = train_pairs + test_pairs
    raw_users = np.asarray([u for u, _ in all_pairs], dtype=np.int64)
    raw_items = np.asarray([i for _, i in all_pairs], dtype=np.int64)

    train_mask = np.zeros(len(all_pairs), dtype=bool)
    test_mask = np.zeros(len(all_pairs), dtype=bool)
    train_mask[: len(train_pairs)] = True
    test_mask[len(train_pairs) :] = True

    timestamps = np.zeros(len(all_pairs), dtype=np.int64)
    return raw_users, raw_items, train_mask, test_mask, timestamps


def load_amazonbook(
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
) -> CanonicalInteractions:
    """Load Amazon-Book from local LightGCN-format split files.

    This avoids PyG download side effects and uses the repository-local data
    folder directly. All interactions are implicit positives.
    """
    del include_optional_features, feature_policy
    raw_dir = _resolve_raw_dir(data_dir)
    train_target = max_rows
    test_target = None
    if max_rows is not None:
        train_target = max(1, int(max_rows * 0.8))
        test_target = max_rows - train_target

    train_pairs = _parse_interaction_file(raw_dir / "train.txt", max_rows=train_target)
    test_pairs = _parse_interaction_file(raw_dir / "test.txt", max_rows=test_target)
    raw_users, raw_items, train_mask, test_mask, timestamps = _build_arrays(
        train_pairs,
        test_pairs,
    )

    unique_users = np.unique(raw_users)
    unique_items = np.unique(raw_items)
    user_map = {int(uid): idx for idx, uid in enumerate(unique_users)}
    item_map = {int(iid): idx for idx, iid in enumerate(unique_items)}

    user_id = np.array([user_map[int(u)] for u in raw_users], dtype=np.int64)
    item_id = np.array([item_map[int(i)] for i in raw_items], dtype=np.int64)

    label = np.ones(len(user_id), dtype=np.float32)
    sign = np.ones(len(user_id), dtype=np.float32)

    n_items = len(unique_items)
    pop_counts = np.bincount(item_id, minlength=n_items).astype(np.float32)
    popularity = pop_counts / pop_counts.max() if pop_counts.max() > 0 else pop_counts

    return CanonicalInteractions(
        user_id=user_id,
        item_id=item_id,
        label=label,
        timestamp=timestamps,
        sign=sign,
        popularity=popularity,
        n_users=len(unique_users),
        n_items=n_items,
        user_map=user_map,
        item_map=item_map,
        train_mask=train_mask,
        test_mask=test_mask,
    )
