"""Amazon-Book loader from local LightGCN-format raw files."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ...utils.dataset_loader_utils import resolve_local_dataset_dir
from ...utils.interaction_indexing import (
    remap_interaction_ids,
)
from ..canonical import (
    CanonicalInteractions,
    build_indexed_canonical_interactions,
)
from ..feature_policy import DEFAULT_FEATURE_POLICY, FeaturePolicyName

logger = logging.getLogger(__name__)


def _parse_interaction_file(
    path: Path,
    max_rows: int | None = None,
) -> list[tuple[int, int]]:
    """Parse LightGCN-format interaction file.

    Each line: ``user_id item1 item2 ...`` (space-separated).
    Returns list of (user_id, item_id) pairs.
    """
    pairs: list[tuple[int, int]] = []
    malformed_rows = 0
    row_count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                malformed_rows += 1
                continue
            try:
                uid = int(tokens[0])
            except ValueError:
                malformed_rows += 1
                continue
            for iid_str in tokens[1:]:
                try:
                    iid = int(iid_str)
                except ValueError:
                    malformed_rows += 1
                    continue
                pairs.append((uid, iid))
                row_count += 1
                if max_rows is not None and row_count >= max_rows:
                    if malformed_rows > 0:
                        logger.warning(
                            "AmazonBook loader skipped %d malformed interaction tokens in %s.",
                            malformed_rows,
                            path.name,
                        )
                    return pairs
    if malformed_rows > 0:
        logger.warning(
            "AmazonBook loader skipped %d malformed interaction tokens in %s.",
            malformed_rows,
            path.name,
        )
    return pairs


def _resolve_raw_dir(data_dir: str) -> Path:
    """Resolve the local Amazon-Book raw directory without triggering downloads."""
    return resolve_local_dataset_dir(
        candidates=[
            Path(data_dir) / "AmazonBook" / "raw",
            Path(data_dir) / "AmazonBook" / "raw" / "amazon-book",
        ],
        required_files=["train.txt", "test.txt", "user_list.txt", "item_list.txt"],
        missing_message=("AmazonBook raw files not found under data/AmazonBook/raw. Expected train.txt, test.txt, user_list.txt, and item_list.txt."),
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
    preprocessing_preset: str | None = None,
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

    indexed = remap_interaction_ids(raw_users, raw_items)
    user_id = indexed.user_id

    label = np.ones(len(user_id), dtype=np.float32)
    sign = np.zeros(len(user_id), dtype=np.float32)
    effective_preset = preprocessing_preset or "amazonbook_graph_only"

    return build_indexed_canonical_interactions(
        indexed,
        label=label,
        timestamp=timestamps,
        sign=sign,
        train_mask=train_mask,
        test_mask=test_mask,
        feedback_type="implicit",
        preprocessing_preset=effective_preset,
    )
