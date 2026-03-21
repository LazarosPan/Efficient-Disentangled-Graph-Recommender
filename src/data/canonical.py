"""CanonicalInteractions: unified intermediate representation for all datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CanonicalInteractions:
    """Dataset-agnostic interaction container.

    All loaders produce this; downstream code never touches raw formats.
    """

    user_id: np.ndarray       # (N,) int64 — re-indexed from 0
    item_id: np.ndarray       # (N,) int64 — re-indexed from 0
    label: np.ndarray         # (N,) float32 — 1.0 positive, 0.0 negative
    timestamp: np.ndarray     # (N,) int64 — unix seconds (0 if unavailable)
    sign: np.ndarray          # (N,) float32 — [-1, 1] continuous sentiment
    popularity: np.ndarray    # (I,) float32 — per-item interaction count (normalized)

    n_users: int
    n_items: int
    user_map: dict[int, int]  # original_id -> reindexed_id
    item_map: dict[int, int]  # original_id -> reindexed_id

    user_features: np.ndarray | None = None  # (n_users, F_u) optional side features
    item_features: np.ndarray | None = None  # (n_items, F_i) optional side features

    # Predefined split masks (set by loaders that have train/test files)
    train_mask: np.ndarray | None = None     # (N,) bool
    val_mask: np.ndarray | None = None       # (N,) bool
    test_mask: np.ndarray | None = None      # (N,) bool

    # Metadata (e.g., is_rand flags for causal analysis)
    metadata: dict | None = None

    def __len__(self) -> int:
        return len(self.user_id)

    def __repr__(self) -> str:
        uf = f", user_feat={self.user_features.shape}" if self.user_features is not None else ""
        itf = f", item_feat={self.item_features.shape}" if self.item_features is not None else ""
        splits = ", predefined_splits=True" if self.train_mask is not None else ""
        return (
            f"CanonicalInteractions(n_users={self.n_users}, n_items={self.n_items}, "
            f"interactions={len(self):,}, pos_rate={self.label.mean():.2%}{uf}{itf}{splits})"
        )

    def temporal_split(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (train_mask, val_mask, test_mask) arrays based on timestamp ordering."""
        n = len(self)
        order = np.argsort(self.timestamp)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)

        train_mask[order[:train_end]] = True
        val_mask[order[train_end:val_end]] = True
        test_mask[order[val_end:]] = True

        return train_mask, val_mask, test_mask

    def _split_existing_train_mask(
        self,
        train_mask: np.ndarray,
        val_ratio: float,
        train_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derive a validation mask from an existing train mask.

        Keeps any predefined test split intact and carves validation out of the
        current train pool using timestamp order when available.
        """
        train_indices = np.flatnonzero(train_mask)
        if train_indices.size == 0:
            return train_mask.copy(), np.zeros(len(self), dtype=bool)

        non_test_ratio = train_ratio + val_ratio
        if non_test_ratio <= 0:
            return train_mask.copy(), np.zeros(len(self), dtype=bool)

        val_fraction_within_train = val_ratio / non_test_ratio
        val_count = int(round(train_indices.size * val_fraction_within_train))
        if val_count <= 0:
            return train_mask.copy(), np.zeros(len(self), dtype=bool)
        if val_count >= train_indices.size:
            val_count = train_indices.size - 1
        if val_count <= 0:
            return train_mask.copy(), np.zeros(len(self), dtype=bool)

        timestamps = self.timestamp[train_indices]
        if timestamps.size > 0 and np.any(timestamps != timestamps[0]):
            ordered_train = train_indices[np.argsort(timestamps)]
        else:
            ordered_train = train_indices

        split_at = ordered_train.size - val_count
        new_train_mask = np.zeros(len(self), dtype=bool)
        val_mask = np.zeros(len(self), dtype=bool)
        new_train_mask[ordered_train[:split_at]] = True
        val_mask[ordered_train[split_at:]] = True
        return new_train_mask, val_mask

    def get_splits(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return split masks, preferring predefined splits over temporal splitting.

        If all three predefined masks are set, returns them directly.
        Otherwise falls back to ``temporal_split()``.
        """
        if (
            self.train_mask is not None
            and self.val_mask is not None
            and self.test_mask is not None
        ):
            return self.train_mask, self.val_mask, self.test_mask
        if self.train_mask is not None and self.test_mask is not None:
            train_mask, val_mask = self._split_existing_train_mask(
                self.train_mask,
                val_ratio,
                train_ratio,
            )
            return train_mask, val_mask, self.test_mask
        return self.temporal_split(train_ratio, val_ratio)
