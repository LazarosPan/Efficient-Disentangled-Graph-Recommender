"""CanonicalInteractions: unified intermediate representation for all datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


DerivedSplitMode = Literal["per_user_temporal", "global_temporal"]


@dataclass
class CanonicalInteractions:
    """Dataset-agnostic interaction container.

    All loaders produce this; downstream code never touches raw formats.
    """

    user_id: np.ndarray  # (N,) integer — re-indexed from 0
    item_id: np.ndarray  # (N,) integer — re-indexed from 0
    label: np.ndarray  # (N,) float32 — 1.0 positive, 0.0 negative
    timestamp: np.ndarray  # (N,) integer — unix seconds (0 if unavailable)
    sign: np.ndarray  # (N,) float32 — [-1, 1] continuous sentiment
    popularity: np.ndarray  # (I,) float32 — per-item interaction count (normalized)

    n_users: int
    n_items: int
    user_map: dict[int, int]  # original_id -> reindexed_id
    item_map: dict[int, int]  # original_id -> reindexed_id

    user_features: np.ndarray | None = None  # (n_users, F_u) optional side features
    item_features: np.ndarray | None = None  # (n_items, F_i) optional side features

    raw_target: np.ndarray | None = (
        None  # (N,) float32/float64 optional pre-binarized target
    )
    behavior_type: np.ndarray | None = None  # (N,) optional behavior labels
    exposure_flag: np.ndarray | None = None  # (N,) optional randomized-exposure mask
    source_domain: np.ndarray | None = None  # (N,) optional interaction-domain labels
    feedback_type: str | None = None  # dataset-level feedback semantics descriptor
    preprocessing_preset: str | None = (
        None  # dataset-specific preprocessing preset label
    )

    # Predefined split masks (set by loaders that have train/test files)
    train_mask: np.ndarray | None = None  # (N,) bool
    val_mask: np.ndarray | None = None  # (N,) bool
    test_mask: np.ndarray | None = None  # (N,) bool

    # Metadata (e.g., is_rand flags for causal analysis)
    metadata: dict | None = None

    def __len__(self) -> int:
        return len(self.user_id)

    def __repr__(self) -> str:
        uf = (
            f", user_feat={self.user_features.shape}"
            if self.user_features is not None
            else ""
        )
        itf = (
            f", item_feat={self.item_features.shape}"
            if self.item_features is not None
            else ""
        )
        splits = ", predefined_splits=True" if self.train_mask is not None else ""
        feedback = f", feedback={self.feedback_type}" if self.feedback_type else ""
        preset = (
            f", preset={self.preprocessing_preset}"
            if self.preprocessing_preset is not None
            else ""
        )
        return (
            f"CanonicalInteractions(n_users={self.n_users}, n_items={self.n_items}, "
            f"interactions={len(self):,}, pos_rate={self.label.mean():.2%}"
            f"{uf}{itf}{feedback}{preset}{splits})"
        )

    def temporal_split(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return global temporal split masks ordered by interaction timestamp.

        Args:
            train_ratio: Fraction of interactions assigned to training.
            val_ratio: Fraction of interactions assigned to validation.

        Returns:
            Tuple of boolean masks ``(train_mask, val_mask, test_mask)``.
        """
        return self._temporal_split_masks(
            interaction_indices=np.arange(len(self), dtype=np.int64),
            user_id=self.user_id,
            timestamp=self.timestamp,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_mode="global_temporal",
        )

    def per_user_temporal_split(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-user temporal split masks ordered within each user history.

        Args:
            train_ratio: Fraction of each user's history assigned to training.
            val_ratio: Fraction of each user's history assigned to validation.

        Returns:
            Tuple of boolean masks ``(train_mask, val_mask, test_mask)``.
        """
        return self._temporal_split_masks(
            interaction_indices=np.arange(len(self), dtype=np.int64),
            user_id=self.user_id,
            timestamp=self.timestamp,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_mode="per_user_temporal",
        )

    def _temporal_split_masks(
        self,
        interaction_indices: np.ndarray,
        user_id: np.ndarray,
        timestamp: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        split_mode: DerivedSplitMode,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build split masks for a full interaction set or a subset view.

        Args:
            interaction_indices: Absolute indices into the canonical interaction table.
            user_id: User IDs aligned to ``interaction_indices``.
            timestamp: Timestamps aligned to ``interaction_indices``.
            train_ratio: Fraction of rows reserved for training.
            val_ratio: Fraction of rows reserved for validation.
            split_mode: Whether to split globally or within each user history.

        Returns:
            Tuple of boolean masks over the full canonical interaction table.
        """
        n_total = len(self)
        train_mask = np.zeros(n_total, dtype=bool)
        val_mask = np.zeros(n_total, dtype=bool)
        test_mask = np.zeros(n_total, dtype=bool)

        if interaction_indices.size == 0:
            return train_mask, val_mask, test_mask
        if interaction_indices.size == 1:
            train_mask[interaction_indices[0]] = True
            return train_mask, val_mask, test_mask

        if split_mode == "per_user_temporal":
            order = np.lexsort((interaction_indices, timestamp, user_id))
            ordered_users = user_id[order]
            user_boundaries = np.flatnonzero(np.diff(ordered_users)) + 1
            starts = np.concatenate(
                [np.array([0], dtype=np.int64), user_boundaries.astype(np.int64)]
            )
            ends = np.concatenate(
                [
                    user_boundaries.astype(np.int64),
                    np.array([order.size], dtype=np.int64),
                ]
            )
            for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
                self._assign_split_slice(
                    ordered_indices=interaction_indices[order[start:end]],
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                )
            return train_mask, val_mask, test_mask

        order = np.lexsort((interaction_indices, timestamp))
        self._assign_split_slice(
            ordered_indices=interaction_indices[order],
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        return train_mask, val_mask, test_mask

    @staticmethod
    def _assign_split_slice(
        ordered_indices: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
        train_ratio: float,
        val_ratio: float,
    ) -> None:
        """Assign one ordered interaction slice into train/val/test masks.

        Args:
            ordered_indices: Interaction indices already sorted temporally.
            train_mask: Output training mask updated in place.
            val_mask: Output validation mask updated in place.
            test_mask: Output test mask updated in place.
            train_ratio: Fraction assigned to train.
            val_ratio: Fraction assigned to validation.

        Returns:
            None. The provided masks are updated in place.
        """
        count = int(ordered_indices.size)
        if count <= 0:
            return
        if count == 1:
            train_mask[ordered_indices[0]] = True
            return

        train_end = int(count * train_ratio)
        val_end = int(count * (train_ratio + val_ratio))

        train_end = max(1, train_end)
        train_end = min(train_end, count - 1)
        val_end = max(train_end, val_end)
        val_end = min(val_end, count - 1)

        train_mask[ordered_indices[:train_end]] = True
        val_mask[ordered_indices[train_end:val_end]] = True
        test_mask[ordered_indices[val_end:]] = True

    def _split_existing_train_mask(
        self,
        train_mask: np.ndarray,
        val_ratio: float,
        train_ratio: float,
        split_mode: DerivedSplitMode,
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

        inner_train_ratio = 1.0 - val_fraction_within_train
        subset_train_mask, subset_val_mask, _ = self._temporal_split_masks(
            interaction_indices=train_indices,
            user_id=self.user_id[train_indices],
            timestamp=self.timestamp[train_indices],
            train_ratio=inner_train_ratio,
            val_ratio=val_fraction_within_train,
            split_mode=split_mode,
        )
        return subset_train_mask, subset_val_mask

    def get_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        derived_split_mode: DerivedSplitMode = "per_user_temporal",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return split masks, preferring predefined splits over temporal splitting.

        If all three predefined masks are set, returns them directly.
        Otherwise falls back to ``temporal_split()``.

        Raises:
            ValueError: If any two of the returned masks share indices, which
                would allow test-set information to leak into training.
        """
        if (
            self.train_mask is not None
            and self.val_mask is not None
            and self.test_mask is not None
        ):
            train_mask, val_mask, test_mask = (
                self.train_mask,
                self.val_mask,
                self.test_mask,
            )
        elif self.train_mask is not None and self.test_mask is not None:
            train_mask, val_mask = self._split_existing_train_mask(
                self.train_mask,
                val_ratio,
                train_ratio,
                derived_split_mode,
            )
            test_mask = self.test_mask
        else:
            split_fn = (
                self.per_user_temporal_split
                if derived_split_mode == "per_user_temporal"
                else self.temporal_split
            )
            train_mask, val_mask, test_mask = split_fn(train_ratio, val_ratio)

        if (
            np.any(train_mask & val_mask)
            or np.any(train_mask & test_mask)
            or np.any(val_mask & test_mask)
        ):
            raise ValueError(
                "Data integrity violation: train, val, and test splits must be "
                "mutually exclusive. Check that the dataset loader assigns each "
                "interaction to exactly one split."
            )

        return train_mask, val_mask, test_mask

    def compute_item_recency(self, interaction_mask: np.ndarray) -> np.ndarray:
        """Return a per-item normalized recency summary for the given split mask.

        Args:
            interaction_mask: Boolean mask selecting the interaction subset whose
                timestamps should define item recency.

        Returns:
            Float array of shape ``(n_items,)`` with values in ``[0, 1]``.
            Items without valid timestamps receive ``0``.
        """
        if interaction_mask.shape[0] != len(self):
            raise ValueError("interaction_mask must have one entry per interaction")

        valid_mask = interaction_mask & (self.timestamp > 0)
        if not np.any(valid_mask):
            return np.zeros(self.n_items, dtype=np.float32)

        latest_timestamp = np.zeros(self.n_items, dtype=np.int64)
        np.maximum.at(
            latest_timestamp,
            self.item_id[valid_mask],
            self.timestamp[valid_mask],
        )

        observed = latest_timestamp > 0
        recency = np.zeros(self.n_items, dtype=np.float32)
        observed_timestamps = latest_timestamp[observed]
        if observed_timestamps.size == 0:
            return recency

        min_timestamp = observed_timestamps.min()
        max_timestamp = observed_timestamps.max()
        if max_timestamp == min_timestamp:
            recency[observed] = 1.0
            return recency

        scale = float(max_timestamp - min_timestamp)
        recency[observed] = (observed_timestamps - min_timestamp) / scale
        return recency
