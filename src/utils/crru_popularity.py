"""Training-popularity reconstruction helpers for CRRU reporting.

CRRU uses raw PyG ``AveragePopularity@K`` as a logged metric, then normalizes
that raw ARP by the largest item interaction count in the run's training graph.
The denominator is deterministic graph metadata, not a learned model outcome, so
historical reports can reconstruct it when the stored run configuration is
sufficient to rebuild the exact train split.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np

from src.data.canonical import (
    CanonicalInteractions,
    filter_canonical_interactions,
    sample_canonical_interactions,
)
from src.data.feature_policy import DEFAULT_FEATURE_POLICY
from src.data.interaction_masks import positive_interaction_mask
from src.data.loaders import default_preprocessing_preset, load_dataset
from src.utils.config import DEFAULT_SEED, EDGRecConfig
from src.utils.interaction_indexing import compute_popularity_counts


class CRRUPopularityReconstructionError(ValueError):
    """Raised when CRRU cannot reconstruct train-popularity metadata safely."""


def resolve_largest_training_item_interaction_count(
    *,
    stored_value: float | None,
    config: Mapping[str, Any] | None,
    dataset: str | None,
) -> float:
    """Return the CRRU popularity denominator from stored or reconstructed data.

    Args:
        stored_value: Logged ``largest_training_item_interaction_count`` when a
            run already has it.
        config: Stored run configuration. Only data/split/preprocessing fields
            are used for reconstruction.
        dataset: Dataset name from the experiment row or trial.

    Returns:
        The largest positive-train item interaction count for the run's training
        graph.

    Raises:
        CRRUPopularityReconstructionError: If the stored value is invalid or the
            configuration does not contain enough information to rebuild the
            train graph deterministically.

    """
    if stored_value is not None:
        value = float(stored_value)
        if not math.isfinite(value) or value < 0.0:
            raise CRRUPopularityReconstructionError(
                "invalid stored largest_training_item_interaction_count",
            )
        return value

    if not dataset and not (config and config.get("dataset")):
        raise CRRUPopularityReconstructionError("missing dataset")

    key = _reconstruction_key(config or {}, dataset)
    return _reconstruct_largest_training_item_interaction_count_cached(key)


def _reconstruction_key(
    config: Mapping[str, Any],
    dataset: str | None,
) -> tuple[
    str,
    str,
    int | None,
    bool,
    str,
    str,
    str,
    str,
    int | None,
    int,
    float,
    float,
    str,
    str,
    float,
]:
    """Build a hashable key from the run fields that define train popularity."""
    defaults = EDGRecConfig()
    dataset_name = str(dataset or config.get("dataset") or defaults.dataset)
    preprocessing_preset = config.get("preprocessing_preset")
    if preprocessing_preset in (None, ""):
        preprocessing_preset = default_preprocessing_preset(dataset_name)
    item_universe_policy = config.get("item_universe_policy") or defaults.item_universe_policy
    graph_policy = str(config.get("graph_policy") or defaults.graph_policy)
    if graph_policy != "observed":
        raise CRRUPopularityReconstructionError(
            f"cannot reconstruct unsupported graph_policy: {graph_policy!r}",
        )
    return (
        dataset_name,
        str(config.get("data_dir") or defaults.data_dir),
        _optional_int(config.get("loader_max_rows")),
        bool(config.get("use_features", defaults.use_features)),
        str(config.get("feature_policy") or DEFAULT_FEATURE_POLICY),
        str(preprocessing_preset),
        str(item_universe_policy),
        graph_policy,
        _optional_int(config.get("sample_interactions")),
        _required_int(config.get("seed", DEFAULT_SEED), "seed"),
        _required_float(config.get("train_ratio", defaults.train_ratio), "train_ratio"),
        _required_float(config.get("val_ratio", defaults.val_ratio), "val_ratio"),
        str(config.get("derived_split_mode") or defaults.derived_split_mode),
        str(config.get("label_mode") or defaults.label_mode),
        _required_float(
            config.get("watch_ratio_proxy_threshold", defaults.watch_ratio_proxy_threshold),
            "watch_ratio_proxy_threshold",
        ),
    )


@lru_cache(maxsize=128)
def _reconstruct_largest_training_item_interaction_count_cached(
    key: tuple[
        str,
        str,
        int | None,
        bool,
        str,
        str,
        str,
        str,
        int | None,
        int,
        float,
        float,
        str,
        str,
        float,
    ],
) -> float:
    """Rebuild the training split and return its maximum item count."""
    (
        dataset,
        data_dir,
        loader_max_rows,
        use_features,
        feature_policy,
        preprocessing_preset,
        item_universe_policy,
        _graph_policy,
        sample_interactions,
        seed,
        train_ratio,
        val_ratio,
        derived_split_mode,
        label_mode,
        watch_ratio_proxy_threshold,
    ) = key

    label_kwargs = (
        {
            "label_mode": label_mode,
            "watch_ratio_proxy_threshold": watch_ratio_proxy_threshold,
        }
        if dataset == "kuairand1k"
        else {}
    )
    try:
        canonical = load_dataset(
            dataset,
            data_dir,
            max_rows=loader_max_rows,
            include_optional_features=use_features,
            feature_policy=feature_policy,
            preprocessing_preset=preprocessing_preset,
            **label_kwargs,
        )
        canonical = _apply_item_universe_policy(canonical, item_universe_policy)
        canonical = sample_canonical_interactions(
            canonical,
            sample_interactions,
            seed,
            train_ratio,
            val_ratio,
            derived_split_mode=derived_split_mode,
        )
        train_mask, _val_mask, _test_mask = canonical.get_splits(
            train_ratio,
            val_ratio,
            derived_split_mode=derived_split_mode,
        )
    except Exception as exc:
        raise CRRUPopularityReconstructionError(
            "cannot reconstruct training graph metadata",
        ) from exc

    train_positive_mask = positive_interaction_mask(train_mask, canonical.label)
    counts = compute_popularity_counts(
        canonical.item_id[train_positive_mask],
        int(canonical.n_items),
    )
    return float(counts.max()) if counts.size else 0.0


def _apply_item_universe_policy(
    canonical: CanonicalInteractions,
    item_universe_policy: str,
) -> CanonicalInteractions:
    """Apply the same item-universe filtering used before graph construction."""
    if item_universe_policy == "all_catalog_items":
        return canonical
    if item_universe_policy == "observed_interaction_items":
        keep_mask = np.ones(len(canonical), dtype=bool)
    elif item_universe_policy == "random_exposure_items_only":
        exposure_flag = getattr(canonical, "exposure_flag", None)
        if exposure_flag is None:
            raise CRRUPopularityReconstructionError(
                "cannot determine item-universe policy: random exposure flags missing",
            )
        keep_mask = np.asarray(exposure_flag, dtype=bool)
    else:
        raise CRRUPopularityReconstructionError(
            f"cannot determine item-universe policy: {item_universe_policy!r}",
        )

    if not np.any(keep_mask):
        raise CRRUPopularityReconstructionError(
            f"item_universe_policy={item_universe_policy!r} selected no interactions",
        )
    return filter_canonical_interactions(canonical, keep_mask)


def _optional_int(value: object) -> int | None:
    """Return an optional integer from a stored config value."""
    if value in (None, ""):
        return None
    return _required_int(value, "integer config value")


def _required_int(value: object, name: str) -> int:
    """Return a strict integer from a stored config value."""
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CRRUPopularityReconstructionError(f"invalid {name}") from exc
    return number


def _required_float(value: object, name: str) -> float:
    """Return a strict finite float from a stored config value."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CRRUPopularityReconstructionError(f"invalid {name}") from exc
    if not math.isfinite(number):
        raise CRRUPopularityReconstructionError(f"invalid {name}")
    return number
