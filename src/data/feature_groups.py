"""Feature-group inference and subset-selection helpers."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

import numpy as np
import torch

from .feature_policy import (
    DEFAULT_FEATURE_POLICY,
    FeaturePolicyName,
)

FeatureSubsetMode = Literal["all", "none", "include_groups", "exclude_groups"]

FEATURE_GROUPS: tuple[str, ...] = (
    "item_genre",
    "user_demographic",
    "item_author_music",
    "item_video_metadata",
    "item_upload_time",
    "item_category",
    "item_resolution",
    "graph_only",
    "other_safe_item_feature",
    "other_safe_user_feature",
)
GRAPH_ONLY_PROFILE = "graph_only"
FEATURE_SUBSET_CONTROL_PROFILES = ("none", "all_gate_neg4", "all_gate0")
FEATURE_SUBSET_SAFE_ROLE = "safe_pre_treatment"
MAX_TRIPLE_COVERAGE_GROUPS = 4


def infer_feature_group(dataset: str, source: str, column: str, role: str) -> str:
    """Infer a deterministic thesis feature group for one encoded column.

    Args:
        dataset: Dataset registry name.
        source: Feature source path or source stem.
        column: Raw feature column or field token.
        role: Leakage-aware feature role from the feature-policy registry.

    Returns:
        Feature group label used by reports and ablations.

    """
    dataset_name = dataset.lower()
    source_name = source.replace("\\", "/").lower()
    column_name = column.lower()
    combined = f"{dataset_name}/{source_name}/{column_name}"

    if source_name == "graph_only" or column_name == "graph_only":
        return "graph_only"
    user_source = "user" in source_name
    if user_source and any(token in column_name for token in ("gender", "age", "occupation")):
        return "user_demographic"
    if "genre" in combined:
        return "item_genre"
    if "author" in column_name or "music" in column_name:
        return "item_author_music"
    if "upload_dt" in column_name or "upload_date" in column_name:
        return "item_upload_time"
    if any(token in column_name for token in ("width", "height", "resolution")):
        return "item_resolution"
    if "category" in combined or "tag" in column_name or column_name == "feat":
        return "item_category"
    if any(
        token in column_name
        for token in (
            "video_type",
            "upload_type",
            "visible_status",
            "duration",
            "server_",
        )
    ):
        return "item_video_metadata"
    if user_source:
        return "other_safe_user_feature"
    if role == "safe_pre_treatment":
        return "other_safe_item_feature"
    return "other_safe_user_feature" if "user" in combined else "other_safe_item_feature"


def _matrix_width(item_features: np.ndarray | torch.Tensor | None) -> int:
    """Return item-feature width, or zero when side features are absent."""
    if item_features is None:
        return 0
    if len(item_features.shape) != 2:
        return 0
    return int(item_features.shape[1])


def loaded_thesis_safe_item_feature_groups(payload: object) -> tuple[str, ...]:
    """Return loaded thesis-safe item feature groups in feature-column order."""
    width = _matrix_width(getattr(payload, "item_features", None))
    groups = getattr(payload, "item_feature_groups", None)
    if width == 0 or not groups:
        return ()
    roles = getattr(payload, "item_feature_roles", None)
    ordered: list[str] = []
    for index, group in enumerate(tuple(groups)[:width]):
        role = (
            roles[index] if roles is not None and index < len(roles) else FEATURE_SUBSET_SAFE_ROLE
        )
        if role != FEATURE_SUBSET_SAFE_ROLE or group == "graph_only":
            continue
        if group not in ordered:
            ordered.append(str(group))
    return tuple(ordered)


@lru_cache(maxsize=32)
def loaded_thesis_safe_item_feature_groups_for_dataset(
    dataset: str,
    *,
    data_dir: str = "data",
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
    preprocessing_preset: str | None = None,
    label_mode: str = "current",
    watch_ratio_proxy_threshold: float = 0.5,
) -> tuple[str, ...]:
    """Load one dataset and return thesis-safe item feature groups actually present."""
    from .loaders import load_dataset

    canonical = load_dataset(
        dataset,
        data_dir=data_dir,
        include_optional_features=True,
        feature_policy=feature_policy,
        preprocessing_preset=preprocessing_preset,
        label_mode=label_mode,
        watch_ratio_proxy_threshold=watch_ratio_proxy_threshold,
    )
    return loaded_thesis_safe_item_feature_groups(canonical)


def required_feature_subset_profiles(groups: Sequence[str]) -> tuple[str, ...]:
    """Return the dataset-specific profiles that must complete for valid evidence."""
    ordered_groups = tuple(dict.fromkeys(str(group) for group in groups if group))
    if not ordered_groups:
        return (GRAPH_ONLY_PROFILE,)
    profiles = list(FEATURE_SUBSET_CONTROL_PROFILES)
    profiles.extend(f"single_{group}" for group in ordered_groups)
    profiles.extend(f"drop_{group}" for group in ordered_groups)
    profiles.extend(
        f"pair_{left}__{right}" for left, right in itertools.combinations(ordered_groups, 2)
    )
    if len(ordered_groups) <= MAX_TRIPLE_COVERAGE_GROUPS:
        profiles.extend(
            "triple_" + "__".join(combo) for combo in itertools.combinations(ordered_groups, 3)
        )
    return tuple(profiles)


def feature_subset_profile_overrides(
    profile: str,
    groups: Sequence[str],
) -> dict[str, object]:
    """Resolve one feature-subset profile into EDGRecConfig overrides."""
    ordered_groups = tuple(dict.fromkeys(str(group) for group in groups if group))
    if profile == GRAPH_ONLY_PROFILE:
        if ordered_groups:
            raise ValueError("graph_only profile is only valid for datasets without side features")
        return {
            "feature_subset_mode": "none",
            "feature_include_groups": [],
            "feature_exclude_groups": [],
            "use_features": False,
            "feature_gate_init": -4.0,
        }
    if not ordered_groups:
        raise ValueError(f"Feature subset profile {profile!r} requires loaded side features")
    if profile == "none":
        return {
            "feature_subset_mode": "none",
            "feature_include_groups": [],
            "feature_exclude_groups": [],
            "use_features": True,
            "feature_gate_init": -4.0,
        }
    if profile == "all_gate_neg4":
        return {
            "feature_subset_mode": "all",
            "feature_include_groups": [],
            "feature_exclude_groups": [],
            "use_features": True,
            "feature_gate_init": -4.0,
        }
    if profile == "all_gate0":
        return {
            "feature_subset_mode": "all",
            "feature_include_groups": [],
            "feature_exclude_groups": [],
            "use_features": True,
            "feature_gate_init": 0.0,
        }
    prefix, _, suffix = profile.partition("_")
    selected_groups = suffix.split("__") if suffix else []
    unknown = sorted(set(selected_groups) - set(ordered_groups))
    if unknown:
        raise ValueError(f"Unknown feature subset group(s): {', '.join(unknown)}")
    if prefix == "single":
        if len(selected_groups) != 1:
            raise ValueError(f"Invalid single-group profile {profile!r}")
        mode = "include_groups"
        include_groups = selected_groups
        exclude_groups: list[str] = []
    elif prefix == "drop":
        if len(selected_groups) != 1:
            raise ValueError(f"Invalid drop-group profile {profile!r}")
        mode = "exclude_groups"
        include_groups = []
        exclude_groups = selected_groups
    elif prefix in {"pair", "triple"}:
        expected_size = 2 if prefix == "pair" else 3
        if len(selected_groups) != expected_size:
            raise ValueError(f"Invalid {prefix}-group profile {profile!r}")
        mode = "include_groups"
        include_groups = selected_groups
        exclude_groups = []
    else:
        raise ValueError(f"Unknown feature subset profile: {profile!r}")
    return {
        "feature_subset_mode": mode,
        "feature_include_groups": include_groups,
        "feature_exclude_groups": exclude_groups,
        "use_features": True,
        "feature_gate_init": -4.0,
    }


def feature_subset_profile_matrix(groups: Sequence[str]) -> dict[str, dict[str, object]]:
    """Return all required profile overrides for one dataset's loaded groups."""
    return {
        profile: feature_subset_profile_overrides(profile, groups)
        for profile in required_feature_subset_profiles(groups)
    }


def feature_subset_profile_group_labels(
    profile: str,
    groups: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return included and excluded group labels for one profile."""
    overrides = feature_subset_profile_overrides(profile, groups)
    mode = str(overrides["feature_subset_mode"])
    if mode == "all":
        return tuple(groups), ()
    if mode == "none":
        return (), tuple(groups)
    if mode == "include_groups":
        included = tuple(str(group) for group in overrides["feature_include_groups"])
        excluded = tuple(group for group in groups if group not in included)
        return included, excluded
    excluded = tuple(str(group) for group in overrides["feature_exclude_groups"])
    included = tuple(group for group in groups if group not in excluded)
    return included, excluded


def selected_item_feature_indices_for_subset(
    *,
    feature_groups: tuple[str, ...] | None,
    mode: FeatureSubsetMode,
    include_groups: list[str],
    exclude_groups: list[str],
) -> list[int]:
    """Return columns zeroed by a feature-subset config."""
    if not feature_groups or mode == "all":
        return []
    if mode == "none":
        return list(range(len(feature_groups)))
    include_targets = set(include_groups)
    exclude_targets = set(exclude_groups)
    if mode == "include_groups":
        return [index for index, group in enumerate(feature_groups) if group not in include_targets]
    if mode == "exclude_groups":
        return [index for index, group in enumerate(feature_groups) if group in exclude_targets]
    raise ValueError(f"Unsupported feature_subset_mode {mode!r}.")


def zero_item_feature_subset_columns(
    item_features: np.ndarray | torch.Tensor,
    *,
    feature_groups: tuple[str, ...] | None,
    mode: FeatureSubsetMode,
    include_groups: list[str],
    exclude_groups: list[str],
) -> np.ndarray | torch.Tensor:
    """Return item features with subset-excluded columns zeroed."""
    indices = (
        list(range(int(item_features.shape[1])))
        if mode == "none"
        else selected_item_feature_indices_for_subset(
            feature_groups=feature_groups,
            mode=mode,
            include_groups=include_groups,
            exclude_groups=exclude_groups,
        )
    )
    if not indices:
        return item_features
    if isinstance(item_features, torch.Tensor):
        subset = item_features.clone()
        subset[:, indices] = 0
        return subset
    subset = item_features.copy()
    subset[:, indices] = 0
    return subset


def apply_graph_item_feature_subset(data: object, config: object) -> None:
    """Apply configured item-feature subset to a graph payload."""
    if getattr(data, "_feature_subset_applied", False):
        return
    item_features = getattr(data, "item_features", None)
    if item_features is None:
        data._feature_subset_applied = True
        return
    subset_mode = getattr(config, "feature_subset_mode", "all")
    data.item_features = zero_item_feature_subset_columns(
        item_features,
        feature_groups=getattr(data, "item_feature_groups", None),
        mode=subset_mode,
        include_groups=list(getattr(config, "feature_include_groups", [])),
        exclude_groups=list(getattr(config, "feature_exclude_groups", [])),
    )
    data._feature_subset_applied = True
