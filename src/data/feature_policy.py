"""Dataset-specific side-feature policies for thesis experiments.

Causal feature-role taxonomy
-----------------------------
``safe_pre_treatment``
    Static content descriptors set before any user-item exposure.  These
    cannot carry post-treatment signal by construction (e.g. genre, author,
    upload date, video format).  They are safe to use as item/user features
    in thesis experiments.

``proxy_only``
    Derived aggregates or user-profile fields that may correlate with
    treatment assignment (e.g. register days, follower counts).  They are
    retained in the ``all_optional`` policy for exploratory analysis but
    excluded from ``thesis_default`` item features to avoid confounding.

``post_treatment_excluded``
    **Leakage risk — never load under thesis_default.**
    Engagement counts such as ``show_cnt``, ``play_cnt``, ``like_cnt``, and
    ``share_cnt`` are aggregated *after* the recommendation system has been
    running.  Including them would let the model observe the downstream effect
    of its own recommendations during training, violating the causal DAG
    assumption and inflating offline metrics.  The entire
    ``video_features_statistic_1k.csv`` (KuaiRand) and the engagement-count
    columns of ``item_daily_features.csv`` (KuaiRec) fall into this category.

The ``_FEATURE_ROLE_REGISTRY`` encodes these decisions at the file level so
every loader and dataset-audit tool derives the same thesis-default column set
from a single source of truth.
"""

from __future__ import annotations

from typing import Literal

FeaturePolicyName = Literal["thesis_default", "all_optional"]
FeatureRoleName = Literal[
    "safe_pre_treatment",
    "proxy_only",
    "post_treatment_excluded",
]

DEFAULT_FEATURE_POLICY: FeaturePolicyName = "thesis_default"


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset names so folder labels map to loader registry entries."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


# File-oriented registry shared by loaders and dataset-audit tooling so both
# surfaces derive thesis-default feature choices from the same source of truth.
_FEATURE_ROLE_REGISTRY: dict[
    str,
    dict[str, dict[str, dict[str, object]]],
] = {
    "movielens1m": {
        "user_features": {
            "raw/users.dat": {
                "columns": {
                    "gender": "safe_pre_treatment",
                    "age": "safe_pre_treatment",
                    "occupation": "safe_pre_treatment",
                    "zip_code": "proxy_only",
                },
            },
        },
        "item_features": {
            "raw/movies.dat": {
                "columns": {
                    "genres": "safe_pre_treatment",
                },
            },
        },
    },
    "movielens20m": {
        "item_features": {
            "raw/movies.csv": {
                "thesis_default_columns": ("genres",),
                "columns": {
                    "genres": "safe_pre_treatment",
                },
            },
            "raw/genome-scores.csv": {
                "thesis_default_columns": (),
                "columns": {
                    "relevance": "safe_pre_treatment",
                },
            },
        },
    },
    "taobao": {
        "item_features": {
            "raw/UserBehavior.csv": {
                "columns": {
                    "category_id": "safe_pre_treatment",
                },
                "default_role": "post_treatment_excluded",
            },
        },
    },
    "kuairecv2": {
        "user_features": {
            "data/user_features.csv": {
                "columns": {},
                "default_role": "proxy_only",
            },
            "data/user_features_raw.csv": {
                "columns": {},
                "default_role": "proxy_only",
            },
        },
        "item_features": {
            # item_daily_features.csv contains both static content descriptors
            # (safe to use) and date-varying engagement counts (post-treatment,
            # MUST be excluded).  thesis_default_columns enumerates the safe
            # subset explicitly; all remaining columns (show_cnt, play_cnt,
            # like_cnt, share_cnt, comment_cnt, follow_cnt, share_cnt, …) are
            # post-treatment engagement aggregates and are excluded by the
            # post_treatment_excluded default_role.
            "data/item_daily_features.csv": {
                "thesis_default_columns": (
                    "author_id",
                    "music_id",
                    "video_type",
                    "upload_dt",
                    "upload_type",
                    "visible_status",
                ),
                "columns": {
                    "author_id": "safe_pre_treatment",
                    "video_type": "safe_pre_treatment",
                    "upload_dt": "safe_pre_treatment",
                    "upload_type": "safe_pre_treatment",
                    "visible_status": "safe_pre_treatment",
                    "music_id": "safe_pre_treatment",
                },
                "default_role": "post_treatment_excluded",
            },
            "data/item_categories.csv": {
                "columns": {
                    "feat": "safe_pre_treatment",
                },
            },
            "data/kuairec_caption_category.csv": {
                "columns": {
                    "first_level_category_id": "safe_pre_treatment",
                    "second_level_category_id": "safe_pre_treatment",
                    "third_level_category_id": "safe_pre_treatment",
                },
                "default_role": "proxy_only",
            },
        },
    },
    "kuairand1k": {
        "user_features": {
            "data/user_features_1k.csv": {
                "columns": {},
                "default_role": "proxy_only",
            },
        },
        "item_features": {
            "data/video_features_basic_1k.csv": {
                "thesis_default_columns": (
                    "author_id",
                    "video_type",
                    "upload_dt",
                    "upload_type",
                    "visible_status",
                    "server_width",
                    "server_height",
                    "music_id",
                    "music_type",
                ),
                "columns": {
                    "author_id": "safe_pre_treatment",
                    "video_type": "safe_pre_treatment",
                    "upload_dt": "safe_pre_treatment",
                    "upload_type": "safe_pre_treatment",
                    "visible_status": "safe_pre_treatment",
                    "server_width": "safe_pre_treatment",
                    "server_height": "safe_pre_treatment",
                    "music_id": "safe_pre_treatment",
                    "music_type": "safe_pre_treatment",
                },
                "default_role": "post_treatment_excluded",
            },
            # video_features_statistic_1k.csv contains only post-treatment
            # engagement aggregates (show_cnt, play_cnt, like_cnt, share_cnt,
            # comment_cnt, follow_cnt, …).  The entire file is excluded under
            # thesis_default to prevent leakage; no columns are safe here.
            "data/video_features_statistic_1k.csv": {
                "columns": {},
                "default_role": "post_treatment_excluded",
            },
        },
    },
}


def _source_policy(
    dataset_name: str,
    aspect: str,
    relative_path: str,
) -> dict[str, object] | None:
    """Return the structured registry entry for one dataset/aspect/file.

    Args:
        dataset_name: Loader registry dataset name.
        aspect: Feature aspect such as ``user_features`` or ``item_features``.
        relative_path: Dataset-relative source path.

    Returns:
        Matching registry entry when available, otherwise ``None``.

    """
    dataset_policy = _FEATURE_ROLE_REGISTRY.get(
        normalize_dataset_name(dataset_name),
        {},
    )
    aspect_policy = dataset_policy.get(aspect, {})
    normalized_path = relative_path.replace("\\", "/")
    for suffix, policy in aspect_policy.items():
        if normalized_path.endswith(suffix):
            return policy
    return None


def thesis_default_columns(
    dataset_name: str,
    aspect: str,
    relative_path: str,
) -> tuple[str, ...] | None:
    """Return the thesis-default allowed columns for a file, if any."""
    policy = _source_policy(dataset_name, aspect, relative_path)
    if policy is None:
        return None
    explicit_columns = policy.get("thesis_default_columns")
    if explicit_columns is not None:
        return tuple(explicit_columns) or None
    columns = policy.get("columns", {})
    safe_columns = tuple(column for column, role in columns.items() if role == "safe_pre_treatment")
    return safe_columns or None


def registered_feature_sources(
    dataset_name: str,
    aspect: str,
) -> tuple[str, ...]:
    """Return the registered feature-source paths for one dataset aspect.

    Args:
        dataset_name: Loader registry dataset name.
        aspect: Feature aspect such as ``user_features`` or ``item_features``.

    Returns:
        Tuple of dataset-relative source paths registered for the aspect.

    """
    dataset_policy = _FEATURE_ROLE_REGISTRY.get(
        normalize_dataset_name(dataset_name),
        {},
    )
    aspect_policy = dataset_policy.get(aspect, {})
    return tuple(aspect_policy)


def enabled_feature_sources(
    feature_policy: FeaturePolicyName,
    dataset_name: str,
    aspect: str,
) -> tuple[str, ...]:
    """Return the registered feature sources enabled by the active policy.

    Args:
        feature_policy: Active feature-policy name.
        dataset_name: Loader registry dataset name.
        aspect: Feature aspect such as ``user_features`` or ``item_features``.

    Returns:
        Tuple of dataset-relative source paths enabled for the aspect.

    """
    return tuple(
        relative_path
        for relative_path in registered_feature_sources(dataset_name, aspect)
        if resolve_feature_source(
            feature_policy,
            dataset_name,
            aspect,
            relative_path,
        )[0]
    )


def supports_feature_utility(dataset_name: str) -> bool:
    """Return whether thesis-default feature loading can add usable features.

    Args:
        dataset_name: Loader registry dataset name.

    Returns:
        True when the dataset registry exposes at least one thesis-default
        feature source, otherwise False.

    """
    dataset_policy = _FEATURE_ROLE_REGISTRY.get(normalize_dataset_name(dataset_name), {})
    return any(
        enabled_feature_sources("thesis_default", dataset_name, aspect) for aspect in dataset_policy
    )


def feature_role_for_column(
    dataset_name: str,
    aspect: str,
    relative_path: str,
    column: str,
) -> FeatureRoleName | None:
    """Return the structured feature role for a dataset/aspect/file column.

    Args:
        dataset_name: Loader registry dataset name.
        aspect: Feature aspect such as ``user_features`` or ``item_features``.
        relative_path: Dataset-relative source path.
        column: Candidate column name.

    Returns:
        Structured feature role when the registry knows the column or file.

    """
    policy = _source_policy(dataset_name, aspect, relative_path)
    if policy is None:
        return None
    columns = policy.get("columns", {})
    normalized = column.lower()
    for registered_column, role in columns.items():
        if registered_column.lower() == normalized:
            return role
    default_role = policy.get("default_role")
    if default_role is None:
        return None
    return default_role


def resolve_feature_source(
    feature_policy: FeaturePolicyName,
    dataset_name: str,
    aspect: str,
    relative_path: str,
) -> tuple[bool, tuple[str, ...] | None]:
    """Resolve whether a feature source is enabled and which columns it may use.

    Args:
        feature_policy: Active feature policy name.
        dataset_name: Loader registry dataset name.
        aspect: Feature aspect such as ``user_features`` or ``item_features``.
        relative_path: Dataset-relative source path.

    Returns:
        Tuple ``(enabled, include_columns)``. ``include_columns`` is ``None``
        for unrestricted policies such as ``all_optional``.

    """
    if feature_policy == "all_optional":
        return True, None
    columns = thesis_default_columns(dataset_name, aspect, relative_path)
    return columns is not None, columns
