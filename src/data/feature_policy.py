"""Dataset-specific side-feature policies for thesis experiments."""

from __future__ import annotations

from typing import Literal


FeaturePolicyName = Literal["thesis_default", "all_optional"]
FeatureRoleName = Literal[
    "safe_pre_treatment",
    "proxy_only",
    "post_treatment_excluded",
]

DEFAULT_FEATURE_POLICY: FeaturePolicyName = "thesis_default"

_FEATURE_UTILITY_DATASETS: tuple[str, ...] = (
    "movielens1m",
    "movielens20m",
    "taobao",
    "kuairec_v2",
    "kuairand1k",
)


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset names so folder labels map to loader registry entries."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


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
                }
            },
        },
        "item_features": {
            "raw/movies.dat": {
                "columns": {
                    "genres": "safe_pre_treatment",
                }
            },
        },
    },
    "movielens20m": {
        "item_features": {
            "raw/movies.csv": {
                "columns": {
                    "genres": "safe_pre_treatment",
                }
            },
            "raw/genome-scores.csv": {
                "columns": {
                    "relevance": "safe_pre_treatment",
                }
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
            "data/item_daily_features.csv": {
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
                }
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
        normalize_dataset_name(dataset_name), {}
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
    columns = policy.get("columns", {})
    safe_columns = tuple(
        column for column, role in columns.items() if role == "safe_pre_treatment"
    )
    return safe_columns or None


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


_NORMALIZED_FEATURE_UTILITY_DATASETS = frozenset(
    normalize_dataset_name(name) for name in _FEATURE_UTILITY_DATASETS
)


def supports_feature_utility(dataset_name: str) -> bool:
    """Return whether a dataset has a meaningful ID-only vs thesis-default comparison."""
    return normalize_dataset_name(dataset_name) in _NORMALIZED_FEATURE_UTILITY_DATASETS
