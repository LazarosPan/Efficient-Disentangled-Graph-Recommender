"""Dataset-specific side-feature policies for thesis experiments."""

from __future__ import annotations

from typing import Literal


FeaturePolicyName = Literal["thesis_default", "all_optional"]

DEFAULT_FEATURE_POLICY: FeaturePolicyName = "thesis_default"

FEATURE_UTILITY_DATASETS: tuple[str, ...] = (
    "movielens1m",
    "movielens20m",
    "taobao",
    "kuairec_v2",
    "kuairand1k",
)

POLICY_ABLATION_DATASETS: tuple[str, ...] = (
    "kuairec_v2",
    "kuairand1k",
)


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset names so folder labels map to loader registry entries."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


_THESIS_DEFAULT_COLUMNS: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "movielens1m": {
        "user_features": {
            "raw/users.dat": ("gender", "age", "occupation", "zip_code"),
        },
        "item_features": {
            "raw/movies.dat": ("genres",),
        },
    },
    "movielens20m": {
        "item_features": {
            "raw/movies.csv": ("genres",),
            "raw/genome-scores.csv": ("relevance",),
        },
    },
    "taobao": {
        "item_features": {
            "raw/UserBehavior.csv": ("category_id",),
        },
    },
    "kuairecv2": {
        "item_features": {
            "data/item_daily_features.csv": (
                "author_id",
                "video_type",
                "upload_dt",
                "upload_type",
                "visible_status",
                "music_id",
            ),
            "data/item_categories.csv": ("feat",),
            "data/kuairec_caption_category.csv": (
                "first_level_category_id",
                "second_level_category_id",
                "third_level_category_id",
            ),
        },
    },
    "kuairand1k": {
        "item_features": {
            "data/video_features_basic_1k.csv": (
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
        },
    },
}


def thesis_default_columns(
    dataset_name: str,
    aspect: str,
    relative_path: str,
) -> tuple[str, ...] | None:
    """Return the thesis-default allowed columns for a file, if any."""
    dataset_policy = _THESIS_DEFAULT_COLUMNS.get(normalize_dataset_name(dataset_name), {})
    aspect_policy = dataset_policy.get(aspect, {})
    normalized_path = relative_path.replace("\\", "/")
    for suffix, columns in aspect_policy.items():
        if normalized_path.endswith(suffix):
            return columns
    return None


def thesis_default_file_enabled(dataset_name: str, aspect: str, relative_path: str) -> bool:
    """Return whether a file contributes to the thesis-default feature path."""
    return thesis_default_columns(dataset_name, aspect, relative_path) is not None


def thesis_default_column_enabled(
    dataset_name: str,
    aspect: str,
    relative_path: str,
    column: str,
) -> bool:
    """Return whether a column is included in the thesis-default feature path."""
    columns = thesis_default_columns(dataset_name, aspect, relative_path)
    if columns is None:
        return False
    return column.lower() in {value.lower() for value in columns}


def datasets_with_feature_utility() -> tuple[str, ...]:
    """Datasets where ID-only vs thesis-default feature probes are meaningful."""
    return FEATURE_UTILITY_DATASETS


def datasets_with_policy_ablation() -> tuple[str, ...]:
    """Datasets where thesis_default and all_optional currently differ."""
    return POLICY_ABLATION_DATASETS


def supports_feature_utility(dataset_name: str) -> bool:
    """Return whether a dataset has a meaningful ID-only vs thesis-default comparison."""
    return normalize_dataset_name(dataset_name) in {
        normalize_dataset_name(name) for name in FEATURE_UTILITY_DATASETS
    }


def supports_policy_ablation(dataset_name: str) -> bool:
    """Return whether thesis_default vs all_optional differs for a dataset."""
    return normalize_dataset_name(dataset_name) in {
        normalize_dataset_name(name) for name in POLICY_ABLATION_DATASETS
    }