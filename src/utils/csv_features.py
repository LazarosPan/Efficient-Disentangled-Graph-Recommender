"""Utilities for loading mixed-type CSV side-feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..data.feature_policy import FeaturePolicyName, resolve_feature_source
from .dataset_loader_utils import downcast_numeric_array

_NUMERIC_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)
_STRING_DTYPES = (pl.String, pl.Categorical)
_TEMPORAL_COLUMN_MARKERS = {"date", "time", "timestamp", "time_ms", "upload_dt", "upload_time"}
_TEMPORAL_COLUMN_SUFFIXES = ("_dt", "_date", "_time", "_timestamp")
_CATEGORICAL_COLUMN_NAMES = {"gender", "occupation", "zip_code"}
_CATEGORICAL_COLUMN_SUFFIXES = ("_id", "_type", "_status")


@dataclass(frozen=True, slots=True)
class PolicyCsvFeatureSpec:
    """Describe one policy-gated CSV feature source.

    Args:
        path: Absolute path to the CSV file.
        relative_path: Dataset-relative source path used by the feature-policy registry.
        id_col: Column containing the raw entity id.

    Returns:
        PolicyCsvFeatureSpec: Immutable metadata for one CSV feature block.

    """

    path: Path
    relative_path: str
    id_col: str


def _normalize_feature_name(column_name: str) -> str:
    """Return a lowercase feature name for representation heuristics.

    Args:
        column_name: Raw CSV feature column name.

    Returns:
        Lowercase normalized name.

    """
    return column_name.strip().lower()


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    """Return whether a Polars dtype is numeric.

    Args:
        dtype: Candidate Polars dtype.

    Returns:
        True when the dtype is numeric.

    """
    return dtype in _NUMERIC_DTYPES


def _is_string_dtype(dtype: pl.DataType) -> bool:
    """Return whether a Polars dtype is string-like.

    Args:
        dtype: Candidate Polars dtype.

    Returns:
        True when the dtype is string-like.

    """
    return dtype in _STRING_DTYPES


def _is_temporal_feature(column_name: str) -> bool:
    """Return whether a feature name should be normalized as time.

    Args:
        column_name: Feature column name.

    Returns:
        True when the name denotes a date/time field.

    """
    normalized = _normalize_feature_name(column_name)
    return normalized in _TEMPORAL_COLUMN_MARKERS or normalized.endswith(
        _TEMPORAL_COLUMN_SUFFIXES,
    )


def _is_categorical_feature(column_name: str, dtype: pl.DataType) -> bool:
    """Return whether a feature should be ordinal-encoded as a category.

    Args:
        column_name: Feature column name.
        dtype: Column dtype.

    Returns:
        True when the feature is categorical-like.

    """
    if _is_temporal_feature(column_name):
        return False
    normalized = _normalize_feature_name(column_name)
    return (
        _is_string_dtype(dtype)
        or normalized in _CATEGORICAL_COLUMN_NAMES
        or normalized.endswith(_CATEGORICAL_COLUMN_SUFFIXES)
    )


def _encode_temporal_series(series: pl.Series) -> np.ndarray:
    """Encode a temporal feature series as integer Unix seconds.

    Args:
        series: Polars series for one temporal feature column.

    Returns:
        Encoded integer timestamp values aligned to the input series.

    """
    frame = pl.DataFrame({"value": series})
    if _is_numeric_dtype(series.dtype):
        return frame.select(
            pl.col("value").cast(pl.Int64, strict=False).fill_null(0).alias("value"),
        )["value"].to_numpy()
    encoded = frame.select(
        pl.coalesce(
            pl.col("value").str.strptime(pl.Datetime, strict=False).dt.epoch("s"),
            pl.col("value").str.strptime(pl.Date, strict=False).cast(pl.Datetime).dt.epoch("s"),
        )
        .fill_null(0)
        .cast(pl.Int64)
        .alias("value"),
    )["value"].to_numpy()
    return encoded


def _encode_categorical_series(series: pl.Series) -> np.ndarray:
    """Encode a categorical feature series as deterministic positive IDs.

    Categories are sorted lexicographically and mapped to contiguous IDs
    ``1..N``. Missing values are reserved as ``0.0`` so the output remains
    dense and deterministic across repeated loads of the same file subset.

    Args:
        series: Polars series for one categorical feature column.

    Returns:
        Float32 array aligned to the input series, with ``0.0`` reserved for
        missing values.

    """
    frame = pl.DataFrame({"value": series})
    unique_values = frame.select(pl.col("value").drop_nulls().unique().sort().alias("value"))[
        "value"
    ]
    n_unique = len(unique_values)
    if unique_values.is_empty():
        return np.zeros(len(series), dtype=np.float32)
    lut = pl.DataFrame(
        {
            "value": unique_values,
            "code": np.arange(1, n_unique + 1, dtype=np.int32),
        },
    )
    return frame.join(lut, on="value", how="left")["code"].fill_null(0).cast(pl.Float32).to_numpy()


def _encode_numeric_series(series: pl.Series) -> np.ndarray:
    """Encode a numeric feature series with null-safe float casting.

    Args:
        series: Polars series for one numeric feature column.

    Returns:
        Float32 values aligned to the input series.

    """
    return (
        pl.DataFrame({"value": series})
        .select(pl.col("value").cast(pl.Float32, strict=False).fill_null(0.0).alias("value"))[
            "value"
        ]
        .to_numpy()
    )


def _encode_feature_series(column_name: str, series: pl.Series) -> np.ndarray:
    """Encode one CSV feature column into a compact numeric vector.

    Args:
        column_name: Feature column name.
        series: Polars series for the feature column.

    Returns:
        Numeric array aligned to the input series.

    """
    if _is_temporal_feature(column_name):
        return _encode_temporal_series(series)
    if _is_categorical_feature(column_name, series.dtype):
        return _encode_categorical_series(series)
    return _encode_numeric_series(series)


def load_csv_features(
    path: Path,
    id_col: str,
    id_map: dict[int, int],
    n_entities: int,
    include_columns: tuple[str, ...] | None = None,
) -> np.ndarray | None:
    """Load CSV side features aligned to a contiguous entity index.

    Args:
        path: CSV file to read.
        id_col: Column containing the raw entity id.
        id_map: Mapping from raw ids to contiguous ids.
        n_entities: Total number of entities in the contiguous index space.
        include_columns: Optional feature-column allowlist.

    Returns:
        A numeric feature matrix of shape ``(n_entities, n_features)`` or ``None``.
        Numeric columns stay numeric, temporal columns are normalized to integer
        Unix seconds, and categorical-like columns are ordinal-encoded with
        deterministic positive IDs while reserving ``0`` for missing values.

    """
    if not path.exists():
        return None

    read_columns = [id_col, *include_columns] if include_columns is not None else None
    df = pl.read_csv(path, columns=read_columns, ignore_errors=True)

    if id_col not in df.columns:
        return None

    feat_cols = [c for c in df.columns if c != id_col]
    if include_columns is not None:
        feat_cols = [c for c in include_columns if c in df.columns and c != id_col]

    if not feat_cols:
        return None

    df = df.select([id_col, *feat_cols]).with_columns(pl.col(id_col).cast(pl.Int64, strict=False))

    # Inner join against id_map; unrecognised entity ids are dropped.
    id_lut = pl.DataFrame(
        {id_col: list(id_map.keys()), "_mapped_id": list(id_map.values())},
        schema={id_col: pl.Int64, "_mapped_id": pl.Int64},
    )
    df = df.join(id_lut, on=id_col, how="inner")

    if df.is_empty():
        return None

    mapped_ids = df["_mapped_id"].to_numpy()
    encoded_columns = [
        _encode_feature_series(column_name, df[column_name]) for column_name in feat_cols
    ]
    feat_matrix = np.column_stack(encoded_columns)

    features = np.zeros((n_entities, len(feat_cols)), dtype=feat_matrix.dtype)
    features[mapped_ids] = feat_matrix

    return downcast_numeric_array(features, allow_float16=True)


def stack_feature_blocks(*feature_blocks: np.ndarray | None) -> np.ndarray | None:
    """Return one compact feature matrix from optional column blocks.

    Args:
        feature_blocks: Optional feature matrices sharing the same row count.

    Returns:
        Combined feature matrix or ``None`` when every block is missing.

    """
    present_blocks = [block for block in feature_blocks if block is not None]
    if not present_blocks:
        return None
    if len(present_blocks) == 1:
        return present_blocks[0]
    return downcast_numeric_array(np.hstack(present_blocks), allow_float16=True)


def build_multi_hot_features(
    token_lists_by_entity: dict[int, list[str] | tuple[str, ...]],
    n_entities: int,
    *,
    token_order: tuple[str, ...] | None = None,
) -> np.ndarray | None:
    """Build a multi-hot feature matrix from mapped token lists.

    Args:
        token_lists_by_entity: Mapping from contiguous entity ID to string tokens.
        n_entities: Total number of entities in the contiguous index space.
        token_order: Optional fixed token order. When omitted, tokens are sorted
            lexicographically across the provided rows.

    Returns:
        Multi-hot matrix of shape ``(n_entities, n_tokens)`` or ``None`` when no
        usable tokens are present.

    """
    if not token_lists_by_entity:
        return None

    resolved_order = token_order
    if resolved_order is None:
        resolved_order = tuple(
            sorted(
                {token for tokens in token_lists_by_entity.values() for token in tokens if token},
            ),
        )
    if not resolved_order:
        return None

    token_to_index = {token: index for index, token in enumerate(resolved_order)}
    features = np.zeros((n_entities, len(resolved_order)), dtype=np.uint8)
    for mapped_id, tokens in token_lists_by_entity.items():
        for token in tokens:
            token_index = token_to_index.get(token)
            if token_index is not None:
                features[mapped_id, token_index] = 1
    return features


def load_policy_csv_features(
    path: Path,
    *,
    feature_policy: FeaturePolicyName,
    dataset_name: str,
    aspect: str,
    relative_path: str,
    id_col: str,
    id_map: dict[int, int],
    n_entities: int,
) -> np.ndarray | None:
    """Load one numeric CSV feature block when the active policy enables it.

    Args:
        path: CSV file to load.
        feature_policy: Active feature-policy name.
        dataset_name: Loader registry dataset name.
        aspect: Feature aspect such as ``user_features`` or ``item_features``.
        relative_path: Dataset-relative source path used by the policy registry.
        id_col: Column containing the raw entity id.
        id_map: Mapping from raw ids to contiguous ids.
        n_entities: Total number of entities in the contiguous index space.

    Returns:
        Compact feature matrix for the enabled source, otherwise ``None``.

    """
    enabled, include_columns = resolve_feature_source(
        feature_policy,
        dataset_name,
        aspect,
        relative_path,
    )
    if not enabled:
        return None
    return load_csv_features(
        path,
        id_col,
        id_map,
        n_entities,
        include_columns=include_columns,
    )


def load_policy_csv_feature_blocks(
    *,
    feature_policy: FeaturePolicyName,
    dataset_name: str,
    aspect: str,
    id_map: dict[int, int],
    n_entities: int,
    sources: tuple[PolicyCsvFeatureSpec, ...],
) -> np.ndarray | None:
    """Load and stack policy-gated numeric CSV feature sources.

    Args:
        feature_policy: Active feature-policy name.
        dataset_name: Loader registry dataset name.
        aspect: Feature aspect such as ``user_features`` or ``item_features``.
        id_map: Mapping from raw ids to contiguous ids.
        n_entities: Total number of entities in the contiguous index space.
        sources: Numeric CSV sources to evaluate and stack in order.

    Returns:
        One stacked feature matrix or ``None`` when every source is disabled or missing.

    """
    return stack_feature_blocks(
        *(
            load_policy_csv_features(
                source.path,
                feature_policy=feature_policy,
                dataset_name=dataset_name,
                aspect=aspect,
                relative_path=source.relative_path,
                id_col=source.id_col,
                id_map=id_map,
                n_entities=n_entities,
            )
            for source in sources
        ),
    )
