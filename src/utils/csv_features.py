"""Utilities for loading numeric CSV side-feature tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset_loader_utils import downcast_numeric_array


def load_csv_features(
    path: Path,
    id_col: str,
    id_map: dict[int, int],
    n_entities: int,
    include_columns: tuple[str, ...] | None = None,
) -> np.ndarray | None:
    """Load numeric CSV side features aligned to a contiguous entity index.

    Args:
        path: CSV file to read.
        id_col: Column containing the raw entity id.
        id_map: Mapping from raw ids to contiguous ids.
        n_entities: Total number of entities in the contiguous index space.
        include_columns: Optional feature-column allowlist.

    Returns:
        A numeric feature matrix of shape ``(n_entities, n_features)`` or ``None``.
        The matrix is narrowed to a compact storage dtype after loading.
    """
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as file_obj:
        header = file_obj.readline().strip().split(",")
        id_idx = header.index(id_col)
        if include_columns is None:
            feat_indices = [idx for idx in range(len(header)) if idx != id_idx]
        else:
            feat_indices = [
                header.index(column)
                for column in include_columns
                if column in header and column != id_col
            ]
        if not feat_indices:
            return None

        rows: dict[int, list[float]] = {}
        for line in file_obj:
            parts = line.strip().split(",")
            if len(parts) < len(header):
                continue
            entity_id = int(parts[id_idx])
            if entity_id not in id_map:
                continue
            mapped_id = id_map[entity_id]
            values: list[float] = []
            for feat_idx in feat_indices:
                try:
                    values.append(float(parts[feat_idx]))
                except (ValueError, IndexError):
                    values.append(0.0)
            rows[mapped_id] = values

    if not rows:
        return None

    features = np.zeros((n_entities, len(feat_indices)), dtype=np.float32)
    for mapped_id, values in rows.items():
        features[mapped_id] = values

    return downcast_numeric_array(features, allow_float16=True)
