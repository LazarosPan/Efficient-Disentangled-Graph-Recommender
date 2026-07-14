#!/usr/bin/env python
"""Export consolidated graph and embedding figures for the thesis datasets.

The single-dataset exporter is useful for diagnostics. This script builds the
thesis-facing comparison figures: one image per visualization type, with four
dataset panels and enough on-figure context to be read outside the repository.
"""

# ruff: noqa: E402, I001, E501

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from experiments.run_experiment import load_runtime_data
from scripts.export_graph_embedding_figures import (
    OUTPUT_DIR,
    _degree_vectors,
    _load_checkpoint_payload,
    _load_feature_projection,
    _load_model_projection,
    _node_size,
    _positive_degrees,
    _positive_train_edges,
    _project_matrix,
    _select_ids_by_degree,
    _train_graph_summary,
    _visualization_config_from_checkpoint,
)
from src.utils.project_paths import RESULTS_DIR


CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
USER_COLOR = "#1f77b4"
ITEM_COLOR = "#f2c230"
POSITIVE_EDGE_COLOR = "#2ca25f"
ZERO_SIGN_EDGE_COLOR = "#6b7280"
NEGATIVE_SIGN_EDGE_COLOR = "#de2d26"
TOPOLOGY_NODE_ALPHA = 0.56
PROJECTION_POINT_ALPHA = 0.36
FEATURE_POINT_ALPHA = 0.48
EXPORT_DPI = 360
TOPOLOGY_CORE_ITEM_LIMIT = 180
TOPOLOGY_DIAGNOSTIC_ROW_FRACTION = 0.30


@dataclass(frozen=True)
class DatasetSpec:
    """One thesis dataset and the checkpoint used for learned embeddings."""

    dataset: str
    label: str
    checkpoint: Path
    note: str
    feature_note: str


@dataclass
class ProjectionRecord:
    """Small projected matrix plus metadata needed for a panel."""

    xy: np.ndarray
    roles: np.ndarray
    degrees: np.ndarray
    neutral_degrees: np.ndarray
    negative_degrees: np.ndarray
    source: str
    n_points: int
    n_features: int
    metadata: dict[str, Any]
    graph_summary: dict[str, int]
    unavailable_reason: str | None = None


@dataclass
class DatasetBundle:
    """All small arrays needed after loading a large runtime graph."""

    spec: DatasetSpec
    config_summary: dict[str, Any]
    topology_edges: np.ndarray
    topology_negative_edges: np.ndarray
    topology_neutral_edges: np.ndarray
    user_degree: np.ndarray
    item_degree: np.ndarray
    negative_user_degree: np.ndarray
    negative_item_degree: np.ndarray
    neutral_user_degree: np.ndarray
    neutral_item_degree: np.ndarray
    graph_summary: dict[str, int]
    negative_summary: dict[str, int]
    neutral_summary: dict[str, int]
    train_label_summary: dict[str, int]
    train_sign_summary: dict[str, int]
    topology_sample_summary: dict[str, Any]
    learned_projection: ProjectionRecord | None
    feature_projection: ProjectionRecord | None
    feature_unavailable_reason: str | None


CORE_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset="amazonbook",
        label="AmazonBook",
        checkpoint=CHECKPOINT_DIR / (
            "amazonbook_edgrec_ep200_bs16384_dim64_layers2_branchL1-2_nbr8-4_feat_"
            "scoremixlearned_lr-plateau_seed13_train-23a6ee0e50a803d1.pt"
        ),
        note="Graph-only reference EDGRec checkpoint.",
        feature_note=(
            "No side-feature panel: AmazonBook is graph-only under the thesis "
            "feature policy."
        ),
    ),
    DatasetSpec(
        dataset="movielens1m",
        label="MovieLens 1M",
        checkpoint=CHECKPOINT_DIR / (
            "movielens1m_edgrec_ep200_bs32768_dim64_layers2_branchL1-2_nbr8-4_"
            "ppresetmovielens_explicit_feat_lr-cosine_seed13_train-1684e72894a70dff.pt"
        ),
        note="Reference EDGRec checkpoint for the explicit-rating protocol.",
        feature_note="Feature panel uses encoded genre descriptors from the thesis feature policy.",
    ),
    DatasetSpec(
        dataset="kuairec_v2",
        label="KuaiRec v2",
        checkpoint=CHECKPOINT_DIR / (
            "kuairec_v2_edgrec_ep300_bs32768_dim64_layers1_nbr8_"
            "ppresetkuairec_watchratio_lr-cosine_seed13_train-f06a6e85c3713807.pt"
        ),
        note="Sparse big-matrix watch-ratio reference checkpoint.",
        feature_note="Feature panel uses safe video metadata descriptors.",
    ),
    DatasetSpec(
        dataset="kuairand1k",
        label="KuaiRand 1K",
        checkpoint=CHECKPOINT_DIR / (
            "kuairand1k_edgrec_ep300_bs1048576_dim64_layers2_branchL1-2_nbr8-4_"
            "ppresetkuairand_causal_iunivrandom_exposure_items_only_lr-cosine_"
            "seed13_train-69f13b80ca3363de.pt"
        ),
        note="Compact randomized-exposure reference checkpoint.",
        feature_note="Feature panel uses the same compact randomized-exposure item universe.",
    ),
)


def _format_count(value: int | float) -> str:
    """Human-readable integer count."""
    return f"{int(value):,}"


def _relative(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _assert_checkpoints_exist(specs: tuple[DatasetSpec, ...]) -> None:
    missing = [str(spec.checkpoint) for spec in specs if not spec.checkpoint.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing reference checkpoint(s):\n{joined}")


def _config_summary(config: Any) -> dict[str, Any]:
    """Return the config fields readers need to interpret a figure."""
    return {
        "dataset": config.dataset,
        "preprocessing_preset": getattr(config, "preprocessing_preset", None),
        "feature_policy": getattr(config, "feature_policy", None),
        "item_universe_policy": getattr(config, "item_universe_policy", None),
        "preset": getattr(config, "preset", None),
        "baseline_family": getattr(config, "baseline_family", None),
        "interest_gnn_layers": getattr(config, "interest_gnn_layers", None),
        "conformity_gnn_layers": getattr(config, "conformity_gnn_layers", None),
        "single_branch_gnn_layers": getattr(config, "single_branch_gnn_layers", None),
        "num_neighbors": list(getattr(config, "num_neighbors", []) or []),
    }


def _observed_train_feedback_edges(
    data: Any,
    feedback: str,
    *,
    interaction_sign: np.ndarray | None,
) -> np.ndarray:
    """Return observed train user->item rows for one label group."""
    train_mask = getattr(data, "train_mask", None)
    labels = getattr(data, "labels", None)
    user_nodes = getattr(data, "user_nodes", None)
    item_nodes = getattr(data, "item_nodes", None)
    if not all(hasattr(value, "detach") for value in (train_mask, labels, user_nodes, item_nodes)):
        return np.empty((0, 2), dtype=np.int64)

    label_values = labels.detach().cpu().numpy()
    sign_values = None
    if interaction_sign is not None and int(interaction_sign.shape[0]) == int(label_values.size):
        sign_values = np.asarray(interaction_sign)
    if feedback == "negative":
        label_mask = label_values < 0 if sign_values is None else sign_values < 0
    elif feedback in {"neutral", "zero_sign"}:
        if sign_values is not None and np.unique(sign_values).size <= 1:
            return np.empty((0, 2), dtype=np.int64)
        label_mask = label_values == 0 if sign_values is None else sign_values == 0
    else:
        raise ValueError(f"Unsupported feedback label group: {feedback}")

    label_mask_tensor = np.asarray(label_mask)
    mask = train_mask.detach().cpu().numpy().astype(bool) & label_mask_tensor
    if not bool(mask.any()):
        return np.empty((0, 2), dtype=np.int64)

    users = user_nodes.detach().cpu().numpy()[mask]
    items = item_nodes.detach().cpu().numpy()[mask]
    return np.column_stack((users, items)).astype(np.int64, copy=False)


def _observed_train_feedback_rows(
    data: Any,
    *,
    interaction_sign: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Return observed train interactions with aligned label and sign values."""
    train_mask = getattr(data, "train_mask", None)
    labels = getattr(data, "labels", None)
    user_nodes = getattr(data, "user_nodes", None)
    item_nodes = getattr(data, "item_nodes", None)
    empty = {
        "users": np.asarray([], dtype=np.int64),
        "items": np.asarray([], dtype=np.int64),
        "labels": np.asarray([], dtype=np.float32),
        "signs": np.asarray([], dtype=np.float32),
        "has_graded_sign": np.asarray([False], dtype=np.bool_),
    }
    if not all(hasattr(value, "detach") for value in (train_mask, labels, user_nodes, item_nodes)):
        return empty

    mask = train_mask.detach().cpu().numpy().astype(bool)
    if not bool(mask.any()):
        return empty
    label_values = labels.detach().cpu().numpy()
    sign_values = (
        np.asarray(interaction_sign)
        if interaction_sign is not None and int(interaction_sign.shape[0]) == int(label_values.size)
        else label_values
    )
    has_graded_sign = bool(np.unique(sign_values[mask]).size > 1)
    return {
        "users": user_nodes.detach().cpu().numpy()[mask].astype(np.int64, copy=False),
        "items": item_nodes.detach().cpu().numpy()[mask].astype(np.int64, copy=False),
        "labels": label_values[mask].astype(np.float32, copy=False),
        "signs": sign_values[mask].astype(np.float32, copy=False),
        "has_graded_sign": np.asarray([has_graded_sign], dtype=np.bool_),
    }


def _feedback_summary(data: Any, feedback_edges: np.ndarray) -> dict[str, int]:
    """Return full observed feedback counts for one label group."""
    if feedback_edges.size == 0:
        return {
            "n_train_interactions_total": 0,
            "n_train_users_total": 0,
            "n_train_items_total": 0,
        }
    n_users = int(data.n_users)
    return {
        "n_train_interactions_total": int(feedback_edges.shape[0]),
        "n_train_users_total": int(np.unique(feedback_edges[:, 0]).shape[0]),
        "n_train_items_total": int(
            np.unique(feedback_edges[:, 1] - n_users).shape[0],
        ),
    }


def _train_label_sign_summary(
    data: Any,
    interaction_sign: np.ndarray | None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return train-interaction counts for binary labels and graded signs."""
    train_mask = getattr(data, "train_mask", None)
    labels = getattr(data, "labels", None)
    if not all(hasattr(value, "detach") for value in (train_mask, labels)):
        return {}, {}

    train = train_mask.detach().cpu().numpy().astype(bool)
    label_values = labels.detach().cpu().numpy()
    sign_values = (
        np.asarray(interaction_sign)
        if interaction_sign is not None and int(interaction_sign.shape[0]) == int(label_values.size)
        else label_values
    )
    positive_label = train & (label_values > 0)
    nonpositive_label = train & (label_values <= 0)
    positive_sign = train & (sign_values > 0)
    zero_sign = train & (sign_values == 0)
    negative_sign = train & (sign_values < 0)
    label_summary = {
        "n_train_rows_total": int(train.sum()),
        "label_positive": int(positive_label.sum()),
        "label_nonpositive": int(nonpositive_label.sum()),
    }
    sign_summary = {
        "n_train_rows_total": int(train.sum()),
        "sign_positive": int(positive_sign.sum()),
        "sign_zero": int(zero_sign.sum()),
        "sign_negative": int(negative_sign.sum()),
        "label_positive_sign_zero_overlap": int((positive_label & zero_sign).sum()),
        "label_positive_sign_negative_overlap": int((positive_label & negative_sign).sum()),
    }
    return label_summary, sign_summary


def _feedback_degree_vectors(data: Any, feedback_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return user/item degree vectors for one observed feedback overlay."""
    n_users = int(data.n_users)
    n_items = int(data.n_items)
    if feedback_edges.size == 0:
        return np.zeros(n_users, dtype=np.float64), np.zeros(n_items, dtype=np.float64)
    user_degree = np.bincount(feedback_edges[:, 0], minlength=n_users).astype(np.float64)
    item_degree = np.bincount(feedback_edges[:, 1] - n_users, minlength=n_items).astype(
        np.float64,
    )
    return user_degree, item_degree


def _exclusive_label_sign_classes(
    labels: np.ndarray,
    signs: np.ndarray,
    *,
    has_graded_sign: bool,
) -> np.ndarray:
    """Return exclusive class labels for stratified topology sampling."""
    label_part = np.where(labels > 0, "label>0", "label<=0")
    if not has_graded_sign:
        return label_part.astype(str, copy=False)
    sign_part = np.where(signs < 0, "sign<0", np.where(signs > 0, "sign>0", "sign=0"))
    return np.char.add(np.char.add(label_part.astype(str, copy=False), "|"), sign_part)


def _stratified_quotas(
    class_ids: np.ndarray,
    *,
    max_count: int,
) -> dict[str, int]:
    """Allocate deterministic largest-remainder quotas from class frequencies."""
    labels, counts = np.unique(class_ids.astype(str, copy=False), return_counts=True)
    if labels.size == 0 or max_count <= 0:
        return {}
    total = int(counts.sum())
    target = min(int(max_count), total)
    raw = counts.astype(np.float64) * (target / float(total))
    quotas = np.floor(raw).astype(np.int64)
    if target >= labels.size:
        quotas = np.maximum(quotas, 1)
    while int(quotas.sum()) > target:
        positive = np.flatnonzero(quotas > 0)
        index = positive[np.argmin(raw[positive] - np.floor(raw[positive]))]
        quotas[index] -= 1
    remainder_order = np.argsort(-(raw - np.floor(raw)), kind="stable")
    for index in remainder_order:
        if int(quotas.sum()) >= target:
            break
        quotas[index] += 1
    return {str(label): int(quota) for label, quota in zip(labels, quotas, strict=True)}


def _select_stratified_rows(
    rows: dict[str, np.ndarray],
    quotas: dict[str, int],
    class_ids: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Select deterministic stratified interaction IDs."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []

    class_order = sorted(
        quotas,
        key=lambda label: int(np.count_nonzero(class_ids == label)),
    )
    for class_label in class_order:
        quota = quotas[class_label]
        if quota <= 0:
            continue
        candidates = np.flatnonzero(class_ids == class_label)
        if candidates.size == 0:
            continue
        n_take = min(int(quota), int(candidates.size))
        selected.extend(rng.choice(candidates, size=n_take, replace=False).astype(int).tolist())
    return np.asarray(selected, dtype=np.int64)


def _edge_keys(edges: np.ndarray) -> set[tuple[int, int]]:
    """Return hashable edge keys for user-item node pairs."""
    if edges.size == 0:
        return set()
    return {(int(user), int(item)) for user, item in edges}


def _select_degree_aware_positive_rows(
    rows: dict[str, np.ndarray],
    *,
    data: Any,
    user_degree: np.ndarray,
    item_degree: np.ndarray,
    max_edges: int,
    seed: int,
) -> np.ndarray:
    """Select a readable positive-train GCN core instead of uniform sparse rows."""
    del seed  # deterministic degree order is preferable for this thesis figure.
    n_users = int(data.n_users)
    positive_ids = np.flatnonzero(rows["labels"] > 0)
    if positive_ids.size == 0:
        return np.empty(0, dtype=np.int64)

    users = rows["users"][positive_ids]
    item_nodes = rows["items"][positive_ids]
    items = item_nodes - n_users
    valid = (items >= 0) & (items < item_degree.shape[0])
    positive_ids = positive_ids[valid]
    users = users[valid]
    items = items[valid]
    if positive_ids.size == 0:
        return np.empty(0, dtype=np.int64)

    target_count = min(int(max_edges), int(positive_ids.size))
    positive_items = np.unique(items)
    item_edge_counts = np.bincount(items, minlength=item_degree.shape[0]).astype(np.float64)
    item_order = positive_items[np.argsort(-item_degree[positive_items], kind="stable")]
    minimum_item_count = min(TOPOLOGY_CORE_ITEM_LIMIT, item_order.size)
    cumulative_item_edges = np.cumsum(item_edge_counts[item_order])
    coverage_item_count = int(
        min(
            item_order.size,
            np.searchsorted(cumulative_item_edges, target_count, side="left") + 1,
        ),
    )
    target_item_count = max(minimum_item_count, coverage_item_count)
    target_items = item_order[:target_item_count]
    target_item_set = set(int(item) for item in target_items)
    per_item_quota = max(2, math.ceil(target_count / max(1, len(target_items))))
    active_users = max(1, int(np.count_nonzero(user_degree > 0)))
    target_user_count = min(
        target_count,
        max(320, min(560, round(math.sqrt(active_users) * 5.0))),
    )

    selected_ids: list[int] = []
    selected_edges: set[tuple[int, int]] = set()
    selected_users: set[int] = set()

    def try_add(local_index: int) -> bool:
        row_id = int(positive_ids[local_index])
        user = int(rows["users"][row_id])
        item_node = int(rows["items"][row_id])
        edge = (user, item_node)
        if edge in selected_edges:
            return False
        if user not in selected_users and len(selected_users) >= target_user_count:
            return False
        selected_edges.add(edge)
        selected_users.add(user)
        selected_ids.append(row_id)
        return True

    for item in target_items:
        local_indices = np.flatnonzero(items == int(item))
        ordered = local_indices[
            np.lexsort(
                (
                    users[local_indices],
                    -user_degree[users[local_indices]],
                ),
            )
        ]
        new_user_order = [
            int(local_index)
            for local_index in ordered
            if int(users[local_index]) not in selected_users
        ]
        existing_user_order = [
            int(local_index)
            for local_index in ordered
            if int(users[local_index]) in selected_users
        ]
        added_for_item = 0
        for local_index in (*new_user_order, *existing_user_order):
            if try_add(local_index):
                added_for_item += 1
            if added_for_item >= per_item_quota or len(selected_ids) >= target_count:
                break
        if len(selected_ids) >= target_count:
            break

    if len(selected_ids) < target_count:
        fill_order = np.lexsort(
            (
                items,
                users,
                -user_degree[users],
                -item_degree[items],
            ),
        )
        for local_index in fill_order:
            if int(items[local_index]) not in target_item_set:
                continue
            try_add(int(local_index))
            if len(selected_ids) >= target_count:
                break

    return np.asarray(selected_ids, dtype=np.int64)


def _select_overlay_rows(
    rows: dict[str, np.ndarray],
    *,
    sign_mask: np.ndarray,
    selected_users: set[int],
    selected_items: set[int],
    selected_positive_edges: set[tuple[int, int]],
    max_count: int,
    seed: int,
) -> np.ndarray:
    """Select sign-overlay rows that fall inside the readable positive core."""
    candidate_ids = np.flatnonzero(sign_mask)
    if candidate_ids.size == 0 or max_count <= 0:
        return np.empty(0, dtype=np.int64)

    users = rows["users"][candidate_ids]
    items = rows["items"][candidate_ids]
    core_edge_mask = np.fromiter(
        (
            (int(user), int(item)) in selected_positive_edges
            for user, item in zip(users, items, strict=True)
        ),
        dtype=np.bool_,
        count=candidate_ids.size,
    )
    core_node_mask = np.fromiter(
        (
            int(user) in selected_users and int(item) in selected_items
            for user, item in zip(users, items, strict=True)
        ),
        dtype=np.bool_,
        count=candidate_ids.size,
    )

    primary = candidate_ids[core_edge_mask]
    secondary = candidate_ids[core_node_mask & ~core_edge_mask]
    ordered = np.concatenate((primary, secondary))
    if ordered.size <= max_count:
        return ordered.astype(np.int64, copy=False)

    rng = np.random.default_rng(seed)
    primary_take = min(primary.size, max_count)
    selected_parts: list[np.ndarray] = []
    if primary_take > 0:
        selected_parts.append(primary[:primary_take])
    remaining = max_count - primary_take
    if remaining > 0 and secondary.size > 0:
        selected_parts.append(
            rng.choice(secondary, size=min(remaining, secondary.size), replace=False),
        )
    if not selected_parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(selected_parts).astype(np.int64, copy=False)


def _append_unique_ids(
    selected_ids: list[int],
    selected_set: set[int],
    candidate_ids: np.ndarray,
    *,
    target_count: int,
) -> None:
    """Append candidate row ids until ``target_count`` unique ids are selected."""
    for row_id_value in candidate_ids:
        row_id = int(row_id_value)
        if row_id in selected_set:
            continue
        selected_ids.append(row_id)
        selected_set.add(row_id)
        if len(selected_ids) >= target_count:
            break


def _select_contextual_row_ids(
    rows: dict[str, np.ndarray],
    candidate_ids: np.ndarray,
    *,
    selected_users: set[int],
    selected_items: set[int],
    selected_positive_edges: set[tuple[int, int]],
    exclude_ids: set[int],
    max_count: int,
    seed: int,
) -> np.ndarray:
    """Select rows near the positive core first, then fill deterministically."""
    if candidate_ids.size == 0 or max_count <= 0:
        return np.empty(0, dtype=np.int64)

    candidate_ids = np.asarray(
        [int(row_id) for row_id in candidate_ids if int(row_id) not in exclude_ids],
        dtype=np.int64,
    )
    if candidate_ids.size == 0:
        return np.empty(0, dtype=np.int64)

    users = rows["users"][candidate_ids]
    items = rows["items"][candidate_ids]
    core_edge_mask = np.fromiter(
        (
            (int(user), int(item)) in selected_positive_edges
            for user, item in zip(users, items, strict=True)
        ),
        dtype=np.bool_,
        count=candidate_ids.size,
    )
    core_node_mask = np.fromiter(
        (
            int(user) in selected_users and int(item) in selected_items
            for user, item in zip(users, items, strict=True)
        ),
        dtype=np.bool_,
        count=candidate_ids.size,
    )
    groups = (
        candidate_ids[core_edge_mask],
        candidate_ids[core_node_mask & ~core_edge_mask],
        candidate_ids[~core_node_mask & ~core_edge_mask],
    )
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    remaining = int(max_count)
    for group in groups:
        if remaining <= 0 or group.size == 0:
            continue
        if group.size <= remaining:
            selected.append(group)
            remaining -= int(group.size)
        else:
            selected.append(rng.choice(group, size=remaining, replace=False))
            remaining = 0
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(selected).astype(np.int64, copy=False)


def _readable_topology_edges(
    data: Any,
    *,
    interaction_sign: np.ndarray | None,
    user_degree: np.ndarray,
    item_degree: np.ndarray,
    max_edges: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return a fixed-budget readable train topology sample with sign diagnostics."""
    rows = _observed_train_feedback_rows(data, interaction_sign=interaction_sign)
    if rows["users"].size == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return empty, empty, empty, {"strategy": "empty"}

    target_rows = min(int(max_edges), int(rows["users"].size))
    has_graded_sign = bool(rows["has_graded_sign"][0])
    class_ids = _exclusive_label_sign_classes(
        rows["labels"],
        rows["signs"],
        has_graded_sign=has_graded_sign,
    )
    full_class_counts = {
        str(label): int(count)
        for label, count in zip(*np.unique(class_ids, return_counts=True), strict=True)
    }
    positive_available = int(np.count_nonzero(rows["labels"] > 0))
    if has_graded_sign:
        positive_target = min(
            positive_available,
            max(1, round(target_rows * (1.0 - TOPOLOGY_DIAGNOSTIC_ROW_FRACTION))),
        )
    else:
        positive_target = min(positive_available, target_rows)

    selected_positive_ids = _select_degree_aware_positive_rows(
        rows,
        data=data,
        user_degree=user_degree,
        item_degree=item_degree,
        max_edges=positive_target,
        seed=seed,
    )

    selected_ids: list[int] = []
    selected_set: set[int] = set()
    _append_unique_ids(
        selected_ids,
        selected_set,
        selected_positive_ids,
        target_count=target_rows,
    )

    selected_positive_edges = (
        np.column_stack(
            (rows["users"][selected_positive_ids], rows["items"][selected_positive_ids]),
        ).astype(np.int64, copy=False)
        if selected_positive_ids.size
        else np.empty((0, 2), dtype=np.int64)
    )
    selected_users = set(int(user) for user in rows["users"][selected_positive_ids])
    selected_items = set(int(item) for item in rows["items"][selected_positive_ids])
    selected_positive_edge_keys = _edge_keys(selected_positive_edges)

    if has_graded_sign and len(selected_ids) < target_rows:
        diagnostic_ids = np.flatnonzero((rows["signs"] == 0) | (rows["signs"] < 0))
        diagnostic_classes = np.where(rows["signs"][diagnostic_ids] < 0, "sign<0", "sign=0")
        quotas = _stratified_quotas(
            diagnostic_classes,
            max_count=target_rows - len(selected_ids),
        )
        for sign_label, sign_mask, seed_offset in (
            ("sign<0", rows["signs"] < 0, 17),
            ("sign=0", rows["signs"] == 0, 11),
        ):
            quota = quotas.get(sign_label, 0)
            if quota <= 0 or len(selected_ids) >= target_rows:
                continue
            contextual_ids = _select_contextual_row_ids(
                rows,
                np.flatnonzero(sign_mask),
                selected_users=selected_users,
                selected_items=selected_items,
                selected_positive_edges=selected_positive_edge_keys,
                exclude_ids=selected_set,
                max_count=min(quota, target_rows - len(selected_ids)),
                seed=seed + seed_offset,
            )
            _append_unique_ids(
                selected_ids,
                selected_set,
                contextual_ids,
                target_count=target_rows,
            )

    if len(selected_ids) < target_rows:
        fill_positive_ids = _select_degree_aware_positive_rows(
            rows,
            data=data,
            user_degree=user_degree,
            item_degree=item_degree,
            max_edges=target_rows,
            seed=seed + 23,
        )
        _append_unique_ids(
            selected_ids,
            selected_set,
            fill_positive_ids,
            target_count=target_rows,
        )

    if len(selected_ids) < target_rows:
        fallback_ids = _select_contextual_row_ids(
            rows,
            np.arange(rows["users"].size, dtype=np.int64),
            selected_users=selected_users,
            selected_items=selected_items,
            selected_positive_edges=selected_positive_edge_keys,
            exclude_ids=selected_set,
            max_count=target_rows - len(selected_ids),
            seed=seed + 31,
        )
        _append_unique_ids(
            selected_ids,
            selected_set,
            fallback_ids,
            target_count=target_rows,
        )

    displayed_ids = np.asarray(selected_ids[:target_rows], dtype=np.int64)
    sample_class_ids = class_ids[displayed_ids]
    sample_class_counts = {
        str(label): int(count)
        for label, count in zip(*np.unique(sample_class_ids, return_counts=True), strict=True)
    }

    def edges_for(row_ids: np.ndarray) -> np.ndarray:
        if row_ids.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        return np.column_stack((rows["users"][row_ids], rows["items"][row_ids])).astype(
            np.int64,
            copy=False,
        )

    positive_ids = displayed_ids[rows["labels"][displayed_ids] > 0]
    zero_ids = (
        displayed_ids[rows["signs"][displayed_ids] == 0]
        if has_graded_sign
        else np.empty(0, dtype=np.int64)
    )
    negative_ids = displayed_ids[rows["signs"][displayed_ids] < 0]
    positive_edges = edges_for(positive_ids)
    zero_edges = edges_for(zero_ids)
    negative_edges = edges_for(negative_ids)
    displayed_edges = edges_for(displayed_ids)
    summary = {
        "strategy": "fixed_display_row_budget_degree_aware_positive_core_with_sign_diagnostics",
        "has_graded_sign": has_graded_sign,
        "target_display_rows": int(target_rows),
        "target_positive_core_edges": int(positive_target),
        "n_sampled_rows": int(displayed_ids.size),
        "n_sampled_users": int(np.unique(displayed_edges[:, 0]).size) if displayed_edges.size else 0,
        "n_sampled_items": int(np.unique(displayed_edges[:, 1] - int(data.n_users)).shape[0])
        if displayed_edges.size
        else 0,
        "n_label_positive_rows": int(positive_ids.size),
        "n_sign_positive_rows": int(np.count_nonzero(rows["signs"][displayed_ids] > 0)),
        "n_sign_zero_rows": int(zero_ids.size) if has_graded_sign else 0,
        "n_sign_negative_rows": int(negative_ids.size),
        "n_positive_core_edges": int(positive_edges.shape[0]),
        "n_zero_overlay_rows": int(zero_edges.shape[0]),
        "n_negative_overlay_rows": int(negative_edges.shape[0]),
        "full_class_counts": full_class_counts,
        "sample_class_counts": sample_class_counts,
        "core_item_limit": int(TOPOLOGY_CORE_ITEM_LIMIT),
        "diagnostic_row_fraction": float(TOPOLOGY_DIAGNOSTIC_ROW_FRACTION)
        if has_graded_sign
        else 0.0,
    }
    return positive_edges, zero_edges, negative_edges, summary


def _topology_node_size(degree: float, max_degree: float) -> float:
    """Smaller log-degree node size for dense topology panels."""
    if max_degree <= 0:
        return 4.0
    return 4.0 + 20.0 * (np.log1p(float(degree)) / np.log1p(float(max_degree)))


def _project_learned_embeddings(
    *,
    config: Any,
    canonical: Any,
    data: Any,
    checkpoint_payload: dict[str, Any],
    user_degree: np.ndarray,
    item_degree: np.ndarray,
    neutral_user_degree: np.ndarray,
    neutral_item_degree: np.ndarray,
    negative_user_degree: np.ndarray,
    negative_item_degree: np.ndarray,
    args: argparse.Namespace,
) -> ProjectionRecord:
    """Project propagated learned interest embeddings for one checkpoint."""
    user_embeddings, item_embeddings, key_label = _load_model_projection(
        config=config,
        data=data,
        canonical=canonical,
        checkpoint_payload=checkpoint_payload,
        view="interest",
    )
    user_ids = _select_ids_by_degree(
        user_degree,
        max_count=args.projection_max_users,
        top_share=args.projection_top_share,
        seed=args.seed,
    )
    item_ids = _select_ids_by_degree(
        item_degree,
        max_count=args.projection_max_items,
        top_share=args.projection_top_share,
        seed=args.seed + 1,
    )
    matrix = np.vstack((user_embeddings[user_ids], item_embeddings[item_ids]))
    roles = np.asarray(["user"] * len(user_ids) + ["item"] * len(item_ids))
    degrees = np.concatenate((user_degree[user_ids], item_degree[item_ids]))
    neutral_degrees = np.concatenate(
        (neutral_user_degree[user_ids], neutral_item_degree[item_ids]),
    )
    negative_degrees = np.concatenate(
        (negative_user_degree[user_ids], negative_item_degree[item_ids]),
    )
    xy, metadata = _project_matrix(
        matrix,
        method="umap",
        seed=args.seed,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    metadata["embedding_keys"] = key_label
    return ProjectionRecord(
        xy=xy,
        roles=roles,
        degrees=degrees,
        neutral_degrees=neutral_degrees,
        negative_degrees=negative_degrees,
        source="propagated learned interest embeddings",
        n_points=int(matrix.shape[0]),
        n_features=int(matrix.shape[1]),
        metadata=metadata,
        graph_summary={},
    )


def _feature_config_from_reference(config: Any) -> Any:
    """Reuse the reference graph/data view but turn safe item features on."""
    feature_config = copy.deepcopy(config)
    feature_config.use_features = True
    feature_config.show_progress_bar = False
    feature_config.use_amp = False
    feature_config.use_torch_compile = False
    feature_config.validate()
    return feature_config


def _project_item_features(
    *,
    config: Any,
    args: argparse.Namespace,
) -> tuple[ProjectionRecord | None, str | None]:
    """Project encoded item features using the same dataset view as the reference run."""
    try:
        feature_config = _feature_config_from_reference(config)
        canonical, data = load_runtime_data(feature_config)
        user_item_edges = _positive_train_edges(data)
        negative_edges = _observed_train_feedback_edges(
            data,
            "negative",
            interaction_sign=canonical.sign,
        )
        neutral_edges = _observed_train_feedback_edges(
            data,
            "neutral",
            interaction_sign=canonical.sign,
        )
        user_degree, item_degree = _degree_vectors(data, user_item_edges)
        _negative_user_degree, negative_item_degree = _feedback_degree_vectors(
            data,
            negative_edges,
        )
        _neutral_user_degree, neutral_item_degree = _feedback_degree_vectors(
            data,
            neutral_edges,
        )
        graph_summary = _train_graph_summary(user_item_edges, user_degree, item_degree)
        item_features, source_label = _load_feature_projection(data)
    except Exception as exc:
        return None, str(exc)

    item_ids = _select_ids_by_degree(
        item_degree,
        max_count=args.projection_max_items,
        top_share=args.projection_top_share,
        seed=args.seed + 1,
    )
    matrix = item_features[item_ids]
    roles = np.asarray(["item"] * len(item_ids))
    degrees = item_degree[item_ids]
    neutral_degrees = neutral_item_degree[item_ids]
    negative_degrees = negative_item_degree[item_ids]
    xy, metadata = _project_matrix(
        matrix,
        method="umap",
        seed=args.seed,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    return (
        ProjectionRecord(
            xy=xy,
            roles=roles,
            degrees=degrees,
            neutral_degrees=neutral_degrees,
            negative_degrees=negative_degrees,
            source=f"encoded {source_label}",
            n_points=int(matrix.shape[0]),
            n_features=int(matrix.shape[1]),
            metadata=metadata,
            graph_summary=graph_summary,
        ),
        None,
    )


def _load_dataset_bundle(spec: DatasetSpec, args: argparse.Namespace) -> DatasetBundle:
    """Load one dataset/checkpoint and reduce it to figure-ready arrays."""
    payload = _load_checkpoint_payload(spec.checkpoint)
    saved_config = payload["config"]
    config = _visualization_config_from_checkpoint(saved_config, args.device)
    canonical, data = load_runtime_data(config)
    user_item_edges = _positive_train_edges(data)
    negative_edges = _observed_train_feedback_edges(
        data,
        "negative",
        interaction_sign=canonical.sign,
    )
    neutral_edges = _observed_train_feedback_edges(
        data,
        "neutral",
        interaction_sign=canonical.sign,
    )
    user_degree, item_degree = _degree_vectors(data, user_item_edges)
    negative_user_degree, negative_item_degree = _feedback_degree_vectors(
        data,
        negative_edges,
    )
    neutral_user_degree, neutral_item_degree = _feedback_degree_vectors(
        data,
        neutral_edges,
    )
    graph_summary = _train_graph_summary(user_item_edges, user_degree, item_degree)
    negative_summary = _feedback_summary(data, negative_edges)
    neutral_summary = _feedback_summary(data, neutral_edges)
    train_label_summary, train_sign_summary = _train_label_sign_summary(data, canonical.sign)
    topology_edges, topology_neutral_edges, topology_negative_edges, topology_sample_summary = (
        _readable_topology_edges(
            data,
            interaction_sign=canonical.sign,
            user_degree=user_degree,
            item_degree=item_degree,
            max_edges=args.max_edges,
            seed=args.seed,
        )
    )

    learned_projection = _project_learned_embeddings(
        config=config,
        canonical=canonical,
        data=data,
        checkpoint_payload=payload,
        user_degree=user_degree,
        item_degree=item_degree,
        neutral_user_degree=neutral_user_degree,
        neutral_item_degree=neutral_item_degree,
        negative_user_degree=negative_user_degree,
        negative_item_degree=negative_item_degree,
        args=args,
    )
    learned_projection.graph_summary = graph_summary

    feature_projection, feature_unavailable_reason = _project_item_features(
        config=config,
        args=args,
    )

    return DatasetBundle(
        spec=spec,
        config_summary=_config_summary(config),
        topology_edges=topology_edges,
        topology_negative_edges=topology_negative_edges,
        topology_neutral_edges=topology_neutral_edges,
        user_degree=user_degree,
        item_degree=item_degree,
        negative_user_degree=negative_user_degree,
        negative_item_degree=negative_item_degree,
        neutral_user_degree=neutral_user_degree,
        neutral_item_degree=neutral_item_degree,
        graph_summary=graph_summary,
        negative_summary=negative_summary,
        neutral_summary=neutral_summary,
        train_label_summary=train_label_summary,
        train_sign_summary=train_sign_summary,
        topology_sample_summary=topology_sample_summary,
        learned_projection=learned_projection,
        feature_projection=feature_projection,
        feature_unavailable_reason=feature_unavailable_reason,
    )


def _add_global_legend(fig: plt.Figure, *, include_edges: bool = True) -> None:
    """Add one shared user/item legend, optionally including topology edges."""
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=USER_COLOR,
            markersize=8,
            label="Users",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ITEM_COLOR,
            markersize=8,
            label="Items",
        ),
    ]
    if include_edges:
        handles.extend(
            (
                Line2D(
                    [0],
                    [0],
                    color=POSITIVE_EDGE_COLOR,
                    alpha=0.80,
                    linewidth=1.8,
                    label="Propagation row (label > 0)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=ZERO_SIGN_EDGE_COLOR,
                    alpha=0.70,
                    linewidth=1.8,
                    label="Feedback row (sign = 0)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=NEGATIVE_SIGN_EDGE_COLOR,
                    alpha=0.80,
                    linewidth=1.8,
                    label="Feedback row (sign < 0)",
                ),
            ),
        )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)


def _panel_note(ax: plt.Axes, text: str) -> None:
    """Add a small readable annotation inside one panel."""
    ax.text(
        0.015,
        0.015,
        text,
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 3.0},
    )


def _draw_topology_panel(ax: plt.Axes, bundle: DatasetBundle, seed: int) -> None:
    """Draw one NetworkX spring topology panel."""
    n_users_total = int(bundle.user_degree.shape[0])
    graph = nx.Graph()
    graph.add_edges_from(
        (int(user), int(item_node)) for user, item_node in bundle.topology_edges
    )
    graph.add_edges_from(
        (int(user), int(item_node)) for user, item_node in bundle.topology_neutral_edges
    )
    graph.add_edges_from(
        (int(user), int(item_node)) for user, item_node in bundle.topology_negative_edges
    )
    selected_users = sorted(node for node in graph.nodes if int(node) < n_users_total)
    selected_item_nodes = sorted(node for node in graph.nodes if int(node) >= n_users_total)
    pos = nx.spring_layout(graph, k=0.38, iterations=120, seed=seed)

    max_degree = max(
        float(bundle.user_degree.max(initial=0.0)),
        float(bundle.item_degree.max(initial=0.0)),
    )
    user_sizes = [
        _topology_node_size(bundle.user_degree[user], max_degree)
        for user in selected_users
    ]
    item_sizes = [
        _topology_node_size(bundle.item_degree[item_node - n_users_total], max_degree)
        for item_node in selected_item_nodes
    ]
    ax.scatter([], [], c=USER_COLOR, s=14, alpha=TOPOLOGY_NODE_ALPHA, label="Users")
    ax.scatter([], [], c=ITEM_COLOR, s=14, alpha=TOPOLOGY_NODE_ALPHA, label="Items")
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=selected_users,
        node_color=USER_COLOR,
        node_size=user_sizes,
        alpha=TOPOLOGY_NODE_ALPHA,
        linewidths=0.0,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edgelist=[tuple(edge) for edge in bundle.topology_edges.tolist()],
        alpha=0.56,
        edge_color=POSITIVE_EDGE_COLOR,
        width=0.62,
    )
    if bundle.topology_neutral_edges.size:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=[tuple(edge) for edge in bundle.topology_neutral_edges.tolist()],
            alpha=0.46,
            edge_color=ZERO_SIGN_EDGE_COLOR,
            width=0.54,
        )
    if bundle.topology_negative_edges.size:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=[tuple(edge) for edge in bundle.topology_negative_edges.tolist()],
            alpha=0.58,
            edge_color=NEGATIVE_SIGN_EDGE_COLOR,
            width=0.58,
        )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=selected_item_nodes,
        node_color=ITEM_COLOR,
        node_size=item_sizes,
        alpha=TOPOLOGY_NODE_ALPHA,
        linewidths=0.0,
    )
    ax.set_title(bundle.spec.label, fontsize=12, fontweight="bold")
    ax.axis("off")
    n_sampled_rows = int(bundle.topology_sample_summary.get("n_sampled_rows", 0))
    n_sampled_users = int(bundle.topology_sample_summary.get("n_sampled_users", 0))
    n_sampled_items = int(bundle.topology_sample_summary.get("n_sampled_items", 0))
    n_label_positive = int(bundle.topology_sample_summary.get("n_label_positive_rows", 0))
    n_sign_positive = int(bundle.topology_sample_summary.get("n_sign_positive_rows", 0))
    n_sign_zero = int(bundle.topology_sample_summary.get("n_sign_zero_rows", 0))
    n_sign_negative = int(bundle.topology_sample_summary.get("n_sign_negative_rows", 0))
    has_graded_sign = bool(bundle.topology_sample_summary.get("has_graded_sign", False))
    sign_line = (
        "shown signs: omitted (constant sign = 0)"
        if not has_graded_sign
        else (
            "shown signs (> 0 / = 0 / < 0): "
            f"{_format_count(n_sign_positive)}/{_format_count(n_sign_zero)}/"
            f"{_format_count(n_sign_negative)}"
        )
    )
    note = (
        f"displayed: {_format_count(n_sampled_users)} users, "
        f"{_format_count(n_sampled_items)} items, "
        f"{_format_count(n_sampled_rows)} train rows\n"
        f"shown label > 0 rows: {_format_count(n_label_positive)}; {sign_line}\n"
        f"full positive train graph: "
        f"{_format_count(bundle.graph_summary['n_train_positive_edges_total'])}\n"
        "node size = log positive degree"
    )
    _panel_note(ax, note)


def _plot_topology_figure(
    bundles: list[DatasetBundle],
    output_dir: Path,
    seed: int,
) -> Path:
    """Write the combined NetworkX GCN topology figure."""
    output_path = output_dir / "core_gcn_topology_networkx_spring.png"
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.8))
    for index, (ax, bundle) in enumerate(zip(axes.ravel(), bundles, strict=True)):
        _draw_topology_panel(ax, bundle, seed + index)
    fig.suptitle(
        "EDGRec GCN Training Topology",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.935,
        "Each panel uses the same displayed train-row budget. Green marks label > 0 propagation rows; gray/red mark sign = 0 / sign < 0 feedback rows.",
        ha="center",
        fontsize=9.5,
    )
    _add_global_legend(fig, include_edges=True)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.92))
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _draw_degree_panel(ax: plt.Axes, bundle: DatasetBundle) -> None:
    """Draw one full train-degree distribution panel."""
    user_positive = _positive_degrees(bundle.user_degree)
    item_positive = _positive_degrees(bundle.item_degree)
    max_degree = max(float(user_positive.max()), float(item_positive.max()), 1.0)
    bins = np.logspace(0.0, np.log10(max_degree + 1.0), 48)
    ax.hist(user_positive, bins=bins, color=USER_COLOR, alpha=0.58, label="Users")
    ax.hist(item_positive, bins=bins, color=ITEM_COLOR, alpha=0.58, label="Items")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        (
            f"{bundle.spec.label}\n"
            f"{_format_count(bundle.graph_summary['n_train_positive_edges_total'])} "
            "label-positive train edges"
        ),
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("positive train degree (log scale)")
    ax.set_ylabel("number of nodes (log scale)")


def _plot_degree_figure(bundles: list[DatasetBundle], output_dir: Path) -> Path:
    """Write the combined full-train degree distribution figure."""
    output_path = output_dir / "core_train_degree_distributions.png"
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.8))
    for ax, bundle in zip(axes.ravel(), bundles, strict=True):
        _draw_degree_panel(ax, bundle)
    handles = [
        Line2D([0], [0], color=USER_COLOR, lw=7, alpha=0.58, label="Users"),
        Line2D([0], [0], color=ITEM_COLOR, lw=7, alpha=0.58, label="Items"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Full Positive-Train Degree Distributions", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.93))
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _point_sizes(degrees: np.ndarray, scale: float = 0.10) -> np.ndarray:
    """Return projection marker sizes from train degree."""
    max_degree = float(degrees.max(initial=0.0))
    return np.asarray([max(5.0, _node_size(degree, max_degree) * scale) for degree in degrees])


def _signed_overlay_masks(projection: ProjectionRecord) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-sign and negative-sign sampled-point masks."""
    negative_mask = projection.negative_degrees > 0
    neutral_mask = (projection.neutral_degrees > 0) & ~negative_mask
    return neutral_mask, negative_mask


def _draw_signed_point_outlines(
    ax: plt.Axes,
    projection: ProjectionRecord,
    sizes: np.ndarray,
) -> None:
    """Overlay sign=0 and sign<0 feedback outlines on a projection panel."""
    neutral_mask, negative_mask = _signed_overlay_masks(projection)
    if bool(neutral_mask.any()):
        ax.scatter(
            projection.xy[neutral_mask, 0],
            projection.xy[neutral_mask, 1],
            s=sizes[neutral_mask] + 8.0,
            facecolors="none",
            edgecolors=ZERO_SIGN_EDGE_COLOR,
            alpha=0.68,
            linewidths=0.42,
        )
    if bool(negative_mask.any()):
        ax.scatter(
            projection.xy[negative_mask, 0],
            projection.xy[negative_mask, 1],
            s=sizes[negative_mask] + 9.0,
            facecolors="none",
            edgecolors=NEGATIVE_SIGN_EDGE_COLOR,
            alpha=0.76,
            linewidths=0.50,
        )


def _draw_learned_projection_panel(ax: plt.Axes, bundle: DatasetBundle) -> None:
    """Draw one learned interest-embedding UMAP panel."""
    projection = bundle.learned_projection
    if projection is None:
        ax.text(0.5, 0.5, "projection unavailable", ha="center", va="center")
        ax.axis("off")
        return
    sizes = _point_sizes(projection.degrees)
    user_mask = projection.roles == "user"
    item_mask = projection.roles == "item"
    ax.scatter(
        projection.xy[user_mask, 0],
        projection.xy[user_mask, 1],
        s=sizes[user_mask],
        c=USER_COLOR,
        alpha=PROJECTION_POINT_ALPHA,
        linewidths=0.0,
        edgecolors="none",
    )
    ax.scatter(
        projection.xy[item_mask, 0],
        projection.xy[item_mask, 1],
        s=sizes[item_mask],
        c=ITEM_COLOR,
        alpha=PROJECTION_POINT_ALPHA,
        linewidths=0.0,
        edgecolors="none",
    )
    _draw_signed_point_outlines(ax, projection, sizes)
    ax.set_title(bundle.spec.label, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    trust = projection.metadata.get("trustworthiness")
    trust_text = f", trustworthiness {trust:.3f}" if isinstance(trust, float) else ""
    note = (
        f"{projection.n_points} nodes{trust_text}\n"
        f"sign = 0 / sign < 0 outlines: {int((projection.neutral_degrees > 0).sum())}/"
        f"{int((projection.negative_degrees > 0).sum())}\n"
        "UMAP: local neighborhoods only"
    )
    _panel_note(ax, note)


def _plot_learned_umap_figure(
    bundles: list[DatasetBundle],
    output_dir: Path,
) -> Path:
    """Write the combined learned-embedding UMAP figure."""
    output_path = output_dir / "core_learned_interest_umap.png"
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2))
    for ax, bundle in zip(axes.ravel(), bundles, strict=True):
        _draw_learned_projection_panel(ax, bundle)
    fig.suptitle(
        "UMAP of Learned EDGRec Interest Embeddings",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.935,
        "Users and items are sampled from the train-active graph; gray/red outlines mark sampled nodes with sign = 0 or sign < 0 train feedback.",
        ha="center",
        fontsize=10,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=USER_COLOR,
            markersize=8,
            label="Users",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ITEM_COLOR,
            markersize=8,
            label="Items",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=ZERO_SIGN_EDGE_COLOR,
            markerfacecolor="none",
            markersize=8,
            label="Has sign = 0 feedback",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=NEGATIVE_SIGN_EDGE_COLOR,
            markerfacecolor="none",
            markersize=8,
            label="Has negative-sign feedback",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.92))
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _draw_feature_projection_panel(
    ax: plt.Axes,
    bundle: DatasetBundle,
    norm: Normalize,
) -> Any:
    """Draw one encoded item-feature UMAP panel."""
    projection = bundle.feature_projection
    if projection is None:
        ax.text(
            0.5,
            0.5,
            "No item-feature projection\nunder this thesis data view",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_title(bundle.spec.label, fontsize=12, fontweight="bold")
        ax.axis("off")
        return None
    colors = np.log1p(projection.degrees)
    sizes = _point_sizes(projection.degrees)
    scatter = ax.scatter(
        projection.xy[:, 0],
        projection.xy[:, 1],
        s=sizes,
        c=colors,
        cmap="viridis",
        norm=norm,
        alpha=FEATURE_POINT_ALPHA,
        linewidths=0.0,
        edgecolors="none",
    )
    _draw_signed_point_outlines(ax, projection, sizes)
    ax.set_title(bundle.spec.label, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    trust = projection.metadata.get("trustworthiness")
    trust_text = f", trustworthiness {trust:.3f}" if isinstance(trust, float) else ""
    note = (
        f"{projection.n_points} items{trust_text}\n"
        f"sign = 0 / sign < 0 outlines: {int((projection.neutral_degrees > 0).sum())}/"
        f"{int((projection.negative_degrees > 0).sum())}\n"
        "color = log1p(train degree)"
    )
    _panel_note(ax, note)
    return scatter


def _plot_feature_umap_figure(
    bundles: list[DatasetBundle],
    output_dir: Path,
) -> Path:
    """Write the combined encoded item-feature UMAP figure."""
    output_path = output_dir / "core_item_feature_umap.png"
    feature_records = [bundle.feature_projection for bundle in bundles if bundle.feature_projection]
    max_color = max(
        (float(np.log1p(record.degrees).max(initial=0.0)) for record in feature_records),
        default=1.0,
    )
    norm = Normalize(vmin=0.0, vmax=max(1.0, max_color))
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2))
    scatter = None
    for ax, bundle in zip(axes.ravel(), bundles, strict=True):
        panel_scatter = _draw_feature_projection_panel(ax, bundle, norm)
        scatter = panel_scatter if panel_scatter is not None else scatter
    fig.suptitle(
        "UMAP of Encoded Thesis-Policy Item Features",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.935,
        "Feature panels use the same dataset/preprocessing/item-universe view as the reference checkpoint, with safe item features enabled.",
        ha="center",
        fontsize=10,
    )
    if scatter is not None:
        colorbar_axis = fig.add_axes([0.91, 0.18, 0.018, 0.62])
        colorbar = fig.colorbar(scatter, cax=colorbar_axis)
        colorbar.set_label("log1p(positive train degree)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=ZERO_SIGN_EDGE_COLOR,
            markerfacecolor="none",
            markersize=8,
            label="Has sign = 0 feedback",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=NEGATIVE_SIGN_EDGE_COLOR,
            markerfacecolor="none",
            markersize=8,
            label="Has negative-sign feedback",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(
        left=0.06,
        right=0.88,
        bottom=0.10,
        top=0.88,
        hspace=0.30,
        wspace=0.18,
    )
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _records_for_json(paths: dict[str, Path], bundles: list[DatasetBundle]) -> dict[str, Any]:
    """Build machine-readable metadata for all generated figures."""
    return {
        "artifacts": {name: _relative(path) for name, path in paths.items()},
        "figure_scope": (
            "Qualitative thesis figures. Positive-label training rows define the "
            "EDGRec/LightGCN propagation graph. The NetworkX topology uses the same "
            "displayed train-row budget per dataset, prioritizes a degree-aware "
            "label-positive propagation core, and uses the remaining rows for "
            "non-positive sign diagnostics when available. Positive-sign rows are "
            "counted in the metadata but are not drawn as a separate overlay because "
            "that would usually duplicate the label-positive propagation rows. "
            "UMAP panels are sampled projections "
            "and must not be used as ranking-performance evidence."
        ),
        "datasets": [
            {
                "dataset": bundle.spec.dataset,
                "label": bundle.spec.label,
                "reference_checkpoint": _relative(bundle.spec.checkpoint),
                "reference_note": bundle.spec.note,
                "feature_note": bundle.spec.feature_note,
                "config": bundle.config_summary,
                "graph": bundle.graph_summary,
                "topology_sample": {
                    "summary": bundle.topology_sample_summary,
                    "n_positive_edges": int(bundle.topology_edges.shape[0]),
                    "n_neutral_edges": int(bundle.topology_neutral_edges.shape[0]),
                    "n_negative_edges": int(bundle.topology_negative_edges.shape[0]),
                    "n_users": int(
                        np.unique(
                            np.concatenate(
                                (
                                    bundle.topology_edges[:, 0],
                                    bundle.topology_neutral_edges[:, 0]
                                    if bundle.topology_neutral_edges.size
                                    else np.asarray([], dtype=np.int64),
                                    bundle.topology_negative_edges[:, 0]
                                    if bundle.topology_negative_edges.size
                                    else np.asarray([], dtype=np.int64),
                                ),
                            ),
                        ).shape[0],
                    ),
                    "n_items": int(
                        np.unique(
                            np.concatenate(
                                (
                                    bundle.topology_edges[:, 1],
                                    bundle.topology_neutral_edges[:, 1]
                                    if bundle.topology_neutral_edges.size
                                    else np.asarray([], dtype=np.int64),
                                    bundle.topology_negative_edges[:, 1]
                                    if bundle.topology_negative_edges.size
                                    else np.asarray([], dtype=np.int64),
                                ),
                            ),
                        ).shape[0],
                    ),
                },
                "zero_sign_train_feedback": bundle.neutral_summary,
                "negative_train_feedback": bundle.negative_summary,
                "train_label_summary": bundle.train_label_summary,
                "train_sign_summary": bundle.train_sign_summary,
                "learned_umap": _projection_json(bundle.learned_projection),
                "feature_umap": _projection_json(bundle.feature_projection)
                if bundle.feature_projection is not None
                else {"available": False, "reason": bundle.feature_unavailable_reason},
            }
            for bundle in bundles
        ],
    }


def _projection_json(projection: ProjectionRecord | None) -> dict[str, Any] | None:
    """Serialize one projection record without large arrays."""
    if projection is None:
        return None
    payload = {
        "available": projection.unavailable_reason is None,
        "source": projection.source,
        "n_points": projection.n_points,
        "n_features": projection.n_features,
        "n_sampled_zero_sign_points": int((projection.neutral_degrees > 0).sum()),
        "n_sampled_negative_sign_points": int((projection.negative_degrees > 0).sum()),
        "metadata": projection.metadata,
    }
    if projection.graph_summary:
        payload["graph"] = projection.graph_summary
    if projection.unavailable_reason:
        payload["reason"] = projection.unavailable_reason
    return payload


def _write_readme(paths: dict[str, Path], bundles: list[DatasetBundle], output_dir: Path) -> Path:
    """Write thesis-oriented interpretation notes for the generated figures."""
    path = output_dir / "README.md"
    lines = [
        "# Graph and Embedding Figures",
        "",
        "This directory contains the consolidated thesis-facing visualization set. The old",
        "per-dataset `*_umap_checkpoint/` and `*_feature_umap/` folders were intentionally",
        "replaced because they repeated the same topology and degree plots.",
        "",
        "These figures are qualitative evidence. They explain the EDGRec data and model",
        "geometry, but ranking, feature-selection, and method-comparison claims still need",
        "the validation/full-data result tables.",
        "",
        "## Artifact Inventory",
        "",
        "| File | What it shows | How to read it |",
        "| --- | --- | --- |",
        f"| `{paths['topology'].name}` | Four NetworkX spring-layout views of fixed-budget train-row samples from the GCN training view. | Blue nodes are users, yellow nodes are items, and green edges are displayed `label > 0` interactions used by GCN propagation. Gray/red edges show displayed `sign = 0`/`sign < 0` feedback rows. `sign > 0` rows are counted in the panel notes and `{paths['metadata'].name}` rather than overdrawn because they usually coincide with label-positive propagation rows. |",
        f"| `{paths['degree'].name}` | Full positive-train user/item degree distributions. | Histograms are the GCN scale check; label/sign row summaries are kept in `{paths['metadata'].name}` instead of crowding the plot. |",
        f"| `{paths['learned_umap'].name}` | UMAP projections of propagated learned EDGRec interest embeddings from reference checkpoints. | Blue/yellow fill separates users/items; gray/red outlines mark sampled nodes with `sign = 0` or `sign < 0` feedback. Use local neighborhoods only. |",
        f"| `{paths['feature_umap'].name}` | UMAP projections of encoded thesis-policy item features. | Points are items, colored by `log1p(train degree)` and outlined when `sign = 0` or `sign < 0` train feedback exists. AmazonBook is blank because it is graph-only in this feature policy. |",
        f"| `{paths['metadata'].name}` | Machine-readable counts, checkpoint names, projection settings, and trustworthiness values. | Use this for exact captions and appendix tables. |",
        "",
        "## Dataset Scale",
        "",
        "| Dataset | Reference view | Full positive train graph | Full sign distribution | Topology core sample shown |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for bundle in bundles:
        config = bundle.config_summary
        view = config.get("preprocessing_preset") or "default preprocessing"
        if config.get("item_universe_policy"):
            view = f"{view}; {config['item_universe_policy']}"
        sample_summary = bundle.topology_sample_summary
        if bool(sample_summary.get("has_graded_sign", False)):
            sample_sign_text = (
                "sign > 0 / = 0 / < 0 "
                f"{_format_count(sample_summary.get('n_sign_positive_rows', 0))}/"
                f"{_format_count(sample_summary.get('n_sign_zero_rows', 0))}/"
                f"{_format_count(sample_summary.get('n_sign_negative_rows', 0))}"
            )
        else:
            sample_sign_text = "sign rows omitted (constant sign = 0)"
        lines.append(
            f"| {bundle.spec.label} | `{view}` | "
            f"{_format_count(bundle.graph_summary['n_train_positive_edges_total'])} edges, "
            f"{_format_count(bundle.graph_summary['n_train_users_total'])} users, "
            f"{_format_count(bundle.graph_summary['n_train_items_total'])} items | "
            f"sign > 0 {_format_count(bundle.train_sign_summary.get('sign_positive', 0))}; "
            f"sign = 0 {_format_count(bundle.train_sign_summary.get('sign_zero', 0))}; "
            f"sign < 0 {_format_count(bundle.train_sign_summary.get('sign_negative', 0))} | "
            f"{_format_count(sample_summary.get('n_sampled_users', 0))} users, "
            f"{_format_count(sample_summary.get('n_sampled_items', 0))} items, "
            f"{_format_count(sample_summary.get('n_sampled_rows', 0))} displayed rows; "
            f"label > 0 {_format_count(sample_summary.get('n_label_positive_rows', 0))}; "
            f"{sample_sign_text} |",
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `label` and `sign` are deliberately separate. `label > 0` defines the observed positive train graph used by LightGCN/EDGRec propagation and by BPR positives. `sign` is the graded feedback descriptor retained for diagnostics and signed-feedback analysis.",
            "- If a dataset has no graded sign variation, as in AmazonBook where the canonical sign is stored as zero for every observed interaction, the topology does not draw a separate `sign = 0` overlay. Otherwise the gray overlay would simply cover the same positive graph and make the propagation edges harder to read.",
            "- The GCN topology figure is a NetworkX visualization of a fixed displayed-row budget, not the full graph and not a uniform random sample. Plotting millions of edges would collapse into an unreadable mass, while uniform row sampling mostly shows isolated one-edge nodes in sparse datasets. The sampler therefore prioritizes a degree-aware `label > 0` propagation core and reserves a smaller budget for `sign = 0` and `sign < 0` diagnostics. The exact displayed counts are printed in each panel and recorded in JSON.",
            "- The degree-distribution figure is the main answer to whether the data scale is real: it is computed over the complete positive training graph, so it should be cited when explaining popularity skew, negative sampling, and why train-only popularity is a controlled input. The binary label and graded sign counts are kept in the metadata table instead of being drawn on top of the histogram.",
            "- The learned-interest UMAP figure can support a qualitative discussion of whether trained user and item embeddings occupy coherent local neighborhoods. Gray/red outlines show whether sampled nodes also receive `sign = 0` or `sign < 0` observed feedback. Do not use it to claim that one method ranks better than another.",
            "- The feature UMAP figure helps explain what the safe item features look like after encoding and whether feature-space neighborhoods align with popularity and signed-feedback exposure. It supports the feature-engineering narrative, while usefulness still comes from `results/feature_analysis/` and matched full-data rows.",
            "",
            "## Projection Metadata",
            "",
            "| Dataset | Learned UMAP | Feature UMAP |",
            "| --- | --- | --- |",
        ],
    )
    for bundle in bundles:
        learned = _projection_table_cell(bundle.learned_projection)
        feature = _projection_table_cell(bundle.feature_projection)
        if bundle.feature_projection is None:
            feature = "not available"
        lines.append(f"| {bundle.spec.label} | {learned} | {feature} |")
    lines.extend(
        [
            "",
            "## Thesis Caption Starters",
            "",
            "- GCN topology: \"NetworkX spring-layout visualizations of fixed-budget readable samples from the observed training view. Green edges are `label > 0` user-item rows used for EDGRec/LightGCN propagation. Gray and red edges show displayed `sign = 0` and `sign < 0` observed training feedback where available. `sign > 0` rows are counted in the panel notes and metadata rather than overdrawn because they usually coincide with the green propagation graph. Sign-colored rows are diagnostic feedback rows, not extra message-passing edges. Node size is log-scaled positive train degree; validation and test interactions are excluded.\"",
            "- Degree distribution: \"Full positive-training-degree distributions show dataset sparsity and popularity skew for the actual propagation graph; aligned label/sign row summaries are reported separately so non-positive rows are not visually confused with GCN edges.\"",
            "- Learned UMAP: \"UMAP projections of propagated EDGRec interest embeddings provide qualitative geometry diagnostics; gray/red outlines indicate sampled nodes with `sign = 0` or `sign < 0` feedback, and the figure is read with the metric tables rather than as standalone evidence.\"",
            "- Feature UMAP: \"Encoded item-feature projections show the structure of the safe thesis-policy descriptors, train-degree popularity, and where signed non-positive feedback occurs in feature space.\"",
            "",
        ],
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _projection_table_cell(projection: ProjectionRecord | None) -> str:
    """Return compact README text for a projection."""
    if projection is None:
        return "not available"
    trust = projection.metadata.get("trustworthiness")
    trust_text = f", trust {trust:.3f}" if isinstance(trust, float) else ""
    return f"{projection.n_points} points, {projection.n_features} dims{trust_text}"


def _clean_old_per_dataset_outputs(output_dir: Path) -> None:
    """Remove duplicated per-dataset folders created by earlier diagnostic runs."""
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_dir() and (
            child.name.endswith("_umap_checkpoint") or child.name.endswith("_feature_umap")
        ):
            for nested in child.rglob("*"):
                if nested.is_file():
                    nested.unlink()
            for nested_dir in sorted(
                (path for path in child.rglob("*") if path.is_dir()),
                reverse=True,
            ):
                nested_dir.rmdir()
            child.rmdir()


def build_parser() -> argparse.ArgumentParser:
    """Return CLI parser."""
    parser = argparse.ArgumentParser(
        description="Export consolidated EDGRec graph and UMAP figures for core datasets.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--projection-max-users", type=int, default=250)
    parser.add_argument("--projection-max-items", type=int, default=250)
    parser.add_argument("--projection-top-share", type=float, default=0.5)
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--keep-old-per-dataset",
        action="store_true",
        help="Keep older diagnostic per-dataset folders instead of removing duplicates.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _assert_checkpoints_exist(CORE_DATASETS)
    if not args.keep_old_per_dataset:
        _clean_old_per_dataset_outputs(args.output_dir)

    bundles: list[DatasetBundle] = []
    for spec in CORE_DATASETS:
        print(f"Loading {spec.dataset} from {spec.checkpoint.name}")
        bundles.append(_load_dataset_bundle(spec, args))

    paths = {
        "topology": _plot_topology_figure(
            bundles,
            args.output_dir,
            args.seed,
        ),
        "degree": _plot_degree_figure(bundles, args.output_dir),
        "learned_umap": _plot_learned_umap_figure(
            bundles,
            args.output_dir,
        ),
        "feature_umap": _plot_feature_umap_figure(
            bundles,
            args.output_dir,
        ),
    }
    metadata_path = args.output_dir / "core_graph_embedding_figures.json"
    paths["metadata"] = metadata_path
    metadata_path.write_text(
        json.dumps(_records_for_json(paths, bundles), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["readme"] = _write_readme(paths, bundles, args.output_dir)

    print("Wrote consolidated graph/embedding figures:")
    for name, path in paths.items():
        print(f"- {name}: {_relative(path)}")


if __name__ == "__main__":
    main()
