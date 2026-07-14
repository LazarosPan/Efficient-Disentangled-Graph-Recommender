#!/usr/bin/env python
"""Export qualitative train-graph and embedding projection figures.

These figures are thesis/defense aids. They visualize the split-safe training
graph or learned embedding geometry; they are not performance evidence by
themselves.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from experiments.run_experiment import build_runtime_model, load_runtime_data
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from src.utils.config import CONFIG_PRESET_METHODS, EDGRecConfig
from src.utils.method_naming import public_preset_name
from src.utils.project_paths import RESULTS_DIR
from src.utils.trainer_runtime import _migrate_model_state

ProjectionMethod = Literal["pca", "umap"]

OUTPUT_DIR = RESULTS_DIR / "graph_embedding_figures"
USER_COLOR = "#2166ac"
ITEM_COLOR = "#b2182b"
EDGE_COLOR = "#6b7280"


def _safe_slug(value: str) -> str:
    """Return a filesystem-friendly slug."""
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "p")
    )


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    """Detach a tensor as a CPU NumPy array."""
    return value.detach().cpu().numpy()


def _positive_train_edges(data: Any) -> np.ndarray:
    """Return directed user->item train-positive edges from the PyG graph."""
    edge_index = getattr(data, "edge_index", None)
    if not isinstance(edge_index, torch.Tensor):
        raise ValueError("Runtime data does not expose tensor edge_index.")

    n_users = int(data.n_users)
    edges = edge_index.detach().cpu()
    src = edges[0].numpy()
    dst = edges[1].numpy()
    user_to_item = (src < n_users) & (dst >= n_users)
    selected = np.column_stack((src[user_to_item], dst[user_to_item]))
    if selected.size == 0:
        raise ValueError("The runtime graph has no positive train user->item edges.")
    return selected.astype(np.int64, copy=False)


def _degree_vectors(data: Any, user_item_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return train-degree vectors for users and items."""
    n_users = int(data.n_users)
    n_items = int(data.n_items)
    user_degree = np.bincount(user_item_edges[:, 0], minlength=n_users).astype(np.float64)
    item_degree = np.bincount(user_item_edges[:, 1] - n_users, minlength=n_items).astype(
        np.float64,
    )
    popularity_count = getattr(data, "popularity_count", None)
    if isinstance(popularity_count, torch.Tensor) and popularity_count.numel() == n_items:
        item_degree = _tensor_to_numpy(popularity_count).astype(np.float64, copy=False)
    return user_degree, item_degree


def _train_graph_summary(
    user_item_edges: np.ndarray,
    user_degree: np.ndarray,
    item_degree: np.ndarray,
) -> dict[str, int]:
    """Return full train-positive graph counts behind the plotted sample."""
    return {
        "n_train_positive_edges_total": int(user_item_edges.shape[0]),
        "n_train_users_total": int(np.count_nonzero(user_degree > 0)),
        "n_train_items_total": int(np.count_nonzero(item_degree > 0)),
    }


def _sample_topology_edges(
    user_item_edges: np.ndarray,
    *,
    n_users: int,
    user_degree: np.ndarray,
    item_degree: np.ndarray,
    max_edges: int,
    max_users: int,
    max_items: int,
) -> np.ndarray:
    """Select a readable high-signal subgraph from full train edges."""
    if max_edges <= 0 or max_users <= 0 or max_items <= 0:
        raise ValueError("max_edges, max_users, and max_items must be positive.")

    users = user_item_edges[:, 0]
    items = user_item_edges[:, 1] - n_users
    positive_items = np.flatnonzero(item_degree > 0)
    if positive_items.size == 0:
        positive_items = np.unique(items)
    item_order = positive_items[np.argsort(-item_degree[positive_items], kind="stable")]
    target_items = item_order[:max_items]
    target_item_set = set(int(item) for item in target_items)

    selected_edges: list[tuple[int, int]] = []
    selected_users: set[int] = set()
    selected_items: set[int] = set()

    def try_add(edge_index: int) -> bool:
        user = int(users[edge_index])
        item = int(items[edge_index])
        if item not in target_item_set:
            return False
        if user not in selected_users and len(selected_users) >= max_users:
            return False
        if item not in selected_items and len(selected_items) >= max_items:
            return False
        selected_users.add(user)
        selected_items.add(item)
        selected_edges.append((user, item + n_users))
        return True

    per_item_quota = max(1, math.ceil(max_edges / max(1, len(target_items))))
    for item in target_items:
        item_edge_indices = np.flatnonzero(items == item)
        item_edge_order = item_edge_indices[
            np.lexsort((users[item_edge_indices], -user_degree[users[item_edge_indices]]))
        ]
        new_user_edges = [
            int(edge_index)
            for edge_index in item_edge_order
            if int(users[edge_index]) not in selected_users
        ]
        existing_user_edges = [
            int(edge_index)
            for edge_index in item_edge_order
            if int(users[edge_index]) in selected_users
        ]
        added_for_item = 0
        for edge_index in (*new_user_edges, *existing_user_edges):
            if try_add(edge_index):
                added_for_item += 1
            if added_for_item >= per_item_quota or len(selected_edges) >= max_edges:
                break
        if len(selected_edges) >= max_edges:
            break

    fill_order = np.lexsort(
        (
            items,
            users,
            -user_degree[users],
            -item_degree[items],
        ),
    )
    seen_edges = set(selected_edges)
    for edge_index in fill_order:
        if len(selected_edges) >= max_edges:
            break
        edge = (int(users[edge_index]), int(items[edge_index]) + n_users)
        if edge in seen_edges:
            continue
        before_count = len(selected_edges)
        if try_add(int(edge_index)):
            seen_edges.add(edge)
        if len(selected_edges) == before_count and len(selected_users) >= max_users:
            continue
        if len(selected_edges) >= max_edges:
            break

    if not selected_edges:
        raise ValueError("Topology sampler selected no edges.")
    return np.asarray(selected_edges, dtype=np.int64)


def _node_size(degree: float, max_degree: float) -> float:
    """Scale node size with log train degree."""
    if max_degree <= 0:
        return 70.0
    return 70.0 + 260.0 * (math.log1p(float(degree)) / math.log1p(float(max_degree)))


def _plot_topology(
    selected_edges: np.ndarray,
    *,
    data: Any,
    dataset: str,
    user_degree: np.ndarray,
    item_degree: np.ndarray,
    graph_summary: Mapping[str, int],
    layout: str,
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    """Write a NetworkX topology plot for selected train-positive graph edges."""
    n_users = int(data.n_users)
    graph = nx.Graph()
    graph.add_edges_from((int(user), int(item_node)) for user, item_node in selected_edges)

    selected_users = sorted(node for node in graph.nodes if int(node) < n_users)
    selected_item_nodes = sorted(node for node in graph.nodes if int(node) >= n_users)

    if layout == "bipartite":
        pos = nx.bipartite_layout(graph, selected_users, align="vertical", scale=1.0)
    elif layout == "spring":
        pos = nx.spring_layout(graph, k=0.35, iterations=90, seed=seed)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    max_degree = max(
        float(user_degree.max(initial=0.0)),
        float(item_degree.max(initial=0.0)),
    )
    user_sizes = [_node_size(user_degree[user], max_degree) for user in selected_users]
    item_sizes = [
        _node_size(item_degree[item_node - n_users], max_degree)
        for item_node in selected_item_nodes
    ]

    plt.figure(figsize=(11.0, 8.0))
    nx.draw_networkx_edges(
        graph,
        pos,
        alpha=0.16,
        edge_color=EDGE_COLOR,
        width=0.8,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=selected_users,
        node_color=USER_COLOR,
        node_size=user_sizes,
        linewidths=0.35,
        edgecolors="white",
        label="Users",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=selected_item_nodes,
        node_color=ITEM_COLOR,
        node_size=item_sizes,
        linewidths=0.35,
        edgecolors="white",
        label="Items",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=USER_COLOR,
            markersize=9,
            label="Users",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ITEM_COLOR,
            markersize=9,
            label="Items",
        ),
        Line2D([0], [0], color=EDGE_COLOR, alpha=0.35, label="Positive train interaction"),
    ]
    plt.legend(handles=legend_handles, loc="upper left", frameon=False)
    total_edges = int(graph_summary["n_train_positive_edges_total"])
    plotted_edges = int(selected_edges.shape[0])
    plt.title(
        f"{dataset}: train-positive GCN topology sample "
        f"({plotted_edges:,}/{total_edges:,} edges)",
    )
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "artifact": str(output_path),
        "dataset": dataset,
        "plot_type": "topology_sample",
        "layout": layout,
        "n_edges": plotted_edges,
        "n_users": len(selected_users),
        "n_items": len(selected_item_nodes),
        "n_plotted_edges": plotted_edges,
        "n_plotted_users": len(selected_users),
        "n_plotted_items": len(selected_item_nodes),
        **dict(graph_summary),
        "meaning": (
            "Bipartite graph sampled from the full positive training edge set only; "
            "node size is log-scaled train degree. This visualizes the "
            "LightGCN/EDGRec propagation topology, not validation/test leakage or "
            "ranking performance."
        ),
    }


def _positive_degrees(degree: np.ndarray) -> np.ndarray:
    """Return positive degree values for plotting."""
    positive = degree[degree > 0]
    return positive if positive.size else np.asarray([1.0])


def _plot_degree_distribution(
    *,
    dataset: str,
    user_degree: np.ndarray,
    item_degree: np.ndarray,
    graph_summary: Mapping[str, int],
    output_path: Path,
) -> dict[str, Any]:
    """Write full-train user/item degree distributions."""
    user_positive = _positive_degrees(user_degree)
    item_positive = _positive_degrees(item_degree)
    max_degree = max(float(user_positive.max()), float(item_positive.max()), 1.0)
    bins = np.logspace(0.0, math.log10(max_degree + 1.0), 48)

    plt.figure(figsize=(9.5, 6.5))
    plt.hist(
        user_positive,
        bins=bins,
        color=USER_COLOR,
        alpha=0.58,
        label="Users",
    )
    plt.hist(
        item_positive,
        bins=bins,
        color=ITEM_COLOR,
        alpha=0.58,
        label="Items",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("positive train degree")
    plt.ylabel("number of nodes")
    plt.title(
        f"{dataset}: full train graph degree distribution "
        f"({graph_summary['n_train_positive_edges_total']:,} edges)",
    )
    plt.legend(frameon=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "artifact": str(output_path),
        "dataset": dataset,
        "plot_type": "degree_distribution",
        **dict(graph_summary),
        "user_degree_max": float(user_positive.max()),
        "user_degree_mean": float(user_positive.mean()),
        "item_degree_max": float(item_positive.max()),
        "item_degree_mean": float(item_positive.mean()),
        "meaning": (
            "Full positive-training-edge degree distribution for users and items. "
            "Unlike the topology visualization, this plot is computed over the full "
            "training graph rather than a sampled edge subset."
        ),
    }


def _project_matrix(
    matrix: np.ndarray,
    *,
    method: ProjectionMethod,
    seed: int,
    umap_neighbors: int,
    umap_min_dist: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a matrix to two dimensions with PCA or UMAP."""
    if matrix.shape[0] < 3:
        raise ValueError("Projection needs at least three sampled rows.")
    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        xy = reducer.fit_transform(matrix)
        metadata = {
            "method": "pca",
            "explained_variance_ratio": [
                float(value) for value in reducer.explained_variance_ratio_
            ],
        }
    elif method == "umap":
        try:
            from umap import UMAP
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "UMAP projection requires the optional dependency 'umap-learn'. "
                "Install it with: uv add umap-learn",
            ) from exc
        n_neighbors = max(2, min(int(umap_neighbors), matrix.shape[0] - 1))
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=float(umap_min_dist),
            metric="cosine",
            random_state=seed,
        )
        xy = reducer.fit_transform(matrix)
        metadata = {
            "method": "umap",
            "n_neighbors": int(n_neighbors),
            "min_dist": float(umap_min_dist),
            "metric": "cosine",
        }
    else:
        raise ValueError(f"Unsupported projection method: {method}")

    tw_neighbors = max(1, min(10, matrix.shape[0] // 3))
    if tw_neighbors >= 1 and matrix.shape[0] > tw_neighbors + 1:
        metadata["trustworthiness"] = float(
            trustworthiness(matrix, xy, n_neighbors=tw_neighbors, metric="cosine"),
        )
        metadata["trustworthiness_neighbors"] = int(tw_neighbors)
    return xy, metadata


def _select_ids_by_degree(
    degree: np.ndarray,
    *,
    max_count: int,
    top_share: float,
    seed: int,
) -> np.ndarray:
    """Select top-degree and random nonzero ids for projection plots."""
    positive_ids = np.flatnonzero(degree > 0)
    if positive_ids.size == 0:
        positive_ids = np.arange(degree.shape[0])
    if positive_ids.size <= max_count:
        return positive_ids.astype(np.int64, copy=False)

    rng = np.random.default_rng(seed)
    top_count = max(1, min(max_count, round(max_count * top_share)))
    top_order = positive_ids[np.argsort(-degree[positive_ids], kind="stable")]
    top_ids = top_order[:top_count]
    remaining = np.setdiff1d(positive_ids, top_ids, assume_unique=False)
    random_count = max_count - top_ids.size
    if random_count > 0 and remaining.size > 0:
        random_ids = rng.choice(remaining, size=min(random_count, remaining.size), replace=False)
        selected = np.concatenate((top_ids, random_ids))
    else:
        selected = top_ids
    return np.sort(selected.astype(np.int64, copy=False))


def _embedding_pair_from_propagated(
    propagated: dict[str, torch.Tensor],
    view: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return user/item tensors for a requested embedding view."""
    if view == "interest":
        user_key = "user_interest" if "user_interest" in propagated else "user"
        item_key = "item_interest" if "item_interest" in propagated else "item"
    elif view == "conformity":
        user_key = "user_conformity"
        item_key = "item_conformity"
    elif view == "base":
        user_key = "user"
        item_key = "item"
    else:
        raise ValueError(f"Unsupported embedding view: {view}")

    if user_key not in propagated or item_key not in propagated:
        available = ", ".join(sorted(propagated))
        raise ValueError(
            f"Propagated embeddings do not include {user_key!r}/{item_key!r}; "
            f"available keys: {available}",
        )
    return propagated[user_key], propagated[item_key], f"{user_key}/{item_key}"


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    """Load a checkpoint dictionary on CPU."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint {path} did not contain a dictionary payload.")
    if "config" not in payload:
        raise KeyError(f"Checkpoint {path} is missing its saved config.")
    return payload


def _visualization_config_from_checkpoint(saved_config: EDGRecConfig, device: str) -> EDGRecConfig:
    """Return a current EDGRecConfig from a possibly legacy checkpoint config."""
    config = EDGRecConfig()
    for field_info in fields(EDGRecConfig):
        if hasattr(saved_config, field_info.name):
            setattr(config, field_info.name, getattr(saved_config, field_info.name))
    config.baseline_family = public_preset_name(config.baseline_family) or config.baseline_family
    config.device = device
    config.show_progress_bar = False
    config.use_amp = False
    config.use_torch_compile = False
    if config.use_ipw and config.loss_weight_propensity_calibration <= 0.0:
        config.use_ipw = False
    config.validate()
    return config


def _resolve_config(args: argparse.Namespace) -> tuple[EDGRecConfig, dict[str, Any] | None]:
    """Resolve runtime config from args or a checkpoint."""
    if args.checkpoint is not None:
        payload = _load_checkpoint_payload(Path(args.checkpoint))
        saved_config = payload["config"]
        if not isinstance(saved_config, EDGRecConfig):
            raise TypeError("Checkpoint config is not an EDGRecConfig instance.")
        return _visualization_config_from_checkpoint(saved_config, args.device), payload

    config = EDGRecConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        device=args.device,
        sample_interactions=args.sample_interactions,
        loader_max_rows=args.loader_max_rows,
        show_progress_bar=False,
        use_amp=False,
        use_torch_compile=False,
    )
    preset_method_name = CONFIG_PRESET_METHODS.get(args.preset)
    if preset_method_name is None:
        choices = ", ".join(CONFIG_PRESET_METHODS)
        raise ValueError(f"Unknown preset {args.preset!r}. Expected one of: {choices}")
    getattr(config, preset_method_name)()
    if args.use_features:
        config.use_features = True
    config.validate()
    return config, None


def _load_model_projection(
    *,
    config: EDGRecConfig,
    data: Any,
    canonical: Any,
    checkpoint_payload: dict[str, Any],
    view: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return sampled learned user/item embeddings from a saved checkpoint."""
    device = torch.device(config.device)
    model = build_runtime_model(config, canonical, data).to(device)
    raw_state = checkpoint_payload.get("best_state") or checkpoint_payload.get("model_state")
    if raw_state is None:
        raise KeyError("Checkpoint has neither best_state nor model_state.")
    incompatible = model.load_state_dict(_migrate_model_state(raw_state), strict=False)
    unsupported_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(("scoring.", "propensity."))
    ]
    if unsupported_missing:
        missing = ", ".join(unsupported_missing)
        raise RuntimeError(f"Checkpoint is missing model keys required for projection: {missing}")
    if incompatible.missing_keys:
        missing = ", ".join(incompatible.missing_keys[:8])
        suffix = " ..." if len(incompatible.missing_keys) > 8 else ""
        print(f"Ignoring current scorer keys absent from legacy checkpoint: {missing}{suffix}")
    if incompatible.unexpected_keys:
        unexpected = ", ".join(incompatible.unexpected_keys[:8])
        suffix = " ..." if len(incompatible.unexpected_keys) > 8 else ""
        print(f"Ignoring checkpoint keys not used by reconstructed model: {unexpected}{suffix}")
    model.eval()

    with torch.no_grad():
        propagated = model.get_propagated_for_eval(
            data.edge_index.to(device),
            edge_sign=getattr(data, "edge_sign", None).to(device)
            if isinstance(getattr(data, "edge_sign", None), torch.Tensor)
            else None,
            edge_norm=getattr(data, "edge_norm", None).to(device)
            if isinstance(getattr(data, "edge_norm", None), torch.Tensor)
            else None,
        )
    user_embeddings, item_embeddings, key_label = _embedding_pair_from_propagated(propagated, view)
    return (
        _tensor_to_numpy(user_embeddings).astype(np.float64, copy=False),
        _tensor_to_numpy(item_embeddings).astype(np.float64, copy=False),
        key_label,
    )


def _load_feature_projection(data: Any) -> tuple[np.ndarray, str]:
    """Return item-feature matrix for projection."""
    item_features = getattr(data, "item_features", None)
    if not isinstance(item_features, torch.Tensor) or item_features.numel() == 0:
        raise ValueError(
            "No item_features tensor is available. Use --use-features for raw "
            "feature projections, or pass --checkpoint for learned embeddings.",
        )
    return _tensor_to_numpy(item_features).astype(np.float64, copy=False), "item features"


def _plot_projection(
    matrix: np.ndarray,
    *,
    roles: np.ndarray,
    degrees: np.ndarray,
    dataset: str,
    source_label: str,
    method: ProjectionMethod,
    seed: int,
    umap_neighbors: int,
    umap_min_dist: float,
    output_path: Path,
) -> dict[str, Any]:
    """Write an embedding or feature projection plot."""
    xy, projection_metadata = _project_matrix(
        matrix,
        method=method,
        seed=seed,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
    )
    max_degree = float(degrees.max(initial=0.0))
    sizes = np.array([_node_size(degree, max_degree) * 0.55 for degree in degrees])

    plt.figure(figsize=(9.5, 7.2))
    user_mask = roles == "user"
    item_mask = roles == "item"
    if bool(user_mask.any()) and bool(item_mask.any()):
        plt.scatter(
            xy[user_mask, 0],
            xy[user_mask, 1],
            s=sizes[user_mask],
            c=USER_COLOR,
            alpha=0.62,
            linewidths=0.2,
            edgecolors="white",
            label="Users",
        )
        plt.scatter(
            xy[item_mask, 0],
            xy[item_mask, 1],
            s=sizes[item_mask],
            c=ITEM_COLOR,
            alpha=0.62,
            linewidths=0.2,
            edgecolors="white",
            label="Items",
        )
        plt.legend(frameon=False, loc="best")
    else:
        role_label = "Items" if bool(item_mask.any()) else "Users"
        degree_colors = np.log1p(degrees)
        scatter = plt.scatter(
            xy[:, 0],
            xy[:, 1],
            s=sizes,
            c=degree_colors,
            cmap="viridis",
            alpha=0.7,
            linewidths=0.2,
            edgecolors="white",
            label=role_label,
        )
        colorbar = plt.colorbar(scatter)
        colorbar.set_label("log1p(train degree)")
    plt.title(f"{dataset}: {method.upper()} projection of {source_label}")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "artifact": str(output_path),
        "dataset": dataset,
        "source": source_label,
        "n_points": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        **projection_metadata,
        "meaning": (
            "Qualitative two-dimensional projection. Local neighborhoods can support "
            "visual interpretation, but clusters/distances must be read alongside the "
            "validation and full-data result tables."
        ),
    }


def _projection_payload(
    *,
    args: argparse.Namespace,
    config: EDGRecConfig,
    data: Any,
    canonical: Any,
    checkpoint_payload: dict[str, Any] | None,
    user_degree: np.ndarray,
    item_degree: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Build matrix, role labels, and degree labels for a projection plot."""
    if args.embedding_source == "learned":
        if checkpoint_payload is None:
            raise ValueError("Learned embedding projection requires --checkpoint.")
        user_embeddings, item_embeddings, _key_label = _load_model_projection(
            config=config,
            data=data,
            canonical=canonical,
            checkpoint_payload=checkpoint_payload,
            view=args.embedding_view,
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
        return matrix, roles, degrees, f"learned {args.embedding_view} embeddings"

    if args.embedding_source == "item_features":
        item_features, source_label = _load_feature_projection(data)
        item_ids = _select_ids_by_degree(
            item_degree,
            max_count=args.projection_max_items,
            top_share=args.projection_top_share,
            seed=args.seed + 1,
        )
        matrix = item_features[item_ids]
        roles = np.asarray(["item"] * len(item_ids))
        degrees = item_degree[item_ids]
        return matrix, roles, degrees, f"encoded {source_label}"

    raise ValueError(f"Unsupported embedding source: {args.embedding_source}")


def _write_report(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    """Write a compact Markdown interpretation report beside generated figures."""
    path = output_dir / "README.md"
    lines = [
        "# Graph and Embedding Figures",
        "",
        "These artifacts are qualitative visualization aids for EDGRec. They should be used ",
        "to explain graph topology, embedding neighborhoods, and feature geometry, not to ",
        "claim ranking improvement without the generated validation/full-data tables.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {Path(record['artifact']).name}",
                "",
                f"- Dataset: `{record['dataset']}`",
                f"- Meaning: {record['meaning']}",
            ],
        )
        if "method" in record:
            lines.append(f"- Projection: `{record['method']}`")
        if record.get("plot_type") == "topology_sample":
            lines.append(
                "- Plotted graph sample: "
                f"{record['n_plotted_edges']:,} train edges, "
                f"{record['n_plotted_users']:,} users, "
                f"{record['n_plotted_items']:,} items from "
                f"{record['n_train_positive_edges_total']:,} train-positive edges, "
                f"{record['n_train_users_total']:,} train users, and "
                f"{record['n_train_items_total']:,} train items.",
            )
        if record.get("plot_type") == "degree_distribution":
            lines.append(
                "- Full train graph: "
                f"{record['n_train_positive_edges_total']:,} train-positive edges, "
                f"{record['n_train_users_total']:,} train users, and "
                f"{record['n_train_items_total']:,} train items.",
            )
            lines.append(
                "- Mean positive train degree: "
                f"users {record['user_degree_mean']:.2f}, "
                f"items {record['item_degree_mean']:.2f}.",
            )
        if "trustworthiness" in record:
            lines.append(
                "- Projection trustworthiness: "
                f"{record['trustworthiness']:.3f} "
                f"(k={record['trustworthiness_neighbors']})",
            )
        if "explained_variance_ratio" in record:
            values = ", ".join(f"{value:.3f}" for value in record["explained_variance_ratio"])
            lines.append(f"- PCA explained variance ratio: {values}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Return CLI parser."""
    parser = argparse.ArgumentParser(
        description="Export EDGRec train-graph topology and optional embedding projections.",
    )
    parser.add_argument("--dataset", default="movielens1m")
    parser.add_argument("--preset", default="edgrec", choices=tuple(CONFIG_PRESET_METHODS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-interactions", type=int)
    parser.add_argument("--loader-max-rows", type=int)
    parser.add_argument("--use-features", action="store_true")
    parser.add_argument("--layout", choices=("spring", "bipartite"), default="spring")
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--max-items", type=int, default=300)
    parser.add_argument("--projection", choices=("none", "pca", "umap"), default="none")
    parser.add_argument(
        "--embedding-source",
        choices=("learned", "item_features"),
        default="learned",
    )
    parser.add_argument(
        "--embedding-view",
        choices=("interest", "conformity", "base"),
        default="interest",
    )
    parser.add_argument("--projection-max-users", type=int, default=600)
    parser.add_argument("--projection-max-items", type=int, default=600)
    parser.add_argument("--projection-top-share", type=float, default=0.5)
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config, checkpoint_payload = _resolve_config(args)
    canonical, data = load_runtime_data(config)
    user_item_edges = _positive_train_edges(data)
    user_degree, item_degree = _degree_vectors(data, user_item_edges)
    graph_summary = _train_graph_summary(user_item_edges, user_degree, item_degree)
    selected_edges = _sample_topology_edges(
        user_item_edges,
        n_users=int(data.n_users),
        user_degree=user_degree,
        item_degree=item_degree,
        max_edges=args.max_edges,
        max_users=args.max_users,
        max_items=args.max_items,
    )

    dataset_slug = _safe_slug(config.dataset)
    records: list[dict[str, Any]] = []
    topology_path = args.output_dir / f"{dataset_slug}_gcn_topology_{args.layout}.png"
    records.append(
        _plot_topology(
            selected_edges,
            data=data,
            dataset=config.dataset,
            user_degree=user_degree,
            item_degree=item_degree,
            graph_summary=graph_summary,
            layout=args.layout,
            seed=args.seed,
            output_path=topology_path,
        ),
    )
    degree_path = args.output_dir / f"{dataset_slug}_train_degree_distribution.png"
    records.append(
        _plot_degree_distribution(
            dataset=config.dataset,
            user_degree=user_degree,
            item_degree=item_degree,
            graph_summary=graph_summary,
            output_path=degree_path,
        ),
    )

    if args.projection != "none":
        matrix, roles, degrees, source_label = _projection_payload(
            args=args,
            config=config,
            data=data,
            canonical=canonical,
            checkpoint_payload=checkpoint_payload,
            user_degree=user_degree,
            item_degree=item_degree,
        )
        source_slug = _safe_slug(source_label)
        projection_path = (
            args.output_dir / f"{dataset_slug}_{args.projection}_{source_slug}_projection.png"
        )
        records.append(
            _plot_projection(
                matrix,
                roles=roles,
                degrees=degrees,
                dataset=config.dataset,
                source_label=source_label,
                method=args.projection,
                seed=args.seed,
                umap_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist,
                output_path=projection_path,
            ),
        )

    metadata_path = args.output_dir / f"{dataset_slug}_graph_embedding_figures.json"
    metadata_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    report_path = _write_report(args.output_dir, records)
    print(f"Wrote {len(records)} figure(s)")
    print(f"Metadata: {metadata_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
