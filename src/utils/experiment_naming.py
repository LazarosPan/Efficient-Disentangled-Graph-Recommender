"""Shared experiment-name construction for runtime and reporting."""

from __future__ import annotations

from collections.abc import Mapping


def _config_value(config: object, field_name: str, default: object = None) -> object:
    """Return one field from a dataclass-like object or mapping."""
    if isinstance(config, Mapping):
        return config.get(field_name, default)
    return getattr(config, field_name, default)


def _config_bool(config: object, field_name: str, default: bool) -> bool:
    """Return one boolean config field."""
    return bool(_config_value(config, field_name, default))


def _config_int(config: object, field_name: str, default: int = 0) -> int:
    """Return one integer config field."""
    value = _config_value(config, field_name, default)
    return int(value) if value is not None else default


def _max_gnn_layers(config: object) -> int:
    """Return the active GNN depth for object and stored-dict configs."""
    explicit = _config_value(config, "max_gnn_layers")
    if explicit is not None:
        return int(explicit)
    if not _config_bool(config, "use_dual_branch", True):
        return _config_int(config, "single_branch_gnn_layers")
    return max(
        _config_int(config, "interest_gnn_layers"),
        _config_int(config, "conformity_gnn_layers"),
    )


def _format_num_neighbors_vector(num_neighbors: object) -> str:
    """Return one fan-out vector as a compact slug fragment."""
    return "-".join(str(value) for value in num_neighbors)


def _format_num_neighbors_payload(num_neighbors: object) -> str | None:
    """Return a compact label for a raw or keyed num_neighbors payload."""
    if num_neighbors is None:
        return None
    if isinstance(num_neighbors, Mapping):
        parts: list[str] = []
        for key in sorted(num_neighbors):
            value = _format_num_neighbors_payload(num_neighbors[key])
            parts.append(f"{key}[{value or 'na'}]")
        return "__".join(parts)
    if isinstance(num_neighbors, (list, tuple)):
        if not num_neighbors:
            return None
        if all(isinstance(item, (list, tuple)) for item in num_neighbors):
            return "+".join(_format_num_neighbors_vector(item) for item in num_neighbors)
        return _format_num_neighbors_vector(num_neighbors)
    return str(num_neighbors)


def _num_neighbors_label(config: object) -> str | None:
    """Return the fan-out label for a config if it has one."""
    return _format_num_neighbors_payload(_config_value(config, "num_neighbors"))


def build_canonical_experiment_name(
    config: object,
    preset: str | None,
    intervention: str | None,
) -> str:
    """Build a descriptive canonical experiment name from an effective config.

    The function accepts both live ``UCaGNNConfig`` objects and stored
    ``config_json`` dictionaries so checkpoint names and result-table labels
    cannot drift apart.
    """
    use_dual_branch = _config_bool(config, "use_dual_branch", True)
    interest_layers = _config_int(config, "interest_gnn_layers")
    conformity_layers = _config_int(config, "conformity_gnn_layers")

    parts = [
        str(_config_value(config, "dataset", "-")),
        preset or "custom",
        f"ep{_config_int(config, 'epochs')}",
        f"bs{_config_int(config, 'batch_size')}",
        f"dim{_config_int(config, 'embed_dim')}",
        f"layers{_max_gnn_layers(config)}",
    ]

    if use_dual_branch and interest_layers != conformity_layers:
        parts.append(f"branchL{interest_layers}-{conformity_layers}")

    num_neighbors_label = _num_neighbors_label(config)
    if num_neighbors_label is not None:
        parts.append(f"nbr{num_neighbors_label}")

    sample_interactions = _config_value(config, "sample_interactions")
    if sample_interactions is not None:
        parts.append(f"sample{int(sample_interactions)}")

    loader_max_rows = _config_value(config, "loader_max_rows")
    if loader_max_rows is not None:
        parts.append(f"loadrows{int(loader_max_rows)}")

    preprocessing_preset = _config_value(config, "preprocessing_preset")
    if preprocessing_preset is not None:
        parts.append(f"ppreset{preprocessing_preset}")

    graph_policy = str(_config_value(config, "graph_policy", "observed"))
    if graph_policy != "observed":
        parts.append(f"graph{graph_policy}")

    derived_split_mode = str(
        _config_value(config, "derived_split_mode", "per_user_temporal"),
    )
    if derived_split_mode != "per_user_temporal":
        parts.append(f"split{derived_split_mode}")

    if _config_bool(config, "use_features", False):
        parts.append("feat")

    feature_policy = _config_value(config, "feature_policy", "thesis_default")
    if feature_policy not in (None, "thesis_default"):
        parts.append(f"fpolicy{feature_policy}")

    parts.append(f"lr-{_config_value(config, 'lr_scheduler', 'none')}")

    if intervention:
        parts.append(intervention)

    parts.append(f"seed{_config_int(config, 'seed')}")
    return "_".join(parts)


__all__ = ["build_canonical_experiment_name"]
