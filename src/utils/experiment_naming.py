"""Shared experiment-name construction for runtime and reporting."""

from __future__ import annotations

from collections.abc import Callable, Mapping

NamePartFormatter = Callable[[object], str]


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


def _format_int_part(value: object) -> str:
    """Format a config value as an integer name fragment."""
    return str(int(value))


def _format_float_part(value: object) -> str:
    """Format a float config value as a filesystem-safe name fragment."""
    return f"{float(value):g}".replace(".", "p")


def _optional_name_part(
    value: object,
    prefix: str,
    formatter: NamePartFormatter = str,
) -> str | None:
    """Return a prefixed name fragment only when ``value`` is present."""
    if value is None:
        return None
    return f"{prefix}{formatter(value)}"


def _non_default_name_part(
    value: object,
    default: object,
    prefix: str,
    formatter: NamePartFormatter = str,
) -> str | None:
    """Return a prefixed name fragment only when ``value`` differs from default."""
    if value in (None, default):
        return None
    return f"{prefix}{formatter(value)}"


def _extend_name_parts(parts: list[str], candidates: list[str | None]) -> None:
    """Append all present candidate name fragments to ``parts``."""
    parts.extend(part for part in candidates if part is not None)


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


def format_num_neighbors_vector(num_neighbors: object) -> str:
    """Return one fan-out vector as a compact slug fragment."""
    return "-".join(str(value) for value in num_neighbors)


def format_num_neighbors_payload(num_neighbors: object) -> str | None:
    """Return a compact label for a raw or keyed num_neighbors payload."""
    if num_neighbors is None:
        return None
    if isinstance(num_neighbors, Mapping):
        parts: list[str] = []
        for key in sorted(num_neighbors):
            value = format_num_neighbors_payload(num_neighbors[key])
            parts.append(f"{key}[{value or 'na'}]")
        return "__".join(parts)
    if isinstance(num_neighbors, (list, tuple)):
        if not num_neighbors:
            return None
        if all(isinstance(item, (list, tuple)) for item in num_neighbors):
            return "+".join(format_num_neighbors_vector(item) for item in num_neighbors)
        return format_num_neighbors_vector(num_neighbors)
    return str(num_neighbors)


def _num_neighbors_label(config: object) -> str | None:
    """Return the fan-out label for a config if it has one."""
    return format_num_neighbors_payload(_config_value(config, "num_neighbors"))


def build_canonical_experiment_name(
    config: object,
    preset: str | None,
    intervention: str | None,
) -> str:
    """Build a descriptive canonical experiment name from an effective config.

    The function accepts both live ``EDGRecConfig`` objects and stored
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
    sample_interactions = _config_value(config, "sample_interactions")
    loader_max_rows = _config_value(config, "loader_max_rows")
    preprocessing_preset = _config_value(config, "preprocessing_preset")
    item_universe_policy = str(
        _config_value(config, "item_universe_policy", "observed_interaction_items"),
    )
    graph_policy = str(_config_value(config, "graph_policy", "observed"))
    derived_split_mode = str(
        _config_value(config, "derived_split_mode", "per_user_temporal"),
    )
    feature_policy = _config_value(config, "feature_policy", "thesis_default")
    feature_subset_mode = _config_value(config, "feature_subset_mode", "all")
    feature_include_groups = _config_value(config, "feature_include_groups")
    feature_exclude_groups = _config_value(config, "feature_exclude_groups")
    _extend_name_parts(
        parts,
        [
            _optional_name_part(num_neighbors_label, "nbr"),
            _optional_name_part(sample_interactions, "sample", _format_int_part),
            _optional_name_part(loader_max_rows, "loadrows", _format_int_part),
            _optional_name_part(preprocessing_preset, "ppreset"),
            _non_default_name_part(
                item_universe_policy,
                "observed_interaction_items",
                "iuniv",
            ),
            _non_default_name_part(graph_policy, "observed", "graph"),
            _non_default_name_part(
                derived_split_mode,
                "per_user_temporal",
                "split",
            ),
            "feat" if _config_bool(config, "use_features", False) else None,
            _non_default_name_part(feature_policy, "thesis_default", "fpolicy"),
            _non_default_name_part(feature_subset_mode, "all", "fsubset"),
            _optional_name_part(
                format_num_neighbors_payload(feature_include_groups),
                "finclude",
            ),
            _optional_name_part(
                format_num_neighbors_payload(feature_exclude_groups),
                "fexclude",
            ),
            _non_default_name_part(
                _config_value(config, "embedding_optimizer", "adamw"),
                "adamw",
                "embopt",
            ),
            _non_default_name_part(
                _config_value(config, "train_edge_keep_prob", 1.0),
                1.0,
                "edgekeep",
                _format_float_part,
            ),
            f"lr-{_config_value(config, 'lr_scheduler', 'none')}",
            intervention,
            f"seed{_config_int(config, 'seed')}",
        ],
    )
    return "_".join(parts)


__all__ = [
    "build_canonical_experiment_name",
    "format_num_neighbors_payload",
    "format_num_neighbors_vector",
]
