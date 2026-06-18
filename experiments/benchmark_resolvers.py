"""Shared benchmark-override resolvers for formal runs and search."""

from __future__ import annotations

from collections.abc import Mapping

from src.utils.benchmark_datasets import benchmark_dataset_lookup_keys
from src.utils.config import GRAPH_POLICY_CHOICES, SUPPORTED_LR_SCHEDULERS, EDGRecConfig

from experiments.recipes import resolve_profile_num_neighbors


def normalize_benchmark_lr_scheduler_override(
    raw_value: object,
) -> list[str] | str:
    """Normalize benchmark ``lr_scheduler`` overrides to one string or a sweep list."""
    if raw_value is None:
        return "plateau"
    if isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.split(",") if part.strip()]
        return values[0] if len(values) == 1 else values
    if isinstance(raw_value, (list, tuple)):
        return [str(value) for value in raw_value]
    return "plateau"


def normalize_benchmark_graph_policy_override(
    raw_value: object,
) -> tuple[str | None, list[str] | None]:
    """Normalize benchmark ``graph_policy`` overrides to one value or a sweep list."""
    if raw_value is None:
        return None, None
    if isinstance(raw_value, str):
        if raw_value not in GRAPH_POLICY_CHOICES:
            raise ValueError(
                f"graph_policy must be one of {GRAPH_POLICY_CHOICES}, got {raw_value!r}",
            )
        return raw_value, None
    if isinstance(raw_value, (list, tuple)):
        if not raw_value:
            raise ValueError("graph_policy sweep must be a non-empty list.")
        graph_policy_options: list[str] = []
        for index, value in enumerate(raw_value):
            graph_policy = str(value)
            if graph_policy not in GRAPH_POLICY_CHOICES:
                raise ValueError(
                    f"graph_policy[{index}] must be one of {GRAPH_POLICY_CHOICES}, "
                    f"got {graph_policy!r}",
                )
            if graph_policy not in graph_policy_options:
                graph_policy_options.append(graph_policy)
        return graph_policy_options[0], graph_policy_options
    raise ValueError(
        "graph_policy must be a string or a list of graph-policy strings.",
    )


def normalize_benchmark_preprocessing_override(
    raw_value: object,
) -> tuple[str | None, list[str] | None]:
    """Normalize benchmark preprocessing overrides to one value or a sweep list."""
    if raw_value is None:
        return None, None
    if isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.split(",") if part.strip()]
    elif isinstance(raw_value, (list, tuple)):
        values = [str(value).strip() for value in raw_value if str(value).strip()]
    else:
        raise ValueError(
            "preprocessing_preset must be a string or a list of preprocessing preset names.",
        )
    if not values:
        raise ValueError("preprocessing_preset sweep must be a non-empty string or list.")
    deduped = list(dict.fromkeys(values))
    return deduped[0], deduped if len(deduped) > 1 else None


def normalize_benchmark_num_neighbors_override(
    raw_value: object,
) -> list[list[int]] | dict[str, list[list[int]]] | None:
    """Normalize benchmark ``num_neighbors`` overrides for one profile payload."""
    if raw_value is None:
        return None
    if isinstance(raw_value, Mapping):
        normalized: dict[str, list[list[int]]] = {}
        for key, value in raw_value.items():
            if isinstance(value, Mapping):
                raise ValueError(
                    f"num_neighbors[{key}] must be a vector or a list of vectors, "
                    "not a nested mapping.",
                )
            resolved = resolve_profile_num_neighbors({"num_neighbors": value})
            if resolved is None:
                raise ValueError(
                    f"num_neighbors[{key}] must be a non-empty fan-out vector "
                    "or a non-empty list of vectors.",
                )
            normalized[str(key)] = resolved
        return normalized

    resolved = resolve_profile_num_neighbors({"num_neighbors": raw_value})
    if resolved is None:
        return None
    return resolved[0] if len(resolved) == 1 else resolved


def resolve_benchmark_lr_scheduler_values(
    benchmark_args: Mapping[str, object],
    *,
    expand_all: bool = True,
) -> list[str]:
    """Return the resolved scheduler sweep for benchmark-style args."""
    raw_values = benchmark_args.get("lr_scheduler")
    if raw_values is None:
        raw_values = EDGRecConfig().lr_scheduler

    if isinstance(raw_values, str):
        scheduler_values = [raw_values]
    elif isinstance(raw_values, (list, tuple)):
        scheduler_values = [str(value) for value in raw_values]
    else:
        raise ValueError("lr_scheduler must be a string or a list of strings.")

    if scheduler_values == ["all"]:
        if not expand_all:
            raise ValueError("lr_scheduler='all' cannot be used in this context.")
        return list(SUPPORTED_LR_SCHEDULERS)
    return scheduler_values


def resolve_benchmark_graph_policy_values(
    benchmark_args: Mapping[str, object],
) -> list[str]:
    """Return the resolved graph-policy values for one benchmark-like plan."""
    graph_policy_options = benchmark_args.get("graph_policy_options")
    if isinstance(graph_policy_options, (list, tuple)) and graph_policy_options:
        return [str(graph_policy) for graph_policy in graph_policy_options]
    graph_policy = benchmark_args.get("graph_policy")
    if graph_policy is not None:
        return [str(graph_policy)]
    return [EDGRecConfig().graph_policy]


def resolve_benchmark_num_neighbor_values(
    benchmark_args: Mapping[str, object],
    *,
    dataset: str | None = None,
) -> list[list[int]]:
    """Return the resolved neighbor-vector sweep for benchmark-style args."""
    raw_num_neighbors = benchmark_args.get("num_neighbors")
    if isinstance(raw_num_neighbors, Mapping):
        if dataset is None:
            raise ValueError(
                "Dataset-specific num_neighbors mappings require a dataset name.",
            )
        for lookup_key in benchmark_dataset_lookup_keys(dataset):
            selected_neighbors = raw_num_neighbors.get(lookup_key)
            if selected_neighbors is None:
                continue
            neighbor_sweep = resolve_profile_num_neighbors(
                {"num_neighbors": selected_neighbors},
            )
            if neighbor_sweep is not None:
                return [list(num_neighbors) for num_neighbors in neighbor_sweep]
        available = ", ".join(sorted(str(key) for key in raw_num_neighbors))
        raise ValueError(
            f"No num_neighbors entry matches dataset '{dataset}'. Available keys: {available}",
        )

    neighbor_sweep = resolve_profile_num_neighbors({"num_neighbors": raw_num_neighbors})
    if neighbor_sweep is not None:
        return [list(num_neighbors) for num_neighbors in neighbor_sweep]
    return [list(EDGRecConfig().num_neighbors)]


def resolve_benchmark_preprocessing_preset_values(
    benchmark_args: Mapping[str, object],
) -> list[str | None]:
    """Return the resolved preprocessing-preset values for one benchmark-like plan."""
    raw_values = benchmark_args.get("preprocessing_preset_options") or [
        benchmark_args.get("preprocessing_preset"),
    ]
    if not isinstance(raw_values, (list, tuple)):
        raw_values = [raw_values]
    return [None if value is None else str(value) for value in raw_values]


__all__ = [
    "normalize_benchmark_graph_policy_override",
    "normalize_benchmark_lr_scheduler_override",
    "normalize_benchmark_num_neighbors_override",
    "normalize_benchmark_preprocessing_override",
    "resolve_benchmark_graph_policy_values",
    "resolve_benchmark_lr_scheduler_values",
    "resolve_benchmark_num_neighbor_values",
    "resolve_benchmark_preprocessing_preset_values",
]
