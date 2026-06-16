"""Declarative experiment catalog helpers for frictionless CLI usage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.experiment_naming import format_num_neighbors_payload

CATALOG_PATH = Path(__file__).with_name("experiment_catalog.json")
SEARCH_SPACES_PATH = Path(__file__).with_name("search_spaces.json")


@lru_cache(maxsize=1)
def load_experiment_catalog() -> dict[str, Any]:
    """Load the experiment catalog from disk."""
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_search_spaces_catalog() -> dict[str, Any]:
    """Load the Optuna search-space catalog from disk."""
    with SEARCH_SPACES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _catalog_entry(
    catalog: Mapping[str, Any],
    *,
    section: str,
    entry_name: str,
    entry_label: str,
) -> dict[str, Any]:
    """Return one named entry from a catalog section with a consistent error."""
    entries = catalog.get(section, {})
    if entry_name not in entries:
        available = ", ".join(sorted(entries))
        raise KeyError(
            f"Unknown {entry_label} '{entry_name}'. Available {entry_label}s: {available}",
        )
    return entries[entry_name]


def recipe_names(include_aliases: bool = True) -> list[str]:
    """Return all available recipe names.

    Args:
        include_aliases: When ``False``, alias entries (``alias_for`` set) are
            excluded so only canonical recipes are returned.

    """
    raw = load_experiment_catalog().get("recipes", {})
    if include_aliases:
        return sorted(raw)
    return sorted(name for name, recipe in raw.items() if "alias_for" not in recipe)


def slugify_fragment(raw: object, *, fallback: str = "") -> str:
    """Return a filesystem-safe slug fragment for generated identifiers."""
    normalized = "".join(
        character.lower() if str(character).isalnum() else "-" for character in str(raw)
    )
    return "-".join(part for part in normalized.split("-") if part) or fallback


def _normalize_num_neighbors_vector(
    raw_value: object,
    *,
    field_name: str,
) -> list[int]:
    """Return one validated ``num_neighbors`` vector from a JSON-like value."""
    if not isinstance(raw_value, (list, tuple)) or not raw_value:
        raise ValueError(f"{field_name} must be a non-empty list of per-hop fan-out values.")
    return [int(value) for value in raw_value]


def _normalize_num_neighbors_options(
    raw_value: object,
    *,
    field_name: str,
) -> list[list[int]]:
    """Return one deduplicated fan-out sweep from a vector or a vector list."""
    if (
        isinstance(raw_value, (list, tuple))
        and raw_value
        and all(not isinstance(value, (list, tuple)) for value in raw_value)
    ):
        resolved = _normalize_num_neighbors_vector(raw_value, field_name=field_name)
        return [list(resolved)]

    if not isinstance(raw_value, (list, tuple)) or not raw_value:
        raise ValueError(
            f"{field_name} must be a non-empty fan-out vector or a non-empty list of vectors.",
        )

    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for index, raw_option in enumerate(raw_value):
        option = _normalize_num_neighbors_vector(
            raw_option,
            field_name=f"{field_name}[{index}]",
        )
        option_key = tuple(option)
        if option_key in seen:
            continue
        seen.add(option_key)
        deduped.append(option)

    return deduped


def resolve_profile_num_neighbors(
    overrides: Mapping[str, Any],
) -> list[list[int]] | dict[str, list[list[int]]] | None:
    """Resolve one or many neighbor vectors from one profile payload.

    Args:
        overrides: Profile override mapping that may contain ``num_neighbors``
            as either one vector, a list of vectors, or a mapping of dataset
            or dataset-tier labels to those shapes.

    Returns:
        Ordered list of per-run fan-out vectors to expand in benchmark
        planning, or a dataset-keyed mapping of such sweeps when the profile
        wants different neighbor sweeps per dataset or dataset class.

    """
    if "num_neighbors_options" in overrides:
        raise ValueError("Use num_neighbors only; num_neighbors_options was removed.")
    raw_neighbors = overrides.get("num_neighbors")
    if raw_neighbors is None:
        return None
    if isinstance(raw_neighbors, Mapping):
        if not raw_neighbors:
            raise ValueError(
                "num_neighbors must be a non-empty mapping when provided as an object.",
            )
        resolved: dict[str, list[list[int]]] = {}
        for key, raw_option in raw_neighbors.items():
            resolved[str(key)] = _normalize_num_neighbors_options(
                raw_option,
                field_name=f"num_neighbors[{key}]",
            )
        return resolved
    if (
        isinstance(raw_neighbors, (list, tuple))
        and raw_neighbors
        and all(not isinstance(value, (list, tuple)) for value in raw_neighbors)
    ):
        resolved = _normalize_num_neighbors_vector(raw_neighbors, field_name="num_neighbors")
        return [list(resolved)]

    return _normalize_num_neighbors_options(raw_neighbors, field_name="num_neighbors")


def _resolved_profile_matrix(profile: dict[str, Any]) -> dict[str, Any]:
    """Normalize the catalog matrix shape for a formal profile."""
    matrix = dict(profile.get("matrix", {}))
    raw_datasets = matrix.get("datasets", "all")
    if isinstance(raw_datasets, str):
        matrix["datasets"] = [raw_datasets]
    else:
        matrix["datasets"] = list(raw_datasets)
    matrix["presets"] = list(matrix.get("presets", []))
    return matrix


def _resolved_profile_overrides(profile: dict[str, Any]) -> dict[str, Any]:
    """Normalize the config override block for a formal profile."""
    overrides = dict(profile.get("config_overrides", {}))
    neighbor_sweep = resolve_profile_num_neighbors(overrides)
    if neighbor_sweep is not None:
        if isinstance(neighbor_sweep, dict):
            overrides["num_neighbors"] = neighbor_sweep
        else:
            overrides["num_neighbors"] = (
                neighbor_sweep[0] if len(neighbor_sweep) == 1 else neighbor_sweep
            )
    return overrides


def _resolved_runtime_probe(profile: dict[str, Any]) -> dict[str, int] | None:
    """Return optional runtime-probe metadata for a formal profile."""
    raw_probe = profile.get("runtime_probe")
    if raw_probe is None:
        return None
    if not isinstance(raw_probe, Mapping):
        raise ValueError("runtime_probe must be an object when provided.")
    target_epochs = int(raw_probe.get("target_epochs", 0))
    if target_epochs < 1:
        raise ValueError("runtime_probe.target_epochs must be >= 1.")
    return {"target_epochs": target_epochs}


def _formal_profile_name(profile: dict[str, Any]) -> str:
    """Build a deterministic semantic profile name from the profile payload."""
    matrix = _resolved_profile_matrix(profile)
    overrides = _resolved_profile_overrides(profile)
    neighbor_options = resolve_profile_num_neighbors(overrides)
    neighbor_slug = format_num_neighbors_payload(neighbor_options) or "na"
    batch_slug = (
        "abauto"
        if overrides.get("auto_batch_size", True)  # True matches UCaGNNConfig default
        else f"bs{slugify_fragment(overrides.get('batch_size', 'na'))}"
    )
    signature = json.dumps(
        {"matrix": matrix, "config_overrides": overrides},
        sort_keys=True,
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    lr_value = str(overrides.get("lr", "na")).replace(".", "p")
    return (
        f"e{slugify_fragment(overrides.get('epochs', 'na'))}-lr"
        f"{slugify_fragment(lr_value)}-{batch_slug}-n{neighbor_slug}-"
        f"{digest}"
    )


def _formal_profile_id(profile: dict[str, Any]) -> str:
    """Return the explicit user-facing identifier for a formal profile."""
    explicit_id = profile.get("id")
    if explicit_id is None:
        return _formal_profile_name(profile)
    return slugify_fragment(explicit_id)


@lru_cache(maxsize=1)
def _resolved_formal_profiles() -> list[dict[str, Any]]:
    """Return formal profiles with resolved names, payloads, and aliases."""
    resolved_profiles: list[dict[str, Any]] = []
    for _index, profile in enumerate(
        load_experiment_catalog().get("formal_profiles", []),
    ):
        resolved_id = _formal_profile_id(profile)
        resolved_name = _formal_profile_name(profile)
        aliases = {resolved_id, resolved_name}
        raw_aliases = profile.get("aliases", [])
        aliases.update(slugify_fragment(alias) for alias in raw_aliases)
        resolved_profiles.append(
            {
                "id": resolved_id,
                "name": resolved_name,
                "description": profile.get("description", ""),
                "matrix": _resolved_profile_matrix(profile),
                "config_overrides": _resolved_profile_overrides(profile),
                "runtime_probe": _resolved_runtime_probe(profile),
                "aliases": aliases,
            },
        )
    return resolved_profiles


@lru_cache(maxsize=1)
def _formal_profile_alias_index() -> dict[str, dict[str, Any]]:
    """Return resolved formal profiles keyed by their supported aliases."""
    profiles_by_alias: dict[str, dict[str, Any]] = {}
    for profile in _resolved_formal_profiles():
        for alias in profile["aliases"]:
            profiles_by_alias[alias] = profile
    return profiles_by_alias


def formal_profile_names() -> list[str]:
    """Return all available formal profile identifiers."""
    return [profile["id"] for profile in _resolved_formal_profiles()]


def default_formal_profile_name() -> str:
    """Return the default formal profile identifier."""
    profiles = _resolved_formal_profiles()
    if not profiles:
        raise ValueError("No formal profiles are defined in the experiment catalog.")
    for profile in profiles:
        if "default" in profile["aliases"]:
            return str(profile["id"])
    return str(profiles[0]["id"])


def search_space_names() -> list[str]:
    """Return available Optuna search-space identifiers."""
    return sorted(load_search_spaces_catalog().get("search_spaces", {}))


def get_search_space(space_name: str) -> dict[str, Any]:
    """Return a named Optuna search-space definition from the search catalog."""
    space = dict(
        _catalog_entry(
            load_search_spaces_catalog(),
            section="search_spaces",
            entry_name=space_name,
            entry_label="search space",
        ),
    )
    return {
        "name": space_name,
        "description": space.get("description", ""),
        "base_profile": space.get("base_profile"),
        "datasets": space.get("datasets"),
        "objective": space.get("objective"),
        "sampler": space.get("sampler"),
        "pruner": space.get("pruner"),
        "max_epochs": space.get("max_epochs"),
        "trials": space.get("trials"),
        "config_overrides": dict(space.get("config_overrides", {})),
        "parameters": dict(space.get("parameters", {})),
        "profile_overrides": dict(space.get("profile_overrides", {})),
        "parameters_by_dataset": dict(space.get("parameters_by_dataset", {})),
    }


def get_formal_profile(profile_name: str) -> dict[str, Any]:
    """Return a named formal profile from the experiment catalog."""
    normalized_name = slugify_fragment(profile_name.strip())
    resolved_profile = _formal_profile_alias_index().get(normalized_name)
    if resolved_profile is not None:
        return {
            "id": resolved_profile["id"],
            "name": resolved_profile["name"],
            "description": resolved_profile["description"],
            "matrix": resolved_profile["matrix"],
            "config_overrides": resolved_profile["config_overrides"],
            "runtime_probe": resolved_profile["runtime_probe"],
        }

    available = ", ".join(formal_profile_names())
    raise KeyError(
        f"Unknown formal profile '{profile_name}'. Available profiles: {available}",
    )


def get_recipe(recipe_name: str) -> dict[str, Any]:
    """Resolve a recipe, following aliases to their canonical target."""
    recipe = dict(
        _catalog_entry(
            load_experiment_catalog(),
            section="recipes",
            entry_name=recipe_name,
            entry_label="recipe",
        ),
    )
    alias_target = recipe.get("alias_for")
    if alias_target is None:
        return {
            "name": recipe_name,
            "preset": recipe.get("preset"),
            "description": recipe.get("description", ""),
            "overrides": dict(recipe.get("overrides", {})),
        }

    resolved = get_recipe(alias_target)
    return {
        "name": recipe_name,
        "preset": recipe.get("preset", resolved.get("preset")),
        "description": recipe.get("description", resolved.get("description", "")),
        "overrides": dict(resolved.get("overrides", {})),
        "alias_for": alias_target,
    }


def recipe_summary_lines() -> list[str]:
    """Return formatted summary lines for ``--list-recipes`` output."""
    lines: list[str] = []
    for name in recipe_names():
        recipe = get_recipe(name)
        overrides = recipe.get("overrides", {})
        parts = [f"preset={recipe.get('preset')}"]
        parts.extend(f"{key}={value}" for key, value in overrides.items())
        if "alias_for" in recipe:
            parts.append(f"alias_for={recipe['alias_for']}")
        description = recipe.get("description")
        if description:
            parts.append(f"desc={description}")
        lines.append(f"  {name:<44} " + ", ".join(parts))
    return lines
