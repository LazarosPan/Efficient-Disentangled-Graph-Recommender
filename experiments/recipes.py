"""Declarative experiment catalog helpers for frictionless CLI usage."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).with_name("experiment_catalog.json")


@lru_cache(maxsize=1)
def load_experiment_catalog() -> dict[str, Any]:
    """Load the experiment catalog from disk."""
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _raw_recipe(recipe_name: str) -> dict[str, Any]:
    catalog = load_experiment_catalog()
    recipes = catalog.get("recipes", {})
    if recipe_name not in recipes:
        available = ", ".join(sorted(recipes))
        raise KeyError(
            f"Unknown recipe '{recipe_name}'. Available recipes: {available}"
        )
    return recipes[recipe_name]


def recipe_names() -> list[str]:
    """Return all available recipe names."""
    return sorted(load_experiment_catalog().get("recipes", {}))


def _slugify_fragment(raw: object) -> str:
    """Return a filesystem-safe slug fragment for profile identifiers."""
    normalized = "".join(
        character.lower() if str(character).isalnum() else "-" for character in str(raw)
    )
    return "-".join(part for part in normalized.split("-") if part)


def _catalog_formal_profiles() -> list[dict[str, Any]]:
    """Return the raw formal profile entries from the experiment catalog."""
    raw_profiles = load_experiment_catalog().get("formal_profiles", [])
    if not isinstance(raw_profiles, list):
        raise TypeError("formal_profiles must be a list")
    return [dict(profile) for profile in raw_profiles]


def _resolved_profile_matrix(profile: dict[str, Any]) -> dict[str, Any]:
    """Normalize the catalog matrix shape for a formal profile."""
    matrix = dict(profile.get("matrix", {}))
    matrix["presets"] = list(matrix.get("presets", []))
    matrix["graph_methods"] = list(matrix.get("graph_methods", []))
    matrix["scoring_weight_modes"] = list(
        matrix.get("scoring_weight_modes", ["learned"])
    )
    return matrix


def _resolved_profile_overrides(profile: dict[str, Any]) -> dict[str, Any]:
    """Normalize the config override block for a formal profile."""
    overrides = dict(profile.get("config_overrides", {}))
    if "num_neighbors" in overrides and overrides["num_neighbors"] is not None:
        overrides["num_neighbors"] = list(overrides["num_neighbors"])
    return overrides


def _formal_profile_name(profile: dict[str, Any]) -> str:
    """Build a deterministic semantic profile name from the profile payload."""
    matrix = _resolved_profile_matrix(profile)
    overrides = _resolved_profile_overrides(profile)
    scoring_modes = matrix["scoring_weight_modes"]
    if set(scoring_modes) == {"fixed", "learned"} and len(scoring_modes) == 2:
        scoring_mode_slug = "both"
    else:
        scoring_mode_slug = "-".join(_slugify_fragment(mode) for mode in scoring_modes)
    neighbors = overrides.get("num_neighbors") or []
    neighbor_slug = "x".join(str(value) for value in neighbors) if neighbors else "na"
    signature = json.dumps(
        {"matrix": matrix, "config_overrides": overrides},
        sort_keys=True,
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    lr_value = str(overrides.get("lr", "na")).replace(".", "p")
    return (
        f"e{_slugify_fragment(overrides.get('epochs', 'na'))}-"
        f"lr{_slugify_fragment(lr_value)}-"
        f"bs{_slugify_fragment(overrides.get('batch_size', 'na'))}-"
        f"n{neighbor_slug}-sw{scoring_mode_slug}-{digest}"
    )


def formal_profile_names() -> list[str]:
    """Return all available formal profile names."""
    return [_formal_profile_name(profile) for profile in _catalog_formal_profiles()]


def default_formal_profile_name() -> str:
    """Return the default formal profile name."""
    names = formal_profile_names()
    if not names:
        raise ValueError("No formal profiles are defined in the experiment catalog.")
    return names[0]


def get_formal_profile(profile_name: str) -> dict[str, Any]:
    """Return a named formal profile from the experiment catalog."""
    profiles = _catalog_formal_profiles()
    for index, profile in enumerate(profiles):
        resolved_name = _formal_profile_name(profile)
        aliases = {resolved_name}
        if index == 0:
            aliases.update({"default", "latest"})
        if profile_name in aliases:
            return {
                "name": resolved_name,
                "description": profile.get("description", ""),
                "matrix": _resolved_profile_matrix(profile),
                "config_overrides": _resolved_profile_overrides(profile),
            }

    available = ", ".join(formal_profile_names())
    raise KeyError(
        f"Unknown formal profile '{profile_name}'. Available profiles: {available}"
    )


def get_recipe(recipe_name: str) -> dict[str, Any]:
    """Resolve a recipe, following aliases to their canonical target."""
    recipe = dict(_raw_recipe(recipe_name))
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
