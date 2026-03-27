"""Declarative experiment catalog helpers for frictionless CLI usage."""

from __future__ import annotations

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


def formal_profile_names() -> list[str]:
    """Return all available formal profile names."""
    return sorted(load_experiment_catalog().get("formal_profiles", {}))


def get_formal_profile(profile_name: str) -> dict[str, Any]:
    """Return a named formal profile from the experiment catalog."""
    catalog = load_experiment_catalog()
    profiles = catalog.get("formal_profiles", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(
            f"Unknown formal profile '{profile_name}'. Available profiles: {available}"
        )

    profile = dict(profiles[profile_name])
    return {
        "name": profile_name,
        "description": profile.get("description", ""),
        "matrix": dict(profile.get("matrix", {})),
        "config_overrides": dict(profile.get("config_overrides", {})),
    }


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
