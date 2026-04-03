"""Loader registry — every loader returns a CanonicalInteractions."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from ..feature_policy import DEFAULT_FEATURE_POLICY, FeaturePolicyName

if TYPE_CHECKING:
    from collections.abc import Callable
    from ..canonical import CanonicalInteractions

from .movielens1m import load_movielens1m
from .movielens20m import load_movielens20m
from .taobao import load_taobao
from .kuairec_v2 import load_kuairec_v2
from .amazonbook import load_amazonbook
from .kuairand1k import load_kuairand1k

LOADERS: dict[str, Callable[..., CanonicalInteractions]] = {
    "movielens1m": load_movielens1m,
    "movielens20m": load_movielens20m,
    "taobao": load_taobao,
    "kuairec_v2": load_kuairec_v2,
    "amazonbook": load_amazonbook,
    "kuairand1k": load_kuairand1k,
}

_PREPROCESSING_PRESETS: dict[str, dict[str, dict[str, object]]] = {
    "movielens1m": {
        "movielens_explicit": {"preprocessing_preset": "movielens_explicit"},
    },
    "movielens20m": {
        "movielens_explicit": {"preprocessing_preset": "movielens_explicit"},
    },
    "taobao": {
        "taobao_multibehavior": {"preprocessing_preset": "taobao_multibehavior"},
    },
    "kuairec_v2": {
        "kuairec_fullobs": {
            "preprocessing_preset": "kuairec_fullobs",
            "matrix_variant": "small_matrix",
        },
    },
    "amazonbook": {
        "amazonbook_graph_only": {"preprocessing_preset": "amazonbook_graph_only"},
    },
    "kuairand1k": {
        "kuairand_causal": {"preprocessing_preset": "kuairand_causal"},
    },
}


def resolve_preprocessing_preset(
    name: str,
    preprocessing_preset: str | None,
) -> dict[str, object]:
    """Resolve preset-specific loader kwargs for a dataset.

    Args:
        name: Loader registry dataset name.
        preprocessing_preset: Optional named preprocessing preset.

    Returns:
        Loader keyword arguments implied by the preset.

    Raises:
        ValueError: If the requested preset is unknown for the dataset.
    """
    if preprocessing_preset is None:
        return {}
    dataset_presets = _PREPROCESSING_PRESETS.get(name, {})
    if preprocessing_preset not in dataset_presets:
        available = ", ".join(sorted(dataset_presets)) or "none"
        raise ValueError(
            f"Unknown preprocessing_preset '{preprocessing_preset}' for dataset "
            f"'{name}'. Available presets: {available}"
        )
    return dict(dataset_presets[preprocessing_preset])


@lru_cache(maxsize=32)
def _load_dataset_cached(
    name: str,
    data_dir: str,
    max_rows: int,
    include_optional_features: bool,
    feature_policy: FeaturePolicyName,
    preprocessing_preset: str | None,
) -> CanonicalInteractions:
    loader_kwargs = resolve_preprocessing_preset(name, preprocessing_preset)
    loader_kwargs.setdefault("preprocessing_preset", preprocessing_preset)
    return LOADERS[name](
        data_dir,
        max_rows=max_rows,
        include_optional_features=include_optional_features,
        feature_policy=feature_policy,
        **loader_kwargs,
    )


def load_dataset(
    name: str,
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
    preprocessing_preset: str | None = None,
) -> CanonicalInteractions:
    """Load a dataset by name."""
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(LOADERS.keys())}")
    loader_kwargs = resolve_preprocessing_preset(name, preprocessing_preset)
    loader_kwargs.setdefault("preprocessing_preset", preprocessing_preset)
    if max_rows is None:
        return LOADERS[name](
            data_dir,
            max_rows=max_rows,
            include_optional_features=include_optional_features,
            feature_policy=feature_policy,
            **loader_kwargs,
        )
    return _load_dataset_cached(
        name,
        data_dir,
        max_rows,
        include_optional_features,
        feature_policy,
        preprocessing_preset,
    )
