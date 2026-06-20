"""Dataset loader registry — maps dataset names to loader callables and presets."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from ..feature_policy import DEFAULT_FEATURE_POLICY, FeaturePolicyName

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..canonical import CanonicalInteractions

from .amazonbook import load_amazonbook
from .kuairand1k import load_kuairand1k
from .kuairec_v2 import load_kuairec_v2
from .movielens1m import load_movielens1m
from .movielens20m import load_movielens20m
from .taobao import load_taobao

LOADERS: dict[str, Callable[..., CanonicalInteractions]] = {
    "movielens1m": load_movielens1m,
    "movielens20m": load_movielens20m,
    "taobao": load_taobao,
    "kuairec_v2": load_kuairec_v2,
    "amazonbook": load_amazonbook,
    "kuairand1k": load_kuairand1k,
}
_DEFAULT_PREPROCESSING_PRESETS: dict[str, str] = {
    "movielens1m": "movielens_explicit",
    "movielens20m": "movielens_explicit",
    "taobao": "taobao_multibehavior",
    "kuairec_v2": "kuairec_watchratio",
    "amazonbook": "amazonbook_graph_only",
    "kuairand1k": "kuairand_causal",
}

_PREPROCESSING_PRESETS: dict[str, dict[str, dict[str, object]]] = {
    "movielens1m": {
        "movielens_explicit": {"preprocessing_preset": "movielens_explicit"},
    },
    "movielens20m": {
        "movielens_explicit": {"preprocessing_preset": "movielens_explicit"},
        "movielens_explicit_dense_genome": {
            "preprocessing_preset": "movielens_explicit_dense_genome",
            "feature_policy": "all_optional",
        },
    },
    "taobao": {
        "taobao_multibehavior": {"preprocessing_preset": "taobao_multibehavior"},
        "taobao_multibehavior_raw": {
            "preprocessing_preset": "taobao_multibehavior_raw",
        },
    },
    "kuairec_v2": {
        "kuairec_watchratio": {
            "preprocessing_preset": "kuairec_watchratio",
        },
        "kuairec_watchratio_raw": {
            "preprocessing_preset": "kuairec_watchratio_raw",
        },
        "kuairec_fullobs": {
            "preprocessing_preset": "kuairec_fullobs",
        },
    },
    "amazonbook": {
        "amazonbook_graph_only": {"preprocessing_preset": "amazonbook_graph_only"},
    },
    "kuairand1k": {
        "kuairand_causal": {"preprocessing_preset": "kuairand_causal"},
        "kuairand_random_only": {"preprocessing_preset": "kuairand_random_only"},
    },
}


def default_preprocessing_preset(name: str) -> str | None:
    """Return the repository default preprocessing preset for one dataset.

    Args:
        name: Loader registry dataset name.

    Returns:
        Default preprocessing preset for the dataset, or ``None`` when the
        dataset intentionally has no separate preset layer.

    """
    return _DEFAULT_PREPROCESSING_PRESETS.get(name)


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
            (
                f"Unknown preprocessing_preset '{preprocessing_preset}' for "
                f"dataset '{name}'. Available presets: {available}"
            ),
        )
    return dict(dataset_presets[preprocessing_preset])


def _load_dataset(
    name: str,
    data_dir: str,
    max_rows: int | None,
    include_optional_features: bool,
    feature_policy: FeaturePolicyName,
    preprocessing_preset: str | None,
    label_mode: str = "current",
    watch_ratio_proxy_threshold: float = 0.5,
) -> CanonicalInteractions:
    """Load a dataset once after resolving preset-specific loader kwargs."""
    effective_preprocessing_preset = preprocessing_preset or _DEFAULT_PREPROCESSING_PRESETS.get(
        name,
    )
    loader_kwargs = resolve_preprocessing_preset(
        name,
        effective_preprocessing_preset,
    )
    loader_kwargs.setdefault("preprocessing_preset", effective_preprocessing_preset)
    if name == "kuairand1k":
        loader_kwargs.setdefault("label_mode", label_mode)
        loader_kwargs.setdefault("watch_ratio_proxy_threshold", watch_ratio_proxy_threshold)
    effective_feature_policy = loader_kwargs.pop("feature_policy", feature_policy)
    return LOADERS[name](
        data_dir,
        max_rows=max_rows,
        include_optional_features=include_optional_features,
        feature_policy=effective_feature_policy,
        **loader_kwargs,
    )


@lru_cache(maxsize=32)
def _load_dataset_cached(
    name: str,
    data_dir: str,
    max_rows: int,
    include_optional_features: bool,
    feature_policy: FeaturePolicyName,
    preprocessing_preset: str | None,
    label_mode: str = "current",
    watch_ratio_proxy_threshold: float = 0.5,
) -> CanonicalInteractions:
    """Cached variant of ``_load_dataset`` keyed on all arguments."""
    return _load_dataset(
        name,
        data_dir,
        max_rows=max_rows,
        include_optional_features=include_optional_features,
        feature_policy=feature_policy,
        preprocessing_preset=preprocessing_preset,
        label_mode=label_mode,
        watch_ratio_proxy_threshold=watch_ratio_proxy_threshold,
    )


def load_dataset(
    name: str,
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
    preprocessing_preset: str | None = None,
    label_mode: str = "current",
    watch_ratio_proxy_threshold: float = 0.5,
) -> CanonicalInteractions:
    """Load a dataset by name.

    When ``max_rows`` is ``None``, the full dataset is loaded without caching.
    When ``max_rows`` is set (e.g. for debug/preflight runs), results are cached
    by argument tuple so repeated calls with the same parameters are free.

    Args:
        name: Dataset identifier; must be a key in ``LOADERS``.
        data_dir: Root directory containing raw dataset folders.
        max_rows: Optional row cap; enables caching when set.
        include_optional_features: Whether to load optional side-features.
        feature_policy: Feature-policy preset name.
        preprocessing_preset: Optional named preprocessing preset. When omitted,
            the repository default for the dataset is used automatically.
        label_mode: KuaiRand-only label ablation mode. Other datasets reject it.
        watch_ratio_proxy_threshold: KuaiRand watch-ratio threshold for
            ``label_mode="watch_ratio_proxy"``.

    Returns:
        Fully preprocessed ``CanonicalInteractions`` for the requested dataset.

    Raises:
        ValueError: If ``name`` is not registered in ``LOADERS``.

    """
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(LOADERS.keys())}")
    if name != "kuairand1k" and (label_mode != "current" or watch_ratio_proxy_threshold != 0.5):
        raise ValueError("label_mode and watch_ratio_proxy_threshold are KuaiRand-only")
    if max_rows is None:
        return _load_dataset(
            name,
            data_dir,
            max_rows=None,
            include_optional_features=include_optional_features,
            feature_policy=feature_policy,
            preprocessing_preset=preprocessing_preset,
            label_mode=label_mode,
            watch_ratio_proxy_threshold=watch_ratio_proxy_threshold,
        )
    return _load_dataset_cached(
        name,
        data_dir,
        max_rows,
        include_optional_features,
        feature_policy,
        preprocessing_preset,
        label_mode,
        watch_ratio_proxy_threshold,
    )
