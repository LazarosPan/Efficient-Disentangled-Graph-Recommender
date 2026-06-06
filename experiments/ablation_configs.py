"""Thesis-facing ablation variants for U-CaGNN component analysis.

Each variant starts from ``preset_full()`` and removes one reviewer-relevant
component from the mainline configuration. The default surface stays intentionally
small so ablation runs remain interpretable and compute-bounded.
"""

from __future__ import annotations

from src.utils.config import DEFAULT_SEED, UCaGNNConfig

# Variants beyond this set require a thesis rationale; do not extend casually.
ABLATION_VARIANTS: dict[str, dict] = {
    "mainline": {},
    "no_popularity_head": {
        "use_popularity_head": False,
        "score_weight_popularity": 0.0,
        "loss_weight_popularity": 0.0,
    },
    "no_independence": {"loss_weight_independence": 0.0},
    "no_features": {"use_features": False},
}
_ABLATION_RECOMMENDED_DEFAULTS: dict[str, object] = {
    "epochs": 100,
    "use_early_stopping": True,
    "patience": 10,
    "lr_scheduler": "cosine",
    "interest_gnn_layers": 1,
    "conformity_gnn_layers": 2,
    "num_neighbors": [6, 3],
}


def build_ablation_base_kwargs(
    *,
    dataset: str,
    data_dir: str,
    device: str,
    seed: int = DEFAULT_SEED,
    epochs: int | None = None,
    batch_size: int | None = None,
    graph_policy: str | None = None,
    sample_interactions: int | None = None,
    loader_max_rows: int | None = None,
) -> dict[str, object]:
    """Build the shared base kwargs for ablation configs.

    Args:
        dataset: Dataset name for the ablation run.
        data_dir: Root data directory.
        device: Execution device.
        seed: Run seed.
        epochs: Optional epoch override.
        batch_size: Optional batch-size override.
        graph_policy: Optional graph-policy override.
        sample_interactions: Optional tiny-run interaction cap.
        loader_max_rows: Optional loader row cap.

    Returns:
        Keyword arguments ready for ``make_ablation_config()``.

    """
    base_kwargs: dict[str, object] = {
        "dataset": dataset,
        "data_dir": data_dir,
        "seed": seed,
        "device": device,
    }
    optional_fields = {
        "epochs": epochs,
        "batch_size": batch_size,
        "graph_policy": graph_policy,
        "sample_interactions": sample_interactions,
        "loader_max_rows": loader_max_rows,
    }
    for field_name, field_value in optional_fields.items():
        if field_value is not None:
            base_kwargs[field_name] = field_value
    return base_kwargs


def make_ablation_config(variant: str, **base_kwargs) -> UCaGNNConfig:
    """Create a UCaGNNConfig for a specific ablation variant.

    Starts from preset_full(), then applies the variant overrides.

    Args:
        variant: Key from ABLATION_VARIANTS.
        **base_kwargs: Additional overrides (dataset, seed, epochs, etc.).

    """
    if variant not in ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown ablation variant '{variant}'. Available: {list(ABLATION_VARIANTS.keys())}",
        )

    config = UCaGNNConfig(**base_kwargs)
    config.preset_full()

    for key, value in _ABLATION_RECOMMENDED_DEFAULTS.items():
        if key not in base_kwargs:
            setattr(config, key, list(value) if isinstance(value, list) else value)

    for key, value in ABLATION_VARIANTS[variant].items():
        setattr(config, key, value)

    config.validate()
    return config
