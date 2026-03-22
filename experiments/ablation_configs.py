"""Ablation variant definitions for U-CaGNN component analysis.

Each variant starts from ``preset_full()`` and disables one component.
"""
from __future__ import annotations

from src.utils.config import UCaGNNConfig

ABLATION_VARIANTS: dict[str, dict] = {
    "full": {},
    "learned_score_weights": {"scoring_weight_mode": "learned"},
    "no_dual_branch": {"use_dual_branch": False},
    "no_sign_aware": {"use_sign_aware": False},
    "no_counterfactual": {
        "use_counterfactual": False,
        "lambda_cf": 0.0,
        "gamma_counterfactual": 0.0,
    },
    "no_ipw": {"use_ipw": False},
    "no_popularity_emb": {
        "use_popularity_emb": False,
        "lambda_pop": 0.0,
    },
    "no_ortho": {"lambda_ortho": 0.0},
    "no_contrastive": {"lambda_contr": 0.0},
}


def make_ablation_config(variant: str, **base_kwargs) -> UCaGNNConfig:
    """Create a UCaGNNConfig for a specific ablation variant.

    Starts from preset_full(), then applies the variant overrides.

    Args:
        variant: Key from ABLATION_VARIANTS.
        **base_kwargs: Additional overrides (dataset, seed, epochs, etc.).
    """
    if variant not in ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown ablation variant '{variant}'. "
            f"Available: {list(ABLATION_VARIANTS.keys())}"
        )

    config = UCaGNNConfig(**base_kwargs)
    config.preset_full()

    for key, value in ABLATION_VARIANTS[variant].items():
        setattr(config, key, value)

    return config
