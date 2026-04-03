"""Ablation variant definitions for wave-1 U-CaGNN v2 component analysis.

Each variant starts from ``preset_full()`` and toggles one scorer, head, or
regularizer decision. Support-parameter variants keep the same two-hop shape
used by the fused-score preset.
"""

from __future__ import annotations

from src.utils.config import UCaGNNConfig

ABLATION_VARIANTS: dict[str, dict] = {
    # ── Component ablations (disable one component) ──────────────────────
    "mainline": {},
    "fixed_fusion": {"scoring_weight_mode": "fixed"},
    "interest_only_scoring": {
        "train_scoring_mode": "interest_only",
        "eval_scoring_mode": "interest_only",
    },
    "conformity_suppressed_scoring": {
        "train_scoring_mode": "conformity_suppressed",
        "eval_scoring_mode": "conformity_suppressed",
    },
    "no_dual_branch": {"use_dual_branch": False},
    "symmetric_two_hop_depth": {
        "interest_gnn_layers": 2,
        "conformity_gnn_layers": 2,
    },
    "no_sign_aware": {"use_sign_aware": False},
    "no_ipw": {"use_ipw": False},
    "no_popularity_head": {
        "use_popularity_head": False,
        "gamma_popularity": 0.0,
        "lambda_pop": 0.0,
    },
    "no_popularity_emb": {
        "use_popularity_emb": False,
    },
    "no_independence": {"lambda_independence": 0.0},
    "no_directau": {"lambda_align": 0.0, "lambda_uniform": 0.0},
    # ── Batch × num_neighbors pairings (GPU utilization exploration) ─────
    "bs2048_nn20": {"batch_size": 2048, "num_neighbors": [20, 10]},
    "bs4096_nn10": {"batch_size": 4096, "num_neighbors": [10, 5]},
    "bs8192_nn5": {"batch_size": 8192, "num_neighbors": [5, 5]},
    "bs16384_nn5": {"batch_size": 16384, "num_neighbors": [5, 5]},
    # ── Embedding dimension ──────────────────────────────────────────────
    "embed_dim_32": {"embed_dim": 32},
    # ── Curriculum schedule ──────────────────────────────────────────────
    "no_curriculum": {"curriculum_phase1_end": 0, "curriculum_phase2_end": 0},
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
