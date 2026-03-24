"""UCaGNNConfig: single dataclass controlling all architecture toggles, hyperparameters, and loss weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..data.feature_policy import DEFAULT_FEATURE_POLICY, FeaturePolicyName


@dataclass
class UCaGNNConfig:
    # ── Architecture toggles ─────────────────────────────────────────────
    use_dual_branch: bool = True
    use_sign_aware: bool = True
    use_counterfactual: bool = True
    use_ipw: bool = True
    use_popularity_emb: bool = True

    # ── Graph construction ───────────────────────────────────────────────
    graph_method: Literal["dense", "knn", "cagra"] = "knn"
    knn_k: int = 20
    cagra_out_degree: int = 32
    cagra_initial_degree: int = 64
    cagra_team_size: int = 8

    # ── Embedding / GNN hyperparameters ──────────────────────────────────
    embed_dim: int = 64
    pop_embed_dim: int = 16
    n_gnn_layers: int = 2
    interest_gnn_layers: int | None = None
    conformity_gnn_layers: int | None = None
    dropout: float = 0.0

    # ── Scoring weights ──────────────────────────────────────────────────
    scoring_weight_mode: Literal["fixed", "learned"] = "fixed"
    alpha_interest: float = 0.5
    beta_conformity: float = 0.3
    gamma_counterfactual: float = 0.2

    # ── Loss lambdas (0.0 = disabled) ────────────────────────────────────
    lambda_rec: float = 1.0
    lambda_ortho: float = 0.02
    lambda_contr: float = 0.1
    lambda_cf: float = 0.08
    lambda_pop: float = 0.15

    # ── Contrastive loss ─────────────────────────────────────────────────
    contrastive_tau: float = 0.1

    # ── Propensity (IPW) ─────────────────────────────────────────────────
    propensity_hidden: int = 128
    propensity_clip_min: float = 0.01
    propensity_clip_max: float = 0.99

    # ── Training ─────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 60
    patience: int = 10
    grad_clip_norm: float = 1.0
    eval_ks: list[int] = field(default_factory=lambda: [10, 20, 50])
    eval_scoring_mode: Literal[
        "default",
        "interest_only",
        "conformity_only",
        "counterfactual_only",
        "conformity_suppressed",
    ] = "default"
    # ── Training mode ─────────────────────────────────────────────────
    training_mode: Literal["full_graph", "cached_propagation", "mini_batch"] = (
        "full_graph"
    )
    num_neighbors: list[int] = field(default_factory=lambda: [10, 10])
    mini_batch_num_workers: int = 0
    sample_interactions: int | None = None
    loader_max_rows: int | None = None

    # ── Negative sampling ────────────────────────────────────────────────
    n_negatives: int = 1
    hard_negative_ratio: float = (
        0.0  # fraction of negatives that are popularity-weighted
    )

    # ── Curriculum schedule (epoch thresholds) ───────────────────────────
    curriculum_phase1_end: int = 0  # 0 = no curriculum (all losses from epoch 0)
    curriculum_phase2_end: int = 0

    # ── Side features ─────────────────────────────────────────────────────
    use_features: bool = True  # load and use user/item side features when available
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY

    # ── Data ─────────────────────────────────────────────────────────────
    dataset: str = "movielens1m"
    data_dir: str = "data"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 13

    # ── Device ───────────────────────────────────────────────────────────
    device: str = "cuda"

    # ── Profiling ────────────────────────────────────────────────────────
    enable_profiling: bool = True
    profiling_cadence: int = 10

    def __post_init__(self) -> None:
        if self.n_gnn_layers < 1:
            raise ValueError("n_gnn_layers must be >= 1")
        if self.interest_gnn_layers is not None and self.interest_gnn_layers < 1:
            raise ValueError("interest_gnn_layers must be >= 1 when provided")
        if self.conformity_gnn_layers is not None and self.conformity_gnn_layers < 1:
            raise ValueError("conformity_gnn_layers must be >= 1 when provided")
        if self.training_mode == "mini_batch":
            if len(self.num_neighbors) != self.max_gnn_layers:
                raise ValueError(
                    f"num_neighbors length ({len(self.num_neighbors)}) must equal "
                    f"max_gnn_layers ({self.max_gnn_layers})"
                )
        if self.sample_interactions is not None and self.sample_interactions <= 0:
            raise ValueError("sample_interactions must be > 0 when provided")
        if self.loader_max_rows is not None and self.loader_max_rows <= 0:
            raise ValueError("loader_max_rows must be > 0 when provided")
        if self.profiling_cadence < 1:
            raise ValueError("profiling_cadence must be >= 1")

    @property
    def use_cagra(self) -> bool:
        return self.graph_method == "cagra"

    @property
    def resolved_interest_gnn_layers(self) -> int:
        return self.interest_gnn_layers or self.n_gnn_layers

    @property
    def resolved_conformity_gnn_layers(self) -> int:
        return self.conformity_gnn_layers or self.n_gnn_layers

    @property
    def max_gnn_layers(self) -> int:
        if not self.use_dual_branch:
            return self.n_gnn_layers
        return max(
            self.resolved_interest_gnn_layers, self.resolved_conformity_gnn_layers
        )

    def preset_lightgcn(self) -> UCaGNNConfig:
        """Non-causal LightGCN baseline."""
        self.use_dual_branch = False
        self.use_sign_aware = False
        self.use_counterfactual = False
        self.use_ipw = False
        self.use_popularity_emb = False
        self.lambda_ortho = 0.0
        self.lambda_contr = 0.0
        self.lambda_cf = 0.0
        self.lambda_pop = 0.0
        self.scoring_weight_mode = "fixed"
        self.beta_conformity = 0.0
        self.gamma_counterfactual = 0.0
        self.graph_method = "dense"
        self.interest_gnn_layers = None
        self.conformity_gnn_layers = None
        self.feature_policy = DEFAULT_FEATURE_POLICY
        return self

    def preset_dice_like(self) -> UCaGNNConfig:
        """DICE-like: dual branch + orthogonality only."""
        self.use_dual_branch = True
        self.use_counterfactual = False
        self.use_ipw = False
        self.lambda_contr = 0.0
        self.lambda_cf = 0.0
        self.scoring_weight_mode = "fixed"
        self.gamma_counterfactual = 0.0
        self.graph_method = "knn"
        self.feature_policy = DEFAULT_FEATURE_POLICY
        return self

    def preset_full(self) -> UCaGNNConfig:
        """Full U-CaGNN with all losses enabled."""
        self.use_dual_branch = True
        self.use_sign_aware = True
        self.use_counterfactual = True
        self.use_ipw = True
        self.use_popularity_emb = True
        return self
