"""UCaGNNConfig: single dataclass controlling model, loss, and runtime policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..data.canonical import DerivedSplitMode
from ..data.feature_policy import DEFAULT_FEATURE_POLICY, FeaturePolicyName


DEFAULT_SEED = 13
ScoringMode = Literal[
    "default",
    "interest_only",
    "conformity_only",
    "counterfactual_only",
    "conformity_suppressed",
]


@dataclass
class UCaGNNConfig:
    # ── Architecture toggles ─────────────────────────────────────────────
    use_dual_branch: bool = True
    use_sign_aware: bool = True
    use_counterfactual: bool = True
    use_ipw: bool = True
    use_popularity_head: bool = True
    use_popularity_emb: bool = True

    # ── Graph construction ───────────────────────────────────────────────
    graph_method: Literal["knn", "cagra"] = "cagra"
    knn_k: int = 20
    cagra_out_degree: int = 32
    cagra_initial_degree: int = 64
    cagra_team_size: int = 8

    # ── Embedding / GNN hyperparameters ──────────────────────────────────
    embed_dim: int = 64
    pop_embed_dim: int = 16
    single_branch_gnn_layers: int = 2
    interest_gnn_layers: int = 1
    conformity_gnn_layers: int = 2
    dropout: float = 0.1

    # ── Scoring weights ──────────────────────────────────────────────────
    scoring_weight_mode: Literal["fixed", "learned"] = "fixed"
    alpha_interest: float = 0.5
    beta_conformity: float = 0.3
    gamma_popularity: float = 0.2
    train_scoring_mode: ScoringMode = "default"

    # ── Loss lambdas (0.0 = disabled) ────────────────────────────────────
    lambda_rec: float = 1.0
    lambda_interest_bpr: float = 0.02
    lambda_conformity_bpr: float = 0.02
    lambda_independence: float = 0.005
    lambda_contrastive: float = 0.02
    lambda_align: float = 0.02
    lambda_uniform: float = 0.02
    lambda_pop: float = 0.02
    auxiliary_loss_schedule: Literal["phased", "linear_ramp"] = "phased"
    auxiliary_ramp_rate: float = 0.001
    independence_ramp_rate: float = 0.00025
    contrastive_temperature: float = 0.2
    contrastive_max_pairs: int = 256
    uniformity_temperature: float = 2.0
    use_conformity_au: bool = False

    # ── Propensity (IPW) ─────────────────────────────────────────────────
    propensity_hidden: int = 128
    propensity_clip_min: float = 0.01
    propensity_clip_max: float = 0.99

    # ── Training ─────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 4096
    epochs: int = 60
    patience: int = 10
    use_early_stopping: bool = True
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: Literal["bfloat16"] = "bfloat16"
    use_torch_compile: bool = False
    use_ema: bool = False
    ema_decay: float = 0.999
    show_progress_bar: bool = True
    progress_bar_loss_cadence: int = 16
    lr_scheduler: Literal["none", "plateau"] = "plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    eval_ks: list[int] = field(default_factory=lambda: [20, 40])
    eval_scoring_mode: ScoringMode = "default"
    # ── Training ─────────────────────────────────────────────────────────
    num_neighbors: list[int] = field(default_factory=lambda: [10, 5])
    sample_interactions: int | None = None
    loader_max_rows: int | None = None

    # ── Negative sampling ────────────────────────────────────────────────
    n_negatives: int = 1
    hard_negative_ratio: float = (
        0.0  # fraction of negatives that are popularity-weighted
    )

    # ── Curriculum schedule (epoch thresholds) ───────────────────────────
    curriculum_phase1_end: int = 15  # CaDCR-inspired staged curriculum (0 = disabled)
    curriculum_phase2_end: int = 30
    loss_schedule: Literal["baseline"] = "baseline"

    # ── Side features ─────────────────────────────────────────────────────
    use_features: bool = True  # load and use user/item side features when available
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY

    # ── Data ─────────────────────────────────────────────────────────────
    dataset: str = "movielens1m"
    data_dir: str = "data"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    derived_split_mode: DerivedSplitMode = "per_user_temporal"
    preprocessing_preset: str | None = None
    popularity_window_seconds: int | None = None
    seed: int = DEFAULT_SEED

    # ── Device ───────────────────────────────────────────────────────────
    device: str = "cuda"

    # ── Profiling ────────────────────────────────────────────────────────
    enable_profiling: bool = False
    profiling_cadence: int = 10

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the active config contract after init or preset mutation."""
        if self.single_branch_gnn_layers < 1:
            raise ValueError("single_branch_gnn_layers must be >= 1")
        if self.interest_gnn_layers < 1:
            raise ValueError("interest_gnn_layers must be >= 1")
        if self.conformity_gnn_layers < 1:
            raise ValueError("conformity_gnn_layers must be >= 1")
        if len(self.num_neighbors) != self.max_gnn_layers:
            raise ValueError(
                f"num_neighbors length ({len(self.num_neighbors)}) must equal "
                f"max_gnn_layers ({self.max_gnn_layers})"
            )
        if self.sample_interactions is not None and self.sample_interactions <= 0:
            raise ValueError("sample_interactions must be > 0 when provided")
        if self.loader_max_rows is not None and self.loader_max_rows <= 0:
            raise ValueError("loader_max_rows must be > 0 when provided")
        if self.progress_bar_loss_cadence < 1:
            raise ValueError("progress_bar_loss_cadence must be >= 1")
        if self.profiling_cadence < 1:
            raise ValueError("profiling_cadence must be >= 1")
        if (
            self.popularity_window_seconds is not None
            and self.popularity_window_seconds <= 0
        ):
            raise ValueError("popularity_window_seconds must be > 0 when provided")
        if self.amp_dtype != "bfloat16":
            raise ValueError("amp_dtype is fixed to 'bfloat16'")
        if self.propensity_clip_min <= 0 or self.propensity_clip_min >= 1:
            raise ValueError("propensity_clip_min must be in (0, 1)")
        if self.propensity_clip_max <= 0 or self.propensity_clip_max > 1:
            raise ValueError("propensity_clip_max must be in (0, 1]")
        if self.propensity_clip_min >= self.propensity_clip_max:
            raise ValueError("propensity_clip_min must be < propensity_clip_max")
        if self.auxiliary_ramp_rate < 0:
            raise ValueError("auxiliary_ramp_rate must be >= 0")
        if self.independence_ramp_rate < 0:
            raise ValueError("independence_ramp_rate must be >= 0")
        if self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be > 0")
        if self.contrastive_max_pairs < 2:
            raise ValueError("contrastive_max_pairs must be >= 2")
        if (
            min(
                self.alpha_interest,
                self.beta_conformity,
                self.gamma_popularity,
            )
            < 0
        ):
            raise ValueError("score weights must be non-negative")

    @property
    def max_gnn_layers(self) -> int:
        if not self.use_dual_branch:
            return self.single_branch_gnn_layers
        return max(self.interest_gnn_layers, self.conformity_gnn_layers)

    def preset_lightgcn(self) -> UCaGNNConfig:
        """Non-causal LightGCN baseline."""
        self.use_dual_branch = False
        self.use_sign_aware = False
        self.use_counterfactual = False
        self.use_ipw = False
        self.use_popularity_head = False
        self.use_popularity_emb = False
        self.lambda_interest_bpr = 0.0
        self.lambda_conformity_bpr = 0.0
        self.lambda_independence = 0.0
        self.lambda_contrastive = 0.0
        self.lambda_align = 0.0
        self.lambda_uniform = 0.0
        self.lambda_pop = 0.0
        self.auxiliary_loss_schedule = "phased"
        self.scoring_weight_mode = "fixed"
        self.beta_conformity = 0.0
        self.gamma_popularity = 0.0
        self.train_scoring_mode = "default"
        self.eval_scoring_mode = "default"
        self.graph_method = "cagra"
        self.feature_policy = DEFAULT_FEATURE_POLICY
        return self

    def preset_dice_like(self) -> UCaGNNConfig:
        """DICE-like: dual branch + orthogonality only."""
        self.use_dual_branch = True
        self.use_counterfactual = False
        self.use_ipw = False
        self.use_popularity_head = False
        self.use_popularity_emb = False
        self.lambda_interest_bpr = 0.0
        self.lambda_conformity_bpr = 0.0
        self.lambda_contrastive = 0.0
        self.lambda_align = 0.0
        self.lambda_uniform = 0.0
        self.lambda_pop = 0.0
        self.auxiliary_loss_schedule = "phased"
        self.scoring_weight_mode = "fixed"
        self.gamma_popularity = 0.0
        self.graph_method = "knn"
        self.feature_policy = DEFAULT_FEATURE_POLICY
        # Evaluate on the interest branch only — conformity captures popularity
        # bias that the dual-branch is designed to separate, so using it in the
        # recommendation score would undermine the causal measurement.
        self.train_scoring_mode = "interest_only"
        self.eval_scoring_mode = "interest_only"
        return self

    def preset_full(self) -> UCaGNNConfig:
        """Wave-1 U-CaGNN v2 mainline: fused scoring with asymmetric depth."""
        self.use_dual_branch = True
        self.use_sign_aware = True
        self.use_counterfactual = True
        self.use_ipw = True
        self.use_popularity_head = True
        self.use_popularity_emb = True
        self.scoring_weight_mode = "learned"
        self.train_scoring_mode = "default"
        self.eval_scoring_mode = "default"
        self.loss_schedule = "baseline"
        self.auxiliary_loss_schedule = "linear_ramp"
        self.interest_gnn_layers = 1
        self.conformity_gnn_layers = 2
        self.num_neighbors = [10, 5]
        self.lambda_contrastive = 0.02
        self.lambda_align = 0.0
        self.lambda_uniform = 0.0
        # Tighter IPW clipping (max weight = 10×) prevents gradient explosion
        # from poorly calibrated propensity estimates early in training.
        self.propensity_clip_min = 0.1
        self.feature_policy = DEFAULT_FEATURE_POLICY
        return self
