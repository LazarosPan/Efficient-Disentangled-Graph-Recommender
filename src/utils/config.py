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
    "conformity_suppressed",
]
GraphPolicy = Literal["observed", "cagra_augmented"]
LRSchedulerName = Literal[
    "none",
    "plateau",
    "step",
    "multi_step",
    "exponential",
    "cosine",
    "cosine_restart",
    "polynomial",
    "linear",
]
SUPPORTED_LR_SCHEDULERS: tuple[LRSchedulerName, ...] = (
    "none",
    "plateau",
    "step",
    "multi_step",
    "exponential",
    "cosine",
    "cosine_restart",
    "polynomial",
    "linear",
)
GRAPH_POLICY_CHOICES: tuple[GraphPolicy, ...] = ("observed", "cagra_augmented")
PresetOverrideValue = bool | int | float | str | list[int]
PresetOverrides = dict[str, PresetOverrideValue]

_NON_CAUSAL_PRESET_OVERRIDES: PresetOverrides = {
    "use_sign_aware": False,
    "use_ipw": False,
    "use_popularity_head": False,
    "use_popularity_emb": False,
    "loss_weight_contrastive": 0.0,
    "loss_weight_align": 0.0,
    "loss_weight_uniform": 0.0,
    "loss_weight_popularity": 0.0,
    "auxiliary_loss_schedule": "phased",
    "scoring_weight_mode": "fixed",
    "score_weight_popularity": 0.0,
    "train_scoring_mode": "default",
    "eval_scoring_mode": "default",
    "use_features": False,
    "feature_policy": DEFAULT_FEATURE_POLICY,
    "propensity_clip_min": 0.01,
}
_LIGHTGCN_PRESET_OVERRIDES: PresetOverrides = _NON_CAUSAL_PRESET_OVERRIDES | {
    "use_dual_branch": False,
    "loss_weight_interest_bpr": 0.0,
    "loss_weight_conformity_bpr": 0.0,
    "loss_weight_independence": 0.0,
    "score_weight_conformity": 0.0,
}
_DICE_LIKE_PRESET_OVERRIDES: PresetOverrides = _NON_CAUSAL_PRESET_OVERRIDES | {
    "use_dual_branch": True,
    "loss_weight_interest_bpr": 0.1,
    "loss_weight_conformity_bpr": 0.1,
    "loss_weight_independence": 0.01,
    "score_weight_interest": 1.0,
    "score_weight_conformity": 1.0,
    "auxiliary_losses_start_epoch": 0,
    "popularity_supervision_start_epoch": 0,
}
_FULL_PRESET_OVERRIDES: PresetOverrides = {
    "use_dual_branch": True,
    "use_sign_aware": True,
    "use_ipw": True,
    "use_popularity_head": True,
    "use_popularity_emb": True,
    "loss_weight_interest_bpr": 0.02,
    "loss_weight_conformity_bpr": 0.02,
    "loss_weight_independence": 0.005,
    "loss_weight_contrastive": 0.0,
    "loss_weight_align": 0.0,
    "loss_weight_uniform": 0.0,
    "loss_weight_popularity": 0.02,
    "auxiliary_loss_schedule": "linear_ramp",
    "scoring_weight_mode": "learned",
    "score_weight_interest": 0.5,
    "score_weight_conformity": 0.3,
    "score_weight_popularity": 0.2,
    "train_scoring_mode": "default",
    "eval_scoring_mode": "default",
    "auxiliary_losses_start_epoch": 15,
    "popularity_supervision_start_epoch": 30,
    "loss_schedule": "baseline",
    "interest_gnn_layers": 1,
    "conformity_gnn_layers": 2,
    "num_neighbors": [10, 5],
    "propensity_clip_min": 0.1,
    "use_features": True,
    "feature_policy": DEFAULT_FEATURE_POLICY,
}


@dataclass
class UCaGNNConfig:
    # ── Architecture toggles ─────────────────────────────────────────────
    use_dual_branch: bool = True
    use_sign_aware: bool = True
    use_ipw: bool = True
    use_popularity_head: bool = True
    use_popularity_emb: bool = True

    # ── Graph construction ───────────────────────────────────────────────
    cagra_k: int = 20
    cagra_out_degree: int = 64  # cuVS-recommended default graph degree
    cagra_initial_degree: int = 128  # cuVS-recommended intermediate graph degree
    cagra_team_size: int = 0  # 0 = auto-select (cuVS default)
    cagra_metric: str = "inner_product"  # dot-product space matches the model's scoring function
    cagra_itopk_size: int = 64  # intermediate candidates per step; higher = better recall
    graph_policy: GraphPolicy = "observed"
    cagra_candidate_k: int = 0  # 0 = full-catalog eval; >0 = restrict to top-K ANN candidates

    # ── Embedding / GNN hyperparameters ──────────────────────────────────
    embed_dim: int = 64
    popularity_embedding_dimensions: int = 16
    single_branch_gnn_layers: int = 2
    interest_gnn_layers: int = 1
    conformity_gnn_layers: int = 2
    dropout: float = 0.1

    # ── Scoring weights ──────────────────────────────────────────────────
    scoring_weight_mode: Literal["fixed", "learned"] = "fixed"
    score_weight_interest: float = 0.5
    score_weight_conformity: float = 0.3
    score_weight_popularity: float = 0.2
    train_scoring_mode: ScoringMode = "default"  # score view optimized by the ranking loss

    # ── Loss lambdas (0.0 = disabled) ────────────────────────────────────
    loss_weight_recommendation: float = 1.0
    loss_weight_interest_bpr: float = 0.02
    loss_weight_conformity_bpr: float = 0.02
    loss_weight_independence: float = 0.005
    loss_weight_contrastive: float = 0.02
    loss_weight_align: float = 0.02
    loss_weight_uniform: float = 0.02
    loss_weight_popularity: float = 0.02
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
    loss_weight_propensity_calibration: float = 0.0

    # ── Training ─────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 4096
    auto_batch_size: bool = True
    batch_size_candidates: list[int] = field(
        default_factory=lambda: [16384, 8192, 4096, 2048, 1024, 512, 256],
    )
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
    lr_scheduler: LRSchedulerName = "plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    eval_ks: list[int] = field(default_factory=lambda: [20, 40])
    eval_scoring_mode: ScoringMode = "default"  # score view used at validation/test time
    # ── Training ─────────────────────────────────────────────────────────
    num_neighbors: list[int] = field(default_factory=lambda: [10, 5])
    sample_interactions: int | None = None
    loader_max_rows: int | None = None

    # ── Negative sampling ────────────────────────────────────────────────
    # n_negatives=1 is the standard BPR setting: one unobserved item per positive
    # interaction, sampled uniformly (or popularity-weighted if hard_negative_ratio > 0).
    # These are BPR "unobserved" negatives, not user-expressed dislikes.
    n_negatives: int = 1
    # Fraction of negatives drawn proportional to item popularity (harder, more popular
    # items) rather than uniformly at random. 0.0 = purely uniform; 0.25 = 25% hard.
    hard_negative_ratio: float = 0.0
    # ── Curriculum schedule (epoch thresholds) ───────────────────────────
    # Phase 1 ends when auxiliary losses begin; phase 2 ends when popularity
    # supervision begins. Both thresholds stay as explicit config fields so
    # checkpoints and experiment logs remain stable.
    auxiliary_losses_start_epoch: int = 15
    popularity_supervision_start_epoch: int = 30
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
    seed: int = DEFAULT_SEED

    # ── Device ───────────────────────────────────────────────────────────
    device: str = "cuda"

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
                (
                    f"num_neighbors length ({len(self.num_neighbors)}) must equal "
                    f"max_gnn_layers ({self.max_gnn_layers})"
                ),
            )
        if self.sample_interactions is not None and self.sample_interactions <= 0:
            raise ValueError("sample_interactions must be > 0 when provided")
        if self.loader_max_rows is not None and self.loader_max_rows <= 0:
            raise ValueError("loader_max_rows must be > 0 when provided")
        if self.progress_bar_loss_cadence < 1:
            raise ValueError("progress_bar_loss_cadence must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not self.batch_size_candidates:
            raise ValueError("batch_size_candidates must contain at least one value")
        if any(candidate < 1 for candidate in self.batch_size_candidates):
            raise ValueError("batch_size_candidates must all be >= 1")
        if self.lr_scheduler not in SUPPORTED_LR_SCHEDULERS:
            raise ValueError(
                f"lr_scheduler must be one of {', '.join(SUPPORTED_LR_SCHEDULERS)}",
            )
        if self.graph_policy not in GRAPH_POLICY_CHOICES:
            raise ValueError(
                f"graph_policy must be one of {', '.join(GRAPH_POLICY_CHOICES)}",
            )
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
                self.score_weight_interest,
                self.score_weight_conformity,
                self.score_weight_popularity,
            )
            < 0
        ):
            raise ValueError("score weights must be non-negative")

    @property
    def max_gnn_layers(self) -> int:
        if not self.use_dual_branch:
            return self.single_branch_gnn_layers
        return max(self.interest_gnn_layers, self.conformity_gnn_layers)

    # ── Scoring contract (authoritative table) ────────────────────────────
    # Each preset fixes the train/eval scoring modes so the model optimises and
    # reports results using a consistent score view.  Same-checkpoint
    # evaluation via ``scripts/evaluate_scoring_modes.py`` may override
    # ``eval_scoring_mode`` without retraining.
    #
    # Preset           | train_scoring_mode | eval_scoring_mode
    # -----------------|--------------------|------------------
    # preset_lightgcn  | "default"          | "default"
    # preset_dice_like | "default"          | "default"
    # preset_full      | "default"          | "default"
    # intervention     | "default" (train)  | overridden at eval time
    #
    # "default"           = interest + conformity (if dual-branch) + popularity (if enabled)
    # "interest_only"     = interest branch only (strips popularity bias from ranking score)
    # "conformity_suppressed" = interest + popularity, no conformity (diagnostic only)
    # ─────────────────────────────────────────────────────────────────────

    def _apply_preset_overrides(
        self,
        overrides: PresetOverrides,
    ) -> UCaGNNConfig:
        """Apply a preset's field overrides in place.

        Args:
            overrides: Mapping from config field names to preset-owned values.

        Returns:
            UCaGNNConfig: The mutated config instance.

        """
        for field_name, value in overrides.items():
            setattr(self, field_name, list(value) if isinstance(value, list) else value)
        return self

    def preset_lightgcn(self) -> UCaGNNConfig:
        """Non-causal LightGCN baseline using the default single-branch score."""
        return self._apply_preset_overrides(_LIGHTGCN_PRESET_OVERRIDES)

    def preset_dice_like(self) -> UCaGNNConfig:
        """DICE-like baseline with fixed interest+conformity scoring."""
        return self._apply_preset_overrides(_DICE_LIKE_PRESET_OVERRIDES)

    def preset_full(self) -> UCaGNNConfig:
        """U-CaGNN mainline: fused scoring with asymmetric depth.

        The trained checkpoint optimizes and evaluates the fused ``default``
        score by default. Same-checkpoint intervention scripts may still
        override ``eval_scoring_mode`` later to measure alternate views without
        retraining.

        """
        return self._apply_preset_overrides(_FULL_PRESET_OVERRIDES)
