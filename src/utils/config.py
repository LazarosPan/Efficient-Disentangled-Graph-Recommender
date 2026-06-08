"""UCaGNNConfig: single dataclass controlling model, loss, and runtime policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..data.canonical import DerivedSplitMode
from ..data.feature_policy import DEFAULT_FEATURE_POLICY, FeaturePolicyName

DEFAULT_SEED = 13
GraphPolicy = Literal["observed", "cagra_augmented"]
TrainingGraphMode = Literal["sampled", "full"]
BranchLossMode = Literal["symmetric_bpr", "dice"]
RecommendationLossMode = Literal["final", "dice_sum"]
NegativeSamplingStrategy = Literal["standard", "dice"]
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
CONFIG_PRESET_METHODS: dict[str, str] = {
    "ucagnn": "preset_full",
    "lightgcn": "preset_lightgcn",
    "lightgcn_paper": "preset_lightgcn_paper",
    "dice_paper": "preset_dice_paper",
    "dice_like": "preset_dice_like",
    "dice_like_ablation": "preset_dice_like",
}
CONFIG_PRESET_CHOICES: tuple[str, ...] = tuple(CONFIG_PRESET_METHODS)
CONFIG_OVERRIDE_FIELDS = (
    "epochs",
    "batch_size",
    "auto_batch_size",
    "batch_size_candidates",
    "embed_dim",
    "single_branch_gnn_layers",
    "interest_gnn_layers",
    "conformity_gnn_layers",
    "dropout",
    "lr",
    "lr_scheduler",
    "lr_scheduler_factor",
    "lr_scheduler_patience",
    "use_early_stopping",
    "patience",
    "use_features",
    "feature_policy",
    "graph_policy",
    "training_graph_mode",
    "branch_loss_mode",
    "recommendation_loss_mode",
    "negative_sampling_strategy",
    "preprocessing_preset",
    "derived_split_mode",
    "num_neighbors",
    "hard_negative_ratio",
    "score_mix_min_weight",
    "score_weight_interest",
    "score_weight_conformity",
    "score_weight_popularity",
    "dice_sampler_margin",
    "dice_sampler_pool",
    "dice_branch_margin",
    "dice_loss_decay",
    "dice_margin_decay",
    "dice_adaptive_decay",
    "n_negatives",
    "distance_correlation_max_pairs",
    "contrastive_max_pairs",
    "contrastive_temperature",
    "uniformity_max_pairs",
    "uniformity_temperature",
    "use_conformity_au",
    "loss_weight_recommendation",
    "loss_weight_interest_bpr",
    "loss_weight_conformity_bpr",
    "loss_weight_independence",
    "loss_weight_contrastive",
    "loss_weight_align",
    "loss_weight_uniform",
    "loss_weight_popularity",
    "loss_weight_propensity_calibration",
    "use_ipw",
    "auxiliary_loss_schedule",
    "auxiliary_ramp_rate",
    "independence_ramp_rate",
    "auxiliary_losses_start_epoch",
    "popularity_supervision_start_epoch",
    "loss_schedule",
    "sample_interactions",
    "loader_max_rows",
)
_BENCHMARK_EXCLUDED_FIELDS = ("embed_dim",)
BENCHMARK_CONFIG_FIELDS = (
    *(
        field_name
        for field_name in CONFIG_OVERRIDE_FIELDS
        if field_name not in _BENCHMARK_EXCLUDED_FIELDS
    ),
    "device",
    "data_dir",
)
PAPER_BASELINE_PRESETS = frozenset(("lightgcn_paper", "dice_paper"))
GRAPH_POLICY_CHOICES: tuple[GraphPolicy, ...] = ("observed", "cagra_augmented")
PresetOverrideValue = bool | int | float | str | list[int]
PresetOverrides = dict[str, PresetOverrideValue]

_NON_CAUSAL_PRESET_OVERRIDES: PresetOverrides = {
    "baseline_family": "non_causal",
    "training_graph_mode": "sampled",
    "branch_loss_mode": "symmetric_bpr",
    "recommendation_loss_mode": "final",
    "negative_sampling_strategy": "standard",
    "score_mix_min_weight": 0.0,
    "use_sign_aware": False,
    "use_ipw": False,
    "use_popularity_head": False,
    "use_popularity_emb": False,
    "use_learned_score_mix": False,
    "loss_weight_contrastive": 0.0,
    "loss_weight_align": 0.0,
    "loss_weight_uniform": 0.0,
    "loss_weight_popularity": 0.0,
    "auxiliary_loss_schedule": "phased",
    "score_weight_popularity": 0.0,
    "use_features": False,
    "feature_policy": DEFAULT_FEATURE_POLICY,
    "propensity_clip_min": 0.01,
}
_LIGHTGCN_PRESET_OVERRIDES: PresetOverrides = _NON_CAUSAL_PRESET_OVERRIDES | {
    "baseline_family": "lightgcn_sampled",
    "use_dual_branch": False,
    "loss_weight_interest_bpr": 0.0,
    "loss_weight_conformity_bpr": 0.0,
    "loss_weight_independence": 0.0,
    "score_weight_conformity": 0.0,
}
_LIGHTGCN_PAPER_PRESET_OVERRIDES: PresetOverrides = _LIGHTGCN_PRESET_OVERRIDES | {
    "baseline_family": "lightgcn_paper",
    "training_graph_mode": "full",
    "graph_policy": "observed",
    "single_branch_gnn_layers": 3,
    "num_neighbors": [10, 5, 5],
    "dropout": 0.0,
    "lr": 0.001,
    "lr_scheduler": "none",
    "weight_decay": 1e-4,
    "batch_size": 2048,
    "auto_batch_size": False,
}
_DICE_LIKE_PRESET_OVERRIDES: PresetOverrides = _NON_CAUSAL_PRESET_OVERRIDES | {
    "baseline_family": "dice_like_ablation",
    "use_dual_branch": True,
    "loss_weight_interest_bpr": 0.1,
    "loss_weight_conformity_bpr": 0.1,
    "loss_weight_independence": 0.01,
    "score_weight_interest": 1.0,
    "score_weight_conformity": 1.0,
    "auxiliary_losses_start_epoch": 0,
    "popularity_supervision_start_epoch": 0,
}
_DICE_PAPER_PRESET_OVERRIDES: PresetOverrides = _NON_CAUSAL_PRESET_OVERRIDES | {
    "baseline_family": "dice_paper",
    "training_graph_mode": "full",
    "graph_policy": "observed",
    "use_dual_branch": True,
    "use_learned_score_mix": False,
    "branch_loss_mode": "dice",
    "recommendation_loss_mode": "dice_sum",
    "negative_sampling_strategy": "dice",
    "n_negatives": 4,
    "single_branch_gnn_layers": 2,
    "interest_gnn_layers": 2,
    "conformity_gnn_layers": 2,
    "num_neighbors": [10, 5],
    "dropout": 0.2,
    "lr": 0.001,
    "lr_scheduler": "none",
    "weight_decay": 5e-8,
    "batch_size": 128,
    "auto_batch_size": False,
    "loss_weight_interest_bpr": 0.1,
    "loss_weight_conformity_bpr": 0.1,
    "loss_weight_independence": 0.01,
    "score_weight_interest": 1.0,
    "score_weight_conformity": 1.0,
    "score_weight_popularity": 0.0,
    "auxiliary_losses_start_epoch": 0,
    "popularity_supervision_start_epoch": 0,
    "dice_sampler_margin": 40.0,
    "dice_sampler_pool": 40,
    "dice_branch_margin": 40.0,
    "dice_loss_decay": 0.9,
    "dice_margin_decay": 0.9,
    "dice_adaptive_decay": True,
}
_LIGHTGCN_PAPER_LOCKED_OVERRIDES: PresetOverrides = {
    key: _LIGHTGCN_PAPER_PRESET_OVERRIDES[key]
    for key in (
        "baseline_family",
        "training_graph_mode",
        "graph_policy",
        "use_dual_branch",
        "use_sign_aware",
        "use_ipw",
        "use_popularity_head",
        "use_popularity_emb",
        "use_learned_score_mix",
        "use_features",
        "feature_policy",
        "branch_loss_mode",
        "recommendation_loss_mode",
        "negative_sampling_strategy",
        "single_branch_gnn_layers",
        "num_neighbors",
        "dropout",
        "lr",
        "lr_scheduler",
        "weight_decay",
        "batch_size",
        "auto_batch_size",
        "loss_weight_interest_bpr",
        "loss_weight_conformity_bpr",
        "loss_weight_independence",
    )
}
_DICE_PAPER_LOCKED_OVERRIDES: PresetOverrides = {
    key: _DICE_PAPER_PRESET_OVERRIDES[key]
    for key in (
        "baseline_family",
        "training_graph_mode",
        "graph_policy",
        "use_dual_branch",
        "use_sign_aware",
        "use_ipw",
        "use_popularity_head",
        "use_popularity_emb",
        "use_learned_score_mix",
        "use_features",
        "feature_policy",
        "branch_loss_mode",
        "recommendation_loss_mode",
        "negative_sampling_strategy",
        "n_negatives",
        "single_branch_gnn_layers",
        "interest_gnn_layers",
        "conformity_gnn_layers",
        "num_neighbors",
        "dropout",
        "lr",
        "lr_scheduler",
        "weight_decay",
        "batch_size",
        "auto_batch_size",
        "dice_sampler_margin",
        "dice_sampler_pool",
        "dice_branch_margin",
        "dice_loss_decay",
        "dice_margin_decay",
        "dice_adaptive_decay",
        "loss_weight_interest_bpr",
        "loss_weight_conformity_bpr",
        "loss_weight_independence",
        "score_weight_interest",
        "score_weight_conformity",
        "score_weight_popularity",
    )
}
_FULL_PRESET_OVERRIDES: PresetOverrides = {
    "baseline_family": "ucagnn",
    "training_graph_mode": "sampled",
    "branch_loss_mode": "dice",
    "recommendation_loss_mode": "final",
    "negative_sampling_strategy": "dice",
    "n_negatives": 1,
    "use_dual_branch": True,
    "use_sign_aware": True,
    "use_ipw": False,
    "use_popularity_head": True,
    "use_popularity_emb": True,
    "use_learned_score_mix": True,
    "loss_weight_interest_bpr": 0.02,
    "loss_weight_conformity_bpr": 0.02,
    "loss_weight_independence": 0.005,
    "loss_weight_contrastive": 0.0,
    "loss_weight_align": 0.0,
    "loss_weight_uniform": 0.0,
    "loss_weight_popularity": 0.02,
    "auxiliary_loss_schedule": "linear_ramp",
    "score_weight_interest": 0.5,
    "score_weight_conformity": 0.3,
    "score_weight_popularity": 0.2,
    "auxiliary_losses_start_epoch": 15,
    "popularity_supervision_start_epoch": 30,
    "loss_schedule": "baseline",
    "interest_gnn_layers": 1,
    "conformity_gnn_layers": 2,
    "num_neighbors": [10, 5],
    "dice_sampler_margin": 40.0,
    "dice_sampler_pool": 40,
    "dice_branch_margin": 40.0,
    "propensity_clip_min": 0.1,
    "use_features": True,
    "feature_policy": DEFAULT_FEATURE_POLICY,
    "score_mix_min_weight": 0.05,
}


@dataclass
class UCaGNNConfig:
    # ── Architecture toggles ─────────────────────────────────────────────
    use_dual_branch: bool = True
    use_sign_aware: bool = True
    use_ipw: bool = False
    use_popularity_head: bool = True
    use_popularity_emb: bool = True
    use_learned_score_mix: bool = True
    baseline_family: str = "ucagnn"
    training_graph_mode: TrainingGraphMode = "sampled"

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

    # ── Scoring priors ───────────────────────────────────────────────────
    score_weight_interest: float = 0.5
    score_weight_conformity: float = 0.3
    score_weight_popularity: float = 0.2
    score_mix_min_weight: float = 0.0

    # ── Loss lambdas (0.0 = disabled) ────────────────────────────────────
    loss_weight_recommendation: float = 1.0
    loss_weight_interest_bpr: float = 0.02
    loss_weight_conformity_bpr: float = 0.02
    loss_weight_independence: float = 0.005
    loss_weight_contrastive: float = 0.02
    loss_weight_align: float = 0.02
    loss_weight_uniform: float = 0.02
    loss_weight_popularity: float = 0.02
    branch_loss_mode: BranchLossMode = "symmetric_bpr"
    recommendation_loss_mode: RecommendationLossMode = "final"
    auxiliary_loss_schedule: Literal["phased", "linear_ramp"] = "phased"
    auxiliary_ramp_rate: float = 0.001
    independence_ramp_rate: float = 0.00025
    contrastive_temperature: float = 0.2
    contrastive_max_pairs: int = 256
    distance_correlation_max_pairs: int = 1024
    uniformity_max_pairs: int = 2048
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
    negative_sampling_strategy: NegativeSamplingStrategy = "standard"
    dice_sampler_margin: float = 40.0
    dice_sampler_pool: int = 40
    dice_branch_margin: float = 0.0
    dice_loss_decay: float = 0.9
    dice_margin_decay: float = 0.9
    dice_adaptive_decay: bool = False
    # ── Curriculum schedule (epoch thresholds) ───────────────────────────
    # These thresholds control when auxiliary losses and popularity
    # supervision activate. They stay explicit so checkpoints and experiment
    # logs keep one stable runtime contract.
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
        if self.training_graph_mode not in ("sampled", "full"):
            raise ValueError("training_graph_mode must be either 'sampled' or 'full'")
        if self.branch_loss_mode not in ("symmetric_bpr", "dice"):
            raise ValueError("branch_loss_mode must be either 'symmetric_bpr' or 'dice'")
        if self.recommendation_loss_mode not in ("final", "dice_sum"):
            raise ValueError("recommendation_loss_mode must be either 'final' or 'dice_sum'")
        if self.negative_sampling_strategy not in ("standard", "dice"):
            raise ValueError("negative_sampling_strategy must be either 'standard' or 'dice'")
        if self.amp_dtype != "bfloat16":
            raise ValueError("amp_dtype is fixed to 'bfloat16'")
        if self.propensity_clip_min <= 0 or self.propensity_clip_min >= 1:
            raise ValueError("propensity_clip_min must be in (0, 1)")
        if self.propensity_clip_max <= 0 or self.propensity_clip_max > 1:
            raise ValueError("propensity_clip_max must be in (0, 1]")
        if self.propensity_clip_min >= self.propensity_clip_max:
            raise ValueError("propensity_clip_min must be < propensity_clip_max")
        if self.use_ipw and self.loss_weight_propensity_calibration <= 0:
            raise ValueError(
                "use_ipw requires loss_weight_propensity_calibration > 0 "
                "so inverse propensity weights are calibrated instead of random.",
            )
        if self.auxiliary_ramp_rate < 0:
            raise ValueError("auxiliary_ramp_rate must be >= 0")
        if self.independence_ramp_rate < 0:
            raise ValueError("independence_ramp_rate must be >= 0")
        if self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be > 0")
        if self.contrastive_max_pairs < 2:
            raise ValueError("contrastive_max_pairs must be >= 2")
        if self.distance_correlation_max_pairs < 2:
            raise ValueError("distance_correlation_max_pairs must be >= 2")
        if self.uniformity_max_pairs < 2:
            raise ValueError("uniformity_max_pairs must be >= 2")
        if (
            min(
                self.score_weight_interest,
                self.score_weight_conformity,
                self.score_weight_popularity,
            )
            < 0
        ):
            raise ValueError("score weights must be non-negative")
        if self.score_mix_min_weight < 0:
            raise ValueError("score_mix_min_weight must be non-negative")
        if self.dice_sampler_margin < 0:
            raise ValueError("dice_sampler_margin must be non-negative")
        if self.dice_sampler_pool < 1:
            raise ValueError("dice_sampler_pool must be >= 1")
        if self.dice_branch_margin < 0:
            raise ValueError("dice_branch_margin must be non-negative")
        if not 0 < self.dice_loss_decay <= 1:
            raise ValueError("dice_loss_decay must be in (0, 1]")
        if not 0 < self.dice_margin_decay <= 1:
            raise ValueError("dice_margin_decay must be in (0, 1]")

    @property
    def max_gnn_layers(self) -> int:
        if not self.use_dual_branch:
            return self.single_branch_gnn_layers
        return max(self.interest_gnn_layers, self.conformity_gnn_layers)

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
        self.validate()
        return self

    def enforce_paper_baseline_contract(self) -> UCaGNNConfig:
        """Re-apply architecture-owned fields for paper baselines.

        Shared benchmark profiles may pass fields such as ``dropout`` and
        ``num_neighbors`` for U-CaGNN. Paper baselines keep their paper-owned
        architecture, scheduler, optimizer, and sampler contract instead of
        silently accepting those shared tuning knobs.
        """
        if self.baseline_family == "lightgcn_paper":
            return self._apply_preset_overrides(_LIGHTGCN_PAPER_LOCKED_OVERRIDES)
        if self.baseline_family == "dice_paper":
            return self._apply_preset_overrides(_DICE_PAPER_LOCKED_OVERRIDES)
        self.validate()
        return self

    def preset_lightgcn(self) -> UCaGNNConfig:
        """Non-causal LightGCN baseline using the single refined scorer."""
        return self._apply_preset_overrides(_LIGHTGCN_PRESET_OVERRIDES)

    def preset_lightgcn_paper(self) -> UCaGNNConfig:
        """Paper-faithful LightGCN baseline with full-graph propagation."""
        return self._apply_preset_overrides(_LIGHTGCN_PAPER_PRESET_OVERRIDES)

    def preset_dice_like(self) -> UCaGNNConfig:
        """DICE-like baseline with the refined scorer and no thesis extras."""
        return self._apply_preset_overrides(_DICE_LIKE_PRESET_OVERRIDES)

    def preset_dice_paper(self) -> UCaGNNConfig:
        """Paper-faithful GCN-DICE baseline using DICE sampling and loss."""
        return self._apply_preset_overrides(_DICE_PAPER_PRESET_OVERRIDES)

    def preset_full(self) -> UCaGNNConfig:
        """U-CaGNN mainline: refined scoring with asymmetric depth."""
        return self._apply_preset_overrides(_FULL_PRESET_OVERRIDES)
