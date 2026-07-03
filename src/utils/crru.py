"""Composite Resource-aware Recommendation Utility (CRRU) helpers.

CRRU is a deterministic, bounded, post-hoc multi-objective utility for comparing
one completed recommender configuration under ranking accuracy, popularity-aware
personalization, and training-resource constraints. It is not a standard
recommender-system metric, causal-effect estimator, fairness metric, debiasing
proof, universal cross-dataset quality score, or report-dependent value.

Thesis CRRU uses a transparent weighted geometric utility:

    RankingAccuracyAtCutoff =
        NDCG@K^{0.50} * Recall@K^{0.35} * HitRatio@K^{0.15}
    PopularityAwarePersonalizationAtCutoff =
        Personalization@K^{0.40}
        * InverseRecommendationPopularity@K^{0.60}
    TrainingResourceUtility =
        PeakGpuMemoryCapacityScore^{0.50}
        * EpochDurationEfficiencyScore^{0.50}
    CRRU@K =
        RankingAccuracyAtCutoff^{0.55}
        * PopularityAwarePersonalizationAtCutoff^{0.30}
        * TrainingResourceUtility^{0.15}

PyG ``LinkPredAveragePopularity`` is logged with raw train-only item interaction
counts, preserving PyG Average Recommendation Popularity semantics. CRRU then
normalizes the logged raw ARP internally:
``CRRUNormalizedAveragePopularity@K =
log(1 + AveragePopularity@K) / log(1 + LargestTrainingItemInteractionCount)``.
``InverseRecommendationPopularity@K = 1 - CRRUNormalizedAveragePopularity@K``.

Resource scores are ``1 / (1 + log(1 + cost))`` for strictly positive finite
peak GPU memory megabytes and epoch duration seconds. ``CRRU_EPSILON`` is only a
numerical lower bound that prevents exact-zero collapse under fractional powers;
invalid or missing inputs raise ``ValueError``. CRRU never normalizes against
other rows, trials, datasets, or reports, so adding another experiment cannot
change an existing experiment's CRRU.

The canonical validation objective name is ``ValidationCRRU@20_40``.

``CRRU_EPSILON`` is not a normalization method. It only keeps multiplicative
fractional-power terms away from exact zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

CRRU_EPSILON = 1e-8
CRRU_FORMULATION_IDENTIFIER = "absolute_weighted_geometric_crru_with_log_normalized_raw_arp"
CRRU_OBJECTIVE_VERSION = CRRU_FORMULATION_IDENTIFIER
VALIDATION_CRRU_METRIC = "ValidationCRRU@20_40"
# Legacy storage aliases are accepted only for reading historical Optuna trials.
# Thesis-facing reports should continue to print the canonical ValidationCRRU names.
LEGACY_VALIDATION_CRRU_METRIC = "ValidationOnlineCRRU@20_40"
VALIDATION_CRRU_METRIC_ALIASES = (
    VALIDATION_CRRU_METRIC,
    LEGACY_VALIDATION_CRRU_METRIC,
)
VALIDATION_ACCURACY_METRIC = "ValidationAccuracy@20_40"
VALIDATION_CRRU_K_METRICS = {
    20: "ValidationCRRU@20",
    40: "ValidationCRRU@40",
}
LEGACY_VALIDATION_CRRU_K_METRICS = {
    20: "ValidationOnlineCRRU@20",
    40: "ValidationOnlineCRRU@40",
}
VALIDATION_CRRU_K_METRIC_ALIASES = {
    cutoff: (
        VALIDATION_CRRU_K_METRICS[cutoff],
        LEGACY_VALIDATION_CRRU_K_METRICS[cutoff],
    )
    for cutoff in VALIDATION_CRRU_K_METRICS
}
CRRU_RECOMMENDATION_METRIC_NAMES_BY_CUTOFF = {
    20: (
        "NDCG@20",
        "Recall@20",
        "HitRatio@20",
        "Personalization@20",
        "AveragePopularity@20",
    ),
    40: (
        "NDCG@40",
        "Recall@40",
        "HitRatio@40",
        "Personalization@40",
        "AveragePopularity@40",
    ),
}
CRRU_RECOMMENDATION_METRIC_NAMES = tuple(
    metric_name
    for cutoff in sorted(CRRU_RECOMMENDATION_METRIC_NAMES_BY_CUTOFF)
    for metric_name in CRRU_RECOMMENDATION_METRIC_NAMES_BY_CUTOFF[cutoff]
)


@dataclass(frozen=True)
class CRRUWeights:
    """Exponent weights for one CRRU utility instantiation."""

    ndcg: float = 0.50
    recall: float = 0.35
    hit: float = 0.15
    personalization: float = 0.40
    inverse_popularity: float = 0.60
    inverse_vram: float = 0.50
    inverse_epoch_time: float = 0.50
    accuracy: float = 0.55
    popularity_aware_personalization: float = 0.30
    efficiency: float = 0.15


THESIS_CRRU_WEIGHTS = CRRUWeights()

CRRU_REPORT_FORMULA_LINES = (
    "CRRU@K - Composite Resource-aware Recommendation Utility at K",
    f"  Formulation: {CRRU_FORMULATION_IDENTIFIER}.",
    "  Family: CRRU@K(m; theta) is parameterized by explicitly stated weights.",
    "  Direction: higher is better; bounded in [0, 1] for valid inputs.",
    "  Scope: absolute per-run utility; independent of other experiments.",
    "  Adding or removing experiments cannot change an already computed CRRU value.",
    (
        "  No row-set, report-row, dataset, trial, or completed-experiment "
        "min-max normalization is used."
    ),
    "  RankingAccuracy@K keeps ranking quality dominant.",
    (
        f"  RankingAccuracy@K = NDCG@K^{THESIS_CRRU_WEIGHTS.ndcg:.2f} "
        f"* Recall@K^{THESIS_CRRU_WEIGHTS.recall:.2f} "
        f"* HitRatio@K^{THESIS_CRRU_WEIGHTS.hit:.2f}"
    ),
    (
        "  HitRatio has the smallest ranking-accuracy weight because it is coarser "
        "than NDCG and Recall."
    ),
    (
        "  PopularityAwarePersonalization@K is personalized recommendation with "
        "reduced popularity concentration."
    ),
    (
        "  Popular items are not inherently bad; CRRU only reflects a thesis "
        "preference against excessive concentration."
    ),
    ("  PyG AveragePopularity@K is logged from raw train-only item interaction counts."),
    (
        "  Reconstructing LargestTrainingItemInteractionCount supplies only the CRRU "
        "denominator; it does not convert a legacy non-raw AveragePopularity@K value "
        "into raw PyG ARP."
    ),
    (
        "  CRRU normalizes raw ARP only inside the utility: "
        "CRRUNormalizedAveragePopularity@K = log(1 + AveragePopularity@K) / "
        "log(1 + LargestTrainingItemInteractionCount)."
    ),
    (
        "  PopularityAwarePersonalization@K = "
        f"Personalization@K^{THESIS_CRRU_WEIGHTS.personalization:.2f} "
        f"* InverseRecommendationPopularity@K^{THESIS_CRRU_WEIGHTS.inverse_popularity:.2f}"
    ),
    ("  InverseRecommendationPopularity@K = 1 - CRRUNormalizedAveragePopularity@K"),
    "  Peak GPU memory is treated as a capacity cost; epoch duration is a throughput cost.",
    "  Average GPU memory may be reported as a diagnostic but is not used in CRRU.",
    (
        "  TrainingResourceUtility = "
        f"PeakGpuMemoryCapacityScore^{THESIS_CRRU_WEIGHTS.inverse_vram:.2f} "
        f"* EpochDurationEfficiencyScore^{THESIS_CRRU_WEIGHTS.inverse_epoch_time:.2f}"
    ),
    "  PeakGpuMemoryCapacityScore = 1 / (1 + log(1 + PeakGpuMemoryMegabytes)).",
    "  EpochDurationEfficiencyScore = 1 / (1 + log(1 + EpochDurationSeconds)).",
    (
        f"  CRRU@K = RankingAccuracy@K^{THESIS_CRRU_WEIGHTS.accuracy:.2f} "
        "* PopularityAwarePersonalization@K^"
        f"{THESIS_CRRU_WEIGHTS.popularity_aware_personalization:.2f} "
        f"* TrainingResourceUtility^{THESIS_CRRU_WEIGHTS.efficiency:.2f}"
    ),
    "  ValidationCRRU@20And40 = arithmetic_mean(CRRU@20, CRRU@40).",
    f"  CRRU_EPSILON={CRRU_EPSILON:g} is only a numerical lower bound, not normalization.",
    "  Missing, NaN, infinite, or out-of-domain inputs raise an error.",
    "  CRRU is a thesis comparison utility, not a causal estimator or standard recommender metric.",
    "  CRRU is not a fairness metric, debiasing proof, or universal cross-dataset quality score.",
)


def _finite_metric(metrics: Mapping[str, float], *names: str) -> float:
    """Return the first finite metric from possible evaluator aliases."""
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            return number
    raise ValueError(f"Missing finite metric; expected one of: {', '.join(names)}.")


def _value_label(name: str, row_index: int | None = None) -> str:
    """Return a clear validation-error label."""
    if row_index is None:
        return name
    return f"{name} at row index {row_index}"


def _finite_required(value: float | None, *, name: str, row_index: int | None = None) -> float:
    """Return a finite float or raise a clear CRRU validation error."""
    if value is None:
        raise ValueError(f"CRRU requires {_value_label(name, row_index)}; got missing value.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"CRRU requires finite {_value_label(name, row_index)}; got {number!r}.")
    return number


def _unit_interval_metric(
    value: float | None,
    *,
    name: str,
    row_index: int | None = None,
) -> float:
    """Validate a finite metric in [0, 1] and apply only epsilon lower bounding."""
    number = _finite_required(value, name=name, row_index=row_index)
    if number < 0.0 or number > 1.0:
        raise ValueError(
            f"CRRU requires {_value_label(name, row_index)} in [0, 1]; got {number!r}.",
        )
    return max(CRRU_EPSILON, number)


def _inverse_recommendation_popularity(
    average_popularity: float | None,
    *,
    largest_training_item_interaction_count: float | None,
    row_index: int | None = None,
) -> float:
    """Return bounded inverse CRRU-normalized recommendation popularity."""
    normalized_popularity = _crru_normalized_average_popularity(
        average_popularity,
        largest_training_item_interaction_count=largest_training_item_interaction_count,
        row_index=row_index,
    )
    return max(CRRU_EPSILON, 1.0 - normalized_popularity)


def _crru_normalized_average_popularity(
    average_popularity: float | None,
    *,
    largest_training_item_interaction_count: float | None,
    row_index: int | None = None,
) -> float:
    """Normalize raw PyG AveragePopularity for CRRU's inverse-popularity term."""
    average_popularity_value = _finite_required(
        average_popularity,
        name="AveragePopularityAtCutoff",
        row_index=row_index,
    )
    if average_popularity_value < 0.0:
        raise ValueError(
            f"CRRU requires {_value_label('AveragePopularityAtCutoff', row_index)} "
            f"to be non-negative; got {average_popularity_value!r}.",
        )
    largest_count = _finite_required(
        largest_training_item_interaction_count,
        name="LargestTrainingItemInteractionCount",
        row_index=row_index,
    )
    if largest_count < 0.0:
        raise ValueError(
            f"CRRU requires {_value_label('LargestTrainingItemInteractionCount', row_index)} "
            f"to be non-negative; got {largest_count!r}.",
        )
    if largest_count == 0.0:
        if average_popularity_value == 0.0:
            return CRRU_EPSILON
        raise ValueError(
            "CRRU cannot normalize positive AveragePopularityAtCutoff when "
            "LargestTrainingItemInteractionCount is zero.",
        )
    normalized = math.log1p(average_popularity_value) / math.log1p(largest_count)
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(
            "CRRU requires CRRUNormalizedAveragePopularityAtCutoff in [0, 1]; "
            f"got {normalized!r} from AveragePopularityAtCutoff={average_popularity_value!r} "
            f"and LargestTrainingItemInteractionCount={largest_count!r}.",
        )
    return max(CRRU_EPSILON, normalized)


def _inverse_log_cost_score(
    value: float | None,
    *,
    name: str,
    row_index: int | None = None,
) -> float:
    """Return a deterministic higher-is-better score for a lower-cost raw quantity."""
    number = _finite_required(value, name=name, row_index=row_index)
    if number <= 0.0:
        raise ValueError(
            f"CRRU requires strictly positive {_value_label(name, row_index)}; got {number!r}.",
        )
    return max(CRRU_EPSILON, 1.0 / (1.0 + math.log1p(number)))


def compute_validation_crru_components_for_k(
    metrics: Mapping[str, float],
    *,
    k: int,
    peak_vram_mb: float | None = None,
    epoch_time_s: float | None = None,
    largest_training_item_interaction_count: float | None = None,
    weights: CRRUWeights = THESIS_CRRU_WEIGHTS,
) -> dict[str, float]:
    """Return ValidationCRRU@K components for one completed validation run.

    This uses the same CRRU exponents as the thesis report:

    ``CRRU@K = RankingAccuracy@K^0.55
    * PopularityAwarePersonalization@K^0.30
    * TrainingResourceUtility^0.15``.

    Validation NDCG/Recall/Hit/Personalization must be finite and in [0, 1].
    AveragePopularity@K must be finite and non-negative. The largest training
    item interaction count, VRAM, and epoch time must be finite valid costs. The
    result does not depend on any other trial or report row.
    """
    cutoff = k
    ndcg = _unit_interval_metric(
        _finite_metric(metrics, f"NDCG@{cutoff}"),
        name=f"NormalizedDiscountedCumulativeGain@{cutoff}",
    )
    recall = _unit_interval_metric(
        _finite_metric(metrics, f"Recall@{cutoff}"),
        name=f"Recall@{cutoff}",
    )
    hit = _unit_interval_metric(
        _finite_metric(metrics, f"Hit@{cutoff}", f"HitRatio@{cutoff}"),
        name=f"HitRatio@{cutoff}",
    )
    personalization = _unit_interval_metric(
        _finite_metric(metrics, f"Personalization@{cutoff}"),
        name=f"Personalization@{cutoff}",
    )
    average_popularity = _finite_metric(metrics, f"AveragePopularity@{cutoff}")

    accuracy = (ndcg**weights.ndcg) * (recall**weights.recall) * (hit**weights.hit)
    popularity_aware_personalization = personalization**weights.personalization * (
        _inverse_recommendation_popularity(
            average_popularity,
            largest_training_item_interaction_count=largest_training_item_interaction_count,
        )
        ** weights.inverse_popularity
    )
    efficiency = (
        _inverse_log_cost_score(peak_vram_mb, name="PeakGpuMemoryMegabytes") ** weights.inverse_vram
    ) * (
        _inverse_log_cost_score(epoch_time_s, name="EpochDurationSeconds")
        ** weights.inverse_epoch_time
    )
    crru = (
        (accuracy**weights.accuracy)
        * (popularity_aware_personalization**weights.popularity_aware_personalization)
        * (efficiency**weights.efficiency)
    )
    _unit_interval_metric(crru, name=f"CRRU@{cutoff}")
    return {
        "accuracy": accuracy,
        "popularity_aware_personalization": popularity_aware_personalization,
        "efficiency": efficiency,
        "crru": crru,
    }


def compute_validation_crru_for_k(
    metrics: Mapping[str, float],
    *,
    k: int,
    peak_vram_mb: float | None = None,
    epoch_time_s: float | None = None,
    largest_training_item_interaction_count: float | None = None,
    weights: CRRUWeights = THESIS_CRRU_WEIGHTS,
) -> float:
    """Return ValidationCRRU@K for one completed validation run."""
    return compute_validation_crru_components_for_k(
        metrics,
        k=k,
        peak_vram_mb=peak_vram_mb,
        epoch_time_s=epoch_time_s,
        largest_training_item_interaction_count=largest_training_item_interaction_count,
        weights=weights,
    )["crru"]


def validation_crru_cutoff_from_metric_name(metric_name: str) -> int | None:
    """Return the cutoff represented by a per-K ValidationCRRU metric name."""
    for cutoff, aliases in VALIDATION_CRRU_K_METRIC_ALIASES.items():
        if metric_name in aliases:
            return cutoff
    return None


def is_validation_crru_metric_name(metric_name: str) -> bool:
    """Return whether a metric name belongs to the ValidationCRRU family."""
    return (
        metric_name in VALIDATION_CRRU_METRIC_ALIASES
        or validation_crru_cutoff_from_metric_name(metric_name) is not None
    )


def compute_validation_crru_objective(
    metrics: Mapping[str, float],
    *,
    peak_vram_mb: float | None = None,
    epoch_time_s: float | None = None,
    largest_training_item_interaction_count: float | None = None,
    cutoffs: Sequence[int] = (20, 40),
    weights: CRRUWeights = THESIS_CRRU_WEIGHTS,
) -> float:
    """Return arithmetic mean of ValidationCRRU over ``cutoffs``."""
    values = [
        compute_validation_crru_for_k(
            metrics,
            k=cutoff,
            peak_vram_mb=peak_vram_mb,
            epoch_time_s=epoch_time_s,
            largest_training_item_interaction_count=largest_training_item_interaction_count,
            weights=weights,
        )
        for cutoff in cutoffs
    ]
    return float(sum(values) / len(values))


def compute_validation_crru_metric_value(
    metric_name: str,
    metrics: Mapping[str, float],
    *,
    peak_vram_mb: float | None = None,
    epoch_time_s: float | None = None,
    largest_training_item_interaction_count: float | None = None,
    weights: CRRUWeights = THESIS_CRRU_WEIGHTS,
) -> float:
    """Return a scalar ValidationCRRU family metric by canonical or legacy name."""
    if metric_name in VALIDATION_CRRU_METRIC_ALIASES:
        return compute_validation_crru_objective(
            metrics,
            peak_vram_mb=peak_vram_mb,
            epoch_time_s=epoch_time_s,
            largest_training_item_interaction_count=largest_training_item_interaction_count,
            weights=weights,
        )
    cutoff = validation_crru_cutoff_from_metric_name(metric_name)
    if cutoff is None:
        raise ValueError(f"{metric_name!r} is not a ValidationCRRU metric name.")
    return compute_validation_crru_for_k(
        metrics,
        k=cutoff,
        peak_vram_mb=peak_vram_mb,
        epoch_time_s=epoch_time_s,
        largest_training_item_interaction_count=largest_training_item_interaction_count,
        weights=weights,
    )


def compute_validation_accuracy_objective(metrics: Mapping[str, float]) -> float:
    """Return the validation-only broad-discovery accuracy objective."""
    return (
        0.50 * _finite_metric(metrics, "NDCG@20")
        + 0.25 * _finite_metric(metrics, "Recall@20")
        + 0.15 * _finite_metric(metrics, "NDCG@40")
        + 0.10 * _finite_metric(metrics, "Recall@40")
    )
