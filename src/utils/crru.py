"""Composite Resource-aware Recommendation Utility (CRRU) helpers.

CRRU is a configurable post-hoc thesis utility family, not a causal-effect
estimator or universal recommender metric. This module exposes the thesis
instantiation through ``THESIS_CRRU_WEIGHTS`` while keeping the aggregation
functions parameterized by those weights. Lower-cost raw quantities such as
average popularity, VRAM, and time/epoch are inverted into higher-is-better
sub-scores before the final multiplicative utility is computed. Higher VRAM use
is still a resource cost, not a reward; a larger batch can improve CRRU only
indirectly when its time/epoch reduction outweighs the VRAM penalty.

Report CRRU@K uses dataset-local, section-row min-max normalization:

    Accuracy@K = NDCG@K^{0.50} * Recall@K^{0.35} * Hit@K^{0.15}
    PopularityDiversity@K = Pers@K^{0.40} * (1 - AvgPop@K_n)^{0.60}
    Efficiency = (1 - log(1 + VRAM)_n)^{0.50}
                 * (1 - log(1 + time/epoch)_n)^{0.50}
    CRRU@K = Accuracy@K^{0.55}
             * PopularityDiversity@K^{0.30}
             * Efficiency^{0.15}

The Optuna search objective cannot use future section-row min-max values while a
trial is running, so ``compute_validation_online_crru_objective`` keeps the same
component/exponent structure but uses trial-local bounded penalties for
popularity and resources. Exact report CRRU should still be recomputed after the
search over completed trials.

``Online`` in this function name means "computable for one trial during Optuna
search". The Optuna `ValidationOnlineCRRU` value is computed on the validation
split and includes validation ranking/popularity metrics plus peak VRAM and
seconds/epoch. It is not an online-serving metric, online evaluation protocol,
or A/B-test result.

``CRRU_EPSILON`` is not a normalization method. It only prevents division by
zero when all values in a report section are identical and keeps multiplicative
fractional-power terms away from exact zero. Min-max is used instead of z-score
because CRRU is a bounded utility in [0, 1]; z-scores can be negative and
unbounded, which is incompatible with the fractional-power product.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

CRRU_EPSILON = 1e-8
VALIDATION_ONLINE_CRRU_METRIC = "ValidationOnlineCRRU@20_40"
VALIDATION_ACCURACY_METRIC = "ValidationAccuracy@20_40"
VALIDATION_ONLINE_CRRU_K_METRICS = {
    20: "ValidationOnlineCRRU@20",
    40: "ValidationOnlineCRRU@40",
}


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
    popularity_diversity: float = 0.30
    efficiency: float = 0.15


THESIS_CRRU_WEIGHTS = CRRUWeights()

CRRU_REPORT_FORMULA_LINES = (
    "CRRU@K — Composite Resource-aware Recommendation Utility at K",
    "  Family: CRRU@K(m; theta) is parameterized; these are thesis weights.",
    "  Direction: higher is better; AvgPop, VRAM, and time/epoch are inverted.",
    "  VRAM is a capacity cost; larger batches help only through lower time/epoch.",
    (
        f"  Accuracy@K = NDCG@K^{THESIS_CRRU_WEIGHTS.ndcg:.2f} "
        f"* Recall@K^{THESIS_CRRU_WEIGHTS.recall:.2f} "
        f"* Hit@K^{THESIS_CRRU_WEIGHTS.hit:.2f}"
    ),
    (
        "  PopularityDiversity@K = "
        f"Pers@K^{THESIS_CRRU_WEIGHTS.personalization:.2f} "
        f"* (1-AvgPop@K_n)^{THESIS_CRRU_WEIGHTS.inverse_popularity:.2f}"
    ),
    (
        "  Efficiency = "
        f"(1-log(1+VRAM)_n)^{THESIS_CRRU_WEIGHTS.inverse_vram:.2f} "
        f"* (1-log(1+time/epoch)_n)^{THESIS_CRRU_WEIGHTS.inverse_epoch_time:.2f}"
    ),
    (
        f"  CRRU@K = Accuracy@K^{THESIS_CRRU_WEIGHTS.accuracy:.2f} "
        f"* PopularityDiversity@K^{THESIS_CRRU_WEIGHTS.popularity_diversity:.2f} "
        f"* Efficiency^{THESIS_CRRU_WEIGHTS.efficiency:.2f}"
    ),
    f"  Normalization: dataset-local section-row min-max with epsilon={CRRU_EPSILON:g}",
    "  Scope: relative within one dataset/report section; not absolute cross-dataset.",
    "  Note: CRRU is not a causal-effect estimator.",
    "  Note: inverse AvgPop is lower popularity concentration, not causal fairness.",
)


def minmax_normalize(
    values: Sequence[float | None],
    *,
    lower_is_better: bool = False,
    epsilon: float = CRRU_EPSILON,
) -> list[float]:
    """Return clamped min-max normalized values for one dataset/report section."""
    cleaned = [value if value is not None else 0.0 for value in values]
    lo, hi = min(cleaned), max(cleaned)
    scale = (hi - lo) + epsilon
    normalized = [(value - lo) / scale for value in cleaned]
    if lower_is_better:
        normalized = [1.0 - value for value in normalized]
    return [max(epsilon, min(1.0, value)) for value in normalized]


def compute_crru_efficiency_scores(
    peak_vram_mb: Sequence[float | None],
    epoch_times_s: Sequence[float | None],
    *,
    weights: CRRUWeights = THESIS_CRRU_WEIGHTS,
) -> list[float]:
    """Return the shared CRRU efficiency term for one dataset/report section."""
    vram_n = minmax_normalize(
        [math.log1p(value) if value else None for value in peak_vram_mb],
        lower_is_better=True,
    )
    time_n = minmax_normalize(
        [math.log1p(value) if value else None for value in epoch_times_s],
        lower_is_better=True,
    )
    return [
        (vram**weights.inverse_vram) * (time**weights.inverse_epoch_time)
        for vram, time in zip(vram_n, time_n, strict=True)
    ]


def compute_crru_scores_for_k(
    *,
    ndcg: Sequence[float | None],
    recall: Sequence[float | None],
    hit: Sequence[float | None],
    personalization: Sequence[float | None],
    average_popularity: Sequence[float | None],
    efficiency_scores: Sequence[float],
    weights: CRRUWeights = THESIS_CRRU_WEIGHTS,
) -> list[float]:
    """Return report CRRU@K values for one normalized dataset/report section."""
    ndcg_n = minmax_normalize(ndcg)
    recall_n = minmax_normalize(recall)
    hit_n = minmax_normalize(hit)
    pers_n = minmax_normalize(personalization)
    avg_pop_n = minmax_normalize(average_popularity, lower_is_better=True)

    scores: list[float] = []
    for ndcg_value, recall_value, hit_value, pers_value, avg_pop_value, efficiency in zip(
        ndcg_n,
        recall_n,
        hit_n,
        pers_n,
        avg_pop_n,
        efficiency_scores,
        strict=True,
    ):
        accuracy = (
            (ndcg_value**weights.ndcg) * (recall_value**weights.recall) * (hit_value**weights.hit)
        )
        popularity_diversity = (pers_value**weights.personalization) * (
            avg_pop_value**weights.inverse_popularity
        )
        scores.append(
            (accuracy**weights.accuracy)
            * (popularity_diversity**weights.popularity_diversity)
            * (efficiency**weights.efficiency),
        )
    return scores


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


def _unit_interval(value: float) -> float:
    """Clamp a metric already expected to live in [0, 1]."""
    return max(CRRU_EPSILON, min(1.0, value))


def _trial_local_lower_cost_score(value: float | None) -> float:
    """Return a bounded higher-is-better score for a lower-cost raw quantity."""
    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        return 1.0
    return max(CRRU_EPSILON, 1.0 / (1.0 + math.log1p(float(value))))


def compute_validation_online_crru_components_for_k(
    metrics: Mapping[str, float],
    *,
    k: int,
    peak_vram_mb: float | None = None,
    epoch_time_s: float | None = None,
    weights: CRRUWeights = THESIS_CRRU_WEIGHTS,
) -> dict[str, float]:
    """Return OnlineCRRU@K components for one Optuna validation trial.

    This uses the same CRRU exponents as the thesis report:

    ``CRRU@K = Accuracy@K^0.55 * PopularityDiversity@K^0.30 * Efficiency^0.15``.

    Validation NDCG/Recall/Hit/Personalization are already bounded metrics.
    Average popularity, VRAM, and epoch time use deterministic trial-local
    lower-cost transforms into higher-is-better sub-scores because section-row
    min-max normalization is only well-defined after a study/report section is
    complete.
    """
    ndcg = _unit_interval(_finite_metric(metrics, f"NDCG@{k}"))
    recall = _unit_interval(_finite_metric(metrics, f"Recall@{k}"))
    hit = _unit_interval(_finite_metric(metrics, f"Hit@{k}", f"HitRatio@{k}"))
    personalization = _unit_interval(_finite_metric(metrics, f"Personalization@{k}"))
    avg_pop = _finite_metric(metrics, f"AveragePopularity@{k}")

    accuracy = (ndcg**weights.ndcg) * (recall**weights.recall) * (hit**weights.hit)
    popularity_diversity = personalization**weights.personalization * (
        _trial_local_lower_cost_score(avg_pop) ** weights.inverse_popularity
    )
    efficiency = (_trial_local_lower_cost_score(peak_vram_mb) ** weights.inverse_vram) * (
        _trial_local_lower_cost_score(epoch_time_s) ** weights.inverse_epoch_time
    )
    online_crru = (
        (accuracy**weights.accuracy)
        * (popularity_diversity**weights.popularity_diversity)
        * (efficiency**weights.efficiency)
    )
    return {
        "accuracy": accuracy,
        "popularity_diversity": popularity_diversity,
        "efficiency": efficiency,
        "online_crru": online_crru,
    }


def compute_validation_online_crru_for_k(
    metrics: Mapping[str, float],
    *,
    k: int,
    peak_vram_mb: float | None = None,
    epoch_time_s: float | None = None,
    weights: CRRUWeights = THESIS_CRRU_WEIGHTS,
) -> float:
    """Return online validation CRRU@K proxy for one Optuna trial."""
    return compute_validation_online_crru_components_for_k(
        metrics,
        k=k,
        peak_vram_mb=peak_vram_mb,
        epoch_time_s=epoch_time_s,
        weights=weights,
    )["online_crru"]


def compute_validation_online_crru_objective(
    metrics: Mapping[str, float],
    *,
    peak_vram_mb: float | None = None,
    epoch_time_s: float | None = None,
    ks: Sequence[int] = (20, 40),
    weights: CRRUWeights = THESIS_CRRU_WEIGHTS,
) -> float:
    """Return the scalar Optuna objective averaging validation CRRU over ``ks``."""
    values = [
        compute_validation_online_crru_for_k(
            metrics,
            k=k,
            peak_vram_mb=peak_vram_mb,
            epoch_time_s=epoch_time_s,
            weights=weights,
        )
        for k in ks
    ]
    return float(sum(values) / len(values))


def compute_validation_accuracy_objective(metrics: Mapping[str, float]) -> float:
    """Return the validation-only broad-discovery accuracy objective."""
    return (
        0.50 * _finite_metric(metrics, "NDCG@20")
        + 0.25 * _finite_metric(metrics, "Recall@20")
        + 0.15 * _finite_metric(metrics, "NDCG@40")
        + 0.10 * _finite_metric(metrics, "Recall@40")
    )
