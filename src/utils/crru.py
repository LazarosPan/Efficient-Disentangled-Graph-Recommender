"""Composite Resource-aware Recommendation Utility (CRRU) helpers.

CRRU is a post-hoc thesis reporting utility, not a causal-effect estimator.
CRRU itself is higher-is-better. Lower-cost raw quantities such as average
popularity, VRAM, and time/epoch are inverted into higher-is-better sub-scores
before the final multiplicative utility is computed.

Report CRRU@K uses dataset-local, section-row min-max normalization:

    Accuracy@K = NDCG@K^{0.50} * Recall@K^{0.35} * Hit@K^{0.15}
    Bias@K = Pers@K^{0.40} * (1 - AvgPop@K_n)^{0.60}
    Efficiency = (1 - log(1 + VRAM)_n)^{0.50}
                 * (1 - log(1 + time/epoch)_n)^{0.50}
    CRRU@K = Accuracy@K^{0.55} * Bias@K^{0.30} * Efficiency^{0.15}

The Optuna search objective cannot use future section-row min-max values while a
trial is running, so ``compute_validation_online_crru_objective`` keeps the same
component/exponent structure but uses trial-local bounded penalties for
popularity and resources. Exact report CRRU should still be recomputed after the
search over completed trials.

``CRRU_EPSILON`` is not a normalization method. It only prevents division by
zero when all values in a report section are identical and keeps multiplicative
fractional-power terms away from exact zero. Min-max is used instead of z-score
because CRRU is a bounded utility in [0, 1]; z-scores can be negative and
unbounded, which is incompatible with the fractional-power product.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

CRRU_EPSILON = 1e-8
VALIDATION_ONLINE_CRRU_METRIC = "ValidationOnlineCRRU@20_40"
CRRU_REPORT_FORMULA_LINES = (
    "CRRU@K — Composite Resource-aware Recommendation Utility at K",
    "  Direction: higher is better; AvgPop, VRAM, and time/epoch are inverted.",
    "  Accuracy@K = NDCG@K^0.50 * Recall@K^0.35 * Hit@K^0.15",
    "  Bias@K     = Pers@K^0.40 * (1-AvgPop@K_n)^0.60",
    "  Efficiency = (1-log(1+VRAM)_n)^0.50 * (1-log(1+time/epoch)_n)^0.50",
    "  CRRU@K     = Accuracy@K^0.55 * Bias@K^0.30 * Efficiency^0.15",
    f"  Normalization: dataset-local section-row min-max with epsilon={CRRU_EPSILON:g}",
    "  Note: CRRU is not a causal-effect estimator.",
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
    return [(vram**0.50) * (time**0.50) for vram, time in zip(vram_n, time_n, strict=True)]


def compute_crru_scores_for_k(
    *,
    ndcg: Sequence[float | None],
    recall: Sequence[float | None],
    hit: Sequence[float | None],
    personalization: Sequence[float | None],
    average_popularity: Sequence[float | None],
    efficiency_scores: Sequence[float],
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
        accuracy = (ndcg_value**0.50) * (recall_value**0.35) * (hit_value**0.15)
        bias = (pers_value**0.40) * (avg_pop_value**0.60)
        scores.append((accuracy**0.55) * (bias**0.30) * (efficiency**0.15))
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


def compute_validation_online_crru_for_k(
    metrics: Mapping[str, float],
    *,
    k: int,
    peak_vram_mb: float | None = None,
    epoch_time_s: float | None = None,
) -> float:
    """Return online validation CRRU@K proxy for one Optuna trial.

    This uses the same CRRU exponents as the thesis report:

    ``CRRU@K = Accuracy@K^0.55 * Bias@K^0.30 * Efficiency^0.15``.

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

    accuracy = (ndcg**0.50) * (recall**0.35) * (hit**0.15)
    bias = (personalization**0.40) * (_trial_local_lower_cost_score(avg_pop) ** 0.60)
    efficiency = (_trial_local_lower_cost_score(peak_vram_mb) ** 0.50) * (
        _trial_local_lower_cost_score(epoch_time_s) ** 0.50
    )
    return (accuracy**0.55) * (bias**0.30) * (efficiency**0.15)


def compute_validation_online_crru_objective(
    metrics: Mapping[str, float],
    *,
    peak_vram_mb: float | None = None,
    epoch_time_s: float | None = None,
    ks: Sequence[int] = (20, 40),
) -> float:
    """Return the scalar Optuna objective averaging validation CRRU over ``ks``."""
    values = [
        compute_validation_online_crru_for_k(
            metrics,
            k=k,
            peak_vram_mb=peak_vram_mb,
            epoch_time_s=epoch_time_s,
        )
        for k in ks
    ]
    return float(sum(values) / len(values))
