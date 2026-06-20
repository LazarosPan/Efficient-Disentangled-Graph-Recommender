#!/usr/bin/env python
"""Export a compact, paper-oriented Optuna figure set."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
from experiments.run_search import DEFAULT_STORAGE
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator
from src.utils.crru import (
    VALIDATION_ONLINE_CRRU_METRIC,
    compute_validation_online_crru_components_for_k,
)

from scripts.report_optuna_optimization import (
    CRRU_FIGURE_STEMS,
    OPTUNA_FIGURES_DIR,
    PAPER_FIGURE_FILENAMES,
    attr_float,
    completed_trials,
    dataset_metric,
    dataset_names,
    load_studies,
    logical_trial_params,
    objective_metric_label,
    trial_objective_metric,
    trial_objective_split,
)

DATASET_LABELS = {
    "amazonbook": "Amazon Book",
    "movielens1m": "MovieLens-1M",
    "kuairec_v2": "KuaiRec v2",
    "kuairand1k": "KuaiRand-1K",
}
DATASET_COLORS = {
    "amazonbook": "#2f6f9f",
    "movielens1m": "#d17a22",
    "kuairec_v2": "#2f8f46",
    "kuairand1k": "#b83b3b",
}
PARAMETER_LABELS = {
    "lr": "Learning rate",
    "weight_decay": "Weight decay",
    "batch_size": "Training batch size",
    "lr_scheduler": "LR scheduler",
    "lr_scheduler_factor": "Plateau decay factor",
    "embedding_optimizer": "Embedding optimizer",
    "grad_clip_norm": "Gradient clipping norm",
    "interest_gnn_layers": "Interest branch depth",
    "conformity_gnn_layers": "Conformity branch depth",
    "num_neighbors": "Neighbor fanout",
    "dropout": "Dropout",
    "score_mix_min_weight": "Minimum branch-mix weight",
    "loss_weight_interest_bpr": "Interest-branch BPR weight",
    "loss_weight_conformity_bpr": "Conformity-branch BPR weight",
    "loss_weight_independence": "Branch independence penalty",
    "loss_weight_contrastive": "Contrastive loss weight",
    "loss_weight_popularity": "Popularity penalty weight",
    "auxiliary_loss_schedule": "Auxiliary-loss schedule",
    "auxiliary_ramp_rate": "Auxiliary ramp slope",
    "independence_ramp_rate": "Independence ramp slope",
    "auxiliary_losses_start_epoch": "Auxiliary losses start epoch",
    "popularity_supervision_start_epoch": "Popularity supervision start epoch",
    "graph_policy": "Training graph policy",
    "item_universe_policy": "Item universe policy",
    "preprocessing_preset": "Preprocessing preset",
    "train_edge_keep_prob": "Train edge keep rate",
    "hard_negative_ratio": "Hard-negative ratio",
    "dice_sampler_margin": "DICE popularity-gap margin",
    "use_features": "Use item/user features",
    "use_popularity_head": "Use popularity head",
}
PROFILE_FALLBACK_PARAMS = (
    "lr",
    "weight_decay",
    "batch_size",
    "embedding_optimizer",
    "conformity_gnn_layers",
    "interest_gnn_layers",
    "num_neighbors",
    "dropout",
    "score_mix_min_weight",
    "loss_weight_interest_bpr",
    "loss_weight_conformity_bpr",
    "loss_weight_independence",
    "loss_weight_contrastive",
    "loss_weight_popularity",
    "auxiliary_loss_schedule",
    "auxiliary_ramp_rate",
    "independence_ramp_rate",
    "graph_policy",
    "item_universe_policy",
    "preprocessing_preset",
    "train_edge_keep_prob",
    "hard_negative_ratio",
    "dice_sampler_margin",
    "grad_clip_norm",
)
EXCLUDED_FIGURE_PARAM_PREFIXES = ("cagra_",)
CRRU_SELECTION_TITLE = "Validation CRRU selection score @20/40"
CRRU_SELECTION_SHORT_LABEL = "CRRU selection score"
CRRU_SELECTION_AXIS_LABEL = "Normalized CRRU score\n(dataset-local, 0=worst, 1=best)"
CRRU_WEIGHT_NOTE = "Final CRRU weights: accuracy 55%, popularity-diversity 30%, efficiency 15%."
FIGURE_FILE_SUFFIXES = frozenset({".png", ".pdf", ".svg", ".jpg", ".jpeg", ".html"})
LOW_SUPPORT_COMPLETED_TRIALS = 10


@dataclass(frozen=True)
class FigureImportanceResult:
    """Importance result used only for thesis-facing overview figures."""

    importances: dict[str, float]
    quality: str
    trial_count: int
    reason: str | None


def _dataset_label(dataset: str) -> str:
    """Return a thesis-facing dataset label."""
    return DATASET_LABELS.get(dataset, dataset)


def _parameter_label(parameter: str, *, width: int = 18) -> str:
    """Return a wrapped thesis-facing parameter label."""
    label = PARAMETER_LABELS.get(parameter, parameter.replace("_", " "))
    return "\n".join(textwrap.wrap(label, width=width))


def _include_figure_param(parameter: str) -> bool:
    """Return whether a parameter belongs in thesis-facing overview figures."""
    return not parameter.startswith(EXCLUDED_FIGURE_PARAM_PREFIXES)


def _objective_priority(metric: str) -> int:
    """Return stable display priority for validation objective families."""
    return {
        VALIDATION_ONLINE_CRRU_METRIC: 0,
        "NDCG@40": 2,
    }.get(metric, 10)


def _objective_label(split: str, metric: str) -> str:
    """Return a compact objective label for plot titles."""
    return f"{split} {metric}" if split and split != "-" else metric


def _figure_objective_label(metric: str) -> str:
    """Return a Matplotlib-friendly objective label."""
    if metric == VALIDATION_ONLINE_CRRU_METRIC:
        return CRRU_SELECTION_TITLE
    return objective_metric_label(metric).replace("@20_40", "@20/40")


def _compact_objective_label(split: str, metric: str) -> str:
    """Return a short objective label for dense figures."""
    compact_metric = {
        VALIDATION_ONLINE_CRRU_METRIC: "CRRU selection@20/40",
    }.get(metric, metric)
    return f"{split} {compact_metric}" if split and split != "-" else compact_metric


def _compact_study_label(study_name: str, *, width: int = 46) -> str:
    """Return a shortened study label for dense figure panels."""
    return textwrap.shorten(study_name, width=width, placeholder="...")


def _record_group_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the comparable objective group for one flat trial record."""
    return (
        str(record["dataset"]),
        str(record.get("objective_split", "-")),
        str(record.get("objective_metric", "-")),
    )


def _record_group_sort_key(group: tuple[str, str, str]) -> tuple[int, str, str]:
    """Return stable sort order for dataset/objective plot groups."""
    dataset, split, metric = group
    return _objective_priority(metric), dataset, split


def _records_by_group(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[tuple[str, str, str], list[Mapping[str, Any]]]]:
    """Group records by dataset and comparable objective family."""
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_record_group_key(record)].append(record)
    return sorted(grouped.items(), key=lambda item: _record_group_sort_key(item[0]))


def _group_title(group: tuple[str, str, str]) -> str:
    """Return a short title for one dataset/objective plot group."""
    dataset, _split, _metric = group
    return _dataset_label(dataset)


def _best_trial_handle() -> Line2D:
    """Return a legend handle for the selected validation trial."""
    return Line2D(
        [],
        [],
        marker="*",
        markersize=12,
        linestyle="None",
        markerfacecolor="#f0c419",
        markeredgecolor="#202020",
        label="Gold star = selected trial",
    )


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Save and close one matplotlib figure without clipping labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    return output_path


def _clear_existing_figures(output_dir: Path) -> None:
    """Remove stale generated figure files from the Optuna figure directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in FIGURE_FILE_SUFFIXES:
            path.unlink()


def _color_for_dataset(dataset: str) -> str:
    """Return a stable plot color for a dataset."""
    return DATASET_COLORS.get(dataset, "#4d6073")


def _trial_epoch_time_s(trial: optuna.trial.FrozenTrial, dataset: str) -> float | None:
    """Return stored or derived seconds per epoch for a trial."""
    avg_epoch_time_s = attr_float(trial, f"{dataset}.avg_epoch_time_s")
    if avg_epoch_time_s is not None and avg_epoch_time_s > 0.0:
        return avg_epoch_time_s
    training_time_s = attr_float(trial, f"{dataset}.training_time_s")
    epochs = attr_float(trial, f"{dataset}.epochs_stopped_at")
    if training_time_s is not None and epochs is not None and epochs > 0.0:
        return training_time_s / epochs
    return None


def _metric_value(
    trial: optuna.trial.FrozenTrial,
    dataset: str,
    metric_name: str,
) -> float | None:
    """Return one stored metric value for plotting."""
    if metric_name == "epoch_time_s":
        return _trial_epoch_time_s(trial, dataset)
    if metric_name == "peak_vram_mb":
        return attr_float(trial, f"{dataset}.peak_vram_mb")
    return dataset_metric(trial, dataset, metric_name)


def _online_crru_validation_metrics(
    trial: optuna.trial.FrozenTrial,
    dataset: str,
) -> dict[str, float]:
    """Return validation metrics needed to reconstruct OnlineCRRU components."""
    names = (
        "NDCG@20",
        "Recall@20",
        "HitRatio@20",
        "Personalization@20",
        "AveragePopularity@20",
        "NDCG@40",
        "Recall@40",
        "HitRatio@40",
        "Personalization@40",
        "AveragePopularity@40",
    )
    metrics: dict[str, float] = {}
    for name in names:
        value = _metric_value(trial, dataset, name)
        if value is None:
            return {}
        metrics[name] = value
    return metrics


def _online_crru_component_summary(
    trial: optuna.trial.FrozenTrial,
    dataset: str,
) -> dict[str, float | None]:
    """Return averaged OnlineCRRU component scores over K=20 and K=40."""
    metrics = _online_crru_validation_metrics(trial, dataset)
    if not metrics:
        return {
            "accuracy_component": None,
            "popularity_diversity_component": None,
            "resource_efficiency_score": None,
            "online_crru_reconstructed": None,
        }
    try:
        by_k = [
            compute_validation_online_crru_components_for_k(
                metrics,
                k=k,
                peak_vram_mb=_metric_value(trial, dataset, "peak_vram_mb"),
                epoch_time_s=_trial_epoch_time_s(trial, dataset),
            )
            for k in (20, 40)
        ]
    except ValueError:
        return {
            "accuracy_component": None,
            "popularity_diversity_component": None,
            "resource_efficiency_score": None,
            "online_crru_reconstructed": None,
        }
    return {
        "accuracy_component": float(sum(item["accuracy"] for item in by_k) / len(by_k)),
        "popularity_diversity_component": float(
            sum(item["popularity_diversity"] for item in by_k) / len(by_k),
        ),
        "resource_efficiency_score": float(sum(item["efficiency"] for item in by_k) / len(by_k)),
        "online_crru_reconstructed": float(
            sum(item["online_crru"] for item in by_k) / len(by_k),
        ),
    }


def _trial_records(studies: Sequence[optuna.Study]) -> list[dict[str, Any]]:
    """Return one flat CRRU-proxy record per dataset-local completed trial."""
    records: list[dict[str, Any]] = []
    for study in studies:
        for trial in sorted(completed_trials(study), key=lambda item: item.number):
            params = logical_trial_params(trial)
            source_metric = trial_objective_metric(trial)
            source_split = trial_objective_split(trial)
            for dataset in dataset_names(trial):
                objective = dataset_metric(trial, dataset, VALIDATION_ONLINE_CRRU_METRIC)
                if objective is None or not math.isfinite(float(objective)):
                    continue
                completed_at = trial.datetime_complete or trial.datetime_start
                component_summary = _online_crru_component_summary(trial, dataset)
                records.append(
                    {
                        "dataset": dataset,
                        "study_name": study.study_name,
                        "source_label": (
                            f"{_compact_study_label(study.study_name, width=34)} "
                            f"trial {trial.number}"
                        ),
                        "study_direction": "maximize",
                        "objective_metric": VALIDATION_ONLINE_CRRU_METRIC,
                        "objective_split": "val",
                        "source_objective_metric": source_metric,
                        "source_objective_split": source_split,
                        "objective_label": _objective_label(
                            "val",
                            VALIDATION_ONLINE_CRRU_METRIC,
                        ),
                        "trial_number": trial.number,
                        "completed_at": completed_at.isoformat() if completed_at else "",
                        "objective": float(objective),
                        "params": params,
                        "lr": _as_float(params.get("lr")),
                        "branch_mix_floor": _as_float(params.get("score_mix_min_weight")),
                        "interest_depth": _as_int(params.get("interest_gnn_layers")),
                        "conformity_depth": _as_int(params.get("conformity_gnn_layers")),
                        "fanout": _fanout_tuple(params.get("num_neighbors")),
                        "dropout": _as_float(params.get("dropout")),
                        "ndcg40": _metric_value(trial, dataset, "NDCG@40"),
                        "recall40": _metric_value(trial, dataset, "Recall@40"),
                        "hit40": _metric_value(trial, dataset, "HitRatio@40"),
                        "personalization40": _metric_value(trial, dataset, "Personalization@40"),
                        "average_popularity40": _metric_value(
                            trial,
                            dataset,
                            "AveragePopularity@40",
                        ),
                        "epoch_time_s": _metric_value(trial, dataset, "epoch_time_s"),
                        "peak_vram_mb": _metric_value(trial, dataset, "peak_vram_mb"),
                        **component_summary,
                    },
                )
    _attach_component_scores(records)
    return records


def _as_float(value: Any) -> float | None:
    """Return a finite float or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    """Return an int or None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fanout_tuple(value: Any) -> tuple[int, ...] | None:
    """Return a fanout tuple or None."""
    if not isinstance(value, list | tuple):
        return None
    try:
        return tuple(int(part) for part in value)
    except (TypeError, ValueError):
        return None


def _stable_param_key(value: Any) -> str:
    """Return a stable categorical key for a parameter value."""
    if value is None:
        return "__missing__"
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return average ranks with ties."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for original_index, _value in indexed[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def _pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return Pearson correlation, or None when undefined."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(lval * rval for lval, rval in zip(left_centered, right_centered, strict=True))
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _squared_pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Return squared Pearson correlation, or zero when undefined."""
    corr = _pearson_correlation(left, right)
    if corr is None:
        return 0.0
    return max(0.0, min(1.0, corr * corr))


def _spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return Spearman rank correlation, or None when undefined."""
    if len(left) != len(right) or len(left) < 3:
        return None
    return _pearson_correlation(_average_ranks(left), _average_ranks(right))


def _eta_squared(categories: Sequence[str], targets: Sequence[float]) -> float:
    """Return between-group variance share for one categorical parameter."""
    if len(categories) != len(targets) or len(set(categories)) < 2:
        return 0.0
    overall_mean = sum(targets) / len(targets)
    total_ss = sum((value - overall_mean) ** 2 for value in targets)
    if total_ss <= 0.0:
        return 0.0
    grouped: dict[str, list[float]] = defaultdict(list)
    for category, target in zip(categories, targets, strict=True):
        grouped[category].append(target)
    between_ss = 0.0
    for values in grouped.values():
        group_mean = sum(values) / len(values)
        between_ss += len(values) * ((group_mean - overall_mean) ** 2)
    return max(0.0, min(1.0, between_ss / total_ss))


def _exploratory_record_importances(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return normalized univariate association scores for global figure records."""
    targets = [_as_float(record.get("objective")) for record in records]
    if any(value is None for value in targets):
        return {}
    target_values = [float(value) for value in targets if value is not None]
    target_ranks = _average_ranks(target_values)
    names = sorted(
        {
            name
            for record in records
            if isinstance(record.get("params"), Mapping)
            for name in record["params"]
            if _include_figure_param(str(name))
        },
    )
    raw_scores: dict[str, float] = {}
    for name in names:
        values = [
            record["params"].get(name) if isinstance(record.get("params"), Mapping) else None
            for record in records
        ]
        categorical_keys = [_stable_param_key(value) for value in values]
        if len(set(categorical_keys)) < 2:
            continue
        numeric_values = [_as_float(value) for value in values]
        if all(value is not None for value in numeric_values):
            value_ranks = _average_ranks(
                [float(value) for value in numeric_values if value is not None]
            )
            score = _squared_pearson(value_ranks, target_ranks)
        else:
            score = _eta_squared(categorical_keys, target_values)
        if math.isfinite(score) and score > 0.0:
            raw_scores[name] = score
    total = sum(raw_scores.values())
    if total <= 0.0:
        return {}
    normalized = {name: score / total for name, score in raw_scores.items()}
    return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))


def _minmax_score(
    values: Sequence[float | None],
    *,
    lower_is_better: bool = False,
) -> list[float | None]:
    """Return dataset-local min-max scores in [0, 1]."""
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return [None for _ in values]
    lo, hi = min(finite), max(finite)
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-12):
        return [1.0 if value is not None else None for value in values]
    scores: list[float | None] = []
    for value in values:
        if value is None or not math.isfinite(float(value)):
            scores.append(None)
            continue
        score = (float(value) - lo) / (hi - lo)
        if lower_is_better:
            score = 1.0 - score
        scores.append(max(0.0, min(1.0, score)))
    return scores


def _attach_component_scores(records: list[dict[str, Any]]) -> None:
    """Attach dataset-local normalized objective scores to flat records."""
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[_record_group_key(record)].append(record)

    for dataset_records in by_group.values():
        objective_percentiles = _minmax_score(
            [record.get("objective") for record in dataset_records],
            lower_is_better=False,
        )
        for index, record in enumerate(dataset_records):
            record["objective_percentile"] = objective_percentiles[index]


def _best_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the best comparable-objective record from a sequence."""
    ranked = [
        record for record in records if _as_float(record.get("objective_percentile")) is not None
    ]
    if not ranked:
        return None
    return max(ranked, key=lambda record: float(record["objective_percentile"]))


COMPONENT_CORRELATION_SPECS = (
    ("accuracy_component", "Accuracy\nNDCG/Recall/Hit"),
    (
        "popularity_diversity_component",
        "Popularity-diversity\nPers + inverse AvgPop",
    ),
    ("resource_efficiency_score", "Resource efficiency\ninverse time + inverse VRAM"),
)


def export_component_correlations_by_dataset(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    objective_metric: str,
    filename: str,
) -> Path | None:
    """Export Spearman associations between validation CRRU and its components."""
    groups = [(group, group_records) for group, group_records in _records_by_group(records)]
    if not groups:
        return None

    matrix = np.full((len(COMPONENT_CORRELATION_SPECS), len(groups)), np.nan)
    counts = np.zeros_like(matrix, dtype=int)
    for x_index, (_group, group_records) in enumerate(groups):
        for y_index, (component_key, _label) in enumerate(COMPONENT_CORRELATION_SPECS):
            paired = [
                (float(component), float(objective))
                for record in group_records
                if (component := _as_float(record.get(component_key))) is not None
                and (objective := _as_float(record.get("objective"))) is not None
            ]
            counts[y_index, x_index] = len(paired)
            if len(paired) < 3:
                continue
            component_values, objective_values = zip(*paired, strict=True)
            corr = _spearman_correlation(component_values, objective_values)
            if corr is not None and math.isfinite(corr):
                matrix[y_index, x_index] = corr

    if not np.isfinite(matrix).any():
        return None

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#eeeeee")
    fig, ax = plt.subplots(figsize=(10.2, 4.7))
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(
        [_dataset_label(group[0]) for group, _records in groups],
        fontsize=9,
        rotation=0,
        ha="center",
    )
    ax.set_yticks(range(len(COMPONENT_CORRELATION_SPECS)))
    ax.set_yticklabels([label for _key, label in COMPONENT_CORRELATION_SPECS])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("CRRU component")
    ax.set_title("Validation CRRU component associations")
    for y_index in range(matrix.shape[0]):
        for x_index in range(matrix.shape[1]):
            if math.isfinite(float(matrix[y_index, x_index])):
                text = f"{matrix[y_index, x_index]:+.2f}\nn={counts[y_index, x_index]}"
                color = "#ffffff" if abs(float(matrix[y_index, x_index])) >= 0.55 else "#202020"
            else:
                text = f"n={counts[y_index, x_index]}\nNA"
                color = "#555555"
            ax.text(
                x_index,
                y_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    fig.colorbar(
        image,
        ax=ax,
        label="Spearman rho",
        fraction=0.055,
        pad=0.04,
        shrink=0.82,
    )
    fig.subplots_adjust(left=0.23, right=0.86, bottom=0.19, top=0.79)
    return _save_figure(fig, output_dir / filename)


def export_selection_frontier_by_dataset(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    objective_metric: str,
    filename: str,
) -> Path | None:
    """Export the accuracy, popularity-diversity, and efficiency trade-off view."""
    groups = [
        (group, comparable_records)
        for group, group_records in _records_by_group(records)
        if (
            comparable_records := [
                record
                for record in group_records
                if _as_float(record.get("accuracy_component")) is not None
                and _as_float(record.get("popularity_diversity_component")) is not None
                and _as_float(record.get("resource_efficiency_score")) is not None
                and _as_float(record.get("objective_percentile")) is not None
            ]
        )
    ]
    if not groups:
        return None

    columns = 2
    rows = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.6, 4.9 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat[len(groups) :]:
        ax.set_visible(False)

    last_scatter = None
    for ax, (group, comparable_records) in zip(axes.flat, groups, strict=False):
        efficiency_ranks = _minmax_score(
            [record.get("resource_efficiency_score") for record in comparable_records],
        )
        last_scatter = ax.scatter(
            [float(record["accuracy_component"]) for record in comparable_records],
            [float(record["popularity_diversity_component"]) for record in comparable_records],
            c=[score if score is not None else 0.0 for score in efficiency_ranks],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=58,
            alpha=0.70,
            edgecolor="#202020",
            linewidth=0.35,
        )
        best = _best_record(comparable_records)
        if best is not None:
            ax.scatter(
                [float(best["accuracy_component"])],
                [float(best["popularity_diversity_component"])],
                marker="*",
                s=205,
                color="#f0c419",
                edgecolor="#202020",
                linewidth=0.9,
                zorder=5,
            )
        ax.set_xlabel("Accuracy component (NDCG, Recall, Hit)")
        ax.set_ylabel("Popularity-diversity component (Personalization, inverse AvgPop)")
        ax.set_title(_group_title(group))
        ax.grid(alpha=0.20)

    if last_scatter is not None:
        fig.colorbar(
            last_scatter,
            ax=axes.ravel().tolist(),
            label="Resource-efficiency percentile\nwithin dataset; higher is more efficient",
            shrink=0.82,
        )
    axes.flat[0].legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                markersize=7,
                linestyle="None",
                markerfacecolor="#4f8f77",
                markeredgecolor="#202020",
                label="Completed trial",
            ),
            _best_trial_handle(),
        ],
        loc="best",
        fontsize=8,
        frameon=True,
    )
    fig.suptitle("Validation CRRU component trade-off by dataset")
    return _save_figure(fig, output_dir / filename)


def _top_importance_params(
    grouped_records: Sequence[tuple[tuple[str, str, str], Sequence[Mapping[str, Any]]]],
    *,
    limit: int,
) -> list[str]:
    """Return compact cross-dataset parameter set by mean importance."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for _group, records in grouped_records:
        result = _figure_importance_result(records)
        for name, importance in result.importances.items():
            totals[name] = totals.get(name, 0.0) + float(importance)
            counts[name] = counts.get(name, 0) + 1
    if not totals:
        return [param for param in PROFILE_FALLBACK_PARAMS if _include_figure_param(param)][:limit]
    ranked = sorted(
        totals,
        key=lambda name: (-(totals[name] / counts[name]), name),
    )
    return ranked[:limit]


def _figure_importance_result(records: Sequence[Mapping[str, Any]]) -> FigureImportanceResult:
    """Return explicitly exploratory importances for one global figure row."""
    if len(records) < 2:
        return FigureImportanceResult(
            {},
            "unavailable",
            len(records),
            "fewer than two completed dataset-local trials",
        )

    importances = _exploratory_record_importances(records)
    if not importances:
        return FigureImportanceResult(
            {},
            "unavailable",
            len(records),
            "no varying sampled parameters with finite objectives",
        )
    return FigureImportanceResult(
        importances,
        "exploratory",
        len(records),
        "univariate association over all loaded completed dataset-local trials",
    )


def export_importance_by_dataset(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    objective_metric: str,
    filename: str,
) -> Path | None:
    """Export one dataset-by-parameter importance heatmap."""
    grouped_records = _records_by_group(records)
    rows: list[tuple[tuple[str, str, str], FigureImportanceResult]] = []
    has_importance = False
    for group, group_records in grouped_records:
        result = _figure_importance_result(group_records)
        if result.importances:
            has_importance = True
        rows.append((group, result))
    if not rows or not has_importance:
        return None

    params = _top_importance_params(grouped_records, limit=11)
    matrix_values = [
        [
            result.importances[param]
            if result.importances and param in result.importances
            else np.nan
            for param in params
        ]
        for _, result in rows
    ]
    matrix = np.ma.masked_invalid(np.array(matrix_values, dtype=float))
    finite_values = [
        float(value) for row in matrix_values for value in row if math.isfinite(float(value))
    ]
    finite_max = max(finite_values) if finite_values else 0.01
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#eeeeee")
    fig, ax = plt.subplots(
        figsize=(15.0, max(5.0, 1.0 * len(rows) + 1.8)),
        constrained_layout=True,
    )
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=max(0.01, finite_max)),
    )
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(
        [_parameter_label(param, width=14) for param in params],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [
            (
                f"{_group_title(group)}\n({result.quality}, n={result.trial_count})"
                if result.importances
                else f"{_group_title(group)}\n(unavailable)"
            )
            for group, result in rows
        ],
    )
    ax.set_title("Exploratory hyperparameter associations with validation CRRU")
    ax.set_xlabel("Sampled hyperparameter")
    for y_index, (_group, result) in enumerate(rows):
        if not result.importances:
            ax.text(
                (len(params) - 1) / 2,
                y_index,
                "unavailable",
                ha="center",
                va="center",
                fontsize=9,
                color="#555555",
            )
            continue
        for x_index, value in enumerate(matrix_values[y_index]):
            if math.isfinite(float(value)):
                text_color = "#ffffff" if float(value) >= (0.55 * finite_max) else "#202020"
                ax.text(
                    x_index,
                    y_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
    fig.colorbar(image, ax=ax, label="Normalized univariate association share")
    ax.legend(
        handles=[
            Patch(
                facecolor="#eeeeee",
                edgecolor="#cccccc",
                label="Gray = no detected association for displayed parameters",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.12),
        fontsize=8,
        frameon=True,
    )
    return _save_figure(fig, output_dir / filename)


def export_crru_components_by_dataset(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    objective_metric: str,
    filename: str,
) -> Path | None:
    """Export CRRU component-vs-objective diagnostics."""
    if not records:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), constrained_layout=True)
    plotted_any = False
    for ax, (key, label) in zip(axes.flat[:3], COMPONENT_CORRELATION_SPECS, strict=True):
        for group, group_records in _records_by_group(records):
            dataset = group[0]
            component_percentiles = _minmax_score([record.get(key) for record in group_records])
            comparable_pairs = [
                (record, component_percentile)
                for record, component_percentile in zip(
                    group_records,
                    component_percentiles,
                    strict=True,
                )
                if component_percentile is not None
                and _as_float(record.get("objective_percentile")) is not None
            ]
            if not comparable_pairs:
                continue
            plotted_any = True
            ax.scatter(
                [float(component_percentile) for _record, component_percentile in comparable_pairs],
                [float(record["objective_percentile"]) for record, _score in comparable_pairs],
                s=36,
                alpha=0.68,
                color=_color_for_dataset(dataset),
                label=_group_title(group).replace("\n", " | "),
            )
            best_pair = max(
                comparable_pairs,
                key=lambda pair: float(pair[0]["objective_percentile"]),
                default=None,
            )
            if best_pair is not None:
                best, best_component_percentile = best_pair
                ax.scatter(
                    [float(best_component_percentile)],
                    [float(best["objective_percentile"])],
                    marker="*",
                    s=150,
                    color="#f0c419",
                    edgecolor="#202020",
                    linewidth=0.8,
                    zorder=5,
                )
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(label.replace("\n", " "))
        ax.set_xlabel("Component percentile within dataset")
        ax.set_ylabel("Validation CRRU percentile\nwithin dataset")
        ax.grid(alpha=0.22)

    axes.flat[3].axis("off")
    axes.flat[3].text(
        0.50,
        0.78,
        "Component meaning",
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="center",
        transform=axes.flat[3].transAxes,
    )
    axes.flat[3].text(
        0.50,
        0.62,
        "Accuracy: NDCG, Recall, and Hit.\n"
        "Popularity-diversity: personalization and inverse AvgPop.\n"
        "Efficiency: inverse log time/epoch and inverse log VRAM.\n\n"
        f"{CRRU_WEIGHT_NOTE}\n"
        "Higher component percentiles indicate stronger within-dataset values.",
        fontsize=9,
        linespacing=1.25,
        va="top",
        ha="center",
        transform=axes.flat[3].transAxes,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[3].legend(
            [*handles, _best_trial_handle()],
            [*labels, _best_trial_handle().get_label()],
            loc="lower center",
            fontsize=8,
            frameon=False,
        )
    fig.suptitle("Validation CRRU response to its component percentiles")
    return _save_figure(fig, output_dir / filename) if plotted_any else None


def _sorted_learning_rates(records: Sequence[Mapping[str, Any]]) -> list[float]:
    """Return sorted unique finite learning rates."""
    return sorted({float(record["lr"]) for record in records if _as_float(record.get("lr"))})


def _format_lr(value: float) -> str:
    """Format a learning rate without scientific notation noise."""
    if value >= 0.001:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _format_lr_tick(value: float, _position: int | None = None) -> str:
    """Return a compact learning-rate tick label."""
    if value <= 0.0:
        return ""
    return _format_lr(value)


def _circle_legend_handle(markersize: float, label: str) -> Line2D:
    """Return one filled-circle legend handle."""
    return Line2D(
        [],
        [],
        marker="o",
        markersize=markersize,
        linestyle="None",
        markerfacecolor="#8a9aa8",
        markeredgecolor="#202020",
        markeredgewidth=0.6,
        alpha=0.75,
        label=label,
    )


def export_lr_branchmix_landscape(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    objective_metric: str,
    filename: str,
) -> Path | None:
    """Export dataset-local LR/branch-mix objective landscapes."""
    groups = [
        (group, comparable_records)
        for group, group_records in _records_by_group(records)
        if (
            comparable_records := [
                record
                for record in group_records
                if _as_float(record.get("lr")) is not None
                and _as_float(record.get("branch_mix_floor")) is not None
                and _as_float(record.get("objective_percentile")) is not None
            ]
        )
    ]
    if not groups:
        return None

    columns = 2
    rows = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.2, 4.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat[len(groups) :]:
        ax.set_visible(False)

    last_scatter = None
    for ax, (group, comparable_records) in zip(axes.flat, groups, strict=False):
        fanout_totals = [
            sum(record["fanout"]) if record.get("fanout") else 1 for record in comparable_records
        ]
        max_fanout_total = max(fanout_totals) if fanout_totals else 1
        sizes = [38.0 + 90.0 * (total / max_fanout_total) for total in fanout_totals]
        last_scatter = ax.scatter(
            [
                float(record["lr"]) * (1.0 + ((int(record["trial_number"]) % 7) - 3) * 0.012)
                for record in comparable_records
            ],
            [float(record["branch_mix_floor"]) for record in comparable_records],
            c=[float(record["objective_percentile"]) for record in comparable_records],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=sizes,
            alpha=0.78,
            edgecolor="#202020",
            linewidth=0.35,
        )
        best = _best_record(comparable_records)
        if best is not None:
            ax.scatter(
                [float(best["lr"])],
                [float(best["branch_mix_floor"])],
                marker="*",
                s=185,
                color="#f0c419",
                edgecolor="#202020",
                linewidth=0.9,
                zorder=5,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0), numticks=12))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_lr_tick))
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Minimum branch-mix weight")
        ax.set_title(_group_title(group))
        ax.grid(alpha=0.20)

    if last_scatter is not None:
        fig.colorbar(
            last_scatter,
            ax=axes.ravel().tolist(),
            label=CRRU_SELECTION_AXIS_LABEL,
            shrink=0.82,
        )
    axes.flat[0].legend(
        handles=[
            _circle_legend_handle(
                5.5,
                "Smaller circle = fewer sampled neighbors",
            ),
            _circle_legend_handle(
                9.5,
                "Larger circle = more sampled neighbors",
            ),
            _best_trial_handle(),
        ],
        loc="best",
        fontsize=8,
        frameon=True,
    )
    fig.suptitle("Validation CRRU landscape: learning rate and branch mixing")
    return _save_figure(fig, output_dir / filename)


def export_branch_depth_heatmaps(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    objective_metric: str,
    filename: str,
) -> Path | None:
    """Export mean objective heatmaps for interest/conformity branch depths."""
    groups = [
        (group, comparable_records)
        for group, group_records in _records_by_group(records)
        if (
            comparable_records := [
                record
                for record in group_records
                if record.get("interest_depth") is not None
                and record.get("conformity_depth") is not None
                and _as_float(record.get("objective_percentile")) is not None
            ]
        )
    ]
    if not groups:
        return None

    columns = 2
    rows = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.6, 4.7 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat[len(groups) :]:
        ax.set_visible(False)

    last_image = None
    for ax, (group, comparable_records) in zip(axes.flat, groups, strict=False):
        interests = sorted({int(record["interest_depth"]) for record in comparable_records})
        conformities = sorted({int(record["conformity_depth"]) for record in comparable_records})
        matrix = np.full((len(conformities), len(interests)), np.nan)
        counts = np.zeros((len(conformities), len(interests)), dtype=int)
        for y_index, conformity in enumerate(conformities):
            for x_index, interest in enumerate(interests):
                values = [
                    float(record["objective_percentile"])
                    for record in comparable_records
                    if int(record["interest_depth"]) == interest
                    and int(record["conformity_depth"]) == conformity
                ]
                if values:
                    matrix[y_index, x_index] = float(sum(values) / len(values))
                    counts[y_index, x_index] = len(values)
        cmap = plt.get_cmap("BuGn").copy()
        cmap.set_bad(color="#eeeeee")
        last_image = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xticks(range(len(interests)))
        ax.set_xticklabels([str(value) for value in interests])
        ax.set_yticks(range(len(conformities)))
        ax.set_yticklabels([str(value) for value in conformities])
        ax.set_xlabel("Interest branch GNN layers")
        ax.set_ylabel("Conformity branch GNN layers")
        ax.set_title(_group_title(group))
        if np.isfinite(matrix).any():
            best_index = int(np.nanargmax(matrix))
            best_y, best_x = np.unravel_index(best_index, matrix.shape)
            ax.scatter(
                [best_x + 0.35],
                [best_y - 0.35],
                marker="*",
                s=120,
                color="#f0c419",
                edgecolor="#202020",
                linewidth=0.8,
                zorder=5,
            )
        for y_index in range(len(conformities)):
            for x_index in range(len(interests)):
                if math.isfinite(matrix[y_index, x_index]):
                    low_support = counts[y_index, x_index] < LOW_SUPPORT_COMPLETED_TRIALS
                    support_label = f"n={counts[y_index, x_index]}{'*' if low_support else ''}"
                    ax.text(
                        x_index,
                        y_index,
                        f"{matrix[y_index, x_index]:.3f}\n{support_label}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#7a3b00" if low_support else "#202020",
                        fontweight="bold" if low_support else "normal",
                    )
    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes.ravel().tolist(),
            label=CRRU_SELECTION_AXIS_LABEL,
            shrink=0.84,
        )
    axes.flat[0].legend(
        handles=[
            Line2D(
                [],
                [],
                marker="*",
                markersize=10,
                linestyle="None",
                markerfacecolor="#f0c419",
                markeredgecolor="#202020",
                label="Gold star = highest mean cell",
            ),
        ],
        loc="upper left",
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(
        "Validation CRRU response by EDGRec branch depth\n"
        f"Cell = mean score; n* = fewer than {LOW_SUPPORT_COMPLETED_TRIALS} completed trials.",
    )
    return _save_figure(fig, output_dir / filename)


def _fanout_label(fanout: tuple[int, ...]) -> str:
    """Return a compact fanout label."""
    return "-".join(str(value) for value in fanout)


def export_fanout_runtime_tradeoffs(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    objective_metric: str,
    filename: str,
) -> Path | None:
    """Export fanout objective and runtime trade-offs."""
    groups = [
        (group, comparable_records)
        for group, group_records in _records_by_group(records)
        if (
            comparable_records := [
                record
                for record in group_records
                if record.get("fanout") is not None
                and _as_float(record.get("objective_percentile")) is not None
            ]
        )
    ]
    if not groups:
        return None

    columns = 2
    rows = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.0, 4.9 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat[len(groups) :]:
        ax.set_visible(False)

    for ax, (group, comparable_records) in zip(axes.flat, groups, strict=False):
        dataset = group[0]
        fanouts = sorted(
            {record["fanout"] for record in comparable_records},
            key=lambda fanout: (len(fanout), sum(fanout), fanout),
        )
        time_scores = _minmax_score(
            [record.get("epoch_time_s") for record in comparable_records],
            lower_is_better=False,
        )
        x_by_fanout = {fanout: index for index, fanout in enumerate(fanouts)}
        for record, time_score in zip(comparable_records, time_scores, strict=True):
            fanout = record["fanout"]
            jitter = ((int(record["trial_number"]) % 7) - 3) * 0.045
            size = 34.0 + 90.0 * (time_score if time_score is not None else 0.3)
            ax.scatter(
                x_by_fanout[fanout] + jitter,
                float(record["objective_percentile"]),
                s=size,
                alpha=0.62,
                color=_color_for_dataset(dataset),
                edgecolor="#202020",
                linewidth=0.3,
            )
        for fanout in fanouts:
            values = [
                float(record["objective_percentile"])
                for record in comparable_records
                if record["fanout"] == fanout
            ]
            if not values:
                continue
            median_value = sorted(values)[len(values) // 2]
            ax.scatter(
                [x_by_fanout[fanout]],
                [median_value],
                marker="D",
                s=58,
                color="#202020",
                edgecolor="#ffffff",
                linewidth=0.6,
                zorder=5,
            )
        best = _best_record(comparable_records)
        if best is not None:
            ax.scatter(
                [x_by_fanout[best["fanout"]]],
                [float(best["objective_percentile"])],
                marker="*",
                s=175,
                color="#f0c419",
                edgecolor="#202020",
                linewidth=0.9,
                zorder=6,
            )
        ax.set_xticks(range(len(fanouts)))
        ax.set_xticklabels([_fanout_label(fanout) for fanout in fanouts], rotation=25, ha="right")
        ax.set_xlabel("Sampled neighbors per GNN layer")
        ax.set_ylabel(CRRU_SELECTION_AXIS_LABEL)
        ax.set_title(_group_title(group))
        ax.grid(alpha=0.20)
    axes.flat[0].legend(
        handles=[
            _circle_legend_handle(4.5, "Smaller circle = shorter epoch"),
            _circle_legend_handle(9.0, "Larger circle = longer epoch"),
            Line2D(
                [],
                [],
                marker="D",
                markersize=7,
                linestyle="None",
                markerfacecolor="#202020",
                markeredgecolor="#ffffff",
                label="Black diamond = median CRRU score",
            ),
            Line2D(
                [],
                [],
                marker="*",
                markersize=10,
                linestyle="None",
                markerfacecolor="#f0c419",
                markeredgecolor="#202020",
                label="Gold star = selected trial",
            ),
        ],
        loc="upper left",
        fontsize=8,
        frameon=True,
    )
    fig.suptitle("Validation CRRU by sampled-neighbor profile")
    return _save_figure(fig, output_dir / filename)


def export_paper_figures(
    studies: Sequence[optuna.Study],
    output_dir: Path,
) -> list[Path]:
    """Export the compact default figure set and remove stale generated figures."""
    _clear_existing_figures(output_dir)
    records = _trial_records(studies)
    if not records:
        return []
    filenames = dict(zip(CRRU_FIGURE_STEMS, PAPER_FIGURE_FILENAMES, strict=True))
    outputs: list[Path | None] = []
    outputs.append(
        export_component_correlations_by_dataset(
            records,
            output_dir,
            objective_metric=VALIDATION_ONLINE_CRRU_METRIC,
            filename=filenames["component_correlations_by_dataset"],
        ),
    )
    outputs.append(
        export_selection_frontier_by_dataset(
            records,
            output_dir,
            objective_metric=VALIDATION_ONLINE_CRRU_METRIC,
            filename=filenames["selection_frontier_by_dataset"],
        ),
    )
    outputs.append(
        export_importance_by_dataset(
            records,
            output_dir,
            objective_metric=VALIDATION_ONLINE_CRRU_METRIC,
            filename=filenames["importance_by_dataset"],
        ),
    )
    outputs.append(
        export_crru_components_by_dataset(
            records,
            output_dir,
            objective_metric=VALIDATION_ONLINE_CRRU_METRIC,
            filename=filenames["components_by_dataset"],
        ),
    )
    outputs.append(
        export_lr_branchmix_landscape(
            records,
            output_dir,
            objective_metric=VALIDATION_ONLINE_CRRU_METRIC,
            filename=filenames["lr_branchmix_landscape"],
        ),
    )
    outputs.append(
        export_branch_depth_heatmaps(
            records,
            output_dir,
            objective_metric=VALIDATION_ONLINE_CRRU_METRIC,
            filename=filenames["branch_depth_heatmaps"],
        ),
    )
    outputs.append(
        export_fanout_runtime_tradeoffs(
            records,
            output_dir,
            objective_metric=VALIDATION_ONLINE_CRRU_METRIC,
            filename=filenames["fanout_runtime_tradeoffs"],
        ),
    )
    return [path for path in outputs if path is not None]


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--study-name", default=None)
    parser.add_argument(
        "--space",
        default=None,
        help="Deprecated no-op; default figures now scan all studies in storage.",
    )
    parser.add_argument("--output-dir", type=Path, default=OPTUNA_FIGURES_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export compact Optuna figures."""
    args = build_parser().parse_args(argv)
    studies = load_studies(args.storage, args.study_name)
    written = export_paper_figures(studies, args.output_dir)
    print(f"Wrote {len(written)} Optuna PNG figure(s) to {args.output_dir.resolve()}")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
