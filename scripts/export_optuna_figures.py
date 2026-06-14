#!/usr/bin/env python
"""Export a compact, paper-oriented Optuna figure set."""

from __future__ import annotations

import argparse
import math
import textwrap
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
from experiments.run_search import DEFAULT_STORAGE, default_study_name, resolve_search_space
from matplotlib.colors import PowerNorm

from scripts.report_optuna_optimization import (
    OPTUNA_FIGURES_DIR,
    attr_float,
    completed_trials,
    dataset_importances,
    dataset_metric,
    dataset_names,
    load_studies,
    logical_trial_params,
)

DEFAULT_SPACE_NAME = "ucagnn-core-optimization"
PAPER_FIGURE_NAMES = (
    "optuna_progress_by_dataset.png",
    "optuna_importance_by_dataset.png",
    "optuna_crru_components_by_dataset.png",
    "optuna_lr_branchmix_landscape.png",
    "optuna_branch_depth_heatmaps.png",
    "optuna_fanout_runtime_tradeoffs.png",
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
    "cagra_candidate_k": "ANN eval candidates",
    "cagra_k": "CAGRA graph neighbors",
    "cagra_out_degree": "CAGRA graph degree",
    "cagra_initial_degree": "CAGRA initial degree",
    "cagra_team_size": "CAGRA team size",
    "cagra_metric": "CAGRA distance metric",
    "cagra_itopk_size": "CAGRA search queue size",
    "hard_negative_ratio": "Hard-negative ratio",
    "dice_sampler_margin": "DICE popularity-gap margin",
    "use_features": "Use item/user features",
    "use_popularity_head": "Use popularity head",
}
PROFILE_FALLBACK_PARAMS = (
    "lr",
    "weight_decay",
    "batch_size",
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
    "cagra_candidate_k",
    "cagra_k",
    "cagra_out_degree",
    "cagra_initial_degree",
    "cagra_team_size",
    "cagra_metric",
    "cagra_itopk_size",
    "hard_negative_ratio",
    "dice_sampler_margin",
    "grad_clip_norm",
)


def _dataset_label(dataset: str) -> str:
    """Return a thesis-facing dataset label."""
    return DATASET_LABELS.get(dataset, dataset)


def _parameter_label(parameter: str, *, width: int = 18) -> str:
    """Return a wrapped thesis-facing parameter label."""
    label = PARAMETER_LABELS.get(parameter, parameter.replace("_", " "))
    return "\n".join(textwrap.wrap(label, width=width))


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Save and close one matplotlib figure without clipping labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    return output_path


def _clear_existing_pngs(output_dir: Path) -> None:
    """Remove stale generated PNGs from the Optuna figure directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.png"):
        path.unlink()


def _load_default_dataset_studies(storage: str, space_name: str) -> list[optuna.Study]:
    """Load the current dataset-local studies for the selected search space."""
    search_space = resolve_search_space(space_name)
    studies: list[optuna.Study] = []
    for dataset in search_space.datasets:
        dataset_space = resolve_search_space(space_name, dataset=dataset)
        study_name = default_study_name(
            space_name,
            dataset_space.datasets,
            search_space=dataset_space,
        )
        try:
            studies.append(optuna.load_study(study_name=study_name, storage=storage))
        except (KeyError, ValueError):
            continue
    return studies


def _study_dataset_pairs(studies: Sequence[optuna.Study]) -> list[tuple[str, optuna.Study]]:
    """Return dataset/study pairs represented by completed trial attrs."""
    pairs: list[tuple[str, optuna.Study]] = []
    seen: set[tuple[str, str]] = set()
    for study in studies:
        datasets = sorted(
            {dataset for trial in completed_trials(study) for dataset in dataset_names(trial)},
        )
        for dataset in datasets:
            key = (dataset, study.study_name)
            if key not in seen:
                pairs.append((dataset, study))
                seen.add(key)
    return pairs


def _objective_points(
    study: optuna.Study,
    dataset: str,
) -> list[tuple[int, float, optuna.trial.FrozenTrial]]:
    """Return trial-indexed dataset objective points for one study/dataset pair."""
    points: list[tuple[int, float, optuna.trial.FrozenTrial]] = []
    trials = sorted(completed_trials(study), key=lambda trial: trial.number)
    for trial in trials:
        value = dataset_metric(trial, dataset, "objective")
        if value is None and len(dataset_names(trial)) <= 1 and trial.value is not None:
            value = float(trial.value)
        if value is not None and math.isfinite(float(value)):
            points.append((len(points) + 1, float(value), trial))
    return points


def _incumbent_values(values: Sequence[float], *, minimize: bool) -> list[float]:
    """Return best-so-far values for an objective series."""
    best = math.inf if minimize else -math.inf
    incumbents: list[float] = []
    for value in values:
        best = min(best, value) if minimize else max(best, value)
        incumbents.append(best)
    return incumbents


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


def _trial_records(pairs: Sequence[tuple[str, optuna.Study]]) -> list[dict[str, Any]]:
    """Return one flat record per dataset-local completed trial."""
    records: list[dict[str, Any]] = []
    for dataset, study in pairs:
        for trial_index, objective, trial in _objective_points(study, dataset):
            params = logical_trial_params(trial)
            records.append(
                {
                    "dataset": dataset,
                    "study_direction": study.direction.name.lower(),
                    "trial_index": trial_index,
                    "trial_number": trial.number,
                    "objective": objective,
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
                    "average_popularity40": _metric_value(trial, dataset, "AveragePopularity@40"),
                    "epoch_time_s": _metric_value(trial, dataset, "epoch_time_s"),
                    "peak_vram_mb": _metric_value(trial, dataset, "peak_vram_mb"),
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


def _finite_values(records: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    """Return finite numeric values for one record key."""
    values: list[float] = []
    for record in records:
        value = _as_float(record.get(key))
        if value is not None:
            values.append(value)
    return values


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
    """Attach CRRU component diagnostics to flat records."""
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[str(record["dataset"])].append(record)

    for dataset_records in by_dataset.values():
        low_pop_scores = _minmax_score(
            [record.get("average_popularity40") for record in dataset_records],
            lower_is_better=True,
        )
        low_time_scores = _minmax_score(
            [record.get("epoch_time_s") for record in dataset_records],
            lower_is_better=True,
        )
        low_vram_scores = _minmax_score(
            [record.get("peak_vram_mb") for record in dataset_records],
            lower_is_better=True,
        )
        objective_percentiles = _minmax_score(
            [record.get("objective") for record in dataset_records],
            lower_is_better=False,
        )
        for index, record in enumerate(dataset_records):
            ndcg = _as_float(record.get("ndcg40"))
            recall = _as_float(record.get("recall40"))
            hit = _as_float(record.get("hit40"))
            personalization = _as_float(record.get("personalization40"))
            low_pop = low_pop_scores[index]
            low_time = low_time_scores[index]
            low_vram = low_vram_scores[index]
            record["low_popularity_score"] = low_pop
            record["resource_efficiency_score"] = _geometric_pair(low_time, low_vram)
            record["objective_percentile"] = objective_percentiles[index]
            if ndcg is None or recall is None or hit is None:
                record["accuracy_component"] = None
            else:
                record["accuracy_component"] = (ndcg**0.50) * (recall**0.35) * (hit**0.15)
            if personalization is None or low_pop is None:
                record["bias_component"] = None
            else:
                record["bias_component"] = (personalization**0.40) * (low_pop**0.60)


def _geometric_pair(left: float | None, right: float | None) -> float | None:
    """Return the geometric mean of two optional scores."""
    if left is None or right is None:
        return None
    return math.sqrt(max(0.0, left) * max(0.0, right))


def _best_record(records: Sequence[Mapping[str, Any]], dataset: str) -> Mapping[str, Any] | None:
    """Return the best stored objective record for a dataset."""
    dataset_records = [record for record in records if record["dataset"] == dataset]
    if not dataset_records:
        return None
    return max(dataset_records, key=lambda record: float(record["objective"]))


def export_progress_by_dataset(
    pairs: Sequence[tuple[str, optuna.Study]],
    output_dir: Path,
) -> Path | None:
    """Export multi-panel dataset-local objective histories."""
    plotted = [(dataset, study, _objective_points(study, dataset)) for dataset, study in pairs]
    plotted = [(dataset, study, points) for dataset, study, points in plotted if points]
    if not plotted:
        return None

    columns = 2
    rows = math.ceil(len(plotted) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.5, 4.2 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat[len(plotted) :]:
        ax.set_visible(False)

    for ax, (dataset, study, points) in zip(axes.flat, plotted, strict=False):
        xs = [index for index, _, _ in points]
        ys = [value for _, value, _ in points]
        color = _color_for_dataset(dataset)
        minimize = study.direction.name.lower() == "minimize"
        best_value = min(ys) if minimize else max(ys)
        best_index = ys.index(best_value)
        ax.scatter(xs, ys, color=color, alpha=0.55, s=28, label="completed trial")
        ax.plot(
            xs,
            _incumbent_values(ys, minimize=minimize),
            color="#202020",
            linewidth=2.0,
            label="best so far",
        )
        ax.scatter(
            [xs[best_index]],
            [best_value],
            marker="*",
            s=160,
            color="#f0c419",
            edgecolor="#202020",
            linewidth=0.8,
            zorder=5,
            label="best",
        )
        ax.set_title(f"{_dataset_label(dataset)}: best objective = {best_value:.4f}")
        ax.set_xlabel("Completed trial index")
        ax.set_ylabel("Validation CRRU objective")
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Dataset-local Optuna progress (higher validation CRRU is better)")
    return _save_figure(fig, output_dir / PAPER_FIGURE_NAMES[0])


def _top_importance_params(
    pairs: Sequence[tuple[str, optuna.Study]],
    *,
    limit: int,
) -> list[str]:
    """Return compact cross-dataset parameter set by mean importance."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for dataset, study in pairs:
        for name, importance in dataset_importances(study, dataset).items():
            totals[name] = totals.get(name, 0.0) + float(importance)
            counts[name] = counts.get(name, 0) + 1
    if not totals:
        return list(PROFILE_FALLBACK_PARAMS[:limit])
    ranked = sorted(
        totals,
        key=lambda name: (-(totals[name] / counts[name]), name),
    )
    return ranked[:limit]


def export_importance_by_dataset(
    pairs: Sequence[tuple[str, optuna.Study]],
    output_dir: Path,
) -> Path | None:
    """Export one dataset-by-parameter importance heatmap."""
    rows: list[tuple[str, dict[str, float]]] = []
    for dataset, study in pairs:
        importances = dataset_importances(study, dataset)
        if importances:
            rows.append((dataset, importances))
    if not rows:
        return None

    params = _top_importance_params(pairs, limit=11)
    matrix = np.array(
        [[importances.get(param, 0.0) for param in params] for _, importances in rows],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(15.0, 5.1), constrained_layout=True)
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="YlGnBu",
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=max(0.01, float(matrix.max()))),
    )
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(
        [_parameter_label(param, width=14) for param in params],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_dataset_label(dataset) for dataset, _ in rows])
    ax.set_title(
        "Hyperparameter importance by dataset\n"
        "Target: dataset-local validation CRRU; fresh homogeneous revisions only",
    )
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            if value >= 0.03:
                ax.text(x_index, y_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Optuna importance share")
    return _save_figure(fig, output_dir / PAPER_FIGURE_NAMES[1])


def export_crru_components_by_dataset(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Export CRRU component-vs-objective diagnostics."""
    component_specs = (
        ("accuracy_component", "Accuracy component\nNDCG/Recall/Hit"),
        ("bias_component", "Bias/diversity component\nPersonalization + low popularity"),
        ("resource_efficiency_score", "Efficiency component\nlower epoch time + lower VRAM"),
    )
    if not records:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), constrained_layout=True)
    plotted_any = False
    for ax, (key, label) in zip(axes.flat[:3], component_specs, strict=True):
        for dataset in sorted({str(record["dataset"]) for record in records}):
            dataset_records = [
                record
                for record in records
                if record["dataset"] == dataset
                and _as_float(record.get(key)) is not None
                and _as_float(record.get("objective")) is not None
            ]
            if not dataset_records:
                continue
            plotted_any = True
            ax.scatter(
                [float(record[key]) for record in dataset_records],
                [float(record["objective"]) for record in dataset_records],
                s=36,
                alpha=0.68,
                color=_color_for_dataset(dataset),
                label=_dataset_label(dataset),
            )
            best = _best_record(dataset_records, dataset)
            if best is not None:
                ax.scatter(
                    [float(best[key])],
                    [float(best["objective"])],
                    marker="*",
                    s=150,
                    color="#f0c419",
                    edgecolor="#202020",
                    linewidth=0.8,
                    zorder=5,
                )
        ax.set_xlabel(label)
        ax.set_ylabel("Validation CRRU objective")
        ax.grid(alpha=0.22)

    axes.flat[3].axis("off")
    axes.flat[3].text(
        0.02,
        0.76,
        "CRRU objective diagnostic",
        fontsize=13,
        fontweight="bold",
        va="top",
        transform=axes.flat[3].transAxes,
    )
    axes.flat[3].text(
        0.02,
        0.54,
        "Accuracy has the largest final weight.\n"
        "Bias/diversity rewards personalization and\n"
        "lower popularity. Efficiency rewards lower\n"
        "time/epoch and lower VRAM.\n\n"
        "Star = best trial for that dataset.",
        fontsize=10,
        linespacing=1.35,
        va="top",
        transform=axes.flat[3].transAxes,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[3].legend(handles, labels, loc="lower left", fontsize=9, frameon=False)
    fig.suptitle("How the validation CRRU objective relates to accuracy, bias, and efficiency")
    return _save_figure(fig, output_dir / PAPER_FIGURE_NAMES[2]) if plotted_any else None


def _sorted_learning_rates(records: Sequence[Mapping[str, Any]]) -> list[float]:
    """Return sorted unique finite learning rates."""
    return sorted({float(record["lr"]) for record in records if _as_float(record.get("lr"))})


def _format_lr(value: float) -> str:
    """Format a learning rate without scientific notation noise."""
    if value >= 0.001:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.5f}".rstrip("0").rstrip(".")


def export_lr_branchmix_landscape(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Export dataset-local LR/branch-mix objective landscapes."""
    datasets = sorted({str(record["dataset"]) for record in records})
    if not datasets:
        return None

    columns = 2
    rows = math.ceil(len(datasets) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.2, 4.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat[len(datasets) :]:
        ax.set_visible(False)

    last_scatter = None
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        dataset_records = [
            record
            for record in records
            if record["dataset"] == dataset
            and _as_float(record.get("lr")) is not None
            and _as_float(record.get("branch_mix_floor")) is not None
            and _as_float(record.get("objective_percentile")) is not None
        ]
        if not dataset_records:
            ax.set_visible(False)
            continue
        lrs = _sorted_learning_rates(dataset_records)
        x_by_lr = {lr: index for index, lr in enumerate(lrs)}
        fanout_budgets = [
            sum(record["fanout"]) if record.get("fanout") else 1 for record in dataset_records
        ]
        max_budget = max(fanout_budgets) if fanout_budgets else 1
        sizes = [38.0 + 90.0 * (budget / max_budget) for budget in fanout_budgets]
        last_scatter = ax.scatter(
            [x_by_lr[float(record["lr"])] for record in dataset_records],
            [float(record["branch_mix_floor"]) for record in dataset_records],
            c=[float(record["objective_percentile"]) for record in dataset_records],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=sizes,
            alpha=0.78,
            edgecolor="#202020",
            linewidth=0.35,
        )
        best = _best_record(dataset_records, dataset)
        if best is not None:
            ax.scatter(
                [x_by_lr[float(best["lr"])]],
                [float(best["branch_mix_floor"])],
                marker="*",
                s=185,
                color="#f0c419",
                edgecolor="#202020",
                linewidth=0.9,
                zorder=5,
            )
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([_format_lr(value) for value in lrs], rotation=35, ha="right")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Minimum branch-mix weight")
        ax.set_title(f"{_dataset_label(dataset)}: objective landscape")
        ax.grid(alpha=0.20)

    if last_scatter is not None:
        fig.colorbar(
            last_scatter,
            ax=axes.ravel().tolist(),
            label="Within-dataset objective percentile",
            shrink=0.82,
        )
    fig.suptitle(
        "Learning-rate and branch-mix search landscape\n"
        "Marker size reflects sampled-neighbor budget; star marks the best trial",
    )
    return _save_figure(fig, output_dir / PAPER_FIGURE_NAMES[3])


def export_branch_depth_heatmaps(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Export mean objective heatmaps for interest/conformity branch depths."""
    datasets = sorted({str(record["dataset"]) for record in records})
    if not datasets:
        return None

    columns = 2
    rows = math.ceil(len(datasets) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.6, 4.7 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat[len(datasets) :]:
        ax.set_visible(False)

    for ax, dataset in zip(axes.flat, datasets, strict=False):
        dataset_records = [
            record
            for record in records
            if record["dataset"] == dataset
            and record.get("interest_depth") is not None
            and record.get("conformity_depth") is not None
        ]
        interests = sorted({int(record["interest_depth"]) for record in dataset_records})
        conformities = sorted({int(record["conformity_depth"]) for record in dataset_records})
        if not interests or not conformities:
            ax.set_visible(False)
            continue
        matrix = np.full((len(conformities), len(interests)), np.nan)
        counts = np.zeros((len(conformities), len(interests)), dtype=int)
        for y_index, conformity in enumerate(conformities):
            for x_index, interest in enumerate(interests):
                values = [
                    float(record["objective"])
                    for record in dataset_records
                    if int(record["interest_depth"]) == interest
                    and int(record["conformity_depth"]) == conformity
                ]
                if values:
                    matrix[y_index, x_index] = float(sum(values) / len(values))
                    counts[y_index, x_index] = len(values)
        image = ax.imshow(matrix, aspect="auto", cmap="BuGn")
        ax.set_xticks(range(len(interests)))
        ax.set_xticklabels([str(value) for value in interests])
        ax.set_yticks(range(len(conformities)))
        ax.set_yticklabels([str(value) for value in conformities])
        ax.set_xlabel("Interest branch GNN layers")
        ax.set_ylabel("Conformity branch GNN layers")
        ax.set_title(f"{_dataset_label(dataset)}: mean objective by branch depth")
        for y_index in range(len(conformities)):
            for x_index in range(len(interests)):
                if math.isfinite(matrix[y_index, x_index]):
                    ax.text(
                        x_index,
                        y_index,
                        f"{matrix[y_index, x_index]:.3f}\nn={counts[y_index, x_index]}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.suptitle("U-CaGNN branch-depth effects; deeper is not automatically better")
    return _save_figure(fig, output_dir / PAPER_FIGURE_NAMES[4])


def _fanout_label(fanout: tuple[int, ...]) -> str:
    """Return a compact fanout label."""
    return "-".join(str(value) for value in fanout)


def export_fanout_runtime_tradeoffs(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Export fanout objective and runtime trade-offs."""
    datasets = sorted({str(record["dataset"]) for record in records})
    if not datasets:
        return None

    columns = 2
    rows = math.ceil(len(datasets) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.0, 4.9 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat[len(datasets) :]:
        ax.set_visible(False)

    for ax, dataset in zip(axes.flat, datasets, strict=False):
        dataset_records = [
            record
            for record in records
            if record["dataset"] == dataset
            and record.get("fanout") is not None
            and _as_float(record.get("objective")) is not None
        ]
        fanouts = sorted(
            {record["fanout"] for record in dataset_records},
            key=lambda fanout: (len(fanout), sum(fanout), fanout),
        )
        if not fanouts:
            ax.set_visible(False)
            continue
        time_scores = _minmax_score(
            [record.get("epoch_time_s") for record in dataset_records],
            lower_is_better=False,
        )
        x_by_fanout = {fanout: index for index, fanout in enumerate(fanouts)}
        for record, time_score in zip(dataset_records, time_scores, strict=True):
            fanout = record["fanout"]
            jitter = ((int(record["trial_number"]) % 7) - 3) * 0.045
            size = 34.0 + 90.0 * (time_score if time_score is not None else 0.3)
            ax.scatter(
                x_by_fanout[fanout] + jitter,
                float(record["objective"]),
                s=size,
                alpha=0.62,
                color=_color_for_dataset(dataset),
                edgecolor="#202020",
                linewidth=0.3,
            )
        for fanout in fanouts:
            values = [
                float(record["objective"])
                for record in dataset_records
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
                zorder=5,
            )
        best = _best_record(dataset_records, dataset)
        if best is not None:
            ax.scatter(
                [x_by_fanout[best["fanout"]]],
                [float(best["objective"])],
                marker="*",
                s=175,
                color="#f0c419",
                edgecolor="#202020",
                linewidth=0.9,
                zorder=6,
            )
        ax.set_xticks(range(len(fanouts)))
        ax.set_xticklabels([_fanout_label(fanout) for fanout in fanouts], rotation=25, ha="right")
        ax.set_xlabel("Neighbor fanout")
        ax.set_ylabel("Validation CRRU objective")
        ax.set_title(f"{_dataset_label(dataset)}: fanout and runtime trade-off")
        ax.grid(alpha=0.20)
    fig.suptitle(
        "Neighbor fanout effects for sampled GNN training\n"
        "Larger circles are slower epochs within each dataset; diamond = fanout median",
    )
    return _save_figure(fig, output_dir / PAPER_FIGURE_NAMES[5])


def export_paper_figures(
    studies: Sequence[optuna.Study],
    output_dir: Path,
) -> list[Path]:
    """Export the compact default figure set and remove stale generated PNGs."""
    _clear_existing_pngs(output_dir)
    pairs = _study_dataset_pairs(studies)
    records = _trial_records(pairs)
    outputs = [
        export_progress_by_dataset(pairs, output_dir),
        export_importance_by_dataset(pairs, output_dir),
        export_crru_components_by_dataset(records, output_dir),
        export_lr_branchmix_landscape(records, output_dir),
        export_branch_depth_heatmaps(records, output_dir),
        export_fanout_runtime_tradeoffs(records, output_dir),
    ]
    return [path for path in outputs if path is not None]


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--space", default=DEFAULT_SPACE_NAME)
    parser.add_argument("--output-dir", type=Path, default=OPTUNA_FIGURES_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export compact Optuna figures."""
    args = build_parser().parse_args(argv)
    if args.study_name:
        studies = load_studies(args.storage, args.study_name)
    else:
        studies = _load_default_dataset_studies(args.storage, args.space)
        if not studies:
            studies = load_studies(args.storage)
    written = export_paper_figures(studies, args.output_dir)
    print(f"Wrote {len(written)} paper-ready Optuna figure(s) to {args.output_dir.resolve()}")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
