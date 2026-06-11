#!/usr/bin/env python
"""Export Optuna optimization figures from Optuna RDB storage."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import optuna
from experiments.run_search import DEFAULT_STORAGE

from scripts.report_optuna_optimization import (
    OPTUNA_FIGURES_DIR,
    average_trial_attr,
    completed_trials,
    dashboard_importances,
    dataset_metric,
    dataset_names,
    fanova_importances,
    load_studies,
    trial_sort_key,
)


def slugify(value: str) -> str:
    """Return a file-name-safe slug."""
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in normalized.split("-") if part) or "study"


def objective_values(study: optuna.Study) -> tuple[list[int], list[float]]:
    """Return completed trial numbers and objective values."""
    trials = completed_trials(study)
    return [trial.number for trial in trials], [float(trial.value) for trial in trials]


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save and close one figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_optimization_history(study: optuna.Study, output_path: Path) -> bool:
    """Plot objective history and incumbent best value."""
    xs, ys = objective_values(study)
    if not xs:
        return False

    minimize = study.direction.name.lower() == "minimize"
    best_values: list[float] = []
    incumbent = math.inf if minimize else -math.inf
    for value in ys:
        incumbent = min(incumbent, value) if minimize else max(incumbent, value)
        best_values.append(incumbent)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, marker="o", linewidth=1.2, label="trial objective", alpha=0.75)
    ax.plot(xs, best_values, linewidth=2.0, label="best so far")
    failed_xs = [trial.number for trial in study.trials if trial.state.name == "FAIL"]
    if failed_xs:
        rug_y = min(ys) - (max(ys) - min(ys) or 1.0) * 0.04
        ax.scatter(failed_xs, [rug_y] * len(failed_xs), marker="x", color="#9b2f2f", label="failed")
    ax.set_title(f"Optuna optimization history: {study.study_name}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective value")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_path)
    return True


def export_param_importances(study: optuna.Study, output_path: Path) -> bool:
    """Plot dashboard-like and deterministic fANOVA parameter importances."""
    default = dashboard_importances(study)
    fanova = fanova_importances(study)
    names = list(default)[:12]
    if not names:
        return False

    y_positions = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.35)))
    ax.barh(
        [position - 0.18 for position in y_positions],
        [default.get(name, 0.0) for name in names],
        height=0.34,
        color="#2f6f73",
        label="Optuna default/dashboard-like",
    )
    ax.barh(
        [position + 0.18 for position in y_positions],
        [fanova.get(name, 0.0) for name in names],
        height=0.34,
        color="#b56b45",
        label="deterministic fANOVA",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names)
    ax.set_title(f"Hyperparameter importances: {study.study_name}")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    save_figure(fig, output_path)
    return True


def export_trial_state_counts(study: optuna.Study, output_path: Path) -> bool:
    """Plot trial states so wasted compute is visible."""
    counts: dict[str, int] = {}
    for trial in study.trials:
        counts[trial.state.name.lower()] = counts.get(trial.state.name.lower(), 0) + 1
    if not counts:
        return False

    labels = sorted(counts)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, [counts[label] for label in labels], color="#4d6073")
    ax.set_title(f"Trial state counts: {study.study_name}")
    ax.set_ylabel("Trials")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path)
    return True


def export_objective_vs_attr(
    study: optuna.Study,
    *,
    suffix: str,
    label: str,
    output_path: Path,
) -> bool:
    """Plot objective against an averaged dataset-scoped user attribute."""
    points = [
        (average_trial_attr(trial, suffix), float(trial.value), trial.number)
        for trial in completed_trials(study)
        if average_trial_attr(trial, suffix) is not None and trial.value is not None
    ]
    if not points:
        return False

    xs = [point[0] for point in points if point[0] is not None]
    ys = [point[1] for point in points]
    labels = [point[2] for point in points]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, color="#7a4d1d", alpha=0.8)
    for x_value, y_value, trial_number in zip(xs, ys, labels, strict=True):
        ax.annotate(str(trial_number), (x_value, y_value), fontsize=7, alpha=0.65)
    ax.set_title(f"Objective vs {label}: {study.study_name}")
    ax.set_xlabel(label)
    ax.set_ylabel("Objective value")
    ax.grid(alpha=0.25)
    save_figure(fig, output_path)
    return True


def export_objective_vs_dataset_metric(
    study: optuna.Study,
    *,
    metric_name: str,
    output_path: Path,
) -> bool:
    """Plot objective against the mean dataset validation metric."""
    points: list[tuple[float, float, int]] = []
    for trial in completed_trials(study):
        values = [
            dataset_metric(trial, dataset, metric_name)
            for dataset in dataset_names(trial)
            if dataset_metric(trial, dataset, metric_name) is not None
        ]
        if not values or trial.value is None:
            continue
        points.append((sum(values) / len(values), float(trial.value), trial.number))
    if not points:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([point[0] for point in points], [point[1] for point in points], alpha=0.8)
    for x_value, y_value, trial_number in points:
        ax.annotate(str(trial_number), (x_value, y_value), fontsize=7, alpha=0.65)
    ax.set_title(f"Objective vs validation {metric_name}: {study.study_name}")
    ax.set_xlabel(f"Mean validation {metric_name}")
    ax.set_ylabel("Objective value")
    ax.grid(alpha=0.25)
    save_figure(fig, output_path)
    return True


def export_param_slices(study: optuna.Study, output_path: Path) -> bool:
    """Plot objective slices for the six most important parameters."""
    importances = dashboard_importances(study)
    params = list(importances)[:6]
    if not params:
        return False

    trials = sorted(completed_trials(study), key=lambda trial: trial_sort_key(study, trial))
    fig, axes = plt.subplots(len(params), 1, figsize=(9, max(4, 2.2 * len(params))))
    if len(params) == 1:
        axes = [axes]

    for ax, param_name in zip(axes, params, strict=True):
        raw_points = [
            (trial.params[param_name], float(trial.value))
            for trial in trials
            if param_name in trial.params and trial.value is not None
        ]
        if raw_points and all(isinstance(point[0], bool | int | float) for point in raw_points):
            points = [(float(value), str(value), objective) for value, objective in raw_points]
        else:
            labels_by_value = {
                str(value): index
                for index, value in enumerate(sorted({str(point[0]) for point in raw_points}))
            }
            points = [
                (float(labels_by_value[str(value)]), str(value), objective)
                for value, objective in raw_points
            ]
        if not points:
            ax.set_visible(False)
            continue
        xs = [point[0] for point in points]
        labels = [point[1] for point in points]
        ys = [point[2] for point in points]
        ax.scatter(xs, ys, alpha=0.8)
        if not all(label.replace(".", "", 1).replace("-", "", 1).isdigit() for label in labels):
            unique = sorted(set(zip(xs, labels, strict=True)))
            ax.set_xticks([value for value, _ in unique])
            ax.set_xticklabels([label for _, label in unique], rotation=30, ha="right")
        ax.set_title(param_name)
        ax.set_ylabel("Objective")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Parameter value")
    fig.suptitle(f"Objective slices: {study.study_name}", y=1.0)
    save_figure(fig, output_path)
    return True


def export_top_parallel_coordinates(study: optuna.Study, output_path: Path) -> bool:
    """Plot normalized top-trial parameter profiles for the most important knobs."""
    importances = dashboard_importances(study)
    params = list(importances)[:6]
    trials = sorted(completed_trials(study), key=lambda trial: trial_sort_key(study, trial))[:10]
    if len(params) < 2 or len(trials) < 2:
        return False

    values_by_param: dict[str, list[object]] = {
        param: [trial.params[param] for trial in trials if param in trial.params]
        for param in params
    }
    if any(not values for values in values_by_param.values()):
        return False

    normalized_by_trial: list[list[float]] = []
    for trial in trials:
        row: list[float] = []
        for param in params:
            value = trial.params.get(param)
            values = values_by_param[param]
            if all(isinstance(item, bool | int | float) for item in values):
                numeric_values = [float(item) for item in values]
                lo, hi = min(numeric_values), max(numeric_values)
                scale = (hi - lo) or 1.0
                row.append((float(value) - lo) / scale)
            else:
                labels = sorted({str(item) for item in values})
                index = labels.index(str(value))
                row.append(index / ((len(labels) - 1) or 1))
        normalized_by_trial.append(row)

    fig, ax = plt.subplots(figsize=(10, 5))
    objectives = [float(trial.value) for trial in trials if trial.value is not None]
    lo_obj, hi_obj = min(objectives), max(objectives)
    obj_scale = (hi_obj - lo_obj) or 1.0
    for trial, row in zip(trials, normalized_by_trial, strict=True):
        objective = float(trial.value) if trial.value is not None else lo_obj
        color_value = (objective - lo_obj) / obj_scale
        ax.plot(
            range(len(params)),
            row,
            marker="o",
            alpha=0.75,
            color=plt.cm.viridis(color_value),
            label=f"trial {trial.number}",
        )
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, rotation=25, ha="right")
    ax.set_ylabel("Normalized parameter value")
    ax.set_title(f"Top completed trial parameter profiles: {study.study_name}")
    ax.grid(alpha=0.25)
    save_figure(fig, output_path)
    return True


def export_study_figures(study: optuna.Study, output_dir: Path) -> list[Path]:
    """Export all supported figures for one study."""
    slug = slugify(study.study_name)
    outputs: list[tuple[str, bool]] = [
        (
            f"{slug}_optimization_history.png",
            export_optimization_history(study, output_dir / f"{slug}_optimization_history.png"),
        ),
        (
            f"{slug}_param_importances.png",
            export_param_importances(study, output_dir / f"{slug}_param_importances.png"),
        ),
        (
            f"{slug}_trial_state_counts.png",
            export_trial_state_counts(study, output_dir / f"{slug}_trial_state_counts.png"),
        ),
        (
            f"{slug}_objective_vs_training_time.png",
            export_objective_vs_attr(
                study,
                suffix="training_time_s",
                label="average training time (s)",
                output_path=output_dir / f"{slug}_objective_vs_training_time.png",
            ),
        ),
        (
            f"{slug}_objective_vs_peak_vram.png",
            export_objective_vs_attr(
                study,
                suffix="peak_vram_mb",
                label="average peak VRAM (MB)",
                output_path=output_dir / f"{slug}_objective_vs_peak_vram.png",
            ),
        ),
        (
            f"{slug}_objective_vs_avgpop40.png",
            export_objective_vs_dataset_metric(
                study,
                metric_name="AveragePopularity@40",
                output_path=output_dir / f"{slug}_objective_vs_avgpop40.png",
            ),
        ),
        (
            f"{slug}_objective_vs_ndcg40.png",
            export_objective_vs_dataset_metric(
                study,
                metric_name="NDCG@40",
                output_path=output_dir / f"{slug}_objective_vs_ndcg40.png",
            ),
        ),
        (
            f"{slug}_objective_vs_recall40.png",
            export_objective_vs_dataset_metric(
                study,
                metric_name="Recall@40",
                output_path=output_dir / f"{slug}_objective_vs_recall40.png",
            ),
        ),
        (
            f"{slug}_objective_vs_hit40.png",
            export_objective_vs_dataset_metric(
                study,
                metric_name="HitRatio@40",
                output_path=output_dir / f"{slug}_objective_vs_hit40.png",
            ),
        ),
        (
            f"{slug}_objective_vs_personalization40.png",
            export_objective_vs_dataset_metric(
                study,
                metric_name="Personalization@40",
                output_path=output_dir / f"{slug}_objective_vs_personalization40.png",
            ),
        ),
        (
            f"{slug}_param_slices.png",
            export_param_slices(study, output_dir / f"{slug}_param_slices.png"),
        ),
        (
            f"{slug}_top_parallel_coordinates.png",
            export_top_parallel_coordinates(
                study,
                output_dir / f"{slug}_top_parallel_coordinates.png",
            ),
        ),
    ]
    return [output_dir / filename for filename, written in outputs if written]


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=OPTUNA_FIGURES_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export Optuna figures."""
    args = build_parser().parse_args(argv)
    studies = load_studies(args.storage, args.study_name)
    written: list[Path] = []
    for study in studies:
        written.extend(export_study_figures(study, args.output_dir))
    print(f"Wrote {len(written)} Optuna figure(s) to {args.output_dir.resolve()}")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
