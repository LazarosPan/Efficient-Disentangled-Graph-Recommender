#!/usr/bin/env python
"""Write a thesis-facing Optuna optimization report from Optuna RDB storage."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import optuna
from experiments.run_search import DEFAULT_STORAGE
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.trial import FrozenTrial, TrialState
from src.utils.crru import VALIDATION_ONLINE_CRRU_K_METRICS, VALIDATION_ONLINE_CRRU_METRIC
from src.utils.project_paths import RESULTS_DIR

OPTUNA_OPTIMIZATION_MARKDOWN_PATH = RESULTS_DIR / "optuna_optimization.md"
OPTUNA_FIGURES_DIR = RESULTS_DIR / "optuna_figures"
DEFAULT_TOP_N = 10
PARAMETER_PRIORITY = (
    "lr",
    "weight_decay",
    "lr_scheduler",
    "lr_scheduler_factor",
    "num_neighbors",
    "num_neighbors_depth_1",
    "num_neighbors_depth_2",
    "num_neighbors_depth_3",
    "interest_gnn_layers",
    "conformity_gnn_layers",
    "dropout",
    "score_mix_min_weight",
    "loss_weight_independence",
    "loss_weight_contrastive",
    "loss_weight_popularity",
    "hard_negative_ratio",
    "dice_sampler_margin",
    "grad_clip_norm",
    "use_features",
    "use_popularity_head",
)
DATASET_METRICS = (
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
    VALIDATION_ONLINE_CRRU_K_METRICS[20],
    VALIDATION_ONLINE_CRRU_K_METRICS[40],
    VALIDATION_ONLINE_CRRU_METRIC,
)


def load_studies(storage: str, study_name: str | None = None) -> list[optuna.Study]:
    """Load one or all studies from Optuna RDB storage."""
    if study_name:
        return [optuna.load_study(study_name=study_name, storage=storage)]
    return [
        optuna.load_study(study_name=summary.study_name, storage=storage)
        for summary in optuna.get_all_study_summaries(storage=storage)
    ]


def completed_trials(study: optuna.Study) -> list[FrozenTrial]:
    """Return completed single-objective trials with finite values."""
    return [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    ]


def trial_sort_key(study: optuna.Study, trial: FrozenTrial) -> tuple[float, int]:
    """Sort trials according to the study direction."""
    value = float(trial.value) if trial.value is not None else float("nan")
    direction = study.direction.name.lower()
    objective_rank = value if direction == "minimize" else -value
    return objective_rank, int(trial.number)


def ordered_params(params: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return sampled params in a stable thesis-friendly order."""
    keys = [key for key in PARAMETER_PRIORITY if key in params]
    keys.extend(sorted(key for key in params if key not in set(PARAMETER_PRIORITY)))
    return [(key, params[key]) for key in keys]


def logical_param_name(param_name: str) -> str:
    """Return the thesis-facing config field for an Optuna storage parameter."""
    base_name = param_name.split("__", 1)[0]
    if base_name.startswith("num_neighbors_depth_"):
        return "num_neighbors"
    return base_name


def logical_trial_params(trial: FrozenTrial) -> Mapping[str, Any]:
    """Return logical sampled params, preferring stored U-CaGNN config fields."""
    sampled_params = trial.user_attrs.get("sampled_params")
    if isinstance(sampled_params, Mapping):
        return sampled_params
    return {logical_param_name(key): value for key, value in trial.params.items()}


def logical_importances(importances: Mapping[str, float]) -> dict[str, float]:
    """Coalesce storage-level Optuna importance names into logical config fields."""
    coalesced: dict[str, float] = {}
    for name, importance in importances.items():
        logical_name = logical_param_name(name)
        coalesced[logical_name] = coalesced.get(logical_name, 0.0) + float(importance)
    return dict(sorted(coalesced.items(), key=lambda item: item[1], reverse=True))


def format_param_value(value: Any) -> str:
    """Return a full, non-truncated parameter value."""
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def format_params(params: Mapping[str, Any]) -> str:
    """Return all parameters without truncation."""
    if not params:
        return "-"
    return ", ".join(f"{key}={format_param_value(value)}" for key, value in ordered_params(params))


def attr_float(trial: FrozenTrial, key: str) -> float | None:
    """Return a finite float user attribute when present."""
    value = trial.user_attrs.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def dataset_names(trial: FrozenTrial) -> list[str]:
    """Return datasets represented by trial user attributes."""
    names = {
        key.rsplit(".", 1)[0]
        for key in trial.user_attrs
        if key.endswith(".objective") and "." in key
    }
    return sorted(names)


def dataset_metric(trial: FrozenTrial, dataset: str, metric_name: str) -> float | None:
    """Return a dataset validation metric stored in Optuna user attrs."""
    if metric_name == "objective":
        return attr_float(trial, f"{dataset}.objective")
    return attr_float(trial, f"{dataset}.val.{metric_name}")


def average_trial_attr(trial: FrozenTrial, suffix: str) -> float | None:
    """Average one dataset-scoped user attribute suffix across available datasets."""
    values = [
        attr_float(trial, f"{dataset}.{suffix}")
        for dataset in dataset_names(trial)
        if attr_float(trial, f"{dataset}.{suffix}") is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def format_float(value: float | None, digits: int = 6) -> str:
    """Return a compact numeric string."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def safe_importances(
    study: optuna.Study,
    *,
    evaluator: object | None = None,
) -> dict[str, float]:
    """Return Optuna parameter importances when enough completed trials exist."""
    if len(completed_trials(study)) < 2:
        return {}
    try:
        if evaluator is None:
            importances = get_param_importances(study)
        else:
            importances = get_param_importances(study, evaluator=evaluator)
        return {name: float(value) for name, value in importances.items()}
    except Exception:
        return {}


def dashboard_importances(study: optuna.Study) -> dict[str, float]:
    """Return Optuna's default importance view, matching dashboard target semantics."""
    return logical_importances(safe_importances(study))


def fanova_importances(study: optuna.Study) -> dict[str, float]:
    """Return deterministic fANOVA importances for sensitivity checking."""
    return logical_importances(
        safe_importances(study, evaluator=FanovaImportanceEvaluator(seed=13)),
    )


def failure_label(trial: FrozenTrial) -> str:
    """Return a compact failure reason for report grouping."""
    reason = trial.user_attrs.get("failure_reason")
    stage = trial.user_attrs.get("failure_stage")
    if reason:
        return f"{stage or 'unknown'}: {reason}"
    return "legacy failure before stored attrs; exact exception unavailable in Optuna RDB"


def render_failure_summary(study: optuna.Study) -> list[str]:
    """Render failed-trial diagnostics from stored Optuna attributes."""
    failed_trials = [trial for trial in study.trials if trial.state == TrialState.FAIL]
    if not failed_trials:
        return []
    counts = Counter(failure_label(trial) for trial in failed_trials)
    example_by_reason = {
        reason: next(trial.number for trial in failed_trials if failure_label(trial) == reason)
        for reason in counts
    }
    lines = [
        "### Failed-trial diagnostics",
        "",
        "| Count | Example trial | Stored reason |",
        "|---:|---:|---|",
    ]
    for reason, count in counts.most_common():
        lines.append(f"| {count} | {example_by_reason[reason]} | `{reason}` |")
    lines.extend(
        [
            "",
            "Legacy failures without stored attributes happened before the current failure "
            "recorder could write `failure_stage` / `failure_reason`; the old runs cannot "
            "be reconstructed from Optuna RDB alone.",
            "",
        ],
    )
    return lines


def render_importance_table(title: str, importances: Mapping[str, float]) -> list[str]:
    """Render one importance table."""
    if not importances:
        return []
    lines = [
        f"### {title}",
        "",
        "| Rank | Parameter | Importance |",
        "|---:|---|---:|",
    ]
    for rank, (name, importance) in enumerate(importances.items(), start=1):
        lines.append(f"| {rank} | `{name}` | {importance:.6f} |")
    lines.append("")
    return lines


def render_study_report(study: optuna.Study, *, top_n: int) -> str:
    """Render one Optuna study as markdown."""
    trials = completed_trials(study)
    state_counts = Counter(trial.state.name.lower() for trial in study.trials)
    objective_metric = "-"
    objective_split = "-"
    if trials:
        objective_metric = str(trials[0].user_attrs.get("objective_metric", "-"))
        objective_split = str(trials[0].user_attrs.get("objective_split", "-"))

    lines = [
        f"## Study: `{study.study_name}`",
        "",
        f"- Direction: `{study.direction.name.lower()}`",
        f"- Objective: `{objective_split} {objective_metric}`",
        f"- Trials: {len(study.trials)} total, {len(trials)} completed, "
        f"{state_counts.get('fail', 0)} failed, {state_counts.get('running', 0)} running, "
        f"{state_counts.get('pruned', 0)} pruned",
        "",
    ]
    if not trials:
        lines.extend(["No completed trials.", ""])
        return "\n".join(lines)

    best_trial = sorted(trials, key=lambda trial: trial_sort_key(study, trial))[0]
    lines.extend(
        [
            "### Best trial",
            "",
            f"- Trial: `{best_trial.number}`",
            f"- Objective value: `{format_float(float(best_trial.value))}`",
            f"- Parameters: `{format_params(logical_trial_params(best_trial))}`",
            "",
        ],
    )

    datasets = dataset_names(best_trial)
    if datasets:
        lines.extend(
            [
                "### Best trial dataset metrics",
                "",
                "| Dataset | Objective | NDCG@20 | Recall@20 | Hit@20 | Pers@20 | AvgPop@20 | "
                "NDCG@40 | Recall@40 | Hit@40 | Pers@40 | AvgPop@40 | "
                "ValCRRU@20 | ValCRRU@40 | ValCRRU@20_40 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ],
        )
        for dataset in datasets:
            values = [dataset_metric(best_trial, dataset, metric) for metric in DATASET_METRICS]
            lines.append(
                f"| {dataset} | {format_float(dataset_metric(best_trial, dataset, 'objective'))} | "
                + " | ".join(format_float(value) for value in values[:-1])
                + f" | {format_float(values[-1])} |",
            )
        lines.append("")

    default_importances = dashboard_importances(study)
    if default_importances:
        if len(trials) < 10:
            lines.extend(
                [
                    "Importance warning: fewer than 10 completed trials are available, "
                    "so rankings are unstable and should be treated as diagnostic only.",
                    "",
                ],
            )
        lines.extend(
            render_importance_table(
                "Dashboard-like Optuna importances",
                default_importances,
            ),
        )
    lines.extend(
        render_importance_table(
            "Deterministic fANOVA importance sensitivity",
            fanova_importances(study),
        ),
    )

    lines.extend(render_failure_summary(study))

    lines.extend(
        [
            f"### Top {min(top_n, len(trials))} completed trials",
            "",
            "| Rank | Trial | Objective | Avg train time (s) | Avg peak VRAM (MB) | Parameters |",
            "|---:|---:|---:|---:|---:|---|",
        ],
    )
    for rank, trial in enumerate(
        sorted(trials, key=lambda item: trial_sort_key(study, item))[:top_n],
        start=1,
    ):
        lines.append(
            f"| {rank} | {trial.number} | {format_float(float(trial.value))} | "
            f"{format_float(average_trial_attr(trial, 'training_time_s'), digits=2)} | "
            f"{format_float(average_trial_attr(trial, 'peak_vram_mb'), digits=1)} | "
            f"`{format_params(logical_trial_params(trial))}` |",
        )
    lines.append("")

    lines.extend(
        [
            "### Formal-promotion candidates",
            "",
            "Promote the top 3 completed trials into formal profiles only after checking "
            "dataset-level balance, runtime, and whether popularity diagnostics are acceptable.",
            "",
        ],
    )
    for trial in sorted(trials, key=lambda item: trial_sort_key(study, item))[:3]:
        lines.append(
            f"- Trial `{trial.number}`: objective `{format_float(float(trial.value))}`; "
            f"params `{format_params(logical_trial_params(trial))}`",
        )
    lines.append("")
    return "\n".join(lines)


def render_report(studies: Sequence[optuna.Study], *, storage: str, top_n: int) -> str:
    """Render all Optuna studies as markdown."""
    lines = [
        "# Optuna Optimization Report",
        "",
        "Generated by `uv run scripts/report_optuna_optimization.py`.",
        "",
        f"- Storage: `{storage}`",
        f"- Figure directory: `{OPTUNA_FIGURES_DIR}`",
        "- Dashboard: `uv run optuna-dashboard sqlite:///results/optuna_studies.db`",
        "",
        "## Interpretation notes",
        "",
        "- Search objectives use validation metrics only; test metrics remain for formal reruns.",
        "- New search-space definitions use `ValidationOnlineCRRU@20_40`, an online validation "
        "proxy with the same CRRU component/exponent structure as thesis CRRU.",
        "- Exact thesis CRRU uses dataset-local section-row min-max normalization and is "
        "recomputed after rows exist; it is not a stable live Optuna objective because "
        "future trials change the min/max range.",
        "- Historical studies may still show `val NDCG@40` or other older objectives.",
        "- Lower average popularity and higher personalization are supporting diagnostics, "
        "not causal-effect estimates.",
        "- If Optuna dashboard importances differ, first confirm the same storage URI, "
        "study name, objective target, and completed-trial subset.",
        "- Optuna RDB storage is the canonical owner for search trials; the thesis SQLite "
        "database keeps formal experiment and training logs.",
        "",
    ]
    if not studies:
        lines.extend(["No Optuna studies found.", ""])
    for study in studies:
        lines.append(render_study_report(study, top_n=top_n))
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--output", type=Path, default=OPTUNA_OPTIMIZATION_MARKDOWN_PATH)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the Optuna optimization markdown report."""
    args = build_parser().parse_args(argv)
    studies = load_studies(args.storage, args.study_name)
    report = render_report(studies, storage=args.storage, top_n=args.top_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote Optuna optimization report to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
