"""Dataset-local feature-subset reporting helpers."""

from __future__ import annotations

import csv
import json
import math
import textwrap
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.data.feature_groups import (
    GRAPH_ONLY_PROFILE,
    feature_subset_profile_group_labels,
    loaded_thesis_safe_item_feature_groups,
    loaded_thesis_safe_item_feature_groups_for_dataset,
    required_feature_subset_profiles,
)
from src.data.loaders import load_dataset
from src.utils.crru import (
    VALIDATION_ACCURACY_METRIC,
    VALIDATION_ONLINE_CRRU_K_METRICS,
    VALIDATION_ONLINE_CRRU_METRIC,
    compute_crru_efficiency_scores,
    compute_crru_scores_for_k,
    compute_validation_accuracy_objective,
    compute_validation_online_crru_for_k,
    compute_validation_online_crru_objective,
)
from src.utils.experiment_logger import RUNTIME_PROBE_METRIC_NAMES
from src.utils.project_paths import RESULTS_DIR

FEATURE_ANALYSIS_DIR = RESULTS_DIR / "feature_analysis"
FEATURE_SUBSET_SEARCH_SPACE = "edgrec-feature-subset-search"
FEATURE_SUBSET_DATASETS = ("amazonbook", "movielens1m", "kuairec_v2", "kuairand1k")
FEATURE_SAFE_ROLE = "safe_pre_treatment"
FEATURE_SUBSET_RESULT_COLUMNS = (
    ("dataset", "Dataset"),
    ("feature_subset_profile", "FeatureSubset"),
    ("included_groups", "IncludedGroups"),
    ("excluded_groups", "ExcludedGroups"),
    ("source_objective", "Source objective"),
    ("validation_accuracy_20_40", "ValidationAccuracy@20_40"),
    ("ndcg_20", "NDCG@20"),
    ("recall_20", "Recall@20"),
    ("hit_20", "Hit@20"),
    ("personalization_20", "Pers@20"),
    ("avgpop_20", "AvgPop@20"),
    ("ndcg_40", "NDCG@40"),
    ("recall_40", "Recall@40"),
    ("hit_40", "Hit@40"),
    ("personalization_40", "Pers@40"),
    ("avgpop_40", "AvgPop@40"),
    ("online_crru_20", "OnlineCRRU@20"),
    ("online_crru_40", "OnlineCRRU@40"),
    ("online_crru_20_40", "OnlineCRRU@20_40"),
    ("posthoc_crru_20", "PosthocCRRU@20"),
    ("posthoc_crru_40", "PosthocCRRU@40"),
    ("time_per_epoch_s", "Time/epoch (s)"),
    ("peak_vram_mb", "Peak VRAM (MB)"),
    ("batch", "Batch"),
    ("completed_trials", "CompletedTrials"),
    ("status", "Status"),
)
STALE_FEATURE_EFFECT_FILES = (
    "feature_inventory.csv",
    "feature_inventory.md",
    "feature_group_summary.csv",
    "feature_group_summary.md",
    "feature_ablation_results.csv",
    "feature_ablation_results.md",
    "feature_ablation_delta_heatmap.png",
    "feature_importance_raw.csv",
    "feature_importance_summary.csv",
    "feature_importance.md",
    "feature_importance_ndcg20.png",
    "feature_importance_crru20.png",
    "feature_gate_diagnostics.csv",
    "feature_gate_diagnostics.md",
    "feature_gate_projection_norms.png",
    "score_mix_context_diagnostics.png",
    "feature_optuna_importance.csv",
    "feature_optuna_importance.md",
    "feature_optuna_importance.png",
    "README.md",
)
PRIMARY_METRIC_NAMES = (
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
METRIC_TO_COLUMN = {
    "NDCG@20": "ndcg_20",
    "Recall@20": "recall_20",
    "HitRatio@20": "hit_20",
    "Personalization@20": "personalization_20",
    "AveragePopularity@20": "avgpop_20",
    "NDCG@40": "ndcg_40",
    "Recall@40": "recall_40",
    "HitRatio@40": "hit_40",
    "Personalization@40": "personalization_40",
    "AveragePopularity@40": "avgpop_40",
}


def _ensure_dir() -> None:
    FEATURE_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def _remove_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def remove_stale_feature_effect_reports() -> None:
    """Remove legacy artifacts replaced by feature-subset search reports."""
    for filename in STALE_FEATURE_EFFECT_FILES:
        _remove_file(FEATURE_ANALYSIS_DIR / filename)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> None:
    _ensure_dir()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_labeled_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[tuple[str, str]],
) -> None:
    export_rows = [{label: row.get(key, "") for key, label in columns} for row in rows]
    _write_csv(path, export_rows, [label for _key, label in columns])


def _format_float(value: object, digits: int = 6) -> str:
    number = _finite_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def _format_markdown_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return _format_float(value)
    return str(value).replace("|", "\\|")


def _finite_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _feature_groups_label(groups: Sequence[str]) -> str:
    return ",".join(groups)


def _trial_attrs(trial: object) -> Mapping[str, Any]:
    attrs = getattr(trial, "user_attrs", {})
    return attrs if isinstance(attrs, Mapping) else {}


def _trial_state_name(trial: object) -> str:
    state = getattr(trial, "state", None)
    return str(getattr(state, "name", state))


def _trial_number(trial: object, fallback: int) -> int:
    value = getattr(trial, "number", fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _trial_value(trial: object) -> float | None:
    return _finite_float(getattr(trial, "value", None))


def _json_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _trial_dataset(trial: object) -> str | None:
    attrs = _trial_attrs(trial)
    datasets = attrs.get("datasets")
    if isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)) and len(datasets):
        return str(datasets[0])
    dataset_keys = sorted(
        key.split(".", 1)[0]
        for key in attrs
        if isinstance(key, str) and key.endswith(".objective") and "." in key
    )
    return dataset_keys[0] if len(dataset_keys) == 1 else None


def _sampled_params(trial: object) -> Mapping[str, Any]:
    return _json_mapping(_trial_attrs(trial).get("sampled_params"))


def _profile_from_trial(trial: object) -> str | None:
    profile = _sampled_params(trial).get("feature_subset_profile")
    return str(profile) if profile else None


def _attr_float(trial: object, key: str) -> float | None:
    return _finite_float(_trial_attrs(trial).get(key))


def _trial_epoch_time_s(trial: object, dataset: str) -> float | None:
    explicit = _attr_float(trial, f"{dataset}.avg_epoch_time_s")
    if explicit is not None and explicit > 0:
        return explicit
    training_time = _attr_float(trial, f"{dataset}.training_time_s")
    epochs = _attr_float(trial, f"{dataset}.epochs_stopped_at")
    if training_time is None or epochs is None or epochs <= 0:
        return None
    return training_time / epochs


def _metric_value(trial: object, dataset: str, metric: str) -> float | None:
    stored = _attr_float(trial, f"{dataset}.val.{metric}")
    if stored is not None:
        return stored
    metrics = _primary_metrics(trial, dataset)
    if len(metrics) != len(PRIMARY_METRIC_NAMES):
        return None
    try:
        if metric == VALIDATION_ACCURACY_METRIC:
            return compute_validation_accuracy_objective(metrics)
        if metric == VALIDATION_ONLINE_CRRU_METRIC:
            return compute_validation_online_crru_objective(
                metrics,
                peak_vram_mb=_attr_float(trial, f"{dataset}.peak_vram_mb"),
                epoch_time_s=_trial_epoch_time_s(trial, dataset),
            )
        for k, name in VALIDATION_ONLINE_CRRU_K_METRICS.items():
            if metric == name:
                return compute_validation_online_crru_for_k(
                    metrics,
                    k=k,
                    peak_vram_mb=_attr_float(trial, f"{dataset}.peak_vram_mb"),
                    epoch_time_s=_trial_epoch_time_s(trial, dataset),
                )
    except ValueError:
        return None
    return None


def _primary_metrics(trial: object, dataset: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in PRIMARY_METRIC_NAMES:
        value = _attr_float(trial, f"{dataset}.val.{name}")
        if value is None:
            return {}
        metrics[name] = value
    return metrics


def _configured_probe_or_smoke(trial: object, dataset: str) -> bool:
    attrs = _trial_attrs(trial)
    if attrs.get("seeded_from_study") is not None:
        return True
    if any(key in attrs for key in RUNTIME_PROBE_METRIC_NAMES):
        return True
    if attrs.get("runtime_probe") or attrs.get(f"{dataset}.runtime_probe"):
        return True
    if attrs.get("runtime_probe_target_epochs") or attrs.get(
        f"{dataset}.runtime_probe_target_epochs",
    ):
        return True
    config = _json_mapping(attrs.get(f"{dataset}.effective_config")) or _json_mapping(
        attrs.get("effective_config"),
    )
    epochs = _finite_float(config.get("epochs") or config.get("max_epochs"))
    if epochs is not None and epochs <= 1:
        return True
    return config.get("sample_interactions") not in (None, "") or config.get(
        "loader_max_rows",
    ) not in (None, "")


def _current_feature_subset_revisions(
    dataset_names: Iterable[str],
    data_dir: str,
) -> dict[str, str]:
    """Return current search-space revisions for feature-subset datasets."""
    from experiments.run_search import resolve_search_space, search_space_revision

    revisions: dict[str, str] = {}
    for dataset in dataset_names:
        search_space = resolve_search_space(
            FEATURE_SUBSET_SEARCH_SPACE,
            dataset=dataset,
            data_dir=data_dir,
        )
        revisions[dataset] = search_space_revision(search_space)
    return revisions


def _latest_feature_subset_revisions_from_trials(
    studies: Sequence[object],
) -> dict[str, str]:
    """Return latest stored feature-subset revision per dataset."""
    latest: dict[str, tuple[str, str, int]] = {}
    fallback_index = 0
    for study in studies:
        for trial in getattr(study, "trials", ()):
            fallback_index += 1
            attrs = _trial_attrs(trial)
            if attrs.get("search_space") != FEATURE_SUBSET_SEARCH_SPACE:
                continue
            dataset = _trial_dataset(trial)
            revision = attrs.get("search_space_revision")
            if dataset is None or not revision:
                continue
            started = str(getattr(trial, "datetime_start", "") or "")
            number = _trial_number(trial, fallback_index)
            candidate = (started, str(revision), number)
            current = latest.get(dataset)
            if current is None or (candidate[0], candidate[2]) > (current[0], current[2]):
                latest[dataset] = candidate
    return {dataset: revision for dataset, (_started, revision, _number) in latest.items()}


def _groups_from_feature_subset_profile(profile: str) -> tuple[str, ...]:
    """Extract feature groups mentioned by one feature-subset profile."""
    for prefix in ("single_", "drop_", "pair_", "triple_"):
        if profile.startswith(prefix):
            return tuple(part for part in profile.removeprefix(prefix).split("__") if part)
    return ()


def _feature_subset_groups_from_trials(
    studies: Sequence[object],
    *,
    current_revisions: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """Derive loaded feature groups from stored feature-subset profiles."""
    groups_by_dataset: dict[str, list[str]] = defaultdict(list)
    for study in studies:
        for trial in getattr(study, "trials", ()):
            attrs = _trial_attrs(trial)
            if attrs.get("search_space") != FEATURE_SUBSET_SEARCH_SPACE:
                continue
            dataset = _trial_dataset(trial)
            if dataset is None:
                continue
            revision = current_revisions.get(dataset)
            if revision is not None and attrs.get("search_space_revision") != revision:
                continue
            profile = _profile_from_trial(trial)
            if profile == GRAPH_ONLY_PROFILE:
                groups_by_dataset.setdefault(dataset, [])
                continue
            for group in _groups_from_feature_subset_profile(profile or ""):
                if group not in groups_by_dataset[dataset]:
                    groups_by_dataset[dataset].append(group)
    return {dataset: tuple(groups) for dataset, groups in groups_by_dataset.items()}


def _is_completed_feature_subset_trial(
    trial: object,
    *,
    current_revisions: Mapping[str, str],
) -> bool:
    dataset = _trial_dataset(trial)
    attrs = _trial_attrs(trial)
    revision = current_revisions.get(dataset or "")
    return (
        _trial_state_name(trial) == "COMPLETE"
        and _trial_value(trial) is not None
        and dataset is not None
        and attrs.get("search_space") == FEATURE_SUBSET_SEARCH_SPACE
        and (revision is None or attrs.get("search_space_revision") == revision)
        and _profile_from_trial(trial) is not None
        and not _configured_probe_or_smoke(trial, dataset)
    )


def _completed_feature_subset_trials(
    studies: Sequence[object],
    *,
    current_revisions: Mapping[str, str],
) -> list[object]:
    trials: list[object] = []
    for study in studies:
        for trial in getattr(study, "trials", ()):
            if _is_completed_feature_subset_trial(trial, current_revisions=current_revisions):
                trials.append(trial)
    return trials


def _posthoc_crru_by_dataset(trials: Sequence[object]) -> dict[tuple[str, int], dict[int, float]]:
    output: dict[tuple[str, int], dict[int, float]] = {}
    indexed_trials = list(enumerate(trials))
    datasets = sorted(
        {dataset for _index, trial in indexed_trials if (dataset := _trial_dataset(trial))},
    )
    for dataset in datasets:
        scoped = [
            (index, trial)
            for index, trial in indexed_trials
            if _trial_dataset(trial) == dataset
            and len(_primary_metrics(trial, dataset)) == len(PRIMARY_METRIC_NAMES)
        ]
        if not scoped:
            continue
        efficiency = compute_crru_efficiency_scores(
            [_attr_float(trial, f"{dataset}.peak_vram_mb") for _index, trial in scoped],
            [_trial_epoch_time_s(trial, dataset) for _index, trial in scoped],
        )
        crru_by_k = {
            k: compute_crru_scores_for_k(
                ndcg=[_metric_value(trial, dataset, f"NDCG@{k}") for _index, trial in scoped],
                recall=[_metric_value(trial, dataset, f"Recall@{k}") for _index, trial in scoped],
                hit=[_metric_value(trial, dataset, f"HitRatio@{k}") for _index, trial in scoped],
                personalization=[
                    _metric_value(trial, dataset, f"Personalization@{k}")
                    for _index, trial in scoped
                ],
                average_popularity=[
                    _metric_value(trial, dataset, f"AveragePopularity@{k}")
                    for _index, trial in scoped
                ],
                efficiency_scores=efficiency,
            )
            for k in (20, 40)
        }
        for row_index, (trial_index, trial) in enumerate(scoped):
            output[(dataset, _trial_number(trial, trial_index))] = {
                k: values[row_index] for k, values in crru_by_k.items()
            }
    return output


def _row_from_trial(
    trial: object,
    *,
    fallback_number: int,
    groups_by_dataset: Mapping[str, tuple[str, ...]],
    completed_counts: Mapping[tuple[str, str], int],
    posthoc_crru: Mapping[tuple[str, int], Mapping[int, float]],
) -> dict[str, object]:
    dataset = _trial_dataset(trial) or ""
    profile = _profile_from_trial(trial) or ""
    groups = groups_by_dataset.get(dataset, ())
    included, excluded = feature_subset_profile_group_labels(profile, groups)
    row: dict[str, object] = {
        "dataset": dataset,
        "feature_subset_profile": profile,
        "included_groups": _feature_groups_label(included),
        "excluded_groups": _feature_groups_label(excluded),
        "source_objective": _trial_value(trial),
        "validation_accuracy_20_40": _metric_value(trial, dataset, VALIDATION_ACCURACY_METRIC),
        "online_crru_20": _metric_value(
            trial,
            dataset,
            VALIDATION_ONLINE_CRRU_K_METRICS[20],
        ),
        "online_crru_40": _metric_value(
            trial,
            dataset,
            VALIDATION_ONLINE_CRRU_K_METRICS[40],
        ),
        "online_crru_20_40": _metric_value(trial, dataset, VALIDATION_ONLINE_CRRU_METRIC),
        "time_per_epoch_s": _trial_epoch_time_s(trial, dataset),
        "peak_vram_mb": _attr_float(trial, f"{dataset}.peak_vram_mb"),
        "batch": _attr_float(trial, f"{dataset}.batch_size"),
        "completed_trials": completed_counts.get((dataset, profile), 0),
        "status": "completed",
    }
    for metric, column in METRIC_TO_COLUMN.items():
        row[column] = _metric_value(trial, dataset, metric)
    trial_number = _trial_number(trial, fallback_number)
    trial_crru = posthoc_crru.get((dataset, trial_number), {})
    row["posthoc_crru_20"] = trial_crru.get(20)
    row["posthoc_crru_40"] = trial_crru.get(40)
    return row


def _empty_profile_row(
    dataset: str,
    profile: str,
    groups: Sequence[str],
    *,
    status: str,
) -> dict[str, object]:
    included, excluded = feature_subset_profile_group_labels(profile, groups)
    return {key: "" for key, _label in FEATURE_SUBSET_RESULT_COLUMNS} | {
        "dataset": dataset,
        "feature_subset_profile": profile,
        "included_groups": _feature_groups_label(included),
        "excluded_groups": _feature_groups_label(excluded),
        "completed_trials": 0,
        "status": status,
    }


def _feature_subset_groups_by_dataset(
    dataset_names: Iterable[str],
    data_dir: str,
) -> dict[str, tuple[str, ...]]:
    return {
        dataset: loaded_thesis_safe_item_feature_groups_for_dataset(dataset, data_dir=data_dir)
        for dataset in dataset_names
    }


def build_feature_subset_result_rows(
    studies: Sequence[object],
    *,
    dataset_names: Iterable[str] = FEATURE_SUBSET_DATASETS,
    data_dir: str = "data",
) -> list[dict[str, object]]:
    """Return all completed trial rows plus missing coverage rows."""
    dataset_tuple = tuple(dataset_names)
    current_revisions = _latest_feature_subset_revisions_from_trials(studies)
    missing_revision_datasets = [
        dataset for dataset in dataset_tuple if dataset not in current_revisions
    ]
    if missing_revision_datasets:
        current_revisions |= _current_feature_subset_revisions(
            missing_revision_datasets,
            data_dir,
        )
    groups_by_dataset = _feature_subset_groups_from_trials(
        studies,
        current_revisions=current_revisions,
    )
    missing_group_datasets = [
        dataset for dataset in dataset_tuple if dataset not in groups_by_dataset
    ]
    if missing_group_datasets:
        groups_by_dataset |= _feature_subset_groups_by_dataset(
            missing_group_datasets,
            data_dir,
        )
    completed_trials = _completed_feature_subset_trials(
        studies,
        current_revisions=current_revisions,
    )
    completed_counts = defaultdict(int)
    for trial in completed_trials:
        dataset = _trial_dataset(trial)
        profile = _profile_from_trial(trial)
        if dataset is not None and profile is not None:
            completed_counts[(dataset, profile)] += 1
    posthoc_crru = _posthoc_crru_by_dataset(completed_trials)
    completed_rows = [
        _row_from_trial(
            trial,
            fallback_number=index,
            groups_by_dataset=groups_by_dataset,
            completed_counts=completed_counts,
            posthoc_crru=posthoc_crru,
        )
        for index, trial in enumerate(completed_trials)
    ]

    rows = sorted(
        completed_rows,
        key=lambda row: (
            str(row["dataset"]),
            str(row["feature_subset_profile"]),
            -float(row.get("source_objective") or 0.0),
        ),
    )
    present = {
        (str(row["dataset"]), str(row["feature_subset_profile"]))
        for row in rows
        if row["status"] == "completed"
    }
    for dataset, groups in groups_by_dataset.items():
        for profile in required_feature_subset_profiles(groups):
            if (dataset, profile) in present:
                continue
            status = "not_applicable" if profile == GRAPH_ONLY_PROFILE else "pending"
            rows.append(_empty_profile_row(dataset, profile, groups, status=status))
    return rows


def _best_rows_by_profile(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["feature_subset_profile"]))].append(row)
    best_rows: list[Mapping[str, object]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        completed = [
            row
            for row in candidates
            if row.get("status") == "completed" and _finite_float(row.get("source_objective"))
        ]
        if completed:
            best_rows.append(
                max(completed, key=lambda row: float(row["source_objective"])),
            )
        else:
            best_rows.append(candidates[0])
    return best_rows


def _write_markdown_table(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[tuple[str, str]],
    *,
    title: str,
    notes: Sequence[str] = (),
) -> None:
    _ensure_dir()
    lines = [f"# {title}", "", *notes]
    if notes:
        lines.append("")
    labels = [label for _key, label in columns]
    lines.append("| " + " | ".join(labels) + " |")
    lines.append("| " + " | ".join("---" for _label in labels) + " |")
    for row in rows:
        values = [_format_markdown_value(row.get(key, "")) for key, _label in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _profile_metric(
    rows: Sequence[Mapping[str, object]],
    dataset: str,
    profile: str,
) -> float | None:
    for row in _best_rows_by_profile(rows):
        if row["dataset"] == dataset and row["feature_subset_profile"] == profile:
            return _finite_float(row.get("online_crru_20_40"))
    return None


def _all_feature_baseline(rows: Sequence[Mapping[str, object]], dataset: str) -> float | None:
    values = [
        value
        for profile in ("all_gate_neg4", "all_gate0")
        if (value := _profile_metric(rows, dataset, profile)) is not None
    ]
    return max(values) if values else None


def _feature_subset_delta_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        none = _profile_metric(rows, dataset, "none")
        all_baseline = _all_feature_baseline(rows, dataset)
        if none is not None and all_baseline is not None:
            output.append(
                {"dataset": dataset, "effect": "side_feature_gain", "delta": all_baseline - none},
            )
        for row in _best_rows_by_profile(rows):
            if row["dataset"] != dataset:
                continue
            profile = str(row["feature_subset_profile"])
            value = _finite_float(row.get("online_crru_20_40"))
            if value is None:
                continue
            if profile.startswith("single_") and none is not None:
                output.append(
                    {
                        "dataset": dataset,
                        "effect": profile.replace("single_", "single_group_gain:"),
                        "delta": value - none,
                    },
                )
            elif profile.startswith("drop_") and all_baseline is not None:
                output.append(
                    {
                        "dataset": dataset,
                        "effect": profile.replace("drop_", "drop_group_effect:"),
                        "delta": all_baseline - value,
                    },
                )
            elif profile.startswith("pair_") and none is not None:
                output.append(
                    {
                        "dataset": dataset,
                        "effect": profile.replace("pair_", "pair_gain:"),
                        "delta": value - none,
                    },
                )
            elif profile.startswith("triple_") and none is not None:
                output.append(
                    {
                        "dataset": dataset,
                        "effect": profile.replace("triple_", "triple_gain:"),
                        "delta": value - none,
                    },
                )
    return output


def _write_feature_subset_best_by_dataset(rows: Sequence[Mapping[str, object]]) -> None:
    best_rows = _best_rows_by_profile(rows)
    delta_rows = _feature_subset_delta_rows(rows)
    lines = [
        "# Feature Subset Best By Dataset",
        "",
        "Ranking: validation OnlineCRRU@20_40 within each dataset.",
        "Positive side_feature_gain means side features helped.",
        "Positive single_group_gain means that group alone beat no features.",
        "Positive drop_group_effect means removing that group hurt.",
        "Positive pair/triple gain means that combination beat no features.",
        "",
    ]
    for dataset in sorted({str(row["dataset"]) for row in best_rows}):
        dataset_rows = [row for row in best_rows if row["dataset"] == dataset]
        completed = [
            row
            for row in dataset_rows
            if row["status"] == "completed" and _finite_float(row.get("online_crru_20_40"))
        ]
        pending = sum(1 for row in dataset_rows if row["status"] == "pending")
        lines.extend([f"## {dataset}", ""])
        if not completed:
            status = (
                "not_applicable"
                if all(row["status"] == "not_applicable" for row in dataset_rows)
                else "PENDING"
            )
            lines.extend([status, "", f"Pending required profiles: {pending}.", ""])
            continue
        best = max(completed, key=lambda row: float(row["online_crru_20_40"]))
        lines.append(
            "Best completed profile: "
            f"`{best['feature_subset_profile']}` "
            f"(OnlineCRRU@20_40={_format_float(best['online_crru_20_40'])}, "
            f"ValidationAccuracy@20_40={_format_float(best['validation_accuracy_20_40'])}, "
            f"NDCG@20={_format_float(best['ndcg_20'])}, "
            f"Recall@20={_format_float(best['recall_20'])}, "
            f"AvgPop@20={_format_float(best['avgpop_20'])}, "
            f"time/epoch={_format_float(best['time_per_epoch_s'], digits=2)}, "
            f"VRAM={_format_float(best['peak_vram_mb'], digits=1)}).",
        )
        lines.append(f"Pending required profiles: {pending}.")
        dataset_deltas = [row for row in delta_rows if row["dataset"] == dataset]
        if dataset_deltas:
            lines.extend(["", "| Effect | Delta OnlineCRRU@20_40 |", "|---|---:|"])
            for row in dataset_deltas:
                lines.append(f"| {row['effect']} | {_format_float(row['delta'])} |")
        lines.append("")
    (FEATURE_ANALYSIS_DIR / "feature_subset_best_by_dataset.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _feature_subset_delta_figure_path(dataset: str) -> Path:
    """Return the dataset-local feature-subset delta figure path."""
    return FEATURE_ANALYSIS_DIR / f"feature_subset_deltas_{dataset}.png"


def _remove_feature_subset_delta_figures() -> None:
    """Remove stale feature-subset delta figures before regenerating them."""
    _remove_file(FEATURE_ANALYSIS_DIR / "feature_subset_delta_heatmap.png")
    for path in FEATURE_ANALYSIS_DIR.glob("feature_subset_deltas_*.png"):
        _remove_file(path)


def _effect_group_label(group: str) -> str:
    """Return a readable feature-group label for plots."""
    return "\n".join(textwrap.wrap(group.replace("_", " "), width=18))


def _combination_effect_label(effect: str) -> str:
    """Return a compact label for a pair/triple gain effect."""
    kind, groups = effect.split(":", 1)
    prefix = "pair" if kind == "pair_gain" else "triple"
    return f"{prefix}: " + "\n".join(
        textwrap.wrap(groups.replace("__", " + ").replace("_", " "), width=28),
    )


def _symmetric_axis_limit(values: Sequence[float]) -> float:
    """Return a readable symmetric x-axis limit for signed delta bars."""
    finite = [abs(value) for value in values if math.isfinite(value)]
    return max(finite, default=0.01) * 1.25


def _annotate_bars(ax: plt.Axes, bars: object, values: Sequence[float], limit: float) -> None:
    """Annotate horizontal bars with signed delta values."""
    offset = limit * 0.025
    for bar, value in zip(bars, values, strict=True):
        if not math.isfinite(value):
            continue
        x = value + offset if value >= 0 else value - offset
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )


def _plot_side_feature_gain(
    ax: plt.Axes,
    side_value: float | None,
    *,
    limit: float,
) -> None:
    """Plot the dataset-local all-features versus no-features gain."""
    if side_value is None:
        ax.text(0.5, 0.5, "PENDING", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    color = "#2166ac" if side_value >= 0 else "#b2182b"
    bars = ax.barh([0], [side_value], color=color)
    _annotate_bars(ax, bars, [side_value], limit)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_yticks([0], labels=["best all\nminus none"])
    ax.set_title("Side-feature gain")
    ax.set_xlabel("delta")


def _plot_group_effects(
    ax: plt.Axes,
    group_values: Mapping[str, Mapping[str, float]],
    *,
    limit: float,
) -> None:
    """Plot single-group gains and drop-group effects."""
    if not group_values:
        ax.text(0.5, 0.5, "No group effects", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    groups = sorted(
        group_values,
        key=lambda group: max(abs(value) for value in group_values[group].values()),
        reverse=True,
    )
    y = np.arange(len(groups), dtype=np.float32)
    single_values = [group_values[group].get("single", math.nan) for group in groups]
    drop_values = [group_values[group].get("drop", math.nan) for group in groups]
    single_bars = ax.barh(
        y - 0.18,
        [0.0 if math.isnan(value) else value for value in single_values],
        height=0.34,
        color="#1b9e77",
        label="single vs none",
    )
    drop_bars = ax.barh(
        y + 0.18,
        [0.0 if math.isnan(value) else value for value in drop_values],
        height=0.34,
        color="#d95f02",
        label="full minus drop",
    )
    _annotate_bars(ax, single_bars, single_values, limit)
    _annotate_bars(ax, drop_bars, drop_values, limit)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_yticks(y, labels=[_effect_group_label(group) for group in groups])
    ax.invert_yaxis()
    ax.set_title("Feature-group effects")
    ax.set_xlabel("delta")
    ax.legend(loc="lower right", fontsize=8)


def _plot_combination_gains(
    ax: plt.Axes,
    combination_rows: Sequence[Mapping[str, object]],
    *,
    limit: float,
) -> None:
    """Plot top pair/triple feature-subset gains."""
    if not combination_rows:
        ax.text(
            0.5,
            0.5,
            "No completed pair/triple profiles",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return
    top_rows = sorted(
        combination_rows,
        key=lambda row: float(row["delta"]),
        reverse=True,
    )[:10]
    values = [float(row["delta"]) for row in top_rows]
    y = np.arange(len(top_rows), dtype=np.float32)
    colors = ["#2166ac" if value >= 0 else "#b2182b" for value in values]
    bars = ax.barh(y, values, color=colors)
    _annotate_bars(ax, bars, values, limit)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_yticks(
        y,
        labels=[_combination_effect_label(str(row["effect"])) for row in top_rows],
    )
    ax.invert_yaxis()
    ax.set_title("Top pair/triple gains")
    ax.set_xlabel("delta")


def _write_dataset_feature_subset_delta_figure(
    dataset: str,
    delta_rows: Sequence[Mapping[str, object]],
) -> None:
    """Write one dataset-local feature-subset delta figure."""
    side_value: float | None = None
    group_values: dict[str, dict[str, float]] = {}
    combination_rows: list[Mapping[str, object]] = []
    all_values: list[float] = []
    for row in delta_rows:
        effect = str(row["effect"])
        value = float(row["delta"])
        all_values.append(value)
        if effect == "side_feature_gain":
            side_value = value
        elif effect.startswith("single_group_gain:"):
            group = effect.split(":", 1)[1]
            group_values.setdefault(group, {})["single"] = value
        elif effect.startswith("drop_group_effect:"):
            group = effect.split(":", 1)[1]
            group_values.setdefault(group, {})["drop"] = value
        elif effect.startswith(("pair_gain:", "triple_gain:")):
            combination_rows.append(row)

    limit = _symmetric_axis_limit(all_values)
    group_count = max(len(group_values), len(combination_rows[:10]), 1)
    if combination_rows:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16.0, max(4.5, group_count * 0.55)),
            gridspec_kw={"width_ratios": [1.0, 2.0, 2.6]},
            constrained_layout=True,
        )
    else:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10.5, max(4.2, group_count * 0.55)),
            gridspec_kw={"width_ratios": [1.0, 2.0]},
            constrained_layout=True,
        )
    fig.suptitle(f"{dataset}: feature-subset validation deltas", fontsize=14)
    _plot_side_feature_gain(axes[0], side_value, limit=limit)
    _plot_group_effects(axes[1], group_values, limit=limit)
    if combination_rows:
        _plot_combination_gains(axes[2], combination_rows, limit=limit)
    fig.savefig(_feature_subset_delta_figure_path(dataset), dpi=170)
    plt.close(fig)


def _write_feature_subset_delta_figures(rows: Sequence[Mapping[str, object]]) -> None:
    """Write dataset-local feature-subset delta figures."""
    _remove_feature_subset_delta_figures()
    delta_rows = _feature_subset_delta_rows(rows)
    by_dataset: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in delta_rows:
        by_dataset[str(row["dataset"])].append(row)
    for dataset, dataset_rows in sorted(by_dataset.items()):
        _write_dataset_feature_subset_delta_figure(dataset, dataset_rows)


def render_feature_subset_report_section(rows: Sequence[Mapping[str, object]]) -> list[str]:
    """Return the feature-subset section for results/optuna_optimization.md."""
    best_rows = _best_rows_by_profile(rows)
    pending = sum(1 for row in best_rows if row.get("status") == "pending")
    lines = [
        "## Feature subset search",
        "",
        "Scope: completed, non-probe trials from `edgrec-feature-subset-search` only.",
        "Selection: best row per dataset and feature subset profile by source objective.",
        "",
    ]
    if pending:
        lines.extend(
            [
                f"PENDING: {pending} required dataset-profile rows have no completed trial yet.",
                "",
            ],
        )
    labels = [label for _key, label in FEATURE_SUBSET_RESULT_COLUMNS]
    lines.append("| " + " | ".join(labels) + " |")
    lines.append("| " + " | ".join("---" for _label in labels) + " |")
    for row in best_rows:
        values = [
            _format_markdown_value(row.get(key, ""))
            for key, _label in FEATURE_SUBSET_RESULT_COLUMNS
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def write_feature_subset_search_reports(
    studies: Sequence[object],
    rows: Sequence[Mapping[str, object]] | None = None,
    *,
    dataset_names: Iterable[str] = FEATURE_SUBSET_DATASETS,
    data_dir: str = "data",
) -> list[dict[str, object]]:
    """Write feature-subset search reports from completed Optuna trials."""
    remove_stale_feature_effect_reports()
    result_rows = [
        dict(row)
        for row in (
            rows
            if rows is not None
            else build_feature_subset_result_rows(
                studies,
                dataset_names=dataset_names,
                data_dir=data_dir,
            )
        )
    ]
    best_rows = _best_rows_by_profile(result_rows)
    _write_labeled_csv(
        FEATURE_ANALYSIS_DIR / "feature_subset_results.csv",
        result_rows,
        FEATURE_SUBSET_RESULT_COLUMNS,
    )
    _write_markdown_table(
        FEATURE_ANALYSIS_DIR / "feature_subset_results.md",
        best_rows,
        FEATURE_SUBSET_RESULT_COLUMNS,
        title="Feature Subset Results",
        notes=(
            "Only completed, non-probe trials from `edgrec-feature-subset-search` are evidence.",
            "Rows marked PENDING or not_applicable do not imply a metric conclusion.",
        ),
    )
    _write_feature_subset_best_by_dataset(result_rows)
    _write_feature_subset_delta_figures(result_rows)
    return result_rows


def _feature_subset_source_rows(dataset: str, canonical: object) -> list[dict[str, object]]:
    groups = loaded_thesis_safe_item_feature_groups(canonical)
    item_status = "pending" if groups else "not_applicable"
    rows: list[dict[str, object]] = []
    for entity in ("item", "user"):
        names = getattr(canonical, f"{entity}_feature_names", None)
        if names is None:
            continue
        sources = getattr(canonical, f"{entity}_feature_sources", None) or ("",) * len(names)
        raw_columns = getattr(canonical, f"{entity}_feature_raw_columns", None) or tuple(
            str(name).split("::", 1)[-1].split("=", 1)[0] for name in names
        )
        roles = getattr(canonical, f"{entity}_feature_roles", None) or ("",) * len(names)
        feature_groups = getattr(canonical, f"{entity}_feature_groups", None) or ("",) * len(names)
        for index, name in enumerate(names):
            role = str(roles[index])
            group = str(feature_groups[index])
            safe_item_group = entity == "item" and role == FEATURE_SAFE_ROLE and group in groups
            rows.append(
                {
                    "dataset": dataset,
                    "feature_name": name,
                    "source_file": sources[index],
                    "raw_column": raw_columns[index],
                    "entity_type": entity,
                    "role": role,
                    "group": group,
                    "encoded_column_index": index,
                    "feature_subset_status": item_status if safe_item_group else "not_searched",
                },
            )
    if rows:
        return rows
    return [
        {
            "dataset": dataset,
            "feature_name": GRAPH_ONLY_PROFILE,
            "source_file": GRAPH_ONLY_PROFILE,
            "raw_column": GRAPH_ONLY_PROFILE,
            "entity_type": "item",
            "role": FEATURE_SAFE_ROLE,
            "group": GRAPH_ONLY_PROFILE,
            "encoded_column_index": "",
            "feature_subset_status": "not_applicable",
        },
    ]


def build_feature_group_inventory_rows(
    *,
    dataset_names: Iterable[str] = FEATURE_SUBSET_DATASETS,
    data_dir: str = "data",
) -> list[dict[str, object]]:
    """Load datasets and return actual feature metadata rows."""
    rows: list[dict[str, object]] = []
    for dataset in dataset_names:
        canonical = load_dataset(
            dataset,
            data_dir=data_dir,
            include_optional_features=True,
            feature_policy="thesis_default",
        )
        rows.extend(_feature_subset_source_rows(dataset, canonical))
    return rows


def _write_feature_group_inventory_markdown(rows: Sequence[Mapping[str, object]]) -> None:
    grouped: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["entity_type"]),
            str(row["group"]),
            str(row["feature_subset_status"]),
        )
        grouped[key] += 1
    lines = [
        "# Feature Group Inventory",
        "",
        "Loaded thesis-default feature columns grouped by dataset and entity.",
        "Feature-effect metrics are intentionally absent from this inventory.",
        "",
        "| Dataset | Entity | Group | LoadedColumns | FeatureSubsetStatus |",
        "|---|---|---|---:|---|",
    ]
    for (dataset, entity, group, status), count in sorted(grouped.items()):
        lines.append(f"| {dataset} | {entity} | {group} | {count} | {status} |")
    lines.append("")
    (FEATURE_ANALYSIS_DIR / "feature_group_inventory.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_feature_group_inventory_reports(
    rows: Sequence[Mapping[str, object]] | None = None,
    *,
    dataset_names: Iterable[str] = FEATURE_SUBSET_DATASETS,
    data_dir: str = "data",
) -> list[dict[str, object]]:
    """Write actual loaded feature-group inventory reports."""
    _ensure_dir()
    inventory_rows = [
        dict(row)
        for row in (
            rows
            if rows is not None
            else build_feature_group_inventory_rows(dataset_names=dataset_names, data_dir=data_dir)
        )
    ]
    _write_csv(
        FEATURE_ANALYSIS_DIR / "feature_group_inventory.csv",
        inventory_rows,
        (
            "dataset",
            "feature_name",
            "source_file",
            "raw_column",
            "entity_type",
            "role",
            "group",
            "encoded_column_index",
            "feature_subset_status",
        ),
    )
    _write_feature_group_inventory_markdown(inventory_rows)
    return inventory_rows


def write_query_feature_analysis_reports() -> None:
    """Refresh descriptive feature inventory from the existing query-results command."""
    write_feature_group_inventory_reports(dataset_names=FEATURE_SUBSET_DATASETS)
    remove_stale_feature_effect_reports()
