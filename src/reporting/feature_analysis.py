"""Dataset-local feature-subset reporting helpers."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.data.feature_groups import (
    GRAPH_ONLY_PROFILE,
    feature_subset_profile_group_labels,
    loaded_thesis_safe_item_feature_groups,
    loaded_thesis_safe_item_feature_groups_for_dataset,
    required_feature_subset_profiles,
)
from src.data.loaders import load_dataset
from src.utils.crru import (
    CRRU_RECOMMENDATION_METRIC_NAMES,
    CRRU_REPORT_FORMULA_LINES,
    VALIDATION_ACCURACY_METRIC,
    VALIDATION_CRRU_K_METRICS,
    VALIDATION_CRRU_METRIC,
    compute_validation_accuracy_objective,
    compute_validation_crru_metric_value,
    is_validation_crru_metric_name,
)
from src.utils.crru_popularity import (
    CRRUPopularityReconstructionError,
    resolve_largest_training_item_interaction_count,
)
from src.utils.experiment_logger import RUNTIME_PROBE_METRIC_NAMES
from src.utils.project_paths import RESULTS_DIR

FEATURE_ANALYSIS_DIR = RESULTS_DIR / "feature_analysis"
DATASET_SUMMARY_PATH = RESULTS_DIR / "dataset_visualizations" / "benchmark_summary.json"
FEATURE_SUBSET_SEARCH_SPACE = "edgrec-feature-subset-search"
FEATURE_SUBSET_DATASETS = ("amazonbook", "movielens1m", "kuairec_v2", "kuairand1k")
FEATURE_SAFE_ROLE = "safe_pre_treatment"
A4_TEXT_WIDTH_IN = 160.0 / 25.4
HORIZONTAL_FIGURE_HEIGHT_IN = A4_TEXT_WIDTH_IN * 2.0 / 3.0
THESIS_FIGURE_DPI = 360
DATASET_DISPLAY_NAMES = {
    "amazonbook": "Amazon Book",
    "movielens1m": "MovieLens-1M",
    "movielens20m": "MovieLens-20M",
    "kuairec_v2": "KuaiRec v2",
    "taobao": "Taobao",
    "kuairand1k": "KuaiRand-1K",
}
FEATURE_GROUP_LABELS = {
    "graph_only": "graph only",
    "item_genre": "item genre",
    "user_demographic": "user demographics",
    "item_author_music": "author/music",
    "item_video_metadata": "video metadata",
    "item_upload_time": "upload time",
    "item_category": "category/tags",
    "item_resolution": "resolution",
    "other_safe_item_feature": "other item",
    "other_safe_user_feature": "other user",
}
DATASET_FEATURE_NOTES = {
    "amazonbook": "No side-feature source; interaction graph only.",
    "movielens1m": (
        "Item genres searched; user demographics loaded but not used by the item-only context head."
    ),
    "movielens20m": "Genres are available; no current feature-subset search or EDGRec test row.",
    "kuairec_v2": (
        "Safe item descriptors: author/music, metadata, resolution, category, upload time."
    ),
    "taobao": "Category id is available; no current feature-subset search or EDGRec test row.",
    "kuairand1k": (
        "Safe item descriptors mirror KuaiRec; randomized-exposure rows remain a separate regime."
    ),
}
DATASET_EXCLUSION_NOTES = {
    "amazonbook": "Nothing feature-bearing was omitted.",
    "movielens1m": (
        "Zip code is proxy-only; demographics are not searched/trained in current EDGRec."
    ),
    "movielens20m": "Genome/tag text evidence is outside the current thesis-default path.",
    "kuairec_v2": (
        "User profiles, captions/free text, and engagement counts are excluded or proxy-only."
    ),
    "taobao": (
        "Behavior labels and timestamps are outcomes/context, not side features for the "
        "current model."
    ),
    "kuairand1k": (
        "Statistic engagement file is excluded; show_cnt is only a propensity target when "
        "IPW is explicit."
    ),
}
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
    ("validation_crru_20", "ValidationCRRU@20"),
    ("validation_crru_40", "ValidationCRRU@40"),
    ("validation_crru_20_40", "ValidationCRRU@20_40"),
    ("posthoc_crru_20", "PosthocCRRU@20"),
    ("posthoc_crru_40", "PosthocCRRU@40"),
    ("time_per_epoch_s", "Time/epoch (s)"),
    ("peak_vram_mb", "Peak VRAM (MB)"),
    ("batch", "Batch"),
    ("completed_trials", "CompletedTrials"),
    ("status", "Status"),
)


def _append_crru_formula(lines: list[str]) -> None:
    """Append the thesis CRRU formula block to a markdown report."""
    lines.extend(["## CRRU Reporting Utility", ""])
    for index, line in enumerate(CRRU_REPORT_FORMULA_LINES):
        prefix = "**" if index == 0 else "- "
        suffix = "**" if index == 0 else ""
        lines.append(f"{prefix}{line.strip()}{suffix}")
    lines.append("")


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
PRIMARY_METRIC_NAMES = CRRU_RECOMMENDATION_METRIC_NAMES
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


def _effective_trial_config(trial: object, dataset: str) -> Mapping[str, Any]:
    attrs = _trial_attrs(trial)
    config = attrs.get(f"{dataset}.effective_config")
    if isinstance(config, Mapping):
        return config
    params = dict(_sampled_params(trial))
    params.setdefault("dataset", dataset)
    return params


def _largest_training_item_interaction_count(trial: object, dataset: str) -> float | None:
    try:
        return resolve_largest_training_item_interaction_count(
            stored_value=_attr_float(
                trial,
                f"{dataset}.largest_training_item_interaction_count",
            ),
            config=_effective_trial_config(trial, dataset),
            dataset=dataset,
        )
    except CRRUPopularityReconstructionError:
        return None


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
    metrics = _primary_metrics(trial, dataset)
    if metric == VALIDATION_ACCURACY_METRIC:
        if len(metrics) != len(PRIMARY_METRIC_NAMES):
            return None
        try:
            return compute_validation_accuracy_objective(metrics)
        except ValueError:
            return None
    if is_validation_crru_metric_name(metric):
        return _validation_crru_metric_value(trial, dataset, metric, metrics)
    stored = _attr_float(trial, f"{dataset}.val.{metric}")
    if stored is not None:
        return stored
    return None


def _validation_crru_metric_value(
    trial: object,
    dataset: str,
    metric: str,
    metrics: Mapping[str, float],
) -> float | None:
    """Return one formal ValidationCRRU value from complete trial inputs."""
    if len(metrics) != len(PRIMARY_METRIC_NAMES):
        return None
    peak_vram_mb = _attr_float(trial, f"{dataset}.peak_vram_mb")
    epoch_time_s = _trial_epoch_time_s(trial, dataset)
    largest_training_item_interaction_count = _largest_training_item_interaction_count(
        trial,
        dataset,
    )
    if (
        peak_vram_mb is None
        or epoch_time_s is None
        or largest_training_item_interaction_count is None
    ):
        return None
    try:
        return compute_validation_crru_metric_value(
            metric,
            metrics,
            peak_vram_mb=peak_vram_mb,
            epoch_time_s=epoch_time_s,
            largest_training_item_interaction_count=largest_training_item_interaction_count,
        )
    except ValueError:
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
        for trial_index, trial in indexed_trials:
            if _trial_dataset(trial) != dataset:
                continue
            values_by_cutoff = {
                cutoff: _metric_value(trial, dataset, metric_name)
                for cutoff, metric_name in VALIDATION_CRRU_K_METRICS.items()
            }
            if any(value is None for value in values_by_cutoff.values()):
                continue
            output[(dataset, _trial_number(trial, trial_index))] = {
                cutoff: float(value)
                for cutoff, value in values_by_cutoff.items()
                if value is not None
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
        "validation_crru_20": _metric_value(
            trial,
            dataset,
            VALIDATION_CRRU_K_METRICS[20],
        ),
        "validation_crru_40": _metric_value(
            trial,
            dataset,
            VALIDATION_CRRU_K_METRICS[40],
        ),
        "validation_crru_20_40": _metric_value(trial, dataset, VALIDATION_CRRU_METRIC),
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
    _append_crru_formula(lines)
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
            return _finite_float(row.get("validation_crru_20_40"))
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
            value = _finite_float(row.get("validation_crru_20_40"))
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
        "Ranking: ValidationCRRU@20_40 within each dataset.",
        "Positive side_feature_gain means side features helped.",
        "Positive single_group_gain means that group alone beat no features.",
        "Positive drop_group_effect means removing that group hurt.",
        "Positive pair/triple gain means that combination beat no features.",
        "",
    ]
    _append_crru_formula(lines)
    for dataset in sorted({str(row["dataset"]) for row in best_rows}):
        dataset_rows = [row for row in best_rows if row["dataset"] == dataset]
        completed = [
            row
            for row in dataset_rows
            if row["status"] == "completed" and _finite_float(row.get("validation_crru_20_40"))
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
        best = max(completed, key=lambda row: float(row["validation_crru_20_40"]))
        lines.append(
            "Best completed profile: "
            f"`{best['feature_subset_profile']}` "
            f"(ValidationCRRU@20_40={_format_float(best['validation_crru_20_40'])}, "
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
            lines.extend(["", "| Effect | Delta ValidationCRRU@20_40 |", "|---|---:|"])
            for row in dataset_deltas:
                lines.append(f"| {row['effect']} | {_format_float(row['delta'])} |")
        lines.append("")
    (FEATURE_ANALYSIS_DIR / "feature_subset_best_by_dataset.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _feature_subset_evidence_matrix_path() -> Path:
    """Return the markdown feature-subset evidence matrix path."""
    return FEATURE_ANALYSIS_DIR / "feature_subset_evidence_matrix.md"


def _remove_feature_subset_legacy_figures() -> None:
    """Remove stale feature-subset PNG figures before regenerating tables."""
    _remove_file(_feature_subset_evidence_matrix_path())
    _remove_file(FEATURE_ANALYSIS_DIR / "feature_subset_delta_heatmap.png")
    _remove_file(FEATURE_ANALYSIS_DIR / "feature_subset_evidence_matrix.png")
    _remove_file(FEATURE_ANALYSIS_DIR / "feature_subset_delta_dotplot.png")
    for path in FEATURE_ANALYSIS_DIR.glob("feature_subset_deltas_*.png"):
        _remove_file(path)


def _best_effect_by_prefix(
    deltas: Mapping[tuple[str, str], float],
    dataset: str,
    prefixes: Sequence[str],
) -> tuple[str, float] | None:
    """Return the largest delta for one dataset across effect prefixes."""
    candidates = [
        (effect, value)
        for (candidate_dataset, effect), value in deltas.items()
        if candidate_dataset == dataset and any(effect.startswith(prefix) for prefix in prefixes)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])


def _feature_effect_markdown_text(effect: str, value: float, *, include_label: bool) -> str:
    """Return markdown cell text for one feature effect."""
    if not include_label:
        return f"{value:+.3f}"
    label = _friendly_group_text(effect.split(":", 1)[1])
    return f"{value:+.3f} {label}"


def _claim_boundary_label(dataset: str, best_profile: str, side_gain: float | None) -> str:
    """Return short claim-boundary text for the feature evidence table."""
    if best_profile == GRAPH_ONLY_PROFILE:
        return "graph-only"
    if best_profile == "none" or (side_gain is not None and side_gain < 0):
        return "negative evidence"
    if dataset in {"kuairec_v2", "kuairand1k"}:
        return "rerun before claim"
    return "validation only"


def _profile_table_label(profile: str) -> str:
    """Return a compact readable profile label for the matrix-side table."""
    for prefix in ("single_", "pair_", "triple_"):
        if profile.startswith(prefix):
            return (
                f"{prefix.removesuffix('_')}: {_friendly_group_text(profile.removeprefix(prefix))}"
            )
    if profile.startswith("drop_"):
        return f"drop: {_friendly_group_text(profile.removeprefix('drop_'))}"
    return _profile_readable_label(profile)


def _write_feature_subset_evidence_matrix(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write a markdown feature-subset evidence matrix."""
    deltas = _feature_delta_lookup(rows)
    best_rows = _best_completed_row_by_dataset(rows)
    datasets = [
        dataset
        for dataset in FEATURE_SUBSET_DATASETS
        if dataset in best_rows
        or any(candidate_dataset == dataset for candidate_dataset, _effect in deltas)
    ]
    if not datasets:
        return

    effect_columns: list[tuple[str, str, Sequence[str]]] = [
        ("All side vs none", "", ("side_feature_gain",)),
        ("Best single\ngroup", "single", ("single_group_gain:",)),
        ("Best pair/triple", "combo", ("pair_gain:", "triple_gain:")),
        ("Best drop importance", "drop", ("drop_group_effect:",)),
    ]
    matrix_rows: list[list[str]] = []
    for dataset in datasets:
        cells: list[str] = [DATASET_DISPLAY_NAMES.get(dataset, dataset)]
        for _title, kind, prefixes in effect_columns:
            if kind == "":
                value = deltas.get((dataset, "side_feature_gain"))
                if value is None:
                    cells.append("n/a")
                    continue
                cells.append(f"{value:+.3f}")
                continue
            best = _best_effect_by_prefix(deltas, dataset, prefixes)
            if best is None:
                cells.append("n/a")
                continue
            effect, value = best
            cells.append(
                _feature_effect_markdown_text(effect, value, include_label=True),
            )
        best = best_rows.get(dataset)
        if best is None:
            cells.extend(["no evidence", "-", "-"])
        else:
            raw_profile = str(best.get("feature_subset_profile", ""))
            cells.extend(
                [
                    _profile_table_label(raw_profile),
                    _format_float(best.get("validation_crru_20_40")),
                    _claim_boundary_label(
                        dataset,
                        raw_profile,
                        deltas.get((dataset, "side_feature_gain")),
                    ),
                ],
            )
        matrix_rows.append(cells)
    lines = [
        "# Feature Subset Evidence Matrix",
        "",
        "Cells are validation/search deltas, not test-set claims. "
        "Drop importance = all-features score minus score after removing the group.",
        "",
        "| Dataset | All side vs none | Best single group | Best pair/triple | "
        "Best drop importance | Best validation profile | ValCRRU | Thesis use |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in matrix_rows:
        lines.append(
            "| " + " | ".join(_format_markdown_value(value) for value in row) + " |",
        )
    _feature_subset_evidence_matrix_path().write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _feature_subset_summary_effect_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return compact feature-effect rows for the thesis-facing dot plot."""
    deltas = _feature_delta_lookup(rows)
    best_rows = _best_completed_row_by_dataset(rows)
    output: list[dict[str, object]] = []
    effect_specs = (
        ("All side vs none", ("side_feature_gain",)),
        ("Best single group", ("single_group_gain:",)),
        ("Best pair/triple", ("pair_gain:", "triple_gain:")),
        ("Best drop importance", ("drop_group_effect:",)),
    )
    for dataset in FEATURE_SUBSET_DATASETS:
        if dataset not in best_rows and not any(
            candidate_dataset == dataset for candidate_dataset, _effect in deltas
        ):
            continue
        for label, prefixes in effect_specs:
            if prefixes == ("side_feature_gain",):
                value = deltas.get((dataset, "side_feature_gain"))
                effect = "side_feature_gain"
            else:
                best = _best_effect_by_prefix(deltas, dataset, prefixes)
                if best is None:
                    continue
                effect, value = best
            if value is None:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "effect_label": label,
                    "effect": effect,
                    "delta": value,
                },
            )
    return output


def _write_feature_subset_delta_dotplot(rows: Sequence[Mapping[str, object]]) -> None:
    """Write a compact feature-subset delta lollipop plot."""
    effect_rows = _feature_subset_summary_effect_rows(rows)
    best_rows = _best_completed_row_by_dataset(rows)
    if not effect_rows and not best_rows:
        return

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    effect_order = (
        "All side vs none",
        "Best single group",
        "Best pair/triple",
        "Best drop importance",
    )
    effect_styles = {
        "All side vs none": ("#1f77b4", "o"),
        "Best single group": ("#2f8f46", "s"),
        "Best pair/triple": ("#7b5aa6", "^"),
        "Best drop importance": ("#d17a22", "D"),
    }
    effect_labels = {
        "All side vs none": "All side",
        "Best single group": "Single group",
        "Best pair/triple": "Pair/triple",
        "Best drop importance": "Drop effect",
        "Graph only": "Graph-only",
    }
    plot_rows: list[dict[str, object]] = []
    group_spans: list[tuple[str, float, float]] = []
    y = 0.0
    for dataset in FEATURE_SUBSET_DATASETS:
        dataset_rows = [row for row in effect_rows if row["dataset"] == dataset]
        dataset_rows.sort(key=lambda row: effect_order.index(str(row["effect_label"])))
        if not dataset_rows and dataset in best_rows:
            dataset_rows = [
                {
                    "dataset": dataset,
                    "effect_label": "Graph only",
                    "effect": "",
                    "delta": None,
                },
            ]
        if not dataset_rows:
            continue
        start = y
        for row in dataset_rows:
            row = dict(row)
            row["y"] = y
            plot_rows.append(row)
            y += 1.0
        group_spans.append((dataset, start, y - 1.0))
        y += 0.45

    if not plot_rows:
        return

    fig, ax = plt.subplots(figsize=(A4_TEXT_WIDTH_IN, HORIZONTAL_FIGURE_HEIGHT_IN))
    ax.axvline(0.0, color="#374151", linewidth=1.0, alpha=0.85)
    all_values = [float(row["delta"]) for row in plot_rows if row["delta"] is not None]
    min_value = min(all_values, default=0.0)
    max_value = max(all_values, default=0.0)
    span = max(0.02, max_value - min_value)
    left_limit = min(-0.07, min_value - 0.30 * span)
    right_limit = max(0.06, max_value + 0.34 * span)
    ax.set_xlim(left_limit, right_limit)

    for row in plot_rows:
        label = str(row["effect_label"])
        value = row["delta"]
        row_y = float(row["y"])
        if value is None:
            ax.text(
                0.0,
                row_y,
                "n/a",
                va="center",
                ha="center",
                fontsize=8,
                color="#6b7280",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "#f3f4f6",
                    "edgecolor": "#d1d5db",
                },
            )
            continue
        color, marker = effect_styles[label]
        numeric_value = float(value)
        ax.hlines(
            row_y,
            min(0.0, numeric_value),
            max(0.0, numeric_value),
            color=color,
            linewidth=1.8,
            alpha=0.58,
        )
        ax.scatter(
            numeric_value,
            row_y,
            s=55,
            color=color,
            marker=marker,
            edgecolor="#111827",
            linewidth=0.45,
            zorder=3,
        )
        ax.text(
            numeric_value + (0.002 if numeric_value >= 0 else -0.002),
            row_y,
            f"{numeric_value:+.3f}",
            va="center",
            ha="left" if numeric_value >= 0 else "right",
            fontsize=7.5,
            color="#111827",
        )

    dataset_label_x = left_limit + 0.012 * (right_limit - left_limit)
    for dataset, start, end in group_spans:
        midpoint = (start + end) / 2.0
        if start > 0:
            ax.axhline(start - 0.5, color="#d1d5db", linewidth=0.7, alpha=0.85)
        ax.text(
            dataset_label_x,
            midpoint,
            DATASET_DISPLAY_NAMES.get(dataset, dataset),
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#111827",
        )
    ax.set_yticks(
        [float(row["y"]) for row in plot_rows],
        [
            effect_labels.get(str(row["effect_label"]), str(row["effect_label"]))
            for row in plot_rows
        ],
    )
    ax.invert_yaxis()
    ax.set_ylim(float(plot_rows[-1]["y"]) + 0.55, -0.55)
    ax.set_xlabel("Delta validation CRRU@20/40", labelpad=3, fontsize=9)
    ax.set_title("Feature-subset validation deltas by dataset", fontsize=11, pad=6)
    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout(pad=0.35)
    output_path = FEATURE_ANALYSIS_DIR / "feature_subset_delta_dotplot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THESIS_FIGURE_DPI)
    plt.close(fig)


def _write_feature_subset_delta_tables(rows: Sequence[Mapping[str, object]]) -> None:
    """Write combined feature-subset delta tables."""
    _remove_feature_subset_legacy_figures()
    _write_feature_subset_evidence_matrix(rows)
    _write_feature_subset_delta_dotplot(rows)


def _read_dataset_summary_payloads() -> dict[str, Mapping[str, object]]:
    """Return generated dataset-summary payloads when available."""
    if not DATASET_SUMMARY_PATH.exists():
        return {}
    try:
        payload = json.loads(DATASET_SUMMARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return {}
    output: dict[str, Mapping[str, object]] = {}
    for row in datasets:
        if not isinstance(row, Mapping):
            continue
        name = row.get("name")
        if isinstance(name, str):
            output[name] = row
    return output


def _read_existing_feature_inventory_rows() -> list[dict[str, object]]:
    """Return current feature-inventory rows from disk when they exist."""
    path = FEATURE_ANALYSIS_DIR / "feature_group_inventory.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _inventory_rows_from_result_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build a minimal feature-inventory fallback from subset result rows."""
    output: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        dataset = str(row.get("dataset", ""))
        groups = [
            group
            for field in ("included_groups", "excluded_groups")
            for group in str(row.get(field, "")).split(",")
            if group
        ]
        if not groups and dataset and dataset not in seen:
            groups = [GRAPH_ONLY_PROFILE]
        for group in groups:
            key = (dataset, group)
            if not dataset or key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "dataset": dataset,
                    "feature_name": group,
                    "source_file": group,
                    "raw_column": group,
                    "entity_type": "item",
                    "role": FEATURE_SAFE_ROLE,
                    "group": group,
                    "encoded_column_index": "",
                    "feature_subset_status": (
                        "not_applicable" if group == GRAPH_ONLY_PROFILE else "search_candidate"
                    ),
                },
            )
    return output


def _dataset_order(
    dataset_payloads: Mapping[str, Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> list[str]:
    """Return thesis-friendly dataset order for feature review artifacts."""
    preferred = [
        "amazonbook",
        "movielens1m",
        "movielens20m",
        "kuairec_v2",
        "taobao",
        "kuairand1k",
    ]
    available = set(dataset_payloads) | {str(row.get("dataset", "")) for row in rows}
    ordered = [dataset for dataset in preferred if dataset in available]
    ordered.extend(sorted(dataset for dataset in available if dataset and dataset not in ordered))
    return ordered


def _feature_group_counts(
    inventory_rows: Sequence[Mapping[str, object]],
    dataset: str,
    *,
    entity: str | None = None,
    status: str | None = None,
) -> dict[str, int]:
    """Return loaded feature-column counts by feature group."""
    counts: dict[str, int] = defaultdict(int)
    for row in inventory_rows:
        if row.get("dataset") != dataset:
            continue
        if entity is not None and row.get("entity_type") != entity:
            continue
        if status is not None and row.get("feature_subset_status") != status:
            continue
        group = str(row.get("group", ""))
        if group:
            counts[group] += 1
    return dict(counts)


def _format_group_count_summary(counts: Mapping[str, int], *, fallback: str) -> str:
    """Format feature-group counts for table cells."""
    if not counts:
        return fallback
    parts = [
        f"{FEATURE_GROUP_LABELS.get(group, group)} ({count})"
        for group, count in sorted(counts.items())
    ]
    return "\n".join(parts)


def _format_dataset_context(payload: Mapping[str, object] | None, dataset: str) -> str:
    """Format scale, sparsity, and feedback semantics for one dataset."""
    label = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    if payload is None:
        return label
    interactions = _finite_float(payload.get("n_interactions"))
    density = _finite_float(payload.get("density"))
    feedback = str(payload.get("feedback_description") or "")
    interactions_text = f"{interactions / 1_000_000:.1f}M interactions" if interactions else ""
    density_text = f"{density * 100:.3g}% density" if density is not None else ""
    signed_feedback_text = _format_signed_feedback_context(payload)
    return "\n".join(
        part
        for part in (label, interactions_text, density_text, feedback, signed_feedback_text)
        if part
    )


def _format_signed_feedback_context(payload: Mapping[str, object]) -> str:
    """Format binary-label and graded-sign context from dataset profiles."""
    label_distribution = payload.get("label_distribution")
    sign_distribution = payload.get("sign_distribution")
    if isinstance(label_distribution, Mapping) and isinstance(sign_distribution, Mapping):
        label_shares = label_distribution.get("shares")
        sign_shares = sign_distribution.get("shares")
        if not isinstance(label_shares, Mapping) or not isinstance(sign_shares, Mapping):
            return ""
        positive_label = _finite_float(label_shares.get("positive_label"))
        positive_sign = _finite_float(sign_shares.get("positive_sign"))
        zero_sign = _finite_float(sign_shares.get("zero_sign"))
        negative_sign = _finite_float(sign_shares.get("negative_sign"))
        if None in (positive_label, positive_sign, zero_sign, negative_sign):
            return ""
        text = (
            f"label/sign: label>0 {positive_label:.1%}; "
            f"sign>0 {positive_sign:.1%}, sign=0 {zero_sign:.1%}, "
            f"sign<0 {negative_sign:.1%}"
        )
        overlap = payload.get("label_sign_overlap")
        if isinstance(overlap, Mapping):
            positive_negative = _finite_float(overlap.get("positive_label_negative_sign"))
            if positive_negative:
                text += f"; label>0 & sign<0 {int(positive_negative):,}"
        return text

    masks = payload.get("canonical_feedback_masks")
    if not isinstance(masks, Mapping):
        return ""
    shares = masks.get("shares")
    if not isinstance(shares, Mapping):
        return ""
    positive = _finite_float(shares.get("positive_label"))
    neutral = _finite_float(shares.get("neutral_nonpositive"))
    negative = _finite_float(shares.get("negative_sign"))
    if positive is None or neutral is None or negative is None:
        return ""
    return (
        f"label/sign: label>0 {positive:.1%}; legacy label=0/sign=0 {neutral:.1%}, "
        f"sign<0 {negative:.1%}"
    )


def _best_completed_row_by_dataset(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Return the highest ValidationCRRU feature-subset row per dataset."""
    output: dict[str, Mapping[str, object]] = {}
    for row in _best_rows_by_profile(rows):
        if row.get("status") != "completed":
            continue
        value = _finite_float(row.get("validation_crru_20_40"))
        dataset = str(row.get("dataset", ""))
        if value is None or not dataset:
            continue
        current = output.get(dataset)
        current_value = _finite_float(current.get("validation_crru_20_40")) if current else None
        if current_value is None or value > current_value:
            output[dataset] = row
    return output


def _feature_delta_lookup(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], float]:
    """Return feature-subset delta values keyed by dataset and effect label."""
    output: dict[tuple[str, str], float] = {}
    for row in _feature_subset_delta_rows(rows):
        value = _finite_float(row.get("delta"))
        dataset = str(row.get("dataset", ""))
        effect = str(row.get("effect", ""))
        if value is not None and dataset and effect:
            output[(dataset, effect)] = value
    return output


def _best_positive_effect(
    deltas: Mapping[tuple[str, str], float],
    dataset: str,
    prefixes: Sequence[str],
) -> tuple[str, float] | None:
    """Return the strongest positive feature effect for one dataset."""
    candidates = [
        (effect, value)
        for (candidate_dataset, effect), value in deltas.items()
        if candidate_dataset == dataset
        and any(effect.startswith(prefix) for prefix in prefixes)
        and value > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])


def _profile_readable_label(profile: str) -> str:
    """Return a readable feature-subset profile label."""
    replacements = {
        "graph_only": "graph only",
        "none": "no side features",
        "all_gate_neg4": "all features, gate -4",
        "all_gate0": "all features, gate 0",
    }
    if profile in replacements:
        return replacements[profile]
    for prefix in ("single_", "drop_", "pair_", "triple_"):
        if profile.startswith(prefix):
            label = profile.removeprefix(prefix).replace("__", " + ").replace("_", " ")
            return f"{prefix[:-1]}: {label}"
    return profile.replace("_", " ")


def _friendly_group_text(groups: str) -> str:
    """Return compact feature-group text for figure annotations."""
    labels = [
        FEATURE_GROUP_LABELS.get(group, group.replace("_", " "))
        for group in groups.split("__")
        if group
    ]
    return " + ".join(labels)


def _feature_validation_result_text(
    dataset: str,
    best_rows: Mapping[str, Mapping[str, object]],
    deltas: Mapping[tuple[str, str], float],
) -> str:
    """Return compact validation result text for one dataset."""
    best = best_rows.get(dataset)
    if best is None:
        return "No completed feature-subset evidence."
    raw_profile = str(best.get("feature_subset_profile", ""))
    profile = _profile_readable_label(raw_profile)
    for prefix in ("single_", "pair_", "triple_"):
        if raw_profile.startswith(prefix):
            profile = _friendly_group_text(raw_profile.removeprefix(prefix))
            break
    score = _format_float(best.get("validation_crru_20_40"))
    side_gain = deltas.get((dataset, "side_feature_gain"))
    side_text = f"side gain {side_gain:+.3f}" if side_gain is not None else "side gain n/a"
    positive = _best_positive_effect(
        deltas,
        dataset,
        ("single_group_gain:", "pair_gain:", "triple_gain:"),
    )
    if positive is None:
        return f"Best: {profile}\nValCRRU {score}\n{side_text}"
    effect, value = positive
    effect_label = _friendly_group_text(effect.split(":", 1)[1])
    return (
        f"Best: {profile}\nValCRRU {score}\n{side_text}\nstrongest: {effect_label} ({value:+.3f})"
    )


def _feature_decision_text(
    dataset: str,
    best_rows: Mapping[str, Mapping[str, object]],
    deltas: Mapping[tuple[str, str], float],
) -> str:
    """Return thesis-facing feature decision or next-step text."""
    best = best_rows.get(dataset)
    if best is None:
        return (
            "Context dataset only for now; run feature-subset search before using feature claims."
        )
    profile = str(best.get("feature_subset_profile", ""))
    side_gain = deltas.get((dataset, "side_feature_gain"))
    if profile == GRAPH_ONLY_PROFILE:
        return "Train graph-only; no side-feature claim."
    if profile == "none" or (side_gain is not None and side_gain < 0):
        return (
            "Prefer no side features in current EDGRec basin; keep feature result as "
            "negative evidence."
        )
    if dataset == "kuairec_v2":
        return (
            "Use resolution/video-metadata feature reruns as candidates; final claim needs "
            "matching test rows."
        )
    if dataset == "kuairand1k":
        return (
            "Use category/triple features as candidates; test outside compact diagnostic "
            "regime before a headline."
        )
    return "Use as validation-only feature evidence until a matching full-data test row exists."


def _feature_review_rows(
    rows: Sequence[Mapping[str, object]],
    inventory_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Return dataset-feature decision rows for Markdown and PNG outputs."""
    dataset_payloads = _read_dataset_summary_payloads()
    best_rows = _best_completed_row_by_dataset(rows)
    deltas = _feature_delta_lookup(rows)
    output: list[dict[str, str]] = []
    for dataset in _dataset_order(dataset_payloads, rows):
        item_groups = _feature_group_counts(
            inventory_rows,
            dataset,
            entity="item",
            status="search_candidate",
        )
        if not item_groups and dataset in {"movielens20m", "taobao"}:
            fallback = DATASET_FEATURE_NOTES[dataset]
        else:
            fallback = DATASET_FEATURE_NOTES.get(dataset, "No loaded thesis-default features.")
        user_not_searched = _feature_group_counts(
            inventory_rows,
            dataset,
            entity="user",
            status="not_searched",
        )
        used_features = _format_group_count_summary(item_groups, fallback=fallback)
        if user_not_searched:
            used_features += "\nNot searched: " + _format_group_count_summary(
                user_not_searched,
                fallback="",
            )
        output.append(
            {
                "dataset": dataset,
                "dataset_context": _format_dataset_context(
                    dataset_payloads.get(dataset),
                    dataset,
                ),
                "feature_inputs": used_features,
                "left_out": DATASET_EXCLUSION_NOTES.get(dataset, ""),
                "validation_result": _feature_validation_result_text(dataset, best_rows, deltas),
                "decision": _feature_decision_text(dataset, best_rows, deltas),
            },
        )
    return output


def _write_dataset_feature_decision_map(review_rows: Sequence[Mapping[str, str]]) -> None:
    """Write the dataset-feature decision map as a markdown table."""
    _remove_file(FEATURE_ANALYSIS_DIR / "dataset_feature_decision_map.png")
    if not review_rows:
        return
    lines = [
        "# Dataset Feature Decision Map",
        "",
        "Feature evidence is validation/search evidence from `edgrec-feature-subset-search` "
        "unless a matching full-data test row exists. Excluded features are omitted by "
        "policy, not by plotting convenience.",
        "",
        "| Dataset regime | Thesis-default feature input | Left outside training | "
        "Feature evidence | Training decision / gap |",
        "|---|---|---|---|---|",
    ]
    for row in review_rows:
        lines.append(
            "| "
            + " | ".join(
                _format_markdown_value(value).replace("\n", "<br/>")
                for value in (
                    row["dataset_context"],
                    row["feature_inputs"],
                    row["left_out"],
                    row["validation_result"],
                    row["decision"],
                )
            )
            + " |",
        )
    (FEATURE_ANALYSIS_DIR / "dataset_feature_decision_map.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_feature_engineering_review_markdown(
    review_rows: Sequence[Mapping[str, str]],
) -> None:
    """Write a thesis-facing feature-engineering review document."""
    lines = [
        "# Feature Engineering Review",
        "",
        "Purpose: explain which dataset features are used for EDGRec training, which are "
        "left outside the thesis-default path, and what evidence still needs a full-data "
        "test rerun before becoming a thesis claim.",
        "",
        "Primary figure/tables/reports: `feature_subset_delta_dotplot.png`, "
        "`dataset_feature_decision_map.md`, "
        "`feature_subset_evidence_matrix.md`, `feature_subset_best_by_dataset.md`, and "
        "`feature_subset_results.csv`.",
        "",
        "## Dataset Decisions",
        "",
        (
            "| Dataset / feedback context | Feature input | Left outside training | "
            "Evidence | Decision |"
        ),
        "|---|---|---|---|---|",
    ]
    for row in review_rows:
        lines.append(
            "| "
            + " | ".join(
                _format_markdown_value(value).replace("\n", "<br/>")
                for value in (
                    row["dataset_context"],
                    row["feature_inputs"],
                    row["left_out"],
                    row["validation_result"],
                    row["decision"],
                )
            )
            + " |",
        )
    lines.extend(
        [
            "",
            "## What This Means",
            "",
            "- AmazonBook is a graph-only recommendation dataset in the current code path; no "
            "side-feature engineering claim should be made.",
            "- MovieLens-1M genre features are negative validation evidence in the current "
            "EDGRec basin; user demographics are loaded as metadata but not used by the "
            "item-only context head.",
            "- KuaiRec v2 has the strongest validation signal for side features, especially "
            "resolution and video metadata, but feature-specific full-data test rows are "
            "needed before a test-set claim.",
            "- KuaiRand-1K has positive validation signal for category/triple features, but "
            "current compact randomized-exposure results are diagnostic; standard-regime "
            "feature reruns are needed for a headline.",
            "- MovieLens-20M and Taobao are dataset-analysis context unless they receive "
            "matching EDGRec feature-subset searches and test rows.",
            "",
            "## Not Done Yet",
            "",
            "- Full-data test reruns for KuaiRec `item_resolution`, `item_video_metadata`, and "
            "`item_video_metadata + item_resolution` candidates.",
            "- Full-data KuaiRand rerun for `item_author_music + item_upload_time + "
            "item_category`, preferably outside the ultra-compact diagnostic-only setup.",
            "- Explicit statement in slides that user-side features are not part of the "
            "current EDGRec scorer; otherwise a committee may ask why demographics were "
            "loaded but not trained.",
            "- If Taobao or MovieLens-20M become headline datasets, add their own "
            "feature-subset searches rather than extrapolating from the four current "
            "feature-search datasets.",
        ],
    )
    (FEATURE_ANALYSIS_DIR / "feature_engineering_review.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_feature_engineering_review_reports(
    rows: Sequence[Mapping[str, object]],
    inventory_rows: Sequence[Mapping[str, object]] | None = None,
) -> None:
    """Write feature-engineering decision artifacts from current report rows."""
    _ensure_dir()
    resolved_inventory_rows = (
        list(inventory_rows)
        if inventory_rows is not None
        else _read_existing_feature_inventory_rows()
    )
    if not resolved_inventory_rows:
        resolved_inventory_rows = _inventory_rows_from_result_rows(rows)
    review_rows = _feature_review_rows(rows, resolved_inventory_rows)
    _write_dataset_feature_decision_map(review_rows)
    _write_feature_engineering_review_markdown(review_rows)


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
    _write_feature_subset_delta_tables(result_rows)
    write_feature_engineering_review_reports(result_rows)
    return result_rows


def _feature_subset_source_rows(dataset: str, canonical: object) -> list[dict[str, object]]:
    groups = loaded_thesis_safe_item_feature_groups(canonical)
    item_status = "search_candidate" if groups else "not_applicable"
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
        "`search_candidate` means the safe item group is eligible for the feature-subset "
        "search; it is not a pending experiment status.",
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
