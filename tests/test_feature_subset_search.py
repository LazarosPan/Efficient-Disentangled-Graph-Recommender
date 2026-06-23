"""Feature-subset search and reporting tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import optuna
from experiments import run_search
from src.data.feature_groups import feature_subset_profile_matrix, required_feature_subset_profiles
from src.reporting import feature_analysis as fa


def _search_space_for_profiles(
    profiles: tuple[str, ...],
    groups: tuple[str, ...],
) -> run_search.SearchSpaceSpec:
    """Return a minimal search-space spec with feature-subset support."""
    return run_search.SearchSpaceSpec(
        name=fa.FEATURE_SUBSET_SEARCH_SPACE,
        description="test",
        base_profile="edgrec-compact-search-prior",
        datasets=("movielens1m",),
        objective=run_search.ObjectiveSpec(),
        max_epochs=10,
        trials=10,
        config_overrides={},
        parameters={
            "feature_subset_profile": {
                "type": "categorical",
                "choices": list(profiles),
            },
        },
        profile_overrides={"feature_subset_profile": feature_subset_profile_matrix(groups)},
    )


def test_required_feature_subset_profile_trials_are_enqueued(monkeypatch) -> None:
    """Required feature-subset profiles are queued before normal sampling."""
    groups = ("item_genre", "item_resolution")
    profiles = required_feature_subset_profiles(groups)
    monkeypatch.setattr(
        run_search,
        "loaded_thesis_safe_item_feature_groups_for_dataset",
        lambda dataset, *, data_dir: groups,
    )
    study = optuna.create_study(direction="maximize")
    search_space = _search_space_for_profiles(profiles, groups)

    enqueued = run_search.enqueue_required_feature_subset_profiles(
        study,
        search_space,
        data_dir="data",
    )

    storage_name = run_search._parameter_storage_name(
        "feature_subset_profile",
        search_space.parameters["feature_subset_profile"],
    )
    queued = {str(trial.system_attrs["fixed_params"][storage_name]) for trial in study.trials}
    assert enqueued == len(profiles)
    assert queued == set(profiles)


def test_required_feature_subset_profiles_limit_triples_to_four_groups() -> None:
    """Triple coverage is exhaustive only for datasets with at most four groups."""
    four_group_profiles = required_feature_subset_profiles(("a", "b", "c", "d"))
    five_group_profiles = required_feature_subset_profiles(("a", "b", "c", "d", "e"))

    assert any(profile.startswith("triple_") for profile in four_group_profiles)
    assert not any(profile.startswith("triple_") for profile in five_group_profiles)


def test_pruned_required_feature_subset_profile_is_requeued(monkeypatch) -> None:
    """A pruned required profile does not satisfy completed coverage."""
    groups = ("item_genre",)
    profile = "single_item_genre"
    monkeypatch.setattr(
        run_search,
        "loaded_thesis_safe_item_feature_groups_for_dataset",
        lambda dataset, *, data_dir: groups,
    )
    study = optuna.create_study(direction="maximize")
    search_space = _search_space_for_profiles(required_feature_subset_profiles(groups), groups)
    revision = run_search.search_space_revision(search_space)
    for completed_profile in set(required_feature_subset_profiles(groups)) - {profile}:
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=0.0,
                user_attrs={
                    "search_space": search_space.name,
                    "search_space_revision": revision,
                    "sampled_params": {"feature_subset_profile": completed_profile},
                },
            ),
        )
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.PRUNED,
            user_attrs={
                "search_space": search_space.name,
                "search_space_revision": revision,
                "feature_subset_required_profile": profile,
                "sampled_params": {"feature_subset_profile": profile},
            },
        ),
    )

    enqueued = run_search.enqueue_required_feature_subset_profiles(
        study,
        search_space,
        data_dir="data",
    )

    storage_name = run_search._parameter_storage_name(
        "feature_subset_profile",
        search_space.parameters["feature_subset_profile"],
    )
    queued = {
        str(trial.system_attrs["fixed_params"][storage_name])
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.WAITING
    }
    assert enqueued == 1
    assert profile in queued


def test_required_feature_subset_callback_reports_without_pruning() -> None:
    """Coverage trials report intermediate values but bypass Optuna pruning."""

    class AlwaysPruneTrial:
        """Tiny trial double that fails if pruning is queried."""

        def __init__(self) -> None:
            self.user_attrs: dict[str, float] = {}
            self.reports: list[tuple[float, int]] = []

        def report(self, value: float, *, step: int) -> None:
            self.reports.append((value, step))

        def set_user_attr(self, key: str, value: float) -> None:
            self.user_attrs[key] = value

        def should_prune(self) -> bool:
            raise AssertionError("coverage trial should not ask the pruner")

    trial = AlwaysPruneTrial()
    search_space = replace(
        _search_space_for_profiles(("none",), ("item_genre",)),
        objective=run_search.ObjectiveSpec(metric="NDCG@20"),
    )
    callback = run_search._build_pruning_epoch_callback(
        trial,  # type: ignore[arg-type]
        search_space=search_space,
        dataset="movielens1m",
        dataset_index=0,
        allow_pruning=False,
    )

    callback(0, {"NDCG@20": 0.25}, 1.0)

    assert trial.reports == [(0.25, 1)]
    assert trial.user_attrs["movielens1m.last_pruning_objective"] == 0.25


def _complete_trial(dataset: str, profile: str, value: float) -> SimpleNamespace:
    """Return a fake completed Optuna trial with feature-subset attrs."""
    metrics = {
        "NDCG@20": value,
        "Recall@20": value / 2,
        "HitRatio@20": value / 3,
        "Personalization@20": 0.7,
        "AveragePopularity@20": 0.2,
        "NDCG@40": value,
        "Recall@40": value / 2,
        "HitRatio@40": value / 3,
        "Personalization@40": 0.75,
        "AveragePopularity@40": 0.25,
        "ValidationOnlineCRRU@20_40": value,
    }
    return SimpleNamespace(
        number=hash((dataset, profile, value)) % 10000,
        state=SimpleNamespace(name="COMPLETE"),
        value=value,
        user_attrs={
            "search_space": fa.FEATURE_SUBSET_SEARCH_SPACE,
            "search_space_revision": "test-revision",
            "datasets": [dataset],
            "sampled_params": {"feature_subset_profile": profile},
            **{f"{dataset}.val.{metric}": metric_value for metric, metric_value in metrics.items()},
            f"{dataset}.effective_config": {"epochs": 150},
            f"{dataset}.avg_epoch_time_s": 1.5,
            f"{dataset}.peak_vram_mb": 128.0,
            f"{dataset}.batch_size": 4096,
        },
    )


def test_feature_subset_reports_mark_missing_profiles_pending(tmp_path, monkeypatch) -> None:
    """Missing required profiles are explicit pending rows with empty metrics."""
    monkeypatch.setattr(fa, "FEATURE_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(
        fa,
        "loaded_thesis_safe_item_feature_groups_for_dataset",
        lambda dataset, *, data_dir: ("item_genre",),
    )
    monkeypatch.setattr(
        fa,
        "_current_feature_subset_revisions",
        lambda dataset_names, data_dir: {"movielens1m": "test-revision"},
    )

    rows = fa.write_feature_subset_search_reports(
        [],
        dataset_names=("movielens1m",),
        data_dir="data",
    )

    assert {row["status"] for row in rows} == {"pending"}
    assert all(row["source_objective"] == "" for row in rows)
    assert "PENDING" in (tmp_path / "feature_subset_results.md").read_text(
        encoding="utf-8",
    )
    assert not (tmp_path / "feature_subset_delta_heatmap.png").exists()
    assert not (tmp_path / "feature_subset_deltas_movielens1m.png").exists()


def test_feature_subset_reports_do_not_pool_datasets(tmp_path, monkeypatch) -> None:
    """Feature-subset reports keep one section/row per dataset."""
    monkeypatch.setattr(fa, "FEATURE_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(
        fa,
        "loaded_thesis_safe_item_feature_groups_for_dataset",
        lambda dataset, *, data_dir: ("item_genre",),
    )
    monkeypatch.setattr(
        fa,
        "_current_feature_subset_revisions",
        lambda dataset_names, data_dir: {
            "movielens1m": "test-revision",
            "kuairec_v2": "test-revision",
        },
    )
    study = SimpleNamespace(
        trials=[
            _complete_trial("movielens1m", "none", 0.20),
            _complete_trial("movielens1m", "all_gate_neg4", 0.30),
            _complete_trial("kuairec_v2", "none", 0.10),
            _complete_trial("kuairec_v2", "all_gate0", 0.12),
        ],
    )

    fa.write_feature_subset_search_reports(
        [study],
        dataset_names=("movielens1m", "kuairec_v2"),
        data_dir="data",
    )

    best = (tmp_path / "feature_subset_best_by_dataset.md").read_text(encoding="utf-8")
    assert "## movielens1m" in best
    assert "## kuairec_v2" in best
    assert "global mean" not in best.lower()
    assert not (tmp_path / "feature_subset_delta_heatmap.png").exists()
    assert (tmp_path / "feature_subset_deltas_movielens1m.png").exists()
    assert (tmp_path / "feature_subset_deltas_kuairec_v2.png").exists()
