"""Feature-analysis report dependency tests."""

from __future__ import annotations

from types import SimpleNamespace

from src.reporting import feature_analysis as fa


def _trial(
    dataset: str,
    profile: str,
    value: float,
    *,
    state: str = "COMPLETE",
    max_epochs: int = 150,
) -> SimpleNamespace:
    """Return a tiny Optuna-like feature-subset trial."""
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
    }
    attrs = {
        "search_space": fa.FEATURE_SUBSET_SEARCH_SPACE,
        "search_space_revision": "test-revision",
        "datasets": [dataset],
        "sampled_params": {"feature_subset_profile": profile},
        "objective_metric": "ValidationOnlineCRRU@20_40",
        "objective_split": "val",
        f"{dataset}.effective_config": {"epochs": max_epochs},
        f"{dataset}.avg_epoch_time_s": 1.5,
        f"{dataset}.peak_vram_mb": 256.0,
        f"{dataset}.batch_size": 4096,
    }
    attrs.update(
        {f"{dataset}.val.{metric}": metric_value for metric, metric_value in metrics.items()},
    )
    return SimpleNamespace(
        number=hash((dataset, profile, value)) % 10000,
        state=SimpleNamespace(name=state),
        value=value,
        user_attrs=attrs,
    )


def test_graph_only_dataset_produces_valid_group_inventory(tmp_path, monkeypatch) -> None:
    """Graph-only datasets produce a not_applicable inventory row."""
    monkeypatch.setattr(fa, "FEATURE_ANALYSIS_DIR", tmp_path)
    rows = [
        {
            "dataset": "amazonbook",
            "feature_name": "graph_only",
            "source_file": "graph_only",
            "raw_column": "graph_only",
            "entity_type": "item",
            "role": "safe_pre_treatment",
            "group": "graph_only",
            "encoded_column_index": "",
            "feature_subset_status": "not_applicable",
        },
    ]

    written = fa.write_feature_group_inventory_reports(rows)

    assert written == rows
    assert "not_applicable" in (tmp_path / "feature_group_inventory.md").read_text(
        encoding="utf-8",
    )
    assert (tmp_path / "feature_group_inventory.csv").exists()


def test_feature_subset_reports_run_on_tiny_synthetic_trials(tmp_path, monkeypatch) -> None:
    """Subset report writers run on tiny completed synthetic data."""
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
    study = SimpleNamespace(
        trials=[
            _trial("movielens1m", "none", 0.2),
            _trial("movielens1m", "all_gate_neg4", 0.3),
        ],
    )

    rows = fa.write_feature_subset_search_reports(
        [study],
        dataset_names=("movielens1m",),
    )

    assert any(row["status"] == "completed" for row in rows)
    assert any(row["status"] == "pending" for row in rows)
    assert (tmp_path / "feature_subset_results.csv").exists()
    assert "FeatureSubset" in (tmp_path / "feature_subset_results.md").read_text(
        encoding="utf-8",
    )
    assert not (tmp_path / "feature_subset_delta_heatmap.png").exists()
    assert (tmp_path / "feature_subset_deltas_movielens1m.png").exists()


def test_missing_experiments_are_marked_pending(tmp_path, monkeypatch) -> None:
    """Missing required profiles are reported as PENDING, not inferred."""
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

    rows = fa.write_feature_subset_search_reports([], dataset_names=("movielens1m",))

    assert {row["status"] for row in rows} == {"pending"}
    assert "PENDING" in (tmp_path / "feature_subset_best_by_dataset.md").read_text(
        encoding="utf-8",
    )


def test_one_epoch_runtime_probe_trials_are_excluded(tmp_path, monkeypatch) -> None:
    """Probe-like one-epoch rows are excluded from feature-subset evidence."""
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
    study = SimpleNamespace(trials=[_trial("movielens1m", "none", 0.9, max_epochs=1)])

    rows = fa.write_feature_subset_search_reports(
        [study],
        dataset_names=("movielens1m",),
    )

    assert {row["status"] for row in rows} == {"pending"}
