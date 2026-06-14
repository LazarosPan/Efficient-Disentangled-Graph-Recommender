"""Regression coverage for the Optuna search controller."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import experiments.run_search as search
import optuna
import scripts.report_optuna_optimization as optuna_report
from experiments.recipes import (
    load_experiment_catalog,
    load_search_spaces_catalog,
    search_space_names,
)
from src.utils.experiment_logger import ExperimentLogger


class SearchSpaceValidationTests(unittest.TestCase):
    """Pin search-space validation and config resolution for Optuna runs."""

    def test_search_spaces_live_outside_experiment_catalog(self) -> None:
        """Optuna configuration should stay in its own focused JSON file."""
        self.assertNotIn("search_spaces", load_experiment_catalog())
        self.assertIn("search_spaces", load_search_spaces_catalog())
        self.assertIn("ucagnn-core-optimization", search_space_names())

    def test_search_spaces_resolve_to_ucagnn_configs(self) -> None:
        """All search spaces should resolve through the shared config builder."""
        self.assertIn("ucagnn-core-optimization", search_space_names())

        for space_name in search_space_names():
            spec = search.resolve_search_space(space_name)
            config = search.build_search_config(
                spec,
                dataset=spec.datasets[0],
                device="cpu",
                data_dir="data",
            )

            self.assertEqual(config.baseline_family, "ucagnn")
            self.assertEqual(config.epochs, spec.max_epochs)
            self.assertEqual(config.device, "cpu")
            self.assertEqual(len(config.num_neighbors), config.max_gnn_layers)

    def test_search_space_resolves_sampler_and_pruner_from_catalog(self) -> None:
        """Sampler/pruner blocks in search_spaces.json should not be dropped."""
        spec = search.resolve_search_space(
            "ucagnn-core-optimization",
            dataset="amazonbook",
        )

        self.assertEqual(spec.sampler.seed, 42)
        self.assertEqual(spec.sampler.name, "tpe")
        self.assertFalse(spec.sampler.multivariate)
        self.assertFalse(spec.sampler.group)
        self.assertEqual(spec.pruner.name, "hyperband")
        self.assertEqual(spec.pruner.min_resource, 6)

    def test_auto_batch_search_treats_batch_size_as_runtime_only(self) -> None:
        """Historical sampled batch sizes should not break logical param matching."""
        spec = search.SearchSpaceSpec(
            name="tiny",
            description="test search space",
            base_profile="core-ucagnn-mainline",
            datasets=("amazonbook",),
            objective=search.ObjectiveSpec(),
            max_epochs=1,
            trials=1,
            config_overrides={"auto_batch_size": True},
            parameters={
                "lr": {
                    "type": "categorical",
                    "choices": [0.003],
                },
            },
        )

        self.assertTrue(
            search._sampled_params_match_search_space(
                {"lr": 0.003, "batch_size": 512},
                spec,
            ),
        )
        self.assertFalse(
            search._sampled_params_match_search_space(
                {"lr": 0.003, "dropout": 0.1},
                spec,
            ),
        )

    def test_search_space_validation_rejects_unknown_parameter_fields(self) -> None:
        """Search-space parameters should stay on the explicit existing-knob allowlist."""
        bad_space = {
            "name": "bad-space",
            "base_profile": "core-ucagnn-mainline",
            "datasets": ["amazonbook"],
            "parameters": {
                "not_a_config_field": {
                    "type": "float",
                    "low": 0.0,
                    "high": 1.0,
                },
            },
        }

        with (
            patch.object(search, "get_search_space", return_value=bad_space),
            self.assertRaisesRegex(ValueError, "unsupported config fields"),
        ):
            search.resolve_search_space("bad-space")

    def test_search_space_validation_rejects_paper_baseline_profiles(self) -> None:
        """Optuna v1 should not suggest paper-baseline presets."""
        bad_space = {
            "name": "bad-paper-space",
            "base_profile": "paper-lightgcn-baselines",
            "datasets": ["amazonbook"],
            "parameters": {
                "lr": {
                    "type": "float",
                    "low": 0.0001,
                    "high": 0.01,
                    "log": True,
                },
            },
        }

        with (
            patch.object(search, "get_search_space", return_value=bad_space),
            self.assertRaisesRegex(ValueError, "U-CaGNN-only"),
        ):
            search.resolve_search_space("bad-paper-space")

    def test_dry_run_resolves_base_config_without_training(self) -> None:
        """Dry-run payloads should build valid configs and avoid run_experiment()."""
        with patch.object(search, "run_experiment") as run_experiment:
            spec = search.resolve_search_space(
                "ucagnn-core-optimization",
                dataset="amazonbook",
            )
            payload = search.build_dry_run_payload(
                spec,
                study_name="dry-study",
                storage="sqlite:///:memory:",
                device="cpu",
                data_dir="data",
            )

        run_experiment.assert_not_called()
        base_config = payload["base_configs"]["amazonbook"]
        self.assertEqual(base_config["baseline_family"], "ucagnn")
        self.assertEqual(base_config["dataset"], "amazonbook")
        self.assertEqual(base_config["epochs"], 60)
        self.assertEqual(base_config["device"], "cpu")
        self.assertTrue(base_config["auto_batch_size"])
        self.assertEqual(base_config["batch_size"], 4096)
        self.assertEqual(
            base_config["batch_size_candidates"],
            [32768, 16384, 8192, 4096, 2048, 1024, 512, 256],
        )
        self.assertNotIn("batch_size", payload["parameters"])

    def test_optuna_report_effective_params_include_runtime_batch(self) -> None:
        """Promotion candidates should show resolved runtime fields, not only sampled knobs."""
        trial = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=0.5,
            params={},
            distributions={},
            user_attrs={
                "datasets": ["kuairand1k"],
                "sampled_params": {"lr": 0.003, "dropout": 0.15},
                "kuairand1k.batch_size": 32768,
                "kuairand1k.auto_batch_size": True,
                "kuairand1k.avg_epoch_time_s": 12.5,
                "kuairand1k.peak_vram_mb": 8192.0,
                "kuairand1k.effective_config": {
                    "epochs": 60,
                    "patience": 8,
                    "batch_size": 32768,
                    "auto_batch_size": True,
                    "batch_size_candidates": [32768, 16384, 8192, 4096],
                    "lr": 0.003,
                    "dropout": 0.15,
                    "use_features": True,
                },
            },
        )

        formatted = optuna_report.format_effective_params(trial, dataset="kuairand1k")

        self.assertIn("lr=0.003", formatted)
        self.assertIn("dropout=0.15", formatted)
        self.assertIn("batch_size=32768", formatted)
        self.assertIn("auto_batch_size=True", formatted)
        self.assertIn("batch_size_candidates=[32768,16384,8192,4096]", formatted)
        self.assertIn("time_per_epoch_s=12.5", formatted)
        self.assertIn("peak_vram_mb=8192.0", formatted)

    def test_search_cli_best_trial_prefers_effective_config(self) -> None:
        """CLI best-trial output should show the resolved config when it exists."""
        trial = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=0.5,
            params={},
            distributions={},
            user_attrs={
                "datasets": ["kuairand1k"],
                "sampled_params": {"lr": 0.003},
                "kuairand1k.effective_config": {
                    "lr": 0.003,
                    "batch_size": 8192,
                    "auto_batch_size": True,
                },
            },
        )

        label, payload = search._best_trial_config_payload(trial)

        self.assertEqual(label, "Best effective config:")
        self.assertEqual(payload["batch_size"], 8192)
        self.assertTrue(payload["auto_batch_size"])

    def test_search_cli_best_trial_labels_historical_params(self) -> None:
        """Old trials without effective config should not be mislabeled as full config."""
        trial = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=0.5,
            params={},
            distributions={},
            user_attrs={
                "datasets": ["movielens1m"],
                "sampled_params": {"lr": 0.003, "batch_size": 32768},
                "movielens1m.batch_size": 32768,
                "movielens1m.auto_batch_size": False,
                "movielens1m.avg_epoch_time_s": 12.5,
            },
        )

        label, payload = search._best_trial_config_payload(trial)

        self.assertIn("historical trial lacks effective_config", label)
        self.assertEqual(payload["batch_size"], 32768)
        self.assertFalse(payload["auto_batch_size"])
        self.assertEqual(payload["avg_epoch_time_s"], 12.5)

    def test_optuna_report_trial_accounting_tracks_budget_semantics(self) -> None:
        """Trial accounting should share the search controller's budget predicates."""
        study = SimpleNamespace(
            trials=[
                optuna.trial.create_trial(
                    state=optuna.trial.TrialState.COMPLETE,
                    value=0.50,
                    params={},
                    distributions={},
                ),
                optuna.trial.create_trial(
                    state=optuna.trial.TrialState.COMPLETE,
                    value=0.55,
                    params={},
                    distributions={},
                    user_attrs={"seeded_from_study": "historical-study"},
                ),
                optuna.trial.create_trial(
                    state=optuna.trial.TrialState.PRUNED,
                    params={},
                    distributions={},
                ),
                optuna.trial.create_trial(
                    state=optuna.trial.TrialState.PRUNED,
                    params={},
                    distributions={},
                    user_attrs={"duplicate_sampled_params": True},
                ),
            ],
        )

        lines = optuna_report.render_trial_accounting(study)

        self.assertIn("| COMPLETE | 1 | 1 | 2 |", lines)
        self.assertIn("| PRUNED | 2 | 0 | 2 |", lines)
        self.assertIn(
            "- Fresh informative budget count: `2` (fresh COMPLETE + real fresh PRUNED).",
            lines,
        )
        self.assertIn("- Duplicate-skip pruned trials excluded from that budget: `1`.", lines)

    def test_trial_overrides_enter_build_config_path(self) -> None:
        """Sampled values should become a valid config without post-build mutation."""
        spec = search.SearchSpaceSpec(
            name="tiny",
            description="test search space",
            base_profile="core-ucagnn-mainline",
            datasets=("amazonbook",),
            objective=search.ObjectiveSpec(),
            max_epochs=3,
            trials=1,
            config_overrides={
                "sample_interactions": 100,
                "loader_max_rows": 100,
            },
            parameters={
                "interest_gnn_layers": {
                    "type": "int",
                    "low": 2,
                    "high": 2,
                },
                "conformity_gnn_layers": {
                    "type": "int",
                    "low": 2,
                    "high": 2,
                },
                "num_neighbors": {
                    "type": "fanout",
                    "choices_by_depth": {
                        "2": [[6, 3]],
                    },
                },
                "use_popularity_head": {
                    "type": "categorical",
                    "choices": [False],
                },
            },
        )
        base_config = search.build_search_config(
            spec,
            dataset="amazonbook",
            device="cpu",
            data_dir="data",
        )
        fixed_trial = optuna.trial.FixedTrial(
            {
                search._parameter_storage_name(
                    "interest_gnn_layers",
                    spec.parameters["interest_gnn_layers"],
                ): 2,
                search._parameter_storage_name(
                    "conformity_gnn_layers",
                    spec.parameters["conformity_gnn_layers"],
                ): 2,
                search._parameter_storage_name(
                    "num_neighbors",
                    spec.parameters["num_neighbors"],
                    depth=2,
                ): "[6,3]",
                search._parameter_storage_name(
                    "use_popularity_head",
                    spec.parameters["use_popularity_head"],
                ): False,
            },
        )

        sampled = search.suggest_trial_overrides(
            fixed_trial,
            spec,
            base_config=base_config,
        )
        config = search.build_search_config(
            spec,
            dataset="amazonbook",
            sampled_overrides=sampled,
            device="cpu",
            data_dir="data",
        )

        self.assertEqual(config.interest_gnn_layers, 2)
        self.assertEqual(config.conformity_gnn_layers, 2)
        self.assertEqual(config.num_neighbors, [6, 3])
        self.assertFalse(config.use_popularity_head)
        self.assertEqual(config.sample_interactions, 100)
        self.assertEqual(config.loader_max_rows, 100)

    def test_grid_float_parameter_samples_from_declared_steps(self) -> None:
        """Segmented float grids should keep human-readable search points."""
        spec = search.SearchSpaceSpec(
            name="tiny-grid",
            description="test search space",
            base_profile="core-ucagnn-mainline",
            datasets=("amazonbook",),
            objective=search.ObjectiveSpec(),
            max_epochs=3,
            trials=1,
            config_overrides={
                "sample_interactions": 100,
                "loader_max_rows": 100,
            },
            parameters={
                "lr": {
                    "type": "grid_float",
                    "low": 0.0001,
                    "high": 0.0003,
                    "step": 0.0001,
                },
            },
        )
        base_config = search.build_search_config(
            spec,
            dataset="amazonbook",
            device="cpu",
            data_dir="data",
        )
        fixed_trial = optuna.trial.FixedTrial(
            {search._parameter_storage_name("lr", spec.parameters["lr"]): 0.0002},
        )

        sampled = search.suggest_trial_overrides(
            fixed_trial,
            spec,
            base_config=base_config,
        )

        self.assertEqual(sampled["lr"], 0.0002)

    def test_default_study_name_is_stable_across_parameter_changes(self) -> None:
        """Default study names should not restart just because the grid changed."""
        common_kwargs = {
            "description": "test search space",
            "base_profile": "core-ucagnn-mainline",
            "datasets": ("amazonbook",),
            "objective": search.ObjectiveSpec(),
            "max_epochs": 1,
            "trials": 1,
            "config_overrides": {},
        }
        continuous_spec = search.SearchSpaceSpec(
            name="tiny",
            parameters={
                "lr": {
                    "type": "float",
                    "low": 0.0001,
                    "high": 0.01,
                    "log": True,
                },
            },
            **common_kwargs,
        )
        grid_spec = search.SearchSpaceSpec(
            name="tiny",
            parameters={
                "lr": {
                    "type": "grid_float",
                    "low": 0.0001,
                    "high": 0.001,
                    "step": 0.0001,
                },
            },
            **common_kwargs,
        )

        continuous_name = search.default_study_name(
            "tiny",
            ("amazonbook",),
            search_space=continuous_spec,
        )
        grid_name = search.default_study_name(
            "tiny",
            ("amazonbook",),
            search_space=grid_spec,
        )

        self.assertEqual(continuous_name, "tiny-amazonbook-val-validationonlinecrru-20-40")
        self.assertEqual(grid_name, continuous_name)

    def test_default_study_name_changes_when_objective_changes(self) -> None:
        """Incomparable objectives should live in separate default studies."""
        spec = search.SearchSpaceSpec(
            name="tiny",
            description="test search space",
            base_profile="core-ucagnn-mainline",
            datasets=("amazonbook",),
            objective=search.ObjectiveSpec(metric="NDCG@40"),
            max_epochs=1,
            trials=1,
            config_overrides={},
            parameters={
                "lr": {
                    "type": "grid_float",
                    "low": 0.0001,
                    "high": 0.001,
                    "step": 0.0001,
                },
            },
        )

        self.assertEqual(
            search.default_study_name("tiny", ("amazonbook",), search_space=spec),
            "tiny-amazonbook-val-ndcg-40",
        )

    def test_fanout_suggestions_use_depth_specific_optuna_parameter_names(self) -> None:
        """Conditional fan-out choices should not reuse one dynamic categorical name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = search.SearchSpaceSpec(
                name="fanout-dynamic",
                description="test dynamic fanout",
                base_profile="core-ucagnn-mainline",
                datasets=("amazonbook",),
                objective=search.ObjectiveSpec(),
                max_epochs=1,
                trials=2,
                config_overrides={},
                parameters={
                    "interest_gnn_layers": {
                        "type": "int",
                        "low": 1,
                        "high": 2,
                    },
                    "conformity_gnn_layers": {
                        "type": "int",
                        "low": 1,
                        "high": 3,
                    },
                    "num_neighbors": {
                        "type": "fanout",
                        "choices_by_depth": {
                            "1": [[8], [16]],
                            "2": [[6, 3], [10, 5]],
                            "3": [[10, 5, 3], [16, 8, 4]],
                        },
                    },
                },
            )
            base_config = search.build_search_config(
                spec,
                dataset="amazonbook",
                device="cpu",
                data_dir="data",
            )
            study = optuna.create_study(
                storage=f"sqlite:///{Path(tmpdir) / 'fanout.db'}",
                direction="maximize",
            )
            study.enqueue_trial(
                {
                    search._parameter_storage_name(
                        "interest_gnn_layers",
                        spec.parameters["interest_gnn_layers"],
                    ): 1,
                    search._parameter_storage_name(
                        "conformity_gnn_layers",
                        spec.parameters["conformity_gnn_layers"],
                    ): 1,
                    search._parameter_storage_name(
                        "num_neighbors",
                        spec.parameters["num_neighbors"],
                        depth=1,
                    ): "[8]",
                },
            )
            study.enqueue_trial(
                {
                    search._parameter_storage_name(
                        "interest_gnn_layers",
                        spec.parameters["interest_gnn_layers"],
                    ): 2,
                    search._parameter_storage_name(
                        "conformity_gnn_layers",
                        spec.parameters["conformity_gnn_layers"],
                    ): 3,
                    search._parameter_storage_name(
                        "num_neighbors",
                        spec.parameters["num_neighbors"],
                        depth=3,
                    ): "[16,8,4]",
                },
            )

            def objective(trial: optuna.Trial) -> float:
                sampled = search.suggest_trial_overrides(
                    trial,
                    spec,
                    base_config=base_config,
                )
                self.assertIn(len(sampled["num_neighbors"]), {1, 3})
                return 1.0

            study.optimize(objective, n_trials=2)

            self.assertEqual(len(study.trials), 2)
            self.assertIn(
                search._parameter_storage_name(
                    "num_neighbors",
                    spec.parameters["num_neighbors"],
                    depth=1,
                ),
                study.trials[0].params,
            )
            self.assertIn(
                search._parameter_storage_name(
                    "num_neighbors",
                    spec.parameters["num_neighbors"],
                    depth=3,
                ),
                study.trials[1].params,
            )

    def test_objective_extraction_uses_validation_metrics_only(self) -> None:
        """A high test score should not influence the Optuna objective."""
        result = {
            "history": {
                "val_metrics": [
                    {"NDCG@40": 0.2, "AveragePopularity@40": 1.5},
                    {"NDCG@40": 0.3, "AveragePopularity@40": 1.7},
                ],
            },
            "test_metrics": {"NDCG@40": 0.99},
        }

        objective = search.extract_validation_objective(
            result,
            search.ObjectiveSpec(metric="NDCG@40", split="val", direction="maximize"),
        )

        self.assertEqual(objective, 0.3)

    def test_validation_crru_objective_uses_validation_metrics_only(self) -> None:
        """The composite search objective should not read test metrics."""
        val_metrics = {
            "NDCG@20": 0.10,
            "Recall@20": 0.20,
            "HitRatio@20": 0.30,
            "Personalization@20": 0.50,
            "AveragePopularity@20": 1.0,
            "NDCG@40": 0.30,
            "Recall@40": 0.40,
            "HitRatio@40": 0.50,
            "Personalization@40": 0.70,
            "AveragePopularity@40": 3.0,
        }
        result = {
            "history": {"val_metrics": [val_metrics]},
            "training_time_s": 10.0,
            "epochs_stopped_at": 2,
            "peak_vram_mb": 512.0,
            "test_metrics": {"NDCG@40": 0.99},
        }

        objective = search.extract_validation_objective(
            result,
            search.ObjectiveSpec(
                metric=search.VALIDATION_ONLINE_CRRU_METRIC,
                split="val",
                direction="maximize",
            ),
        )

        self.assertAlmostEqual(
            objective,
            search.compute_validation_online_crru_objective(
                val_metrics,
                peak_vram_mb=512.0,
                epoch_time_s=5.0,
            ),
        )


class SearchExecutionTests(unittest.TestCase):
    """Smoke the search runner without doing real model training."""

    def test_search_trials_are_target_completed_count_not_extra_attempts(self) -> None:
        """A repeated command with the same compatible study should not rerun training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optuna_db = Path(tmpdir) / "optuna.db"
            storage = f"sqlite:///{optuna_db}"
            spec = search.SearchSpaceSpec(
                name="tiny-search",
                description="test search space",
                base_profile="core-ucagnn-mainline",
                datasets=("amazonbook",),
                objective=search.ObjectiveSpec(metric="NDCG@40"),
                max_epochs=1,
                trials=1,
                config_overrides={
                    "sample_interactions": 50,
                    "loader_max_rows": 50,
                },
                parameters={
                    "lr_scheduler": {
                        "type": "categorical",
                        "choices": ["cosine"],
                    },
                },
            )
            study = optuna.create_study(
                study_name="existing-study",
                storage=storage,
                direction="maximize",
            )
            study.add_trial(
                optuna.trial.create_trial(
                    value=0.42,
                    params={
                        search._parameter_storage_name(
                            "lr_scheduler",
                            spec.parameters["lr_scheduler"],
                        ): "cosine",
                    },
                    distributions={
                        search._parameter_storage_name(
                            "lr_scheduler",
                            spec.parameters["lr_scheduler"],
                        ): optuna.distributions.CategoricalDistribution(["cosine"]),
                    },
                    user_attrs={
                        "search_space": "tiny-search",
                        "objective_metric": "NDCG@40",
                        "objective_split": "val",
                        "sampled_params": {"lr_scheduler": "cosine"},
                        "amazonbook.objective": 0.42,
                    },
                    state=optuna.trial.TrialState.COMPLETE,
                ),
            )
            args = SimpleNamespace(
                list_spaces=False,
                space="tiny-search",
                dataset=None,
                trials=1,
                study_name="existing-study",
                storage=storage,
                dry_run=False,
                device="cpu",
                data_dir="data",
                no_mlflow=True,
                mlflow_tracking_uri=None,
                mlflow_experiment_name="ucagnn-search-test",
            )

            with (
                patch.object(search, "resolve_search_space", return_value=spec),
                patch.object(search, "run_experiment") as run_experiment,
            ):
                exit_code = search.run_search(args)

            self.assertEqual(exit_code, 0)
            run_experiment.assert_not_called()

    def test_real_pruned_trials_count_toward_search_budget(self) -> None:
        """A pruned training trial is informative and should satisfy --trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optuna_db = Path(tmpdir) / "optuna.db"
            storage = f"sqlite:///{optuna_db}"
            spec = search.SearchSpaceSpec(
                name="tiny-search",
                description="test search space",
                base_profile="core-ucagnn-mainline",
                datasets=("amazonbook",),
                objective=search.ObjectiveSpec(metric="NDCG@40"),
                max_epochs=1,
                trials=1,
                config_overrides={
                    "sample_interactions": 50,
                    "loader_max_rows": 50,
                },
                parameters={
                    "lr_scheduler": {
                        "type": "categorical",
                        "choices": ["cosine"],
                    },
                },
            )
            storage_param = search._parameter_storage_name(
                "lr_scheduler",
                spec.parameters["lr_scheduler"],
            )
            study = optuna.create_study(
                study_name="existing-study",
                storage=storage,
                direction="maximize",
            )
            study.add_trial(
                optuna.trial.create_trial(
                    params={storage_param: "cosine"},
                    distributions={
                        storage_param: optuna.distributions.CategoricalDistribution(["cosine"]),
                    },
                    user_attrs={
                        "search_space": "tiny-search",
                        "objective_metric": "NDCG@40",
                        "objective_split": "val",
                        "sampled_params": {"lr_scheduler": "cosine"},
                        "amazonbook.pruned": True,
                        "amazonbook.last_pruning_objective": 0.25,
                    },
                    state=optuna.trial.TrialState.PRUNED,
                ),
            )
            args = SimpleNamespace(
                list_spaces=False,
                space="tiny-search",
                dataset=None,
                trials=1,
                study_name="existing-study",
                storage=storage,
                dry_run=False,
                device="cpu",
                data_dir="data",
                no_mlflow=True,
                mlflow_tracking_uri=None,
                mlflow_experiment_name="ucagnn-search-test",
            )

            with (
                patch.object(search, "resolve_search_space", return_value=spec),
                patch.object(search, "run_experiment") as run_experiment,
            ):
                exit_code = search.run_search(args)

            self.assertEqual(exit_code, 0)
            run_experiment.assert_not_called()

    def test_multi_dataset_default_runs_dataset_local_studies(self) -> None:
        """A multi-dataset command should optimize one independent study per dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optuna_db = Path(tmpdir) / "optuna.db"
            storage = f"sqlite:///{optuna_db}"
            common_kwargs = {
                "name": "tiny-search",
                "description": "test search space",
                "base_profile": "core-ucagnn-mainline",
                "objective": search.ObjectiveSpec(metric="NDCG@40"),
                "max_epochs": 1,
                "trials": 1,
                "config_overrides": {
                    "sample_interactions": 50,
                    "loader_max_rows": 50,
                },
                "parameters": {
                    "lr_scheduler": {
                        "type": "categorical",
                        "choices": ["cosine"],
                    },
                },
            }
            all_spec = search.SearchSpaceSpec(
                datasets=("amazonbook", "movielens1m"),
                **common_kwargs,
            )
            dataset_specs = {
                dataset: search.SearchSpaceSpec(datasets=(dataset,), **common_kwargs)
                for dataset in all_spec.datasets
            }
            args = SimpleNamespace(
                list_spaces=False,
                space="tiny-search",
                dataset=None,
                trials=1,
                study_name=None,
                storage=storage,
                dry_run=False,
                device="cpu",
                data_dir="data",
                no_mlflow=True,
                mlflow_tracking_uri=None,
                mlflow_experiment_name="ucagnn-search-test",
            )

            def resolve_space(_space_name: str, dataset: str | None = None):
                if dataset is None:
                    return all_spec
                return dataset_specs[dataset]

            def fake_run_experiment(config, **_kwargs):
                value = 0.41 if config.dataset == "amazonbook" else 0.52
                return {
                    "exp_id": None,
                    "canonical_name": f"{config.dataset}-canonical",
                    "checkpoint_path": None,
                    "epochs_stopped_at": 1,
                    "training_time_s": 0.1,
                    "peak_vram_mb": 0.0,
                    "history": {
                        "val_metrics": [
                            {
                                "NDCG@40": value,
                                "AveragePopularity@40": 1.1,
                            },
                        ],
                    },
                    "test_metrics": {},
                }

            with (
                patch.object(search, "resolve_search_space", side_effect=resolve_space),
                patch.object(search, "run_experiment", side_effect=fake_run_experiment),
            ):
                exit_code = search.run_search(args)

            self.assertEqual(exit_code, 0)
            for dataset, expected in (("amazonbook", 0.41), ("movielens1m", 0.52)):
                study = optuna.load_study(
                    study_name=f"tiny-search-{dataset}-val-ndcg-40",
                    storage=storage,
                )
                self.assertEqual(len(study.trials), 1)
                trial = study.trials[0]
                self.assertAlmostEqual(float(trial.value), expected)
                self.assertEqual(trial.user_attrs["datasets"], [dataset])
                self.assertAlmostEqual(trial.user_attrs[f"{dataset}.objective"], expected)

    def test_dataset_local_study_ignores_existing_imported_rows_for_budget(self) -> None:
        """Previously imported rows should not satisfy the fresh local target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optuna_db = Path(tmpdir) / "optuna.db"
            storage = f"sqlite:///{optuna_db}"
            spec = search.SearchSpaceSpec(
                name="tiny-search",
                description="test search space",
                base_profile="core-ucagnn-mainline",
                datasets=("amazonbook",),
                objective=search.ObjectiveSpec(metric="NDCG@40"),
                max_epochs=1,
                trials=1,
                config_overrides={
                    "sample_interactions": 50,
                    "loader_max_rows": 50,
                },
                parameters={
                    "lr_scheduler": {
                        "type": "categorical",
                        "choices": ["cosine"],
                    },
                },
            )
            storage_param = search._parameter_storage_name(
                "lr_scheduler",
                spec.parameters["lr_scheduler"],
            )
            dataset_study = optuna.create_study(
                study_name="tiny-search-amazonbook-val-ndcg-40",
                storage=storage,
                direction="maximize",
            )
            dataset_study.add_trial(
                optuna.trial.create_trial(
                    value=0.70,
                    params={storage_param: "cosine"},
                    distributions={
                        storage_param: optuna.distributions.CategoricalDistribution(
                            ["cosine"],
                        ),
                    },
                    user_attrs={
                        "search_space": "tiny-search",
                        "study_name": "tiny-search-amazonbook-val-ndcg-40",
                        "datasets": ["amazonbook"],
                        "objective_metric": "NDCG@40",
                        "objective_split": "val",
                        "objective_value": 0.70,
                        "sampled_params": {"lr_scheduler": "cosine"},
                        "amazonbook.objective": 0.70,
                        "seeded_from_study": "tiny-search-all-val-ndcg-40",
                        "seeded_from_trial": 0,
                    },
                    state=optuna.trial.TrialState.COMPLETE,
                ),
            )
            args = SimpleNamespace(
                list_spaces=False,
                space="tiny-search",
                dataset="amazonbook",
                trials=1,
                study_name=None,
                storage=storage,
                dry_run=False,
                device="cpu",
                data_dir="data",
                no_mlflow=True,
                mlflow_tracking_uri=None,
                mlflow_experiment_name="ucagnn-search-test",
            )

            def fake_run_experiment(config, **_kwargs):
                return {
                    "exp_id": None,
                    "canonical_name": f"{config.dataset}-fresh",
                    "checkpoint_path": None,
                    "epochs_stopped_at": 1,
                    "training_time_s": 0.2,
                    "peak_vram_mb": 0.0,
                    "history": {
                        "val_metrics": [
                            {
                                "NDCG@40": 0.72,
                                "AveragePopularity@40": 1.0,
                            },
                        ],
                    },
                    "test_metrics": {},
                }

            with (
                patch.object(search, "resolve_search_space", return_value=spec),
                patch.object(
                    search, "run_experiment", side_effect=fake_run_experiment
                ) as run_experiment,
            ):
                exit_code = search.run_search(args)

            self.assertEqual(exit_code, 0)
            run_experiment.assert_called_once()
            study = optuna.load_study(
                study_name="tiny-search-amazonbook-val-ndcg-40",
                storage=storage,
            )
            self.assertEqual(len(study.trials), 2)
            seeded_trial = study.trials[0]
            fresh_trial = study.trials[1]
            self.assertAlmostEqual(float(seeded_trial.value), 0.70)
            self.assertEqual(seeded_trial.user_attrs["datasets"], ["amazonbook"])
            self.assertEqual(
                seeded_trial.user_attrs["seeded_from_study"],
                "tiny-search-all-val-ndcg-40",
            )
            self.assertEqual(seeded_trial.user_attrs["seeded_from_trial"], 0)
            self.assertAlmostEqual(float(fresh_trial.value), 0.72)
            self.assertNotIn("seeded_from_study", fresh_trial.user_attrs)
            self.assertEqual(fresh_trial.user_attrs["sampled_params"], {"lr_scheduler": "cosine"})

    def test_one_trial_search_creates_storage_without_thesis_trial_mirror(self) -> None:
        """The controller should keep Optuna trial metadata in Optuna storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            optuna_db = tmp_path / "optuna.db"
            thesis_db = tmp_path / "thesis.db"
            spec = search.SearchSpaceSpec(
                name="tiny-search",
                description="test search space",
                base_profile="core-ucagnn-mainline",
                datasets=("amazonbook",),
                objective=search.ObjectiveSpec(metric="NDCG@40"),
                max_epochs=1,
                trials=1,
                config_overrides={
                    "sample_interactions": 50,
                    "loader_max_rows": 50,
                },
                parameters={
                    "lr_scheduler": {
                        "type": "categorical",
                        "choices": ["cosine"],
                    },
                },
            )
            args = SimpleNamespace(
                list_spaces=False,
                space="tiny-search",
                dataset=None,
                trials=1,
                study_name="tiny-study",
                storage=f"sqlite:///{optuna_db}",
                dry_run=False,
                device="cpu",
                data_dir="data",
                no_mlflow=True,
                mlflow_tracking_uri=None,
                mlflow_experiment_name="ucagnn-search-test",
            )

            def fake_run_experiment(config, **kwargs):
                self.assertFalse(kwargs["save_checkpoint"])
                self.assertFalse(kwargs["enable_mlflow"])
                self.assertFalse(kwargs["auto_resume"])
                self.assertEqual(kwargs["checkpoint_every"], 0)
                self.assertFalse(kwargs["include_refined_diagnostics"])
                self.assertFalse(kwargs["evaluate_test"])
                tracker = ExperimentLogger(db_path=str(thesis_db))
                try:
                    exp_id = tracker.log_experiment(
                        config.dataset,
                        config,
                        preset="ucagnn",
                        training_mode="mini_batch",
                        status="running",
                        batch_id=kwargs["batch_id"],
                        profile_name=kwargs["profile_name"],
                        change_note=kwargs["change_note"],
                    )
                    tracker.log_metric(exp_id, "gpu_utilization_pct", 12.0, split="train")
                    tracker.update_experiment_status(exp_id, status="completed")
                finally:
                    tracker.close()
                return {
                    "exp_id": exp_id,
                    "canonical_name": "tiny-canonical",
                    "checkpoint_path": None,
                    "epochs_stopped_at": 1,
                    "training_time_s": 0.1,
                    "peak_vram_mb": 0.0,
                    "history": {
                        "val_metrics": [
                            {
                                "NDCG@40": 0.42,
                                "AveragePopularity@40": 1.1,
                            },
                        ],
                    },
                    "test_metrics": {},
                }

            with (
                patch.object(search, "resolve_search_space", return_value=spec),
                patch.object(search, "run_experiment", side_effect=fake_run_experiment),
                patch.object(search, "THESIS_DB_PATH", thesis_db),
            ):
                exit_code = search.run_search(args)

            self.assertEqual(exit_code, 0)
            self.assertTrue(optuna_db.exists())
            with sqlite3.connect(thesis_db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT batch_id, profile_name, change_note, config_json
                    FROM experiments
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                ).fetchone()
                optuna_table = conn.execute(
                    """
                    SELECT 1
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name = 'optuna_search_trials'
                    """
                ).fetchone()

            self.assertIsNotNone(row)
            self.assertEqual(row["batch_id"], "optuna-tiny-study-trial-0")
            self.assertEqual(row["profile_name"], "tiny-search")
            change_note = json.loads(row["change_note"])
            self.assertEqual(change_note["search_space"], "tiny-search")
            self.assertEqual(change_note["study_name"], "tiny-study")
            self.assertEqual(change_note["trial_number"], 0)
            config_json = json.loads(row["config_json"])
            self.assertEqual(config_json["sample_interactions"], 50)
            self.assertEqual(config_json["loader_max_rows"], 50)
            self.assertIsNone(optuna_table)

            study = optuna.load_study(study_name="tiny-study", storage=f"sqlite:///{optuna_db}")
            trial = study.trials[0]
            self.assertEqual(trial.user_attrs["search_space"], "tiny-search")
            self.assertEqual(trial.user_attrs["objective_metric"], "NDCG@40")
            self.assertAlmostEqual(trial.user_attrs["amazonbook.objective"], 0.42)
            self.assertEqual(trial.user_attrs["sampled_params"], {"lr_scheduler": "cosine"})

    def test_search_returns_failure_when_only_new_trials_fail(self) -> None:
        """Historical completed trials must not hide failures from this invocation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optuna_db = Path(tmpdir) / "optuna.db"
            storage = f"sqlite:///{optuna_db}"
            study = optuna.create_study(
                study_name="existing-study",
                storage=storage,
                direction="maximize",
            )
            study.add_trial(
                optuna.trial.create_trial(
                    value=0.1,
                    params={},
                    distributions={},
                    state=optuna.trial.TrialState.COMPLETE,
                ),
            )
            spec = search.SearchSpaceSpec(
                name="tiny-search",
                description="test search space",
                base_profile="core-ucagnn-mainline",
                datasets=("amazonbook",),
                objective=search.ObjectiveSpec(metric="NDCG@40"),
                max_epochs=1,
                trials=1,
                config_overrides={
                    "sample_interactions": 50,
                    "loader_max_rows": 50,
                },
                parameters={
                    "lr_scheduler": {
                        "type": "categorical",
                        "choices": ["cosine"],
                    },
                },
            )
            args = SimpleNamespace(
                list_spaces=False,
                space="tiny-search",
                dataset=None,
                trials=1,
                study_name="existing-study",
                storage=storage,
                dry_run=False,
                device="cpu",
                data_dir="data",
                no_mlflow=True,
                mlflow_tracking_uri=None,
                mlflow_experiment_name="ucagnn-search-test",
            )

            with (
                patch.object(search, "resolve_search_space", return_value=spec),
                patch.object(
                    search,
                    "suggest_trial_overrides",
                    side_effect=ValueError("bad search contract"),
                ),
            ):
                exit_code = search.run_search(args)

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
