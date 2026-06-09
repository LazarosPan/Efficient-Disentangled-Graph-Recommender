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
        spec = search.resolve_search_space(
            "ucagnn-amazonbook-recovery",
            dataset="amazonbook",
        )

        with patch.object(search, "run_experiment") as run_experiment:
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
        self.assertEqual(base_config["epochs"], 80)
        self.assertEqual(base_config["device"], "cpu")

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
                "interest_gnn_layers": 2,
                "conformity_gnn_layers": 2,
                "num_neighbors_depth_2": "[6,3]",
                "use_popularity_head": False,
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
                    "interest_gnn_layers": 1,
                    "conformity_gnn_layers": 1,
                    "num_neighbors_depth_1": "[8]",
                },
            )
            study.enqueue_trial(
                {
                    "interest_gnn_layers": 2,
                    "conformity_gnn_layers": 3,
                    "num_neighbors_depth_3": "[16,8,4]",
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
            self.assertIn("num_neighbors_depth_1", study.trials[0].params)
            self.assertIn("num_neighbors_depth_3", study.trials[1].params)

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


class SearchExecutionTests(unittest.TestCase):
    """Smoke the search runner without doing real model training."""

    def test_one_trial_search_creates_storage_and_logs_search_metadata(self) -> None:
        """The controller should create Optuna storage and mirror trial rows to SQLite."""
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
                overwrite_checkpoint=False,
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
                            {"NDCG@40": 0.42, "AveragePopularity@40": 1.1},
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
                search_row = conn.execute(
                    """
                    SELECT study_name, search_space, trial_number, dataset,
                           experiment_id, batch_id, objective_metric, objective_split,
                           objective_direction, objective_value, dataset_objective_value,
                           params_json, state, runtime_s, peak_vram_mb,
                           average_popularity_40
                    FROM optuna_search_trials
                    ORDER BY id DESC
                    LIMIT 1
                    """,
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
            self.assertIsNotNone(search_row)
            self.assertEqual(search_row["study_name"], "tiny-study")
            self.assertEqual(search_row["search_space"], "tiny-search")
            self.assertEqual(search_row["trial_number"], 0)
            self.assertEqual(search_row["dataset"], "amazonbook")
            self.assertEqual(search_row["batch_id"], "optuna-tiny-study-trial-0")
            self.assertEqual(search_row["objective_metric"], "NDCG@40")
            self.assertEqual(search_row["objective_split"], "val")
            self.assertEqual(search_row["objective_direction"], "maximize")
            self.assertEqual(search_row["state"], "completed")
            self.assertAlmostEqual(search_row["objective_value"], 0.42)
            self.assertAlmostEqual(search_row["dataset_objective_value"], 0.42)
            self.assertAlmostEqual(search_row["runtime_s"], 0.1)
            self.assertAlmostEqual(search_row["peak_vram_mb"], 0.0)
            self.assertAlmostEqual(search_row["average_popularity_40"], 1.1)
            self.assertEqual(json.loads(search_row["params_json"]), {"lr_scheduler": "cosine"})


if __name__ == "__main__":
    unittest.main()
