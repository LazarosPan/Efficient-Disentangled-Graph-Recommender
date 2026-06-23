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
from src.data.feature_groups import feature_subset_profile_matrix, required_feature_subset_profiles
from src.utils.crru import compute_validation_online_crru_components_for_k
from src.utils.experiment_logger import ExperimentLogger
from src.utils.method_naming import EDGREC_LEGACY_PRESET


class SearchSpaceValidationTests(unittest.TestCase):
    """Pin search-space validation and config resolution for Optuna runs."""

    def test_search_spaces_live_outside_experiment_catalog(self) -> None:
        """Optuna configuration should stay in its own focused JSON file."""
        self.assertNotIn("search_spaces", load_experiment_catalog())
        self.assertIn("search_spaces", load_search_spaces_catalog())
        self.assertIn("edgrec-core-optimization", search_space_names())
        self.assertIn("edgrec-mechanism-coarse", search_space_names())
        self.assertIn("edgrec-lite-kuairec-search", search_space_names())
        self.assertIn("edgrec-lite-movielens-search", search_space_names())
        self.assertIn("edgrec-lite-kuairand-search", search_space_names())
        self.assertIn("amazonbook-edgrec-candidate-search", search_space_names())

    def test_search_spaces_resolve_to_edgrec_configs(self) -> None:
        """All search spaces should resolve through the shared config builder."""
        self.assertIn("edgrec-core-optimization", search_space_names())
        data_dir = "/tmp/edgrec-search-test-data"

        for space_name in search_space_names():
            spec = search.resolve_search_space(space_name)
            config = search.build_search_config(
                spec,
                dataset=spec.datasets[0],
                device="cpu",
                data_dir=data_dir,
            )

            self.assertEqual(config.baseline_family, "edgrec")
            self.assertEqual(config.epochs, spec.max_epochs)
            self.assertEqual(config.device, "cpu")
            self.assertEqual(config.data_dir, data_dir)
            self.assertEqual(len(config.num_neighbors), config.max_gnn_layers)

    def test_search_spaces_resolve_sampled_sparse_embedding_optimizer(self) -> None:
        """Search spaces that sample sparse optimizers should build valid configs."""
        for space_name in search_space_names():
            spec = search.resolve_search_space(space_name)
            if "embedding_optimizer" not in spec.parameters:
                continue
            for dataset in spec.datasets:
                config = search.build_search_config(
                    spec,
                    dataset=dataset,
                    sampled_overrides={"embedding_optimizer": "sparseadam"},
                    device="cpu",
                    data_dir="data",
                )

                self.assertEqual(config.embedding_optimizer, "sparseadam")
                self.assertTrue(config.embedding_sparse_optimizer)
                self.assertEqual(config.training_graph_mode, "sampled")

    def test_mechanism_sparse_optimizer_replays_profile_trial_shape(self) -> None:
        """Mechanism-search profile labels should build with sparse embedding mode."""
        spec = search.resolve_search_space("edgrec-mechanism-coarse")
        data_dir = "/tmp/edgrec-search-test-data"
        sampled_overrides = {
            "use_learned_score_mix": False,
            "score_weight_interest": 0.4,
            "score_weight_conformity": 0.3,
            "score_weight_popularity": 0.3,
            "separate_item_branch_embeddings": True,
            "use_popularity_head": False,
            "use_features": False,
            "loss_weight_interest_bpr": 0.03,
            "loss_weight_conformity_bpr": 0.03,
            "loss_weight_independence": 0.0,
            "loss_weight_contrastive": 0.0,
            "loss_weight_popularity": 0.025,
            "interest_gnn_layers": 1,
            "conformity_gnn_layers": 1,
            "num_neighbors": [8],
            "lr": 0.0008,
            "weight_decay": 0.0,
            "dropout": 0.1,
            "grad_clip_norm": 2.0,
            "embedding_optimizer": "sparseadam",
            "train_edge_keep_prob": 0.6,
            "n_negatives": 3,
            "dice_sampler_margin": 80.0,
            "score_mix_min_weight": 0.05,
            "loss_normalization": "none",
            "item_universe_policy": "random_exposure_items_only",
        }

        config = search.build_search_config(
            spec,
            dataset="kuairand1k",
            sampled_overrides=sampled_overrides,
            device="cpu",
            data_dir=data_dir,
        )

        self.assertEqual(config.data_dir, data_dir)
        self.assertEqual(config.embedding_optimizer, "sparseadam")
        self.assertTrue(config.embedding_sparse_optimizer)
        self.assertEqual(config.training_graph_mode, "sampled")

    def test_old_edgrec_search_space_name_is_not_a_catalog_alias(self) -> None:
        """Catalog CLIs should expose only the public EDGRec search-space names."""
        with self.assertRaises(KeyError):
            search.resolve_search_space(
                f"{EDGREC_LEGACY_PRESET}-core-optimization",
                dataset="amazonbook",
            )

    def test_legacy_edgrec_trial_metadata_counts_for_public_budget(self) -> None:
        """Old Optuna trial labels should still count against the EDGRec budget."""
        spec = search.resolve_search_space(
            "edgrec-core-optimization",
            dataset="amazonbook",
        )
        study = optuna.create_study(direction=spec.objective.direction)
        legacy_trial = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=1.0,
            user_attrs={
                "search_space": f"{EDGREC_LEGACY_PRESET}-core-optimization",
                "search_space_revision": search.search_space_revision(spec),
                "objective_metric": spec.objective.metric,
                "objective_split": spec.objective.split,
            },
        )

        study.add_trial(legacy_trial)

        self.assertEqual(search._budget_informative_trials(study, spec), [study.trials[0]])

    def test_mechanism_search_space_uses_profile_layer_and_accuracy_objective(self) -> None:
        """The broad mechanism search should keep profile labels out of model config."""
        spec = search.resolve_search_space(
            "edgrec-mechanism-coarse",
            dataset="kuairec_v2",
        )
        all_dataset_spec = search.resolve_search_space("edgrec-mechanism-coarse")
        payload = search.build_dry_run_payload(
            spec,
            study_name="mechanism-dry-run",
            storage="sqlite:///:memory:",
            device="cpu",
            data_dir="data",
        )

        self.assertEqual(spec.datasets, ("kuairec_v2",))
        self.assertIn("kuairand1k", all_dataset_spec.datasets)
        self.assertEqual(spec.objective.metric, search.VALIDATION_ACCURACY_METRIC)
        self.assertEqual(spec.sampler.n_startup_trials, 40)
        self.assertNotIn("amazonbook", spec.datasets)
        self.assertIn("score_fusion_profile", payload["parameters"])
        self.assertIn("score_fusion_profile", payload["profile_overrides"])
        self.assertEqual(payload["profile_overrides"], spec.profile_overrides)
        self.assertEqual(
            spec.profile_overrides["score_fusion_profile"]["fixed_interest_context"],
            {
                "use_learned_score_mix": False,
                "score_weight_interest": 0.7,
                "score_weight_conformity": 0.0,
                "score_weight_popularity": 0.3,
            },
        )
        self.assertEqual(spec.parameters["dropout"]["choices"], [0.0, 0.1, 0.2, 0.3])
        self.assertEqual(
            spec.parameters["dice_sampler_margin"]["choices"],
            [10.0, 20.0, 40.0],
        )
        base_config = payload["base_configs"]["kuairec_v2"]
        self.assertFalse(base_config["separate_item_branch_embeddings"])
        self.assertEqual(base_config["loss_normalization"], "none")
        self.assertTrue(base_config["use_learned_score_mix"])

    def test_mechanism_search_space_applies_dataset_local_scalar_ladders(self) -> None:
        """Dataset-local scalar choices should cover known useful regions only."""
        movielens_spec = search.resolve_search_space(
            "edgrec-mechanism-coarse",
            dataset="movielens1m",
        )
        kuairand_spec = search.resolve_search_space(
            "edgrec-mechanism-coarse",
            dataset="kuairand1k",
        )

        self.assertEqual(
            movielens_spec.parameters["lr"]["choices"],
            [0.0004, 0.0008, 0.0015, 0.003],
        )
        self.assertEqual(
            movielens_spec.parameters["dice_sampler_margin"]["choices"],
            [20.0, 40.0, 80.0],
        )
        self.assertEqual(
            movielens_spec.parameters["score_mix_min_weight"]["choices"],
            [0.02, 0.05, 0.1],
        )
        self.assertEqual(
            kuairand_spec.parameters["dice_sampler_margin"]["choices"],
            [50.0, 70.0, 80.0],
        )
        self.assertEqual(
            kuairand_spec.parameters["score_mix_min_weight"]["choices"],
            [0.02, 0.05],
        )
        self.assertEqual(
            kuairand_spec.parameters["item_universe_policy"]["choices"],
            ["random_exposure_items_only"],
        )

    def test_feature_subset_search_uses_dataset_local_trust_regions(self) -> None:
        """Feature-subset search should not fall back to broad generic scalar ranges."""
        raw_space = load_search_spaces_catalog()["search_spaces"]["edgrec-feature-subset-search"]
        by_dataset = raw_space["parameters_by_dataset"]

        self.assertEqual(raw_space["trials"], 32)
        self.assertEqual(
            by_dataset["kuairec_v2"]["lr"]["choices"],
            [0.0003, 0.0004, 0.0008],
        )
        self.assertEqual(
            by_dataset["kuairec_v2"]["weight_decay"]["choices"],
            [0.0, 0.00000003, 0.000001],
        )
        self.assertEqual(
            by_dataset["kuairand1k"]["dice_sampler_margin"]["choices"],
            [50.0, 70.0, 80.0],
        )
        self.assertEqual(
            by_dataset["movielens1m"]["weight_decay"]["choices"],
            [0.0, 0.00001, 0.0001],
        )
        self.assertEqual(
            by_dataset["amazonbook"]["lr"]["choices"],
            [0.003, 0.01],
        )

    def test_lite_kuairec_search_samples_watch_ratio_threshold_presets(self) -> None:
        """KuaiRec-lite Optuna should test the sparse watch-ratio threshold labels."""
        spec = search.resolve_search_space(
            "edgrec-lite-kuairec-search",
            dataset="kuairec_v2",
        )

        self.assertEqual(
            spec.parameters["preprocessing_preset"]["choices"],
            [
                "kuairec_big_matrix_watch_ratio_threshold_0_5",
                "kuairec_big_matrix_watch_ratio_threshold_0_75",
                "kuairec_big_matrix_watch_ratio_threshold_1_0",
            ],
        )

        base_config = search.build_search_config(
            spec,
            dataset="kuairec_v2",
            device="cpu",
            data_dir="data",
        )
        fixed_trial = optuna.trial.FixedTrial(
            {
                search._parameter_storage_name(
                    "preprocessing_preset",
                    spec.parameters["preprocessing_preset"],
                ): "kuairec_big_matrix_watch_ratio_threshold_0_75",
                search._parameter_storage_name(
                    "graph_profile",
                    spec.parameters["graph_profile"],
                ): "shallow_i1_c1",
                search._parameter_storage_name(
                    "n_negatives",
                    spec.parameters["n_negatives"],
                ): 1,
                search._parameter_storage_name(
                    "loss_weight_interest_bpr",
                    spec.parameters["loss_weight_interest_bpr"],
                ): 0.01,
                search._parameter_storage_name(
                    "loss_weight_conformity_bpr",
                    spec.parameters["loss_weight_conformity_bpr"],
                ): 0.01,
                search._parameter_storage_name(
                    "loss_weight_independence",
                    spec.parameters["loss_weight_independence"],
                ): 0.0,
                search._parameter_storage_name(
                    "loss_weight_popularity",
                    spec.parameters["loss_weight_popularity"],
                ): 0.0,
            },
        )

        resolution = search.resolve_trial_parameters(
            fixed_trial,
            spec,
            base_config=base_config,
        )
        config = search.build_search_config(
            spec,
            dataset="kuairec_v2",
            sampled_overrides=resolution.config_overrides,
            device="cpu",
            data_dir="data",
        )

        self.assertEqual(
            resolution.sampled_params["preprocessing_preset"],
            "kuairec_big_matrix_watch_ratio_threshold_0_75",
        )
        self.assertEqual(
            config.preprocessing_preset,
            "kuairec_big_matrix_watch_ratio_threshold_0_75",
        )
        self.assertEqual(resolution.sampled_params["graph_profile"], "shallow_i1_c1")
        self.assertFalse(config.use_features)
        self.assertEqual(config.interest_gnn_layers, 1)
        self.assertEqual(config.conformity_gnn_layers, 1)
        self.assertEqual(config.num_neighbors, [8])
        self.assertEqual(
            len(search.search_space_revision(spec)),
            search.SEARCH_SPACE_REVISION_HASH_LENGTH,
        )

    def test_amazonbook_candidate_search_compares_compact_and_deep_families(self) -> None:
        """AmazonBook should have a dataset-local compact-vs-deep EDGRec search."""
        spec = search.resolve_search_space(
            "amazonbook-edgrec-candidate-search",
            dataset="amazonbook",
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
                    "graph_profile",
                    spec.parameters["graph_profile"],
                ): "deep_features",
                search._parameter_storage_name(
                    "n_negatives",
                    spec.parameters["n_negatives"],
                ): 1,
                search._parameter_storage_name(
                    "score_mix_min_weight",
                    spec.parameters["score_mix_min_weight"],
                ): 0.02,
                search._parameter_storage_name(
                    "loss_weight_popularity",
                    spec.parameters["loss_weight_popularity"],
                ): 0.025,
            },
        )

        self.assertEqual(spec.datasets, ("amazonbook",))
        self.assertEqual(spec.base_profile, "amazonbook-edgrec-compact-candidate")
        self.assertEqual(spec.parameters["graph_profile"]["choices"], ["compact", "deep_features"])
        self.assertFalse(base_config.use_features)
        self.assertEqual(base_config.interest_gnn_layers, 1)
        self.assertEqual(base_config.conformity_gnn_layers, 2)
        self.assertEqual(base_config.num_neighbors, [10, 5])

        resolution = search.resolve_trial_parameters(
            fixed_trial,
            spec,
            base_config=base_config,
        )
        config = search.build_search_config(
            spec,
            dataset="amazonbook",
            sampled_overrides=resolution.config_overrides,
            device="cpu",
            data_dir="data",
        )

        self.assertEqual(resolution.sampled_params["graph_profile"], "deep_features")
        self.assertTrue(config.use_features)
        self.assertEqual(config.feature_gate_init, -4.0)
        self.assertEqual(config.interest_gnn_layers, 2)
        self.assertEqual(config.conformity_gnn_layers, 3)
        self.assertEqual(config.num_neighbors, [8, 4, 2])
        self.assertEqual(config.loss_weight_contrastive, 0.025)

    def test_search_parser_accepts_comma_separated_space_queue(self) -> None:
        """search-experiments should mirror formal-run's queue syntax."""
        parser = search.build_search_parser()

        args = parser.parse_args(
            [
                "--space",
                "edgrec-mechanism-coarse,edgrec-core-optimization",
                "--dry-run",
            ],
        )

        self.assertEqual(
            args.space,
            "edgrec-mechanism-coarse,edgrec-core-optimization",
        )
        self.assertEqual(
            search._parse_search_space_sequence(args.space),
            ["edgrec-mechanism-coarse", "edgrec-core-optimization"],
        )

    def test_search_main_runs_comma_separated_spaces_in_order(self) -> None:
        """Multiple search spaces should execute sequentially and report aggregate failure."""
        args = SimpleNamespace(
            list_spaces=False,
            space="edgrec-mechanism-coarse, edgrec-core-optimization",
            dataset="kuairec_v2",
            study_name=None,
            dry_run=False,
            data_dir="data",
        )
        seen: list[str] = []

        def fake_run_search_space(_args, *, space_name: str) -> int:
            seen.append(space_name)
            return 0 if space_name == "edgrec-mechanism-coarse" else 1

        with patch.object(
            search,
            "_run_search_space",
            side_effect=fake_run_search_space,
        ):
            exit_code = search.run_search(args)

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            seen,
            ["edgrec-mechanism-coarse", "edgrec-core-optimization"],
        )

    def test_search_space_queue_rejects_ambiguous_explicit_study_name(self) -> None:
        """One explicit study name should not be reused across queued spaces."""
        args = SimpleNamespace(
            list_spaces=False,
            space="edgrec-mechanism-coarse,edgrec-core-optimization",
            study_name="shared-study",
            dry_run=False,
        )

        with self.assertRaisesRegex(ValueError, "--study-name is ambiguous"):
            search.run_search(args)

    def test_search_space_resolves_sampler_and_pruner_from_catalog(self) -> None:
        """Sampler/pruner blocks in search_spaces.json should not be dropped."""
        spec = search.resolve_search_space(
            "edgrec-core-optimization",
            dataset="amazonbook",
        )

        self.assertEqual(spec.sampler.seed, 42)
        self.assertEqual(spec.sampler.name, "tpe")
        self.assertFalse(spec.sampler.multivariate)
        self.assertFalse(spec.sampler.group)
        self.assertEqual(spec.pruner.name, "hyperband")
        self.assertEqual(spec.pruner.min_resource, 15)

    def test_auto_batch_search_treats_batch_size_as_runtime_only(self) -> None:
        """Historical sampled batch sizes should not break logical param matching."""
        spec = search.SearchSpaceSpec(
            name="tiny",
            description="test search space",
            base_profile="core-edgrec-mainline",
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
            "base_profile": "core-edgrec-mainline",
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
            self.assertRaisesRegex(ValueError, "EDGRec-only"),
        ):
            search.resolve_search_space("bad-paper-space")

    def test_search_space_validation_requires_json_owned_profile_overrides(self) -> None:
        """Profile labels should be backed by profile_overrides in search_spaces.json."""
        bad_space = {
            "name": "bad-profile-space",
            "base_profile": "core-edgrec-mainline",
            "datasets": ["kuairec_v2"],
            "parameters": {
                "score_fusion_profile": {
                    "type": "categorical",
                    "choices": ["fixed_interest_context"],
                },
            },
        }

        with (
            patch.object(search, "get_search_space", return_value=bad_space),
            self.assertRaisesRegex(ValueError, "profile_overrides.score_fusion_profile"),
        ):
            search.resolve_search_space("bad-profile-space")

    def test_dry_run_resolves_base_config_without_training(self) -> None:
        """Dry-run payloads should build valid configs and avoid run_experiment()."""
        with patch.object(search, "run_experiment") as run_experiment:
            spec = search.resolve_search_space(
                "edgrec-core-optimization",
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
        self.assertEqual(base_config["baseline_family"], "edgrec")
        self.assertEqual(base_config["dataset"], "amazonbook")
        self.assertEqual(base_config["epochs"], 150)
        self.assertEqual(base_config["device"], "cpu")
        self.assertTrue(base_config["auto_batch_size"])
        self.assertEqual(base_config["batch_size"], 4096)
        self.assertEqual(
            base_config["batch_size_candidates"],
            [
                1048576,
                524288,
                262144,
                131072,
                65536,
                32768,
                16384,
                8192,
                4096,
                2048,
                1024,
                512,
                256,
            ],
        )
        self.assertNotIn("batch_size", payload["parameters"])
        self.assertNotIn("hard_negative_ratio", payload["parameters"])
        self.assertIn("dice_mask_reduction", payload["parameters"])
        self.assertNotIn("feature_gate_init", payload["parameters"])
        self.assertIn("n_negatives", payload["parameters"])

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
            "- Fresh informative target count: `2` (fresh COMPLETE + real fresh PRUNED).",
            lines,
        )
        self.assertIn("- Duplicate-skip pruned trials excluded from that target count: `1`.", lines)

    def test_optuna_report_importances_average_eligible_revision_subsets(self) -> None:
        """Importance tables should not jump to only the latest revision group."""
        study = optuna.create_study(direction="maximize")

        def add_revision_trials(revision: str) -> None:
            for index in range(optuna_report.MIN_IMPORTANCE_TRIALS):
                study.add_trial(
                    optuna.trial.create_trial(
                        state=optuna.trial.TrialState.COMPLETE,
                        value=float(index),
                        params={},
                        distributions={},
                        user_attrs={
                            "search_space_revision": revision,
                            "sampled_params": {
                                "lr": 0.001 if index % 2 else 0.002,
                                "dropout": 0.0 if index % 3 else 0.1,
                                "n_negatives": 1 if index % 2 else 2,
                            },
                            "amazonbook.objective": float(index),
                        },
                    ),
                )

        add_revision_trials("rev-a")
        add_revision_trials("rev-b")

        def fake_importances(_study, *, trials, **_kwargs):
            revision = trials[0].user_attrs["search_space_revision"]
            if revision == "rev-a":
                return {"lr": 0.8, "dropout": 0.2}
            return {"lr": 0.2, "n_negatives": 0.8}

        with patch.object(optuna_report, "safe_importances", side_effect=fake_importances):
            result = optuna_report.dataset_importance_result(study, "amazonbook")

        self.assertEqual(result.subset_count, 2)
        self.assertEqual(result.trial_count, optuna_report.MIN_IMPORTANCE_TRIALS * 2)
        self.assertEqual(result.revision, "rev-a, rev-b")
        self.assertGreater(result.importances["n_negatives"], result.importances["lr"])
        self.assertGreater(result.importances["lr"], result.importances["dropout"])
        self.assertAlmostEqual(sum(result.importances.values()), 1.0)

        rendered = optuna_report.render_importance_result("Importances", result)
        self.assertIn(
            "Aggregation: trial-weighted mean of deterministic per-revision fANOVA "
            "importances; parameters absent from a revision are not counted as zero.",
            rendered,
        )

    def test_optuna_report_renders_side_feature_analysis(self) -> None:
        """Feature diagnostics should render dataset-local subset rows."""
        study = optuna.create_study(direction="maximize")
        rows = [
            {
                "dataset": "movielens1m",
                "feature_subset_profile": "all_gate_neg4",
                "included_groups": "item_genre",
                "excluded_groups": "",
                "source_objective": 0.1,
                "validation_accuracy_20_40": 0.2,
                "ndcg_20": 0.3,
                "recall_20": 0.4,
                "hit_20": 0.5,
                "personalization_20": 0.6,
                "avgpop_20": 0.7,
                "ndcg_40": 0.31,
                "recall_40": 0.41,
                "hit_40": 0.51,
                "personalization_40": 0.61,
                "avgpop_40": 0.71,
                "online_crru_20": 0.08,
                "online_crru_40": 0.09,
                "online_crru_20_40": 0.1,
                "posthoc_crru_20": 0.11,
                "posthoc_crru_40": 0.12,
                "time_per_epoch_s": 1.5,
                "peak_vram_mb": 128.0,
                "batch": 4096,
                "completed_trials": 2,
                "status": "completed",
            },
        ]

        with patch.object(optuna_report, "build_feature_subset_result_rows", return_value=rows):
            rendered = "\n".join(optuna_report.render_side_feature_analysis([study]))

        self.assertIn("## Feature subset search", rendered)
        self.assertIn("FeatureSubset", rendered)
        self.assertIn("ValidationAccuracy@20_40", rendered)
        self.assertIn("OnlineCRRU@20_40", rendered)
        self.assertIn("| movielens1m | all_gate_neg4 | item_genre |", rendered)

    def test_trial_overrides_enter_build_config_path(self) -> None:
        """Sampled values should become a valid config without post-build mutation."""
        spec = search.SearchSpaceSpec(
            name="tiny",
            description="test search space",
            base_profile="core-edgrec-mainline",
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
                "dice_mask_reduction": {
                    "type": "categorical",
                    "choices": ["active_mean"],
                },
                "feature_gate_init": {
                    "type": "categorical",
                    "choices": [-2.0],
                },
                "n_negatives": {
                    "type": "categorical",
                    "choices": [2],
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
                search._parameter_storage_name(
                    "dice_mask_reduction",
                    spec.parameters["dice_mask_reduction"],
                ): "active_mean",
                search._parameter_storage_name(
                    "feature_gate_init",
                    spec.parameters["feature_gate_init"],
                ): -2.0,
                search._parameter_storage_name(
                    "n_negatives",
                    spec.parameters["n_negatives"],
                ): 2,
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
        self.assertEqual(config.dice_mask_reduction, "active_mean")
        self.assertEqual(config.feature_gate_init, -2.0)
        self.assertEqual(config.n_negatives, 2)
        self.assertEqual(config.sample_interactions, 100)
        self.assertEqual(config.loader_max_rows, 100)

    def test_profile_trial_resolution_records_logical_and_concrete_params(self) -> None:
        """Profile labels should resolve before EDGRecConfig construction."""
        catalog_profile_overrides = search.resolve_search_space(
            "edgrec-mechanism-coarse",
            dataset="kuairec_v2",
        ).profile_overrides
        spec = search.SearchSpaceSpec(
            name="tiny-profile",
            description="test profile search space",
            base_profile="core-edgrec-mainline",
            datasets=("kuairec_v2",),
            objective=search.ObjectiveSpec(metric=search.VALIDATION_ACCURACY_METRIC),
            max_epochs=3,
            trials=1,
            config_overrides={
                "sample_interactions": 100,
                "loader_max_rows": 100,
            },
            parameters={
                "score_fusion_profile": {
                    "type": "categorical",
                    "choices": ["fixed_interest_context"],
                },
                "item_branch_profile": {
                    "type": "categorical",
                    "choices": ["separate_item_branch_embeddings"],
                },
                "context_feature_profile": {
                    "type": "categorical",
                    "choices": ["context_only"],
                },
                "loss_profile": {
                    "type": "categorical",
                    "choices": ["dice_asym_conformity"],
                },
                "graph_profile": {
                    "type": "categorical",
                    "choices": ["medium"],
                },
                "loss_normalization": {
                    "type": "categorical",
                    "choices": ["none"],
                },
            },
            profile_overrides=catalog_profile_overrides,
        )
        base_config = search.build_search_config(
            spec,
            dataset="kuairec_v2",
            device="cpu",
            data_dir="data",
        )
        fixed_trial = optuna.trial.FixedTrial(
            {
                search._parameter_storage_name(
                    "score_fusion_profile",
                    spec.parameters["score_fusion_profile"],
                ): "fixed_interest_context",
                search._parameter_storage_name(
                    "item_branch_profile",
                    spec.parameters["item_branch_profile"],
                ): "separate_item_branch_embeddings",
                search._parameter_storage_name(
                    "context_feature_profile",
                    spec.parameters["context_feature_profile"],
                ): "context_only",
                search._parameter_storage_name(
                    "loss_profile",
                    spec.parameters["loss_profile"],
                ): "dice_asym_conformity",
                search._parameter_storage_name(
                    "graph_profile",
                    spec.parameters["graph_profile"],
                ): "medium",
                search._parameter_storage_name(
                    "loss_normalization",
                    spec.parameters["loss_normalization"],
                ): "none",
            },
        )

        resolution = search.resolve_trial_parameters(
            fixed_trial,
            spec,
            base_config=base_config,
        )
        config = search.build_search_config(
            spec,
            dataset="kuairec_v2",
            sampled_overrides=resolution.config_overrides,
            device="cpu",
            data_dir="data",
        )

        self.assertEqual(
            resolution.sampled_params["score_fusion_profile"],
            "fixed_interest_context",
        )
        self.assertFalse(config.use_learned_score_mix)
        self.assertEqual(config.score_weight_interest, 0.7)
        self.assertEqual(config.score_weight_conformity, 0.0)
        self.assertEqual(config.score_weight_popularity, 0.3)
        self.assertTrue(config.separate_item_branch_embeddings)
        self.assertFalse(config.use_features)
        self.assertTrue(config.use_popularity_head)
        self.assertEqual(config.loss_weight_conformity_bpr, 0.05)
        self.assertEqual(config.interest_gnn_layers, 2)
        self.assertEqual(config.conformity_gnn_layers, 2)
        self.assertEqual(config.num_neighbors, [10, 5])
        self.assertEqual(config.loss_normalization, "none")

    def test_grid_float_parameter_samples_from_declared_steps(self) -> None:
        """Segmented float grids should keep human-readable search points."""
        spec = search.SearchSpaceSpec(
            name="tiny-grid",
            description="test search space",
            base_profile="core-edgrec-mainline",
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
            "base_profile": "core-edgrec-mainline",
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
            base_profile="core-edgrec-mainline",
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
                base_profile="core-edgrec-mainline",
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

    def test_validation_crru_components_reconstruct_per_k_objective(self) -> None:
        """OnlineCRRU component diagnostics should share the objective formula."""
        val_metrics = {
            "NDCG@20": 0.10,
            "Recall@20": 0.20,
            "HitRatio@20": 0.30,
            "Personalization@20": 0.50,
            "AveragePopularity@20": 1.0,
        }

        components = compute_validation_online_crru_components_for_k(
            val_metrics,
            k=20,
            peak_vram_mb=512.0,
            epoch_time_s=5.0,
        )

        self.assertAlmostEqual(
            components["online_crru"],
            search.compute_validation_online_crru_for_k(
                val_metrics,
                k=20,
                peak_vram_mb=512.0,
                epoch_time_s=5.0,
            ),
        )
        self.assertGreater(components["accuracy"], 0.0)
        self.assertGreater(components["popularity_diversity"], 0.0)
        self.assertGreater(components["efficiency"], 0.0)

    def test_validation_accuracy_objective_uses_validation_metrics_only(self) -> None:
        """The accuracy-first objective should not read test metrics."""
        val_metrics = {
            "NDCG@20": 0.10,
            "Recall@20": 0.20,
            "NDCG@40": 0.30,
            "Recall@40": 0.40,
        }
        result = {
            "history": {"val_metrics": [val_metrics]},
            "test_metrics": {"NDCG@20": 0.99, "Recall@20": 0.99},
        }

        objective = search.extract_validation_objective(
            result,
            search.ObjectiveSpec(
                metric=search.VALIDATION_ACCURACY_METRIC,
                split="val",
                direction="maximize",
            ),
        )

        self.assertAlmostEqual(
            objective,
            0.50 * 0.10 + 0.25 * 0.20 + 0.15 * 0.30 + 0.10 * 0.40,
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
                base_profile="core-edgrec-mainline",
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
            current_revision = search.search_space_revision(spec)
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
                        "search_space_revision": current_revision,
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
                mlflow_experiment_name="edgrec-search-test",
            )

            with (
                patch.object(search, "resolve_search_space", return_value=spec),
                patch.object(search, "run_experiment") as run_experiment,
            ):
                exit_code = search.run_search(args)

            self.assertEqual(exit_code, 0)
            run_experiment.assert_not_called()

    def test_different_revision_trials_do_not_satisfy_search_budget(self) -> None:
        """--trials N should target fresh informative trials for this exact hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optuna_db = Path(tmpdir) / "optuna.db"
            storage = f"sqlite:///{optuna_db}"
            spec = search.SearchSpaceSpec(
                name="tiny-search",
                description="test search space",
                base_profile="core-edgrec-mainline",
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
                    value=0.42,
                    params={storage_param: "cosine"},
                    distributions={
                        storage_param: optuna.distributions.CategoricalDistribution(["cosine"]),
                    },
                    user_attrs={
                        "search_space": "tiny-search",
                        "search_space_revision": "differenthash",
                        "objective_metric": "NDCG@40",
                        "objective_split": "val",
                        "sampled_params": {"lr_scheduler": "cosine"},
                        "amazonbook.objective": 0.42,
                    },
                    state=optuna.trial.TrialState.COMPLETE,
                ),
            )

            self.assertEqual(search._budget_informative_trials(study, spec), [])
            self.assertEqual(search._compatible_completed_trials(study, spec), [study.trials[0]])

    def test_real_pruned_trials_count_toward_search_budget(self) -> None:
        """A pruned training trial is informative and should satisfy --trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optuna_db = Path(tmpdir) / "optuna.db"
            storage = f"sqlite:///{optuna_db}"
            spec = search.SearchSpaceSpec(
                name="tiny-search",
                description="test search space",
                base_profile="core-edgrec-mainline",
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
            current_revision = search.search_space_revision(spec)
            study.add_trial(
                optuna.trial.create_trial(
                    params={storage_param: "cosine"},
                    distributions={
                        storage_param: optuna.distributions.CategoricalDistribution(["cosine"]),
                    },
                    user_attrs={
                        "search_space": "tiny-search",
                        "search_space_revision": current_revision,
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
                mlflow_experiment_name="edgrec-search-test",
            )

            with (
                patch.object(search, "resolve_search_space", return_value=spec),
                patch.object(search, "run_experiment") as run_experiment,
            ):
                exit_code = search.run_search(args)

            self.assertEqual(exit_code, 0)
            run_experiment.assert_not_called()

    def test_missing_feature_subset_coverage_runs_even_when_budget_is_met(self) -> None:
        """Required feature profiles should run before the normal budget can stop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optuna_db = Path(tmpdir) / "optuna.db"
            storage = f"sqlite:///{optuna_db}"
            groups = ("item_genre",)
            profiles = required_feature_subset_profiles(groups)
            spec = search.SearchSpaceSpec(
                name="edgrec-feature-subset-search",
                description="test search space",
                base_profile="edgrec-compact-search-prior",
                datasets=("movielens1m",),
                objective=search.ObjectiveSpec(metric="NDCG@40"),
                max_epochs=1,
                trials=1,
                config_overrides={
                    "sample_interactions": 50,
                    "loader_max_rows": 50,
                },
                parameters={
                    "feature_subset_profile": {
                        "type": "categorical",
                        "choices": list(profiles),
                    },
                },
                profile_overrides={
                    "feature_subset_profile": feature_subset_profile_matrix(groups),
                },
            )
            storage_param = search._parameter_storage_name(
                "feature_subset_profile",
                spec.parameters["feature_subset_profile"],
            )
            study = optuna.create_study(
                study_name="existing-study",
                storage=storage,
                direction="maximize",
            )
            current_revision = search.search_space_revision(spec)
            study.add_trial(
                optuna.trial.create_trial(
                    params={storage_param: "none"},
                    distributions={
                        storage_param: optuna.distributions.CategoricalDistribution(
                            list(profiles),
                        ),
                    },
                    user_attrs={
                        "search_space": spec.name,
                        "search_space_revision": current_revision,
                        "objective_metric": "NDCG@40",
                        "objective_split": "val",
                        "sampled_params": {"feature_subset_profile": "none"},
                        "movielens1m.pruned": True,
                        "movielens1m.last_pruning_objective": 0.25,
                    },
                    state=optuna.trial.TrialState.PRUNED,
                ),
            )
            args = SimpleNamespace(
                list_spaces=False,
                space=spec.name,
                dataset=None,
                trials=1,
                study_name="existing-study",
                storage=storage,
                dry_run=False,
                device="cpu",
                data_dir="data",
                no_mlflow=True,
                mlflow_tracking_uri=None,
                mlflow_experiment_name="edgrec-search-test",
            )
            result = {
                "history": {"val_metrics": [{"NDCG@40": 0.5}]},
                "avg_epoch_time_s": 1.0,
                "peak_vram_mb": 128.0,
                "batch_size": 1024,
            }

            with (
                patch.object(search, "resolve_search_space", return_value=spec),
                patch.object(
                    search,
                    "loaded_thesis_safe_item_feature_groups_for_dataset",
                    return_value=groups,
                ),
                patch.object(search, "run_experiment", return_value=result) as run_experiment,
            ):
                exit_code = search.run_search(args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(run_experiment.call_count, len(profiles))

    def test_multi_dataset_default_runs_dataset_local_studies(self) -> None:
        """A multi-dataset command should optimize one independent study per dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optuna_db = Path(tmpdir) / "optuna.db"
            storage = f"sqlite:///{optuna_db}"
            common_kwargs = {
                "name": "tiny-search",
                "description": "test search space",
                "base_profile": "core-edgrec-mainline",
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
                mlflow_experiment_name="edgrec-search-test",
            )

            def resolve_space(
                _space_name: str,
                dataset: str | None = None,
                data_dir: str = "data",
            ):
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
                base_profile="core-edgrec-mainline",
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
                mlflow_experiment_name="edgrec-search-test",
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
                base_profile="core-edgrec-mainline",
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
                mlflow_experiment_name="edgrec-search-test",
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
                        preset="edgrec",
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
                base_profile="core-edgrec-mainline",
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
                mlflow_experiment_name="edgrec-search-test",
            )

            with (
                patch.object(search, "resolve_search_space", return_value=spec),
                patch.object(
                    search,
                    "resolve_trial_parameters",
                    side_effect=ValueError("bad search contract"),
                ),
            ):
                exit_code = search.run_search(args)

            self.assertEqual(exit_code, 1)

    def test_search_aborts_after_cuda_device_assert(self) -> None:
        """A device-side assert should stop the study before the CUDA context is reused."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optuna_db = Path(tmpdir) / "optuna.db"
            storage = f"sqlite:///{optuna_db}"
            spec = search.SearchSpaceSpec(
                name="tiny-search",
                description="test search space",
                base_profile="core-edgrec-mainline",
                datasets=("amazonbook",),
                objective=search.ObjectiveSpec(metric="NDCG@40"),
                max_epochs=1,
                trials=4,
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
                trials=4,
                study_name="fatal-study",
                storage=storage,
                dry_run=False,
                device="cpu",
                data_dir="data",
                no_mlflow=True,
                mlflow_tracking_uri=None,
                mlflow_experiment_name="edgrec-search-test",
            )

            with (
                patch.object(search, "resolve_search_space", return_value=spec),
                patch.object(
                    search,
                    "run_experiment",
                    side_effect=RuntimeError("CUDA error: device-side assert triggered"),
                ) as run_experiment,
            ):
                exit_code = search.run_search(args)

            self.assertEqual(exit_code, 1)
            run_experiment.assert_called_once()
            study = optuna.load_study(study_name="fatal-study", storage=storage)
            self.assertEqual(len(study.trials), 1)
            self.assertEqual(
                study.trials[0].user_attrs["fatal_failure"],
                "cuda_context_poisoned",
            )


if __name__ == "__main__":
    unittest.main()
