"""Regression coverage for centralized experiment CLI parser builders."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import scripts.quick_validate as quick_validate
from experiments.ablation_configs import (
    ABLATION_VARIANTS,
    build_ablation_base_kwargs,
    make_ablation_config,
)
from experiments.cli_parsers import (
    build_ablation_parser,
    build_benchmark_parser,
    build_formal_run_parser,
    build_run_experiment_parser,
    build_search_parser,
)
from src.utils.benchmark_datasets import resolve_benchmark_datasets
from src.utils.cli_parsers import (
    build_data_information_parser,
    build_explore_all_datasets_parser,
    build_query_results_parser,
    build_quick_validate_parser,
)

EXPECTED_ABLATION_VARIANTS = [
    "mainline",
    "with_contrastive",
    "no_popularity_head",
    "no_independence",
    "no_features",
]


class ExperimentParserTests(unittest.TestCase):
    """Pin the public single-run parser defaults and toggles."""

    def test_experiment_parser_defaults_preserve_tracking_flags(self) -> None:
        """Single-run defaults should keep the fixed tracking/runtime defaults."""
        args = build_run_experiment_parser().parse_args([])

        self.assertEqual(args.dataset, "movielens1m")
        self.assertTrue(args.enable_mlflow)
        self.assertTrue(args.auto_resume)
        self.assertEqual(args.data_dir, "data")
        self.assertEqual(args.device, "cuda")
        self.assertIsNone(args.change_note)
        self.assertFalse(args.list_recipes)
        self.assertFalse(hasattr(args, "epochs"))
        self.assertFalse(hasattr(args, "graph_policy"))

    def test_experiment_parser_accepts_checkpoint_overwrite_flag(self) -> None:
        """Single-run parser should expose explicit checkpoint replacement."""
        args = build_run_experiment_parser().parse_args(["--overwrite-checkpoint"])

        self.assertTrue(args.overwrite_checkpoint)


class AblationParserTests(unittest.TestCase):
    """Pin the public ablation parser contract."""

    def test_ablation_parser_keeps_only_selection_surface(self) -> None:
        """Ablation runs should expose only dataset, variant, and overwrite selection."""
        args = build_ablation_parser().parse_args(["--datasets", "movielens1m"])

        self.assertEqual(args.datasets, ["movielens1m"])
        self.assertFalse(hasattr(args, "dataset"))
        self.assertFalse(hasattr(args, "graph_policy"))
        self.assertFalse(hasattr(args, "epochs"))
        self.assertFalse(hasattr(args, "device"))
        self.assertFalse(hasattr(args, "data_dir"))
        self.assertFalse(hasattr(args, "no_mlflow"))
        self.assertFalse(hasattr(args, "resume_batch"))
        self.assertFalse(hasattr(args, "dry_run"))
        self.assertEqual(args.variants, EXPECTED_ABLATION_VARIANTS)


class AblationConfigTests(unittest.TestCase):
    """Pin the minimal thesis-facing ablation surface."""

    def test_ablation_variants_match_minimal_thesis_matrix(self) -> None:
        """The public ablation variant order should remain stable."""
        self.assertEqual(
            list(ABLATION_VARIANTS),
            EXPECTED_ABLATION_VARIANTS,
        )

    def test_no_popularity_head_ablation_disables_popularity_path(self) -> None:
        """Popularity-head ablation should remove both scoring and supervision."""
        config = make_ablation_config("no_popularity_head")

        self.assertFalse(config.use_popularity_head)
        self.assertEqual(config.score_weight_popularity, 0.0)
        self.assertEqual(config.loss_weight_popularity, 0.0)

    def test_with_contrastive_ablation_enables_bounded_contrastive_loss(self) -> None:
        """Contrastive ablation should enable the literature-backed causal auxiliary."""
        config = make_ablation_config("with_contrastive")

        self.assertGreater(config.loss_weight_contrastive, 0.0)
        self.assertEqual(config.contrastive_max_pairs, 256)
        self.assertEqual(config.contrastive_temperature, 0.2)
        self.assertEqual(config.auxiliary_loss_schedule, "linear_ramp")

    def test_no_features_ablation_disables_side_features(self) -> None:
        """Feature ablation should switch off item/user side features entirely."""
        config = make_ablation_config("no_features")

        self.assertFalse(config.use_features)

    def test_ablation_base_kwargs_omit_unset_optional_overrides(self) -> None:
        """Shared ablation kwargs should keep required fields and skip unset optionals."""
        kwargs = build_ablation_base_kwargs(
            dataset="movielens1m",
            data_dir="data",
            device="cpu",
            batch_size=64,
        )

        self.assertEqual(kwargs["dataset"], "movielens1m")
        self.assertEqual(kwargs["data_dir"], "data")
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["batch_size"], 64)
        self.assertNotIn("epochs", kwargs)
        self.assertNotIn("graph_policy", kwargs)
        self.assertNotIn("sample_interactions", kwargs)

    def test_ablation_base_kwargs_include_graph_policy_when_set(self) -> None:
        """Shared ablation kwargs should forward an explicit graph-policy override."""
        kwargs = build_ablation_base_kwargs(
            dataset="movielens1m",
            data_dir="data",
            device="cpu",
            graph_policy="observed",
        )

        self.assertEqual(kwargs["graph_policy"], "observed")


class FormalRunParserTests(unittest.TestCase):
    """Pin the simple formal-run parser defaults."""

    def test_formal_run_parser_defaults_keep_optional_overrides_unset(self) -> None:
        """Formal-run should stay focused on profile selection only."""
        args = build_formal_run_parser().parse_args([])

        self.assertIsNone(args.profile)
        self.assertFalse(args.list_profiles)
        self.assertFalse(hasattr(args, "device"))
        self.assertFalse(hasattr(args, "data_dir"))
        self.assertFalse(hasattr(args, "no_mlflow"))
        self.assertFalse(hasattr(args, "dry_run"))

    def test_formal_run_parser_accepts_checkpoint_overwrite_flag(self) -> None:
        """Formal-run parser should expose explicit checkpoint replacement."""
        args = build_formal_run_parser().parse_args(["--overwrite-checkpoint"])

        self.assertTrue(args.overwrite_checkpoint)


class SearchParserTests(unittest.TestCase):
    """Pin the Optuna search parser contract."""

    def test_search_parser_defaults(self) -> None:
        """Search CLI should keep study execution optional until a space is selected."""
        args = build_search_parser().parse_args([])

        self.assertIsNone(args.space)
        self.assertIsNone(args.dataset)
        self.assertIsNone(args.trials)
        self.assertIsNone(args.study_name)
        self.assertEqual(args.storage, "sqlite:///results/optuna_studies.db")
        self.assertFalse(args.dry_run)
        self.assertTrue(args.no_mlflow)
        self.assertFalse(hasattr(args, "overwrite_checkpoint"))
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.data_dir, "data")
        self.assertEqual(args.mlflow_experiment_name, "edgrec-optuna")

    def test_search_parser_mlflow_is_explicit_opt_in(self) -> None:
        """Search should avoid MLflow artifacts unless requested explicitly."""
        default_args = build_search_parser().parse_args([])
        enabled_args = build_search_parser().parse_args(["--mlflow"])

        self.assertTrue(default_args.no_mlflow)
        self.assertFalse(enabled_args.no_mlflow)

    def test_search_parser_accepts_dry_run_space(self) -> None:
        """Search CLI should expose a no-training dry-run path."""
        args = build_search_parser().parse_args(
            [
                "--space",
                "edgrec-core-optimization",
                "--trials",
                "1",
                "--dry-run",
            ],
        )

        self.assertEqual(args.space, "edgrec-core-optimization")
        self.assertEqual(args.trials, 1)
        self.assertTrue(args.dry_run)
        self.assertFalse(hasattr(args, "overwrite_checkpoint"))


class BenchmarkParserTests(unittest.TestCase):
    """Pin the benchmark parser defaults that moved behind the shared builder."""

    def test_benchmark_parser_default_matrix_is_unchanged(self) -> None:
        """Benchmark defaults should still expose the canonical matrix sweep."""
        args = build_benchmark_parser().parse_args([])

        self.assertEqual(args.datasets, "small,medium")
        self.assertFalse(hasattr(args, "scoring_weight_modes"))
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.data_dir, "data")
        self.assertFalse(args.no_mlflow)
        self.assertFalse(args.resume_batch)
        self.assertFalse(args.dry_run)
        self.assertIsNone(args.change_note)
        self.assertFalse(hasattr(args, "epochs"))
        self.assertFalse(hasattr(args, "graph_policy"))
        self.assertFalse(hasattr(args, "graph_policy_options"))

    def test_benchmark_parser_accepts_checkpoint_overwrite_flag(self) -> None:
        """Benchmark parser should expose explicit checkpoint replacement."""
        args = build_benchmark_parser().parse_args(["--overwrite-checkpoint"])

        self.assertTrue(args.overwrite_checkpoint)

    def test_benchmark_dataset_selector_error_names_unknown_value(self) -> None:
        """Shared selector expansion should report the actual bad selector."""
        with self.assertRaisesRegex(ValueError, "not_a_tier"):
            resolve_benchmark_datasets(["not_a_tier"])


class UtilityParserTests(unittest.TestCase):
    """Pin the centralized utility parser defaults."""

    def test_quick_validate_parser_defaults(self) -> None:
        """Quick-validate should keep its fixed non-persistent smoke surface."""
        args = build_quick_validate_parser().parse_args([])

        self.assertEqual(
            args.datasets,
            [
                "amazonbook",
                "movielens1m",
                "movielens20m",
                "kuairec_v2",
                "taobao",
                "kuairand1k",
            ],
        )
        self.assertEqual(
            args.categories,
            ["recipes", "ablations", "observability", "evaluation"],
        )
        self.assertEqual(args.data_dir, "data")
        self.assertFalse(args.mlflow)
        self.assertFalse(args.fail_fast)
        self.assertFalse(hasattr(args, "epochs"))
        self.assertFalse(hasattr(args, "graph_policy"))

    def test_query_results_defaults(self) -> None:
        """Query-results should default to the thesis summary view."""
        args = build_query_results_parser().parse_args([])

        self.assertIsNone(args.view)
        self.assertFalse(hasattr(args, "batch_id"))
        self.assertFalse(hasattr(args, "status"))

    def test_explore_all_datasets_defaults(self) -> None:
        """Dataset-visualization parser should preserve shared defaults."""
        args = build_explore_all_datasets_parser().parse_args([])

        self.assertEqual(args.data_dir, "data")
        self.assertEqual(args.output_dir, Path("results") / "dataset_visualizations")
        self.assertEqual(
            args.datasets,
            [
                "amazonbook",
                "movielens1m",
                "movielens20m",
                "kuairec_v2",
                "taobao",
                "kuairand1k",
            ],
        )
        self.assertEqual(args.dpi, 180)

    def test_data_information_defaults(self) -> None:
        """Dataset-information parser should keep both optional outputs unset."""
        args = build_data_information_parser().parse_args([])

        self.assertIsNone(args.output)
        self.assertIsNone(args.audit_json)

    def test_quick_validate_tiny_runtime_defaults_are_shared(self) -> None:
        """Quick-validate tiny recipe and ablation configs should share runtime knobs."""
        args = build_quick_validate_parser().parse_args([])

        recipe_config = quick_validate._build_tiny_recipe_config(
            args,
            "movielens1m",
            recipe="edgrec",
        )
        ablation_config = quick_validate._build_tiny_ablation_config(
            args,
            "movielens1m",
            variant="mainline",
        )

        for config in (recipe_config, ablation_config):
            self.assertEqual(config.epochs, quick_validate.QUICK_VALIDATE_EPOCHS)
            self.assertEqual(config.batch_size, quick_validate.QUICK_VALIDATE_BATCH_SIZE)
            self.assertEqual(config.sample_interactions, 100)
            self.assertEqual(config.loader_max_rows, 100)
            self.assertEqual(config.patience, 1)
            self.assertFalse(config.use_torch_compile)

    def test_quick_validate_ablation_category_uses_tiny_config_path(self) -> None:
        """Ablation quick-validate should build and run the tiny config path."""
        args = SimpleNamespace(
            datasets=["movielens1m"],
            ablation_variants=["mainline"],
            data_dir="data",
            fail_fast=False,
        )
        results: list[dict] = []
        captured: list[object] = []

        def _capture_run_single_case(*, config, **kwargs):
            captured.append(config)
            return {}, 0.0

        with mock.patch.object(
            quick_validate,
            "_run_single_case",
            side_effect=_capture_run_single_case,
        ):
            quick_validate._run_ablation_category(args, results)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "pass")
        self.assertEqual(len(captured), 1)
        config = captured[0]
        self.assertEqual(config.epochs, quick_validate.QUICK_VALIDATE_EPOCHS)
        self.assertEqual(config.batch_size, quick_validate.QUICK_VALIDATE_BATCH_SIZE)
        self.assertEqual(config.sample_interactions, 100)
        self.assertEqual(config.loader_max_rows, 100)
        self.assertEqual(config.patience, 1)
        self.assertFalse(config.use_torch_compile)

    def test_quick_validate_run_single_case_disables_refined_diagnostics(self) -> None:
        """Quick validation should run without optional diagnostics or persistence."""
        config = SimpleNamespace()

        with mock.patch.object(
            quick_validate,
            "run_experiment",
            return_value={"exp_id": None, "test_metrics": {}},
        ) as run_experiment:
            quick_validate._run_single_case(
                category="recipes",
                dataset="movielens1m",
                label="recipe:edgrec",
                config=config,
                preset="edgrec",
                intervention="quick_recipe_edgrec",
            )

        self.assertFalse(run_experiment.call_args.kwargs["include_refined_diagnostics"])
        self.assertFalse(run_experiment.call_args.kwargs["enable_mlflow"])
        self.assertFalse(run_experiment.call_args.kwargs["log_to_sqlite"])

    def test_quick_validate_resume_probe_disables_refined_diagnostics(self) -> None:
        """The direct resume probe should share the quick validation metric contract."""
        args = SimpleNamespace(
            datasets=["movielens1m"],
            data_dir="data",
            fail_fast=False,
            mlflow=False,
        )
        results: list[dict] = []

        with (
            mock.patch.object(quick_validate, "_run_single_case", return_value=({}, 0.0)),
            mock.patch.object(
                quick_validate,
                "run_experiment",
                return_value={"resumed": True},
            ) as run_experiment,
        ):
            quick_validate._run_observability_category(args, results)

        self.assertTrue(run_experiment.called)
        self.assertFalse(run_experiment.call_args.kwargs["include_refined_diagnostics"])
        self.assertFalse(run_experiment.call_args.kwargs["enable_mlflow"])
        self.assertFalse(run_experiment.call_args.kwargs["log_to_sqlite"])


if __name__ == "__main__":
    unittest.main()
