"""Regression coverage for centralized experiment CLI parser builders."""

from __future__ import annotations

import unittest
from pathlib import Path

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
)
from src.utils.cli_parsers import (
    build_data_information_parser,
    build_evaluate_scoring_modes_parser,
    build_explore_all_datasets_parser,
    build_query_results_parser,
    build_quick_validate_parser,
)


class ExperimentParserTests(unittest.TestCase):
    """Pin the public single-run parser defaults and toggles."""

    def test_experiment_parser_defaults_preserve_tracking_flags(self) -> None:
        """Single-run defaults should keep MLflow and auto-resume enabled."""
        args = build_run_experiment_parser().parse_args([])

        self.assertEqual(args.dataset, "movielens1m")
        self.assertTrue(args.enable_mlflow)
        self.assertTrue(args.auto_resume)
        self.assertIsNone(args.use_features)
        self.assertIsNone(args.change_note)
        self.assertFalse(args.list_recipes)

    def test_experiment_parser_disable_flags_override_defaults(self) -> None:
        """Explicit negative flags should flip the centralized bool pairs."""
        args = build_run_experiment_parser().parse_args(
            ["--no-mlflow", "--no-auto-resume", "--no-features"],
        )

        self.assertFalse(args.enable_mlflow)
        self.assertFalse(args.auto_resume)
        self.assertFalse(args.use_features)

    def test_experiment_parser_accepts_checkpoint_overwrite_flag(self) -> None:
        """Single-run parser should expose explicit checkpoint replacement."""
        args = build_run_experiment_parser().parse_args(["--overwrite-checkpoint"])

        self.assertTrue(args.overwrite_checkpoint)


class AblationParserTests(unittest.TestCase):
    """Pin the public ablation parser contract."""

    def test_ablation_parser_keeps_runtime_defaults(self) -> None:
        """Ablation runs should keep their existing runtime defaults."""
        args = build_ablation_parser().parse_args(["--dataset", "movielens1m"])

        self.assertEqual(args.dataset, "movielens1m")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.data_dir, "data")
        self.assertFalse(args.no_mlflow)
        self.assertFalse(args.resume_batch)
        self.assertFalse(args.dry_run)
        self.assertEqual(
            args.variants,
            ["mainline", "no_popularity_head", "no_independence", "no_features"],
        )


class AblationConfigTests(unittest.TestCase):
    """Pin the minimal thesis-facing ablation surface."""

    def test_ablation_variants_match_minimal_thesis_matrix(self) -> None:
        """Only the reviewer-defensible ablations should remain public."""
        self.assertEqual(
            list(ABLATION_VARIANTS),
            ["mainline", "no_popularity_head", "no_independence", "no_features"],
        )

    def test_no_popularity_head_ablation_disables_popularity_path(self) -> None:
        """Popularity-head ablation should remove both scoring and supervision."""
        config = make_ablation_config("no_popularity_head")

        self.assertFalse(config.use_popularity_head)
        self.assertEqual(config.gamma_popularity, 0.0)
        self.assertEqual(config.lambda_pop, 0.0)

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
        self.assertNotIn("sample_interactions", kwargs)


class FormalRunParserTests(unittest.TestCase):
    """Pin the simple formal-run parser defaults."""

    def test_formal_run_parser_defaults_keep_optional_overrides_unset(self) -> None:
        """Formal-run should not invent device or data-dir overrides by default."""
        args = build_formal_run_parser().parse_args([])

        self.assertIsNone(args.profile)
        self.assertIsNone(args.device)
        self.assertIsNone(args.data_dir)
        self.assertIsNone(args.change_note)
        self.assertFalse(args.no_mlflow)
        self.assertFalse(args.dry_run)

    def test_formal_run_parser_accepts_checkpoint_overwrite_flag(self) -> None:
        """Formal-run parser should expose explicit checkpoint replacement."""
        args = build_formal_run_parser().parse_args(["--overwrite-checkpoint"])

        self.assertTrue(args.overwrite_checkpoint)


class BenchmarkParserTests(unittest.TestCase):
    """Pin the benchmark parser defaults that moved behind the shared builder."""

    def test_benchmark_parser_default_matrix_is_unchanged(self) -> None:
        """Benchmark defaults should still expose the canonical matrix sweep."""
        args = build_benchmark_parser().parse_args([])

        self.assertEqual(args.datasets, "small,medium")
        self.assertEqual(args.scoring_weight_modes, ["learned"])
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.data_dir, "data")
        self.assertFalse(args.no_mlflow)
        self.assertFalse(args.resume_batch)
        self.assertFalse(args.dry_run)
        self.assertIsNone(args.use_early_stopping)
        self.assertIsNone(args.auto_batch_size)
        self.assertIsNone(args.batch_size_candidates)
        self.assertIsNone(args.change_note)

    def test_benchmark_parser_accepts_checkpoint_overwrite_flag(self) -> None:
        """Benchmark parser should expose explicit checkpoint replacement."""
        args = build_benchmark_parser().parse_args(["--overwrite-checkpoint"])

        self.assertTrue(args.overwrite_checkpoint)


class UtilityParserTests(unittest.TestCase):
    """Pin the centralized utility parser defaults."""

    def test_quick_validate_parser_defaults(self) -> None:
        """Quick-validate should keep its default categories and MLflow opt-in."""
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

    def test_evaluate_scoring_modes_defaults(self) -> None:
        """Scoring-mode evaluation should preserve its public defaults."""
        args = build_evaluate_scoring_modes_parser().parse_args(
            ["--checkpoint-path", "checkpoint.pt"],
        )

        self.assertEqual(args.checkpoint_path, "checkpoint.pt")
        self.assertEqual(
            args.modes,
            ["default", "interest_only", "conformity_suppressed"],
        )
        self.assertEqual(args.split, "test")
        self.assertEqual(args.batch_size, 512)
        self.assertIsNone(args.device)

    def test_query_results_defaults(self) -> None:
        """Query-results should default to the broadest exploration view."""
        args = build_query_results_parser().parse_args([])

        self.assertIsNone(args.view)
        self.assertIsNone(args.batch_id)
        self.assertIsNone(args.status)
        self.assertIsNone(args.exp)

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


if __name__ == "__main__":
    unittest.main()
