"""Regression coverage for centralized experiment CLI parser builders."""

from __future__ import annotations

import unittest
from pathlib import Path

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
        self.assertFalse(args.list_recipes)

    def test_experiment_parser_disable_flags_override_defaults(self) -> None:
        """Explicit negative flags should flip the centralized bool pairs."""
        args = build_run_experiment_parser().parse_args(
            ["--no-mlflow", "--no-auto-resume", "--no-features"]
        )

        self.assertFalse(args.enable_mlflow)
        self.assertFalse(args.auto_resume)
        self.assertFalse(args.use_features)


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


class FormalRunParserTests(unittest.TestCase):
    """Pin the simple formal-run parser defaults."""

    def test_formal_run_parser_defaults_keep_optional_overrides_unset(self) -> None:
        """Formal-run should not invent device or data-dir overrides by default."""
        args = build_formal_run_parser().parse_args([])

        self.assertIsNone(args.profile)
        self.assertIsNone(args.device)
        self.assertIsNone(args.data_dir)
        self.assertFalse(args.no_mlflow)
        self.assertFalse(args.dry_run)


class BenchmarkParserTests(unittest.TestCase):
    """Pin the benchmark parser defaults that moved behind the shared builder."""

    def test_benchmark_parser_default_matrix_is_unchanged(self) -> None:
        """Benchmark defaults should still expose the canonical matrix sweep."""
        args = build_benchmark_parser().parse_args([])

        self.assertEqual(args.tier, "small")
        self.assertEqual(args.graph_methods, ["cagra", "knn"])
        self.assertEqual(args.scoring_weight_modes, ["learned"])
        self.assertIsNone(args.use_early_stopping)


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
            ["--checkpoint-path", "checkpoint.pt"]
        )

        self.assertEqual(args.checkpoint_path, "checkpoint.pt")
        self.assertEqual(
            args.modes, ["default", "interest_only", "conformity_suppressed"]
        )
        self.assertEqual(args.split, "test")
        self.assertEqual(args.batch_size, 512)
        self.assertIsNone(args.device)

    def test_query_results_defaults(self) -> None:
        """Query-results should default to the broadest exploration view."""
        args = build_query_results_parser().parse_args([])

        self.assertEqual(args.view, "all")
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
