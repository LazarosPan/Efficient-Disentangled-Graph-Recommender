"""Regression coverage for formal training policy defaults."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import experiments.run_benchmark as formal_main

from experiments.run_benchmark import (
    build_benchmark_plan,
    build_parser as build_benchmark_parser,
)
from experiments.recipes import (
    default_formal_profile_name,
    formal_profile_names,
    get_formal_profile,
    get_recipe,
)
from experiments.run_experiment import build_config
from src.profiling.gpu_profiler import GPUProfiler
from src.utils.config import UCaGNNConfig
from src.utils.trainer_runtime import TrainerRuntime


def _experiment_args(**overrides: object) -> SimpleNamespace:
    """Return a minimal namespace accepted by build_config()."""
    base = {
        "dataset": "movielens1m",
        "data_dir": "data",
        "device": "cuda",
        "recipe": None,
        "preset": None,
        "seed": 13,
        "epochs": None,
        "batch_size": None,
        "embed_dim": None,
        "single_branch_gnn_layers": None,
        "interest_gnn_layers": None,
        "conformity_gnn_layers": None,
        "dropout": None,
        "lr": None,
        "use_early_stopping": None,
        "eval_scoring_mode": None,
        "scoring_weight_mode": None,
        "use_features": None,
        "feature_policy": None,
        "graph_method": None,
        "num_neighbors": None,
        "hard_negative_ratio": None,
        "curriculum_phase1_end": None,
        "curriculum_phase2_end": None,
        "loss_schedule": None,
        "sample_interactions": None,
        "loader_max_rows": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FormalTrainingPolicyTests(unittest.TestCase):
    """Pin the formal-run no-early-stopping default."""

    def test_gpu_profiler_defaults_disabled(self) -> None:
        """Direct profiler construction should stay disabled until runtime opts in."""
        self.assertFalse(GPUProfiler()._enabled)

    def test_loss_schedule_defaults_to_baseline(self) -> None:
        """loss_schedule must default to baseline so existing runs are unaffected."""
        config = UCaGNNConfig()
        self.assertEqual(config.loss_schedule, "baseline")

    def test_mini_batch_runtime_defaults_keep_torch_compile_opt_in(self) -> None:
        """Dynamic mini-batch subgraphs should not default to torch.compile."""
        config = UCaGNNConfig(device="cuda")

        self.assertFalse(config.use_torch_compile)

    def test_mini_batch_runtime_defaults_keep_the_two_hop_shape_valid(self) -> None:
        """Default mini-batch settings should keep the 1/2-hop fan-out valid."""
        config = UCaGNNConfig(device="cuda")

        self.assertEqual(config.batch_size, 4096)
        self.assertEqual(config.single_branch_gnn_layers, 2)
        self.assertEqual(config.interest_gnn_layers, 1)
        self.assertEqual(config.conformity_gnn_layers, 2)
        self.assertEqual(config.num_neighbors, [10, 5])
        self.assertEqual(config.max_gnn_layers, 2)
        self.assertEqual(len(config.num_neighbors), config.max_gnn_layers)
        self.assertEqual(config.dropout, 0.1)

    def test_build_config_respects_dropout_override(self) -> None:
        """Catalog/runtime plumbing should pass dropout through to the config."""
        config = build_config(_experiment_args(dropout=0.25))

        self.assertEqual(config.dropout, 0.25)

    def test_ucagnn_preset_applies_wave1_fused_scoring_defaults(self) -> None:
        """The ucagnn preset should target the fused-score wave-1 contract."""
        config = build_config(_experiment_args(preset="ucagnn"))

        self.assertEqual(config.scoring_weight_mode, "learned")
        self.assertEqual(config.train_scoring_mode, "default")
        self.assertEqual(config.eval_scoring_mode, "default")
        self.assertTrue(config.use_popularity_head)
        self.assertEqual(config.interest_gnn_layers, 1)
        self.assertEqual(config.conformity_gnn_layers, 2)
        self.assertEqual(config.max_gnn_layers, 2)
        self.assertEqual(config.num_neighbors, [10, 5])
        self.assertEqual(config.loss_schedule, "baseline")
        self.assertEqual(config.auxiliary_loss_schedule, "linear_ramp")
        self.assertGreater(config.lambda_contrastive, 0.0)
        self.assertEqual(config.lambda_align, 0.0)
        self.assertEqual(config.lambda_uniform, 0.0)

    def test_removed_full_alias_is_rejected(self) -> None:
        """The public preset surface should no longer accept the legacy full alias."""
        self.assertEqual(get_recipe("ucagnn")["preset"], "ucagnn")
        with self.assertRaises(KeyError):
            get_recipe("full")
        with self.assertRaises(ValueError):
            build_config(_experiment_args(preset="full"))

    def test_build_config_rejects_legacy_loss_schedule_override(self) -> None:
        """New runs should reject the removed staged-BPR schedule override."""
        with self.assertRaises(ValueError):
            build_config(_experiment_args(loss_schedule="causal_then_bpr"))

    def test_build_config_respects_use_early_stopping_flag(self) -> None:
        """CLI/config plumbing should allow disabling early stopping explicitly."""
        config = build_config(_experiment_args(use_early_stopping=False))

        self.assertFalse(config.use_early_stopping)

    def test_lightgcn_preset_keeps_single_branch_depth_override(self) -> None:
        """LightGCN should use its dedicated single-branch depth field."""
        config = build_config(
            _experiment_args(
                preset="lightgcn",
                single_branch_gnn_layers=2,
                num_neighbors=[10, 5],
            )
        )

        self.assertFalse(config.use_dual_branch)
        self.assertEqual(config.single_branch_gnn_layers, 2)
        self.assertEqual(config.max_gnn_layers, 2)

    def test_dice_like_preset_keeps_explicit_branch_depth_overrides(self) -> None:
        """DICE-like should preserve explicit branch depths without a shared fallback."""
        config = build_config(
            _experiment_args(
                preset="dice_like",
                interest_gnn_layers=2,
                conformity_gnn_layers=2,
                num_neighbors=[10, 5],
            )
        )

        self.assertTrue(config.use_dual_branch)
        self.assertEqual(config.interest_gnn_layers, 2)
        self.assertEqual(config.conformity_gnn_layers, 2)
        self.assertEqual(config.max_gnn_layers, 2)

    def test_formal_profile_defaults_disable_early_stopping(self) -> None:
        """The default formal profile should own the formal support-parameter bundle."""
        profile_name = default_formal_profile_name()
        profile = get_formal_profile(profile_name)
        benchmark_args = formal_main._build_new_run_args(
            SimpleNamespace(
                device=None,
                data_dir=None,
                no_mlflow=False,
                mlflow_tracking_uri=None,
                mlflow_experiment_name=None,
                dry_run=True,
            ),
            profile_name,
        )

        self.assertFalse(profile["config_overrides"]["use_early_stopping"])
        self.assertEqual(profile["matrix"]["presets"], ["ucagnn"])
        self.assertEqual(profile["matrix"]["graph_methods"], ["cagra", "knn"])
        self.assertEqual(profile["matrix"]["scoring_weight_modes"], ["learned"])
        self.assertEqual(profile["config_overrides"]["batch_size"], 4096)
        self.assertEqual(profile["config_overrides"]["interest_gnn_layers"], 1)
        self.assertEqual(profile["config_overrides"]["conformity_gnn_layers"], 2)
        self.assertEqual(profile["config_overrides"]["dropout"], 0.1)
        self.assertEqual(profile["config_overrides"]["num_neighbors"], [10, 5])
        self.assertEqual(profile["config_overrides"]["hard_negative_ratio"], 0.0)
        self.assertEqual(profile["config_overrides"]["loss_schedule"], "baseline")
        self.assertFalse(benchmark_args.use_early_stopping)
        self.assertEqual(benchmark_args.presets, ["ucagnn"])
        self.assertEqual(benchmark_args.graph_methods, ["cagra", "knn"])
        self.assertEqual(benchmark_args.scoring_weight_modes, ["learned"])
        self.assertEqual(benchmark_args.batch_size, 4096)
        self.assertIsNone(benchmark_args.single_branch_gnn_layers)
        self.assertEqual(benchmark_args.interest_gnn_layers, 1)
        self.assertEqual(benchmark_args.conformity_gnn_layers, 2)
        self.assertEqual(benchmark_args.dropout, 0.1)
        self.assertEqual(benchmark_args.num_neighbors, [10, 5])
        self.assertEqual(benchmark_args.hard_negative_ratio, 0.0)
        self.assertEqual(benchmark_args.loss_schedule, "baseline")
        self.assertEqual(benchmark_args.curriculum_phase1_end, 15)
        self.assertEqual(benchmark_args.curriculum_phase2_end, 30)

    def test_second_formal_profile_is_reserved_for_final_comparison(self) -> None:
        """The non-default profile should keep baselines out of day-to-day runs."""
        profile_name = formal_profile_names()[1]
        profile = get_formal_profile(profile_name)
        benchmark_args = formal_main._build_new_run_args(
            SimpleNamespace(
                device=None,
                data_dir=None,
                no_mlflow=False,
                mlflow_tracking_uri=None,
                mlflow_experiment_name=None,
                dry_run=True,
            ),
            profile_name,
        )

        self.assertEqual(
            profile["matrix"]["presets"], ["ucagnn", "lightgcn", "dice_like"]
        )
        self.assertEqual(profile["matrix"]["graph_methods"], ["cagra"])
        self.assertEqual(profile["matrix"]["scoring_weight_modes"], ["fixed"])
        self.assertEqual(profile["config_overrides"]["single_branch_gnn_layers"], 2)
        self.assertEqual(profile["config_overrides"]["interest_gnn_layers"], 2)
        self.assertEqual(profile["config_overrides"]["conformity_gnn_layers"], 2)
        self.assertEqual(benchmark_args.single_branch_gnn_layers, 2)
        self.assertEqual(benchmark_args.interest_gnn_layers, 2)
        self.assertEqual(benchmark_args.conformity_gnn_layers, 2)

    def test_stale_saved_profile_falls_back_to_default_bundle(self) -> None:
        """Plain formal-run should survive a removed profile in saved state."""
        cli_args = SimpleNamespace(
            profile=None,
            resume_latest=False,
            new_run=False,
            restart=False,
            device=None,
            data_dir=None,
            no_mlflow=False,
            mlflow_tracking_uri=None,
            mlflow_experiment_name=None,
            dry_run=True,
        )

        with patch.object(
            formal_main,
            "_load_state",
            return_value={"profile_name": "removed-profile"},
        ):
            benchmark_args, profile_name, resumed = formal_main._resolve_benchmark_args(
                cli_args
            )

        self.assertEqual(profile_name, default_formal_profile_name())
        self.assertFalse(resumed)
        self.assertEqual(benchmark_args.presets, ["ucagnn"])

    def test_build_config_respects_formal_support_parameter_overrides(self) -> None:
        """Formal support-parameter overrides must flow into the runtime config."""
        config = build_config(
            _experiment_args(
                hard_negative_ratio=0.25,
                curriculum_phase1_end=3,
                curriculum_phase2_end=7,
            )
        )

        self.assertEqual(config.hard_negative_ratio, 0.25)
        self.assertEqual(config.curriculum_phase1_end, 3)
        self.assertEqual(config.curriculum_phase2_end, 7)

    def test_runtime_does_not_stop_when_early_stopping_disabled(self) -> None:
        """TrainerRuntime should keep training when the stop policy is off."""
        runtime = TrainerRuntime.__new__(TrainerRuntime)
        runtime.config = SimpleNamespace(
            use_early_stopping=False,
            curriculum_phase1_end=15,
            curriculum_phase2_end=30,
            patience=10,
        )
        runtime.best_ndcg = 0.5
        runtime.patience_counter = 7
        runtime.best_state = None

        should_stop = runtime._update_early_stopping(
            current_ndcg=0.4,
            primary_metric="NDCG@40",
            epoch=40,
            history={"train_loss": [], "val_metrics": []},
            checkpoint_path=None,
            checkpoint_every=None,
        )

        self.assertFalse(should_stop)
        self.assertEqual(runtime.patience_counter, 0)


class BenchmarkPlanTests(unittest.TestCase):
    """Pin the formal benchmark execution order and parser defaults."""

    def test_parser_defaults_use_canonical_preset_names(self) -> None:
        """Benchmark defaults should not duplicate the legacy full alias."""
        args = build_benchmark_parser().parse_args([])

        self.assertEqual(args.presets, ["ucagnn", "lightgcn", "dice_like"])
        self.assertEqual(args.scoring_weight_modes, ["learned"])

    def test_plan_sweeps_datasets_within_each_method_combo(self) -> None:
        """Datasets should be the innermost loop of the execution plan."""
        from types import SimpleNamespace

        args = SimpleNamespace(
            tier="small",
            presets=["ucagnn", "lightgcn"],
            graph_methods=["cagra", "knn"],
            scoring_weight_modes=["fixed", "learned"],
        )

        plan = build_benchmark_plan(args)

        expected_prefix = [
            ("amazonbook", "ucagnn", "cagra", "fixed"),
            ("movielens1m", "ucagnn", "cagra", "fixed"),
            ("amazonbook", "ucagnn", "cagra", "learned"),
            ("movielens1m", "ucagnn", "cagra", "learned"),
            ("amazonbook", "ucagnn", "knn", "fixed"),
            ("movielens1m", "ucagnn", "knn", "fixed"),
        ]

        self.assertEqual(plan[: len(expected_prefix)], expected_prefix)
        self.assertEqual(
            plan[-2:],
            [
                ("amazonbook", "lightgcn", "knn", "fixed"),
                ("movielens1m", "lightgcn", "knn", "fixed"),
            ],
        )
        self.assertEqual(len(plan), 12)


if __name__ == "__main__":
    unittest.main()
