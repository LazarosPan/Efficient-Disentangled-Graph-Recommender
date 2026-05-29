"""Regression coverage for formal training policy defaults."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import experiments.run_benchmark as formal_main
import numpy as np
import torch
from experiments.cli_parsers import build_benchmark_parser
from experiments.recipes import (
    default_formal_profile_name,
    formal_profile_names,
    get_formal_profile,
    get_recipe,
)
from experiments.run_benchmark import (
    _resolve_benchmark_num_neighbors_for_preset,
    build_benchmark_plan,
)
from experiments.run_experiment import (
    _auto_batch_probe_candidates,
    _auto_batch_probe_interactions,
    _bootstrap_cagra_embeddings,
    _build_canonical_name,
    _build_evaluation_identity,
    _build_training_identity,
    _release_cuda_probe_memory,
    _train_mask_numpy_from_data,
    build_benchmark_config_inputs,
    build_config,
    build_runtime_config_inputs,
    normalize_benchmark_config_overrides,
)
from src.profiling.gpu_profiler import GPUProfiler
from src.training.mini_batch_trainer import MiniBatchTrainer
from src.utils.config import UCaGNNConfig
from src.utils.reproducibility import build_torch_generator
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
        "auto_batch_size": None,
        "batch_size_candidates": None,
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
        "num_neighbors": None,
        "hard_negative_ratio": None,
        "auxiliary_losses_start_epoch": None,
        "popularity_supervision_start_epoch": None,
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

    def test_curriculum_aliases_match_threshold_fields(self) -> None:
        """Semantic curriculum aliases should mirror the underlying thresholds."""
        config = UCaGNNConfig()

        self.assertEqual(config.auxiliary_losses_start_epoch, config.auxiliary_losses_start_epoch)
        self.assertEqual(
            config.popularity_supervision_start_epoch, config.popularity_supervision_start_epoch
        )

    def test_build_config_respects_dropout_override(self) -> None:
        """Catalog/runtime plumbing should pass dropout through to the config."""
        config = build_config(_experiment_args(dropout=0.25))

        self.assertEqual(config.dropout, 0.25)

    def test_build_config_accepts_mapping_inputs(self) -> None:
        """Non-CLI callers should be able to pass plain mappings into build_config."""
        config = build_config(vars(_experiment_args(dropout=0.25, preset="ucagnn")))

        self.assertEqual(config.dropout, 0.25)
        self.assertEqual(config.scoring_weight_mode, "learned")

    def test_build_config_resolves_kuairec_default_preprocessing_preset(self) -> None:
        """Default config assembly should pin the causal-ready KuaiRec view."""
        config = build_config(_experiment_args(dataset="kuairec_v2"))

        self.assertEqual(config.preprocessing_preset, "kuairec_watchratio")
        self.assertIn(
            "ppresetkuairec_watchratio",
            _build_canonical_name(config, None, None),
        )

    def test_build_config_accepts_lr_scheduler_override(self) -> None:
        """CLI/config plumbing should allow scheduler selection."""
        config = build_config(
            _experiment_args(
                preset="ucagnn",
                lr_scheduler="cosine",
                lr_scheduler_factor=0.9,
                lr_scheduler_patience=3,
            ),
        )

        self.assertEqual(config.lr_scheduler, "cosine")
        self.assertEqual(config.lr_scheduler_factor, 0.9)
        self.assertEqual(config.lr_scheduler_patience, 3)

    def test_canonical_name_includes_lr_scheduler(self) -> None:
        """The canonical checkpoint name should expose the selected LR scheduler."""
        config = build_config(
            _experiment_args(
                preset="ucagnn",
                lr_scheduler="cosine",
            ),
        )
        canonical = _build_canonical_name(config, "ucagnn", None)

        self.assertIn("lr-cosine", canonical)

    def test_build_config_rejects_invalid_lr_scheduler_override(self) -> None:
        """Invalid scheduler names should be rejected by config validation."""
        with self.assertRaises(ValueError):
            build_config(
                _experiment_args(
                    preset="ucagnn",
                    lr_scheduler="invalid_scheduler",
                ),
            )

    def test_auto_batch_probe_candidates_follow_configured_ladder(self) -> None:
        """Auto batch-size probing should use the configured candidate ladder."""
        ml1m = UCaGNNConfig(dataset="movielens1m", device="cuda", auto_batch_size=True)
        kuairand = UCaGNNConfig(dataset="kuairand1k", device="cuda", auto_batch_size=True)

        expected = [16384, 8192, 4096, 2048, 1024, 512, 256]
        self.assertEqual(_auto_batch_probe_candidates(ml1m), expected)
        self.assertEqual(_auto_batch_probe_candidates(kuairand), expected)

    def test_auto_batch_probe_interactions_match_epoch_zero_shuffle(self) -> None:
        """Auto-batch probing should mirror the epoch-0 training shuffle."""
        config = UCaGNNConfig(seed=13)
        train_users = torch.arange(8, dtype=torch.long)
        train_items = torch.arange(100, 108, dtype=torch.long)

        shuffled_users, shuffled_items = _auto_batch_probe_interactions(
            train_users,
            train_items,
            config,
        )
        perm = torch.randperm(
            train_users.size(0),
            generator=build_torch_generator(config.seed, train_users.device),
            device=train_users.device,
        )

        self.assertTrue(torch.equal(shuffled_users, train_users[perm]))
        self.assertTrue(torch.equal(shuffled_items, train_items[perm]))

    def test_probe_cleanup_synchronizes_before_releasing_cuda_cache(self) -> None:
        """Probe cleanup should flush pending CUDA work before emptying the cache."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.synchronize") as synchronize,
            patch("torch.cuda.empty_cache") as empty_cache,
        ):
            _release_cuda_probe_memory()

        synchronize.assert_called_once_with()
        empty_cache.assert_called_once_with()

    def test_ucagnn_preset_applies_fused_scoring_defaults(self) -> None:
        """The ucagnn preset should target the fused-score contract."""
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
        self.assertEqual(config.loss_weight_contrastive, 0.0)
        self.assertEqual(config.loss_weight_align, 0.0)
        self.assertEqual(config.loss_weight_uniform, 0.0)

    def test_ucagnn_preset_keeps_explicit_branch_depth_and_neighbor_overrides(self) -> None:
        """Explicit depth overrides must survive preset application for checkpoint identity."""
        config = build_config(
            _experiment_args(
                preset="ucagnn",
                interest_gnn_layers=2,
                conformity_gnn_layers=3,
                num_neighbors=[10, 5, 3],
            ),
        )

        self.assertEqual(config.interest_gnn_layers, 2)
        self.assertEqual(config.conformity_gnn_layers, 3)
        self.assertEqual(config.max_gnn_layers, 3)
        self.assertEqual(config.num_neighbors, [10, 5, 3])

    def test_ucagnn_preset_rejects_short_neighbor_lists_after_depth_override(self) -> None:
        """Invalid fan-out shapes should fail loudly instead of falling back to preset depths."""
        with self.assertRaises(ValueError):
            build_config(
                _experiment_args(
                    preset="ucagnn",
                    interest_gnn_layers=2,
                    conformity_gnn_layers=3,
                    num_neighbors=[10, 5],
                ),
            )

    def test_removed_full_alias_is_rejected(self) -> None:
        """The public preset surface should no longer accept the removed full alias."""
        self.assertEqual(get_recipe("ucagnn")["preset"], "ucagnn")
        with self.assertRaises(KeyError):
            get_recipe("full")
        with self.assertRaises(ValueError):
            build_config(_experiment_args(preset="full"))

    def test_build_config_rejects_removed_loss_schedule_override(self) -> None:
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
            ),
        )

        self.assertFalse(config.use_dual_branch)
        self.assertFalse(config.use_features)
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
            ),
        )

        self.assertTrue(config.use_dual_branch)
        self.assertFalse(config.use_sign_aware)
        self.assertFalse(config.use_features)
        self.assertEqual(config.scoring_weight_mode, "fixed")
        self.assertEqual(config.score_weight_interest, 1.0)
        self.assertEqual(config.score_weight_conformity, 1.0)
        self.assertEqual(config.train_scoring_mode, "default")
        self.assertEqual(config.eval_scoring_mode, "default")
        self.assertEqual(config.interest_gnn_layers, 2)
        self.assertEqual(config.conformity_gnn_layers, 2)
        self.assertEqual(config.max_gnn_layers, 2)

    def test_training_identity_ignores_eval_only_overrides(self) -> None:
        """Resume compatibility should ignore evaluation-only config changes."""
        base = build_config(_experiment_args(preset="ucagnn"))
        eval_override = build_config(_experiment_args(preset="ucagnn"))
        eval_override.eval_scoring_mode = "conformity_suppressed"

        base_identity, base_hash = _build_training_identity(base, "ucagnn", None)
        override_identity, override_hash = _build_training_identity(
            eval_override,
            "ucagnn",
            None,
        )
        _, base_eval_hash = _build_evaluation_identity(base, base_hash)
        _, override_eval_hash = _build_evaluation_identity(
            eval_override,
            override_hash,
        )

        self.assertEqual(base_identity, override_identity)
        self.assertEqual(base_hash, override_hash)
        self.assertNotEqual(base_eval_hash, override_eval_hash)

    def test_training_identity_changes_when_training_config_changes(self) -> None:
        """Resume compatibility should change with training-defining config fields."""
        base = build_config(_experiment_args(preset="ucagnn"))
        changed = build_config(
            _experiment_args(
                preset="ucagnn",
                graph_policy="cagra_augmented",
            ),
        )

        _, base_hash = _build_training_identity(base, "ucagnn", None)
        _, changed_hash = _build_training_identity(changed, "ucagnn", None)

        self.assertNotEqual(base_hash, changed_hash)

    def test_formal_profile_defaults_disable_early_stopping(self) -> None:
        """The default formal profile should own the formal support-parameter bundle."""
        profile_name = default_formal_profile_name()
        profile = get_formal_profile(profile_name)
        benchmark_args = formal_main._build_new_run_args(
            SimpleNamespace(overwrite_checkpoint=False),
            profile_name,
        )

        self.assertFalse(profile["config_overrides"]["use_early_stopping"])
        self.assertEqual(profile["id"], "dev-ucagnn")
        self.assertEqual(profile["matrix"]["presets"], ["ucagnn"])
        self.assertEqual(profile["matrix"]["scoring_weight_modes"], ["learned"])
        self.assertNotIn("batch_size", profile["config_overrides"])
        self.assertNotIn("auto_batch_size", profile["config_overrides"])
        self.assertEqual(
            profile["config_overrides"]["batch_size_candidates"],
            [16384, 8192, 4096, 2048, 1024, 512, 256],
        )
        self.assertEqual(profile["config_overrides"]["interest_gnn_layers"], 1)
        self.assertEqual(profile["config_overrides"]["conformity_gnn_layers"], 2)
        self.assertEqual(profile["config_overrides"]["dropout"], 0.1)
        self.assertEqual(profile["config_overrides"]["num_neighbors"], [10, 5])
        self.assertNotIn("hard_negative_ratio", profile["config_overrides"])
        self.assertNotIn("loss_schedule", profile["config_overrides"])
        self.assertFalse(benchmark_args["use_early_stopping"])
        self.assertEqual(benchmark_args["presets"], ["ucagnn"])
        self.assertEqual(benchmark_args["scoring_weight_modes"], ["learned"])
        self.assertEqual(benchmark_args["batch_size"], 4096)
        self.assertTrue(benchmark_args["auto_batch_size"])
        self.assertEqual(
            benchmark_args["batch_size_candidates"],
            [16384, 8192, 4096, 2048, 1024, 512, 256],
        )
        self.assertIsNone(benchmark_args["single_branch_gnn_layers"])
        self.assertEqual(benchmark_args["interest_gnn_layers"], 1)
        self.assertEqual(benchmark_args["conformity_gnn_layers"], 2)
        self.assertEqual(benchmark_args["dropout"], 0.1)
        self.assertEqual(benchmark_args["num_neighbors"], [10, 5])
        self.assertEqual(benchmark_args["graph_policy"], "observed")
        self.assertIsNone(benchmark_args["graph_policy_options"])
        self.assertEqual(benchmark_args["hard_negative_ratio"], 0.0)
        self.assertIsNone(benchmark_args["loss_schedule"])
        self.assertEqual(benchmark_args["auxiliary_losses_start_epoch"], 15)
        self.assertEqual(benchmark_args["popularity_supervision_start_epoch"], 30)

    def test_benchmark_config_inputs_bridge_into_build_config(self) -> None:
        """Formal benchmark args should rebuild one run through the shared config contract."""
        profile_name = default_formal_profile_name()
        benchmark_args = formal_main._build_new_run_args(
            SimpleNamespace(overwrite_checkpoint=False),
            profile_name,
        )

        config_inputs = build_benchmark_config_inputs(
            benchmark_args,
            dataset="movielens1m",
            preset="ucagnn",
            lr_scheduler="plateau",
            scoring_weight_mode="learned",
            num_neighbors=[10, 5],
            graph_policy="cagra_augmented",
        )

        self.assertEqual(config_inputs["dataset"], "movielens1m")
        self.assertEqual(config_inputs["preset"], "ucagnn")
        self.assertEqual(config_inputs["device"], "cuda")
        self.assertEqual(config_inputs["data_dir"], "data")
        self.assertNotIn("num_neighbors_options", config_inputs)

        config = build_config(
            config_inputs,
        )

        self.assertEqual(config.dataset, "movielens1m")
        self.assertEqual(config.lr_scheduler, "plateau")
        self.assertTrue(config.auto_batch_size)
        self.assertFalse(config.use_early_stopping)
        self.assertEqual(config.graph_policy, "cagra_augmented")
        self.assertEqual(config.num_neighbors, [10, 5])

    def test_runtime_config_inputs_bridge_into_build_config(self) -> None:
        """Quick/runtime config mappings should reuse the shared config-input builder."""
        config = build_config(
            build_runtime_config_inputs(
                dataset="movielens1m",
                preset="ucagnn",
                data_dir="data",
                device="cpu",
                epochs=2,
                batch_size=64,
                auto_batch_size=False,
                graph_policy="cagra_augmented",
                eval_scoring_mode="interest_only",
                sample_interactions=100,
                loader_max_rows=100,
            ),
        )

        self.assertEqual(config.dataset, "movielens1m")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.epochs, 2)
        self.assertEqual(config.batch_size, 64)
        self.assertFalse(config.auto_batch_size)
        self.assertEqual(config.graph_policy, "cagra_augmented")
        self.assertEqual(config.eval_scoring_mode, "interest_only")
        self.assertEqual(config.sample_interactions, 100)
        self.assertEqual(config.loader_max_rows, 100)

    def test_train_mask_numpy_from_data_reuses_graph_mask(self) -> None:
        """Runtime model prep should reuse the graph's existing train mask."""
        train_mask = _train_mask_numpy_from_data(
            SimpleNamespace(
                train_mask=torch.tensor([True, False, True], dtype=torch.bool),
            ),
        )

        self.assertTrue(
            np.array_equal(train_mask, np.array([True, False, True], dtype=bool)),
        )

    def test_bootstrap_cagra_embeddings_explains_feature_requirement(self) -> None:
        """CAGRA bootstrap should explain why featureless datasets are rejected."""
        with self.assertRaisesRegex(
            ValueError,
            "combines CAGRA edges with the observed train-interaction graph",
        ):
            _bootstrap_cagra_embeddings(
                UCaGNNConfig(device="cpu"),
                canonical=SimpleNamespace(),
                observed_data=SimpleNamespace(item_features=None),
            )

    def test_formal_profile_lookup_normalizes_user_facing_labels(self) -> None:
        """Formal profile lookup should normalize user-facing aliases centrally."""
        default_profile = get_formal_profile("DEFAULT")

        self.assertEqual(default_profile["id"], default_formal_profile_name())
        self.assertIn("abauto", default_profile["name"])
        self.assertEqual(get_formal_profile("dev-ucagnn")["id"], "dev-ucagnn")

    def test_second_formal_profile_is_reserved_for_final_comparison(self) -> None:
        """The non-default profile should keep baselines out of day-to-day runs."""
        profile_name = formal_profile_names()[1]
        profile = get_formal_profile(profile_name)
        benchmark_args = formal_main._build_new_run_args(
            SimpleNamespace(overwrite_checkpoint=False),
            profile_name,
        )

        self.assertEqual(
            profile["matrix"]["presets"],
            ["ucagnn", "lightgcn", "dice_like"],
        )
        self.assertEqual(
            profile["id"],
            "dev-matched-comparison-i1-c2-nn8-4-ep200-lr0-01",
        )
        self.assertEqual(
            profile["matrix"]["scoring_weight_modes"],
            ["learned", "fixed"],
        )
        self.assertEqual(profile["config_overrides"]["single_branch_gnn_layers"], 2)
        self.assertNotIn("batch_size", profile["config_overrides"])
        self.assertNotIn("auto_batch_size", profile["config_overrides"])
        self.assertEqual(profile["config_overrides"]["interest_gnn_layers"], 1)
        self.assertEqual(profile["config_overrides"]["conformity_gnn_layers"], 2)
        self.assertEqual(profile["config_overrides"]["num_neighbors"], [8, 4])
        self.assertEqual(
            benchmark_args["scoring_weight_modes"],
            ["learned", "fixed"],
        )
        self.assertTrue(benchmark_args["auto_batch_size"])
        self.assertEqual(benchmark_args["single_branch_gnn_layers"], 2)
        self.assertEqual(benchmark_args["interest_gnn_layers"], 1)
        self.assertEqual(benchmark_args["conformity_gnn_layers"], 2)
        self.assertEqual(benchmark_args["num_neighbors"], [8, 4])
        self.assertIsNone(benchmark_args["sample_interactions"])
        self.assertIsNone(benchmark_args["loader_max_rows"])

    def test_removed_saved_profile_requires_fresh_formal_run(self) -> None:
        """Saved state should fail loudly when the referenced profile was removed."""
        cli_args = SimpleNamespace(
            profile=None,
            overwrite_checkpoint=False,
        )

        state_path = Path(self.id().replace(".", "_") + "_state.json")
        state_path.write_text('{"profile_name": "removed-profile"}')
        self.addCleanup(lambda: state_path.unlink(missing_ok=True))

        with (
            patch.object(formal_main, "STATE_PATH", state_path),
            self.assertRaisesRegex(ValueError, "no longer defined"),
        ):
            formal_main._resolve_benchmark_args(cli_args)

    def test_resolve_benchmark_args_builds_fresh_run_when_state_is_missing(self) -> None:
        """Missing saved state should force a fresh formal benchmark plan."""
        cli_args = SimpleNamespace(profile=None, overwrite_checkpoint=False)

        with (
            patch.object(formal_main, "_load_saved_formal_state", return_value=None),
            patch.object(
                formal_main,
                "_build_new_run_args",
                return_value={"profile_name": default_formal_profile_name(), "fresh": True},
            ) as build_new_run_args,
        ):
            benchmark_args, profile_name, resumed = formal_main._resolve_benchmark_args(
                cli_args,
            )

        build_new_run_args.assert_called_once_with(cli_args, default_formal_profile_name())
        self.assertFalse(resumed)
        self.assertEqual(profile_name, default_formal_profile_name())
        self.assertEqual(
            benchmark_args,
            {"profile_name": default_formal_profile_name(), "fresh": True},
        )

    def test_resolve_benchmark_args_builds_fresh_run_when_requested_profile_differs(
        self,
    ) -> None:
        """A different requested profile should restart instead of resuming saved args."""
        saved_profile_name = default_formal_profile_name()
        requested_profile_name = next(
            profile_name
            for profile_name in formal_profile_names()
            if profile_name != saved_profile_name
        )
        cli_args = SimpleNamespace(profile=requested_profile_name, overwrite_checkpoint=False)
        saved_state = {
            "profile_name": saved_profile_name,
            "profile_slug": "development-signature",
            "benchmark_args": {"datasets": ["small"], "presets": ["ucagnn"]},
        }

        with (
            patch.object(formal_main, "_load_saved_formal_state", return_value=saved_state),
            patch.object(
                formal_main,
                "_build_new_run_args",
                return_value={"profile_name": requested_profile_name, "fresh": True},
            ) as build_new_run_args,
        ):
            benchmark_args, profile_name, resumed = formal_main._resolve_benchmark_args(
                cli_args,
            )

        build_new_run_args.assert_called_once_with(cli_args, requested_profile_name)
        self.assertFalse(resumed)
        self.assertEqual(profile_name, requested_profile_name)
        self.assertEqual(
            benchmark_args,
            {"profile_name": requested_profile_name, "fresh": True},
        )

    def test_resolve_benchmark_args_resumes_saved_run_when_state_matches_without_requested_profile(
        self,
    ) -> None:
        """Unspecified profiles should resume when the saved semantic plan still matches."""
        saved_profile_name = default_formal_profile_name()
        cli_args = SimpleNamespace(profile=None, overwrite_checkpoint=True)
        expected_args = formal_main._build_new_run_args(
            SimpleNamespace(overwrite_checkpoint=False),
            saved_profile_name,
        )
        saved_state = {
            "profile_name": saved_profile_name,
            "profile_slug": expected_args["profile_slug"],
            "benchmark_args": dict(expected_args),
        }

        with (
            patch.object(formal_main, "_load_saved_formal_state", return_value=saved_state),
            patch.object(
                formal_main,
                "_build_new_run_args",
                return_value=dict(expected_args),
            ) as build_new_run_args,
        ):
            benchmark_args, profile_name, resumed = formal_main._resolve_benchmark_args(
                cli_args,
            )

        build_new_run_args.assert_called_once_with(cli_args, saved_profile_name)
        self.assertTrue(resumed)
        self.assertEqual(profile_name, saved_profile_name)
        self.assertEqual(benchmark_args["profile_name"], saved_profile_name)
        self.assertEqual(benchmark_args["profile_slug"], expected_args["profile_slug"])
        self.assertTrue(benchmark_args["overwrite_checkpoint"])
        self.assertTrue(benchmark_args["resume_batch"])

    def test_resolve_benchmark_args_restarts_when_latest_saved_plan_differs(
        self,
    ) -> None:
        """Unspecified profiles should restart when the saved semantic plan changed."""
        saved_profile_name = default_formal_profile_name()
        cli_args = SimpleNamespace(profile=None, overwrite_checkpoint=False)
        expected_args = formal_main._build_new_run_args(
            SimpleNamespace(overwrite_checkpoint=False),
            saved_profile_name,
        )
        saved_args = dict(expected_args)
        saved_args["datasets"] = ["__mismatched__"]
        saved_state = {
            "profile_name": saved_profile_name,
            "profile_slug": expected_args["profile_slug"],
            "benchmark_args": saved_args,
        }

        with (
            patch.object(formal_main, "_load_saved_formal_state", return_value=saved_state),
            patch.object(
                formal_main,
                "_build_new_run_args",
                return_value=dict(expected_args),
            ) as build_new_run_args,
        ):
            benchmark_args, profile_name, resumed = formal_main._resolve_benchmark_args(
                cli_args,
            )

        build_new_run_args.assert_called_once_with(cli_args, saved_profile_name)
        self.assertFalse(resumed)
        self.assertEqual(profile_name, saved_profile_name)
        self.assertEqual(benchmark_args, expected_args)

    def test_resolve_benchmark_args_resumes_requested_profile_when_saved_plan_matches(
        self,
    ) -> None:
        """Matching requested and saved profiles should resume when the semantic plan matches."""
        requested_profile_name = default_formal_profile_name()
        cli_args = SimpleNamespace(profile=requested_profile_name, overwrite_checkpoint=True)
        expected_args = formal_main._build_new_run_args(
            SimpleNamespace(overwrite_checkpoint=False),
            requested_profile_name,
        )
        saved_state = {
            "profile_name": requested_profile_name,
            "profile_slug": expected_args["profile_slug"],
            "benchmark_args": dict(expected_args),
        }

        with (
            patch.object(formal_main, "_load_saved_formal_state", return_value=saved_state),
            patch.object(
                formal_main,
                "_build_new_run_args",
                return_value=dict(expected_args),
            ) as build_new_run_args,
        ):
            benchmark_args, profile_name, resumed = formal_main._resolve_benchmark_args(
                cli_args,
            )

        build_new_run_args.assert_called_once_with(cli_args, requested_profile_name)
        self.assertTrue(resumed)
        self.assertEqual(profile_name, requested_profile_name)
        self.assertEqual(benchmark_args["profile_name"], requested_profile_name)
        self.assertEqual(benchmark_args["profile_slug"], expected_args["profile_slug"])
        self.assertTrue(benchmark_args["overwrite_checkpoint"])
        self.assertTrue(benchmark_args["resume_batch"])

    def test_resolve_benchmark_args_restarts_when_requested_profile_plan_differs(
        self,
    ) -> None:
        """Matching profiles should still restart if the saved semantic plan changed."""
        saved_profile_name = default_formal_profile_name()
        requested_profile_name = saved_profile_name
        cli_args = SimpleNamespace(profile=requested_profile_name, overwrite_checkpoint=False)
        expected_args = formal_main._build_new_run_args(
            SimpleNamespace(overwrite_checkpoint=False),
            requested_profile_name,
        )
        saved_args = dict(expected_args)
        saved_args["datasets"] = ["__mismatched__"]
        saved_state = {
            "profile_name": saved_profile_name,
            "profile_slug": expected_args["profile_slug"],
            "benchmark_args": saved_args,
        }

        with (
            patch.object(formal_main, "_load_saved_formal_state", return_value=saved_state),
            patch.object(
                formal_main,
                "_build_new_run_args",
                return_value=dict(expected_args),
            ) as build_new_run_args,
        ):
            benchmark_args, profile_name, resumed = formal_main._resolve_benchmark_args(
                cli_args,
            )

        build_new_run_args.assert_called_once_with(cli_args, requested_profile_name)
        self.assertFalse(resumed)
        self.assertEqual(profile_name, requested_profile_name)
        self.assertEqual(benchmark_args, expected_args)

    def test_build_new_run_args_ignores_smoke_only_profile_caps(self) -> None:
        """Formal profiles should not turn the formal wrapper into a sampled run."""
        cli_args = SimpleNamespace(
            overwrite_checkpoint=False,
        )
        profile_bundle = {
            "id": "dev-smoke",
            "name": "dev-smoke-signature",
            "description": "diagnostic profile",
            "matrix": {
                "datasets": ["movielens1m"],
                "presets": ["ucagnn"],
                "scoring_weight_modes": ["learned"],
            },
            "config_overrides": {
                "epochs": 5,
                "sample_interactions": 128,
                "loader_max_rows": 128,
            },
        }

        with patch.object(formal_main, "get_formal_profile", return_value=profile_bundle):
            benchmark_args = formal_main._build_new_run_args(cli_args, "dev-smoke")

        self.assertIsNone(benchmark_args["sample_interactions"])
        self.assertIsNone(benchmark_args["loader_max_rows"])

    def test_normalize_benchmark_args_rejects_unexpected_saved_fields(self) -> None:
        """Saved formal-run state should reject fields outside the current shape."""
        with self.assertRaises(ValueError):
            formal_main._normalize_benchmark_args(
                {
                    "tier": "small",
                    "presets": ["ucagnn"],
                    "epochs": 60,
                    "batch_size": 4096,
                    "lr": 1e-3,
                    "unexpected_depth_field": 2,
                    "num_neighbors": [10, 5],
                    "device": "cuda",
                    "data_dir": "data",
                    "no_mlflow": False,
                    "mlflow_tracking_uri": None,
                    "mlflow_experiment_name": "ucagnn-formal",
                    "batch_id": "formal-dev-batch",
                    "resume_batch": True,
                    "dry_run": False,
                },
                fallback_profile_name="development",
            )

    def test_normalize_benchmark_args_rejects_removed_graph_method_field(self) -> None:
        """Saved formal-run state should reject the removed graph_method field."""
        with self.assertRaises(ValueError):
            formal_main._normalize_benchmark_args(
                {
                    "tier": "small",
                    "presets": ["ucagnn"],
                    "graph_method": "cagra",
                    "epochs": 60,
                    "batch_size": 4096,
                    "lr": 1e-3,
                    "num_neighbors": [10, 5],
                    "device": "cuda",
                    "data_dir": "data",
                    "no_mlflow": False,
                    "mlflow_tracking_uri": None,
                    "mlflow_experiment_name": "ucagnn-formal",
                    "batch_id": "formal-dev-batch",
                    "resume_batch": True,
                    "dry_run": False,
                },
                fallback_profile_name="development",
            )

    def test_normalize_benchmark_args_rejects_removed_neighbor_options_field(self) -> None:
        """Saved formal-run state should reject the removed num_neighbors_options field."""
        with self.assertRaises(ValueError):
            formal_main._normalize_benchmark_args(
                {
                    "datasets": ["small"],
                    "presets": ["ucagnn"],
                    "scoring_weight_modes": ["learned"],
                    "epochs": 60,
                    "batch_size": 4096,
                    "lr": 1e-3,
                    "num_neighbors": [10, 5],
                    "num_neighbors_options": [[10, 5], [5, 3]],
                    "device": "cuda",
                    "data_dir": "data",
                    "no_mlflow": False,
                    "mlflow_tracking_uri": None,
                    "mlflow_experiment_name": "ucagnn-formal",
                    "batch_id": "formal-dev-batch",
                    "resume_batch": True,
                    "dry_run": False,
                },
                fallback_profile_name="development",
            )

    def test_normalize_benchmark_args_rejects_removed_popularity_window_field(self) -> None:
        """Saved formal-run state should reject the removed popularity window field."""
        with self.assertRaises(ValueError):
            formal_main._normalize_benchmark_args(
                {
                    "datasets": ["small"],
                    "presets": ["ucagnn"],
                    "scoring_weight_modes": ["learned"],
                    "epochs": 60,
                    "batch_size": 4096,
                    "lr": 1e-3,
                    "num_neighbors": [10, 5],
                    "popularity_window_seconds": None,
                    "device": "cuda",
                    "data_dir": "data",
                    "no_mlflow": False,
                    "mlflow_tracking_uri": None,
                    "mlflow_experiment_name": "ucagnn-formal",
                    "batch_id": "formal-dev-batch",
                    "resume_batch": True,
                    "dry_run": False,
                },
                fallback_profile_name="development",
            )

    def test_saved_benchmark_resolution_resumes_matching_plan_with_runtime_overrides(self) -> None:
        """Shared saved-run resolution should resume matching plans and apply runtime overrides."""
        cli_args = SimpleNamespace(overwrite_checkpoint=True)
        saved_args = {
            "datasets": ["small"],
            "presets": ["ucagnn"],
            "scoring_weight_modes": ["learned"],
            "profile_name": "development",
            "profile_slug": "development-signature",
            "epochs": 60,
            "use_early_stopping": False,
            "batch_size": 4096,
            "auto_batch_size": True,
            "batch_size_candidates": [16384, 8192, 4096, 2048, 1024, 512, 256],
            "lr": 1e-3,
            "single_branch_gnn_layers": None,
            "interest_gnn_layers": 1,
            "conformity_gnn_layers": 2,
            "dropout": 0.1,
            "num_neighbors": [10, 5],
            "hard_negative_ratio": 0.0,
            "auxiliary_losses_start_epoch": 15,
            "popularity_supervision_start_epoch": 30,
            "loss_schedule": "baseline",
            "loader_max_rows": 128,
            "sample_interactions": 128,
            "device": "cuda",
            "data_dir": "data",
            "no_mlflow": False,
            "mlflow_tracking_uri": None,
            "mlflow_experiment_name": "ucagnn-formal",
            "batch_id": "formal-dev-a",
            "resume_batch": False,
            "dry_run": True,
            "overwrite_checkpoint": False,
            "change_note": None,
            "graph_policy_options": None,
            "preprocessing_preset_options": None,
        }
        expected_args = dict(saved_args)

        resolved_args, profile_name, resumed = formal_main._resolve_saved_benchmark_args(
            saved_args,
            expected_args,
            cli_args,
            profile_name="development",
            profile_slug="development-signature",
        )

        self.assertTrue(resumed)
        self.assertEqual(profile_name, "development")
        self.assertTrue(resolved_args["overwrite_checkpoint"])
        self.assertTrue(resolved_args["resume_batch"])
        self.assertFalse(resolved_args["dry_run"])
        self.assertIsNone(resolved_args["sample_interactions"])
        self.assertIsNone(resolved_args["loader_max_rows"])
        self.assertEqual(resolved_args["profile_name"], "development")
        self.assertEqual(resolved_args["profile_slug"], "development-signature")

    def test_saved_benchmark_resolution_falls_back_when_plan_differs(self) -> None:
        """Saved-run resolution should restart when the semantic plan no longer matches."""
        cli_args = SimpleNamespace(overwrite_checkpoint=False)
        saved_args = {
            "datasets": ["small"],
            "presets": ["ucagnn"],
            "scoring_weight_modes": ["learned"],
            "profile_name": "development",
            "profile_slug": "development-signature",
            "epochs": 59,
            "use_early_stopping": False,
            "batch_size": 4096,
            "auto_batch_size": True,
            "batch_size_candidates": [16384, 8192, 4096, 2048, 1024, 512, 256],
            "lr": 1e-3,
            "single_branch_gnn_layers": None,
            "interest_gnn_layers": 1,
            "conformity_gnn_layers": 2,
            "dropout": 0.1,
            "num_neighbors": [10, 5],
            "hard_negative_ratio": 0.0,
            "auxiliary_losses_start_epoch": 15,
            "popularity_supervision_start_epoch": 30,
            "loss_schedule": "baseline",
            "loader_max_rows": None,
            "sample_interactions": None,
            "device": "cuda",
            "data_dir": "data",
            "no_mlflow": False,
            "mlflow_tracking_uri": None,
            "mlflow_experiment_name": "ucagnn-formal",
            "batch_id": "formal-dev-a",
            "resume_batch": True,
            "dry_run": False,
            "overwrite_checkpoint": False,
            "change_note": None,
            "graph_policy_options": None,
            "preprocessing_preset_options": None,
        }
        expected_args = dict(saved_args, epochs=60)

        resolved_args, profile_name, resumed = formal_main._resolve_saved_benchmark_args(
            saved_args,
            expected_args,
            cli_args,
            profile_name="development",
            profile_slug="development-signature",
        )

        self.assertFalse(resumed)
        self.assertEqual(profile_name, "development")
        self.assertEqual(resolved_args, expected_args)

    def test_benchmark_plan_signature_ignores_runtime_overrides(self) -> None:
        """Resume matching should compare the semantic plan, not runtime routing flags."""
        base_args = argparse.Namespace(
            datasets=["small"],
            presets=["ucagnn"],
            scoring_weight_modes=["learned"],
            profile_name="development",
            epochs=60,
            use_early_stopping=False,
            batch_size=4096,
            auto_batch_size=True,
            batch_size_candidates=[16384, 8192, 4096, 2048, 1024, 512, 256],
            lr=1e-3,
            single_branch_gnn_layers=None,
            interest_gnn_layers=1,
            conformity_gnn_layers=2,
            dropout=0.1,
            num_neighbors=[10, 5],
            hard_negative_ratio=0.0,
            auxiliary_losses_start_epoch=15,
            popularity_supervision_start_epoch=30,
            loss_schedule="baseline",
            loader_max_rows=None,
            sample_interactions=None,
            device="cuda",
            data_dir="data",
            no_mlflow=False,
            mlflow_tracking_uri=None,
            mlflow_experiment_name="ucagnn-formal",
            batch_id="formal-dev-a",
            resume_batch=True,
            dry_run=False,
        )
        runtime_override_args = argparse.Namespace(
            **{
                **vars(base_args),
                "device": "cpu",
                "data_dir": "/tmp/data",
                "no_mlflow": True,
                "mlflow_tracking_uri": "sqlite:////tmp/mlflow.db",
                "mlflow_experiment_name": "scratch",
                "batch_id": "formal-dev-b",
                "dry_run": True,
            },
        )

        self.assertEqual(
            formal_main._benchmark_plan_signature(base_args),
            formal_main._benchmark_plan_signature(runtime_override_args),
        )

    def test_build_config_respects_formal_support_parameter_overrides(self) -> None:
        """Formal support-parameter overrides must flow into the runtime config."""
        config = build_config(
            _experiment_args(
                hard_negative_ratio=0.25,
                auxiliary_losses_start_epoch=3,
                popularity_supervision_start_epoch=7,
            ),
        )

        self.assertEqual(config.hard_negative_ratio, 0.25)
        self.assertEqual(config.auxiliary_losses_start_epoch, 3)
        self.assertEqual(config.popularity_supervision_start_epoch, 7)

    def test_normalize_benchmark_config_overrides_uses_shared_defaults(self) -> None:
        """Benchmark payload normalization should stay centralized and JSON-safe."""
        normalized = normalize_benchmark_config_overrides(
            {
                "lr_scheduler": "plateau,cosine",
                "num_neighbors": [[10, 5], [5, 3]],
                "hard_negative_ratio": "0.25",
            },
        )

        self.assertEqual(normalized["lr_scheduler"], ["plateau", "cosine"])
        self.assertEqual(normalized["num_neighbors"], [[10, 5], [5, 3]])
        self.assertEqual(normalized["batch_size"], UCaGNNConfig().batch_size)
        self.assertTrue(normalized["auto_batch_size"])
        self.assertIsNone(normalized["graph_policy"])
        self.assertIsNone(normalized["graph_policy_options"])
        self.assertEqual(normalized["hard_negative_ratio"], 0.25)

    def test_normalize_benchmark_config_overrides_supports_graph_policy_sweeps(self) -> None:
        """Benchmark payload normalization should expand graph-policy sweep lists safely."""
        normalized = normalize_benchmark_config_overrides(
            {
                "graph_policy": ["observed", "cagra_augmented", "observed"],
            },
        )

        self.assertEqual(normalized["graph_policy"], "observed")
        self.assertEqual(
            normalized["graph_policy_options"],
            ["observed", "cagra_augmented"],
        )

    def test_normalize_benchmark_config_overrides_supports_preprocessing_sweeps(self) -> None:
        """Benchmark payload normalization should expand preprocessing sweep lists safely."""
        normalized = normalize_benchmark_config_overrides(
            {
                "preprocessing_preset": [
                    "kuairec_fullobs",
                    "kuairec_watchratio",
                    "kuairec_fullobs",
                ],
            },
        )

        self.assertEqual(normalized["preprocessing_preset"], "kuairec_fullobs")
        self.assertEqual(
            normalized["preprocessing_preset_options"],
            ["kuairec_fullobs", "kuairec_watchratio"],
        )

    def test_runtime_does_not_stop_when_early_stopping_disabled(self) -> None:
        """TrainerRuntime should keep training when the stop policy is off."""
        runtime = TrainerRuntime.__new__(TrainerRuntime)
        runtime.config = SimpleNamespace(
            use_early_stopping=False,
            auxiliary_losses_start_epoch=15,
            popularity_supervision_start_epoch=30,
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

    def test_validation_eval_retries_after_cuda_oom_with_optimizer_offload(self) -> None:
        """Validation should retry once after offloading optimizer state on CUDA OOM."""
        runtime = TrainerRuntime.__new__(TrainerRuntime)
        runtime.device = torch.device("cuda")
        runtime.config = SimpleNamespace(use_amp=False)
        runtime.use_amp = False
        runtime.amp_dtype = torch.float16
        runtime.ema_model = None
        runtime.model = object()
        runtime.data = SimpleNamespace(val_mask="val-mask")
        runtime.evaluator = Mock()
        runtime.evaluator.evaluate.side_effect = [
            torch.OutOfMemoryError("oom"),
            {"NDCG@40": 0.5},
        ]
        runtime.optimizer = Mock()
        runtime._move_optimizer_state = Mock()

        with patch("torch.cuda.empty_cache") as empty_cache:
            metrics = runtime._evaluate_validation_metrics()

        self.assertEqual(metrics, {"NDCG@40": 0.5})
        self.assertEqual(runtime.evaluator.evaluate.call_count, 2)
        runtime._move_optimizer_state.assert_any_call(torch.device("cpu"))
        runtime._move_optimizer_state.assert_any_call(runtime.device)
        runtime.optimizer.zero_grad.assert_called_once_with(set_to_none=True)
        self.assertGreaterEqual(empty_cache.call_count, 2)

    def test_validation_eval_falls_back_to_cpu_after_second_cuda_oom(self) -> None:
        """Validation should use a CPU retry when GPU evaluation still does not fit."""
        runtime = TrainerRuntime.__new__(TrainerRuntime)
        runtime.device = torch.device("cuda")
        runtime.config = SimpleNamespace(use_amp=False)
        runtime.use_amp = False
        runtime.amp_dtype = torch.float16
        runtime.ema_model = None
        runtime.data = SimpleNamespace(val_mask=torch.tensor([True, False]))
        runtime.evaluator = Mock()
        runtime.evaluator.evaluate.side_effect = [
            torch.OutOfMemoryError("oom"),
            torch.OutOfMemoryError("oom again"),
            {"NDCG@40": 0.7},
        ]
        runtime.optimizer = Mock()
        runtime._move_optimizer_state = Mock()
        runtime.model = Mock()
        runtime.model.embedding = Mock()
        runtime.model.embedding.invalidate_feature_cache = Mock()

        with patch("torch.cuda.empty_cache") as empty_cache:
            metrics = runtime._evaluate_validation_metrics()

        self.assertEqual(metrics, {"NDCG@40": 0.7})
        self.assertEqual(runtime.evaluator.evaluate.call_count, 3)
        runtime._move_optimizer_state.assert_any_call(torch.device("cpu"))
        runtime._move_optimizer_state.assert_any_call(runtime.device)
        runtime.model.to.assert_any_call(torch.device("cpu"))
        runtime.model.to.assert_any_call(runtime.device)
        self.assertGreaterEqual(
            runtime.model.embedding.invalidate_feature_cache.call_count,
            2,
        )
        self.assertGreaterEqual(empty_cache.call_count, 3)

    def test_prepare_batch_falls_back_to_cpu_sampler_after_cuda_oom(self) -> None:
        """Batch preparation should switch back to the CPU sampler after a CUDA OOM."""
        trainer = MiniBatchTrainer.__new__(MiniBatchTrainer)
        trainer.sampler_device = torch.device("cuda")
        trainer.device = torch.device("cuda")
        trainer.data = object()
        trainer._force_cpu_sampler = False
        trainer._build_subgraph_sampler = Mock(
            return_value=(Mock(), torch.device("cpu")),
        )
        expected_batch = Mock()
        trainer._prepare_batch_on_sampler_device = Mock(
            side_effect=[torch.OutOfMemoryError("oom"), expected_batch],
        )

        with patch("torch.cuda.empty_cache") as empty_cache:
            actual_batch = trainer._prepare_batch(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                random_seed=13,
            )

        self.assertIs(actual_batch, expected_batch)
        self.assertTrue(trainer._force_cpu_sampler)
        self.assertEqual(trainer.sampler_device, torch.device("cpu"))
        trainer._build_subgraph_sampler.assert_called_once_with(trainer.data)
        self.assertEqual(trainer._prepare_batch_on_sampler_device.call_count, 2)
        empty_cache.assert_called()


class BenchmarkPlanTests(unittest.TestCase):
    """Pin the formal benchmark execution order and parser defaults."""

    def test_lightgcn_uses_two_hop_prefix_from_deeper_profile_bundle(self) -> None:
        """Benchmark wiring should truncate deeper profile fan-out for LightGCN."""
        resolved = _resolve_benchmark_num_neighbors_for_preset(
            {
                "single_branch_gnn_layers": 2,
                "interest_gnn_layers": 2,
                "conformity_gnn_layers": 3,
            },
            "lightgcn",
            [10, 5, 3],
        )

        self.assertEqual(resolved, [10, 5])

    def test_parser_defaults_use_canonical_preset_names(self) -> None:
        """Benchmark defaults should not duplicate the removed full alias."""
        args = build_benchmark_parser().parse_args([])

        self.assertEqual(args.presets, ["ucagnn", "lightgcn", "dice_like"])
        self.assertEqual(args.scoring_weight_modes, ["learned"])

    def test_dice_like_score_mix_is_fixed_only(self) -> None:
        """DICE-like should stay on the fixed score-mix path in formal sweeps."""
        self.assertEqual(
            formal_main._scoring_weight_modes_for_preset(
                "dice_like",
                ["fixed", "learned"],
            ),
            ["fixed"],
        )

    def test_plan_sweeps_datasets_within_each_method_combo(self) -> None:
        """Datasets should be the innermost loop of the execution plan."""
        from types import SimpleNamespace

        args = SimpleNamespace(
            datasets=["small"],
            presets=["ucagnn", "lightgcn"],
            scoring_weight_modes=["fixed", "learned"],
            num_neighbors=[[10, 5], [5, 3]],
        )

        plan = build_benchmark_plan(args)

        expected_prefix = [
            ("amazonbook", "ucagnn", "fixed", "plateau", None, "observed", (10, 5)),
            ("amazonbook", "ucagnn", "fixed", "plateau", None, "observed", (5, 3)),
            ("movielens1m", "ucagnn", "fixed", "plateau", None, "observed", (10, 5)),
            ("movielens1m", "ucagnn", "fixed", "plateau", None, "observed", (5, 3)),
            ("amazonbook", "ucagnn", "learned", "plateau", None, "observed", (10, 5)),
            ("amazonbook", "ucagnn", "learned", "plateau", None, "observed", (5, 3)),
        ]

        self.assertEqual(plan[: len(expected_prefix)], expected_prefix)
        self.assertEqual(len(plan), 12)

    def test_build_benchmark_plan_sweeps_graph_policies(self) -> None:
        """Benchmark planning should expand graph-policy sweeps into separate runs."""
        args = SimpleNamespace(
            datasets=["movielens1m"],
            presets=["ucagnn"],
            scoring_weight_modes=["learned"],
            graph_policy="observed",
            graph_policy_options=["observed", "cagra_augmented"],
            num_neighbors=[10, 5],
            lr_scheduler="plateau",
        )

        plan = build_benchmark_plan(args)

        self.assertEqual(
            plan,
            [
                ("movielens1m", "ucagnn", "learned", "plateau", None, "observed", (10, 5)),
                (
                    "movielens1m",
                    "ucagnn",
                    "learned",
                    "plateau",
                    None,
                    "cagra_augmented",
                    (10, 5),
                ),
            ],
        )

    def test_build_benchmark_plan_sweeps_preprocessing_presets(self) -> None:
        """Benchmark planning should expand preprocessing sweeps into separate runs."""
        args = formal_main._normalize_benchmark_args(
            SimpleNamespace(
                datasets=["kuairec_v2"],
                presets=["ucagnn"],
                scoring_weight_modes=["learned"],
                preprocessing_preset=["kuairec_fullobs", "kuairec_watchratio"],
                num_neighbors=[10, 5],
                lr_scheduler="plateau",
            ),
        )

        plan = build_benchmark_plan(args)

        self.assertEqual(
            plan,
            [
                (
                    "kuairec_v2",
                    "ucagnn",
                    "learned",
                    "plateau",
                    "kuairec_fullobs",
                    "observed",
                    (10, 5),
                ),
                (
                    "kuairec_v2",
                    "ucagnn",
                    "learned",
                    "plateau",
                    "kuairec_watchratio",
                    "observed",
                    (10, 5),
                ),
            ],
        )

    def test_build_benchmark_plan_accepts_explicit_dataset_names(self) -> None:
        """Benchmark plan should accept explicit dataset names alongside tier labels."""
        args = SimpleNamespace(
            datasets=["movielens1m", "small"],
            presets=["ucagnn"],
            scoring_weight_modes=["learned"],
            num_neighbors=[10, 5],
            lr_scheduler="cosine",
        )

        plan = build_benchmark_plan(args)

        self.assertIn(
            ("movielens1m", "ucagnn", "learned", "cosine", None, "observed", (10, 5)),
            plan,
        )
        self.assertIn(
            ("amazonbook", "ucagnn", "learned", "cosine", None, "observed", (10, 5)),
            plan,
        )

    def test_build_benchmark_plan_resolves_all_lr_schedulers(self) -> None:
        """The lr_scheduler='all' shorthand should expand to all supported schedulers."""
        args = SimpleNamespace(
            datasets=["movielens1m"],
            presets=["ucagnn"],
            scoring_weight_modes=["learned"],
            num_neighbors=[10, 5],
            lr_scheduler="all",
        )

        plan = build_benchmark_plan(args)
        schedulers = {entry[3] for entry in plan}

        self.assertEqual(schedulers, set(formal_main.SUPPORTED_LR_SCHEDULERS))

    def test_run_benchmark_reuses_pre_normalized_payload_for_dry_run(self) -> None:
        """Dry-run benchmark execution should not renormalize an internal payload."""
        normalized_args = formal_main._normalize_benchmark_args(
            SimpleNamespace(
                datasets=["movielens1m"],
                presets=["ucagnn"],
                scoring_weight_modes=["learned"],
                num_neighbors=[10, 5],
                dry_run=True,
            ),
        )

        with patch.object(
            formal_main,
            "_normalize_benchmark_args",
            side_effect=AssertionError("unexpected renormalization"),
        ):
            exit_code = formal_main.run_benchmark(normalized_args)

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
