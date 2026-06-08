"""Regression coverage for formal training policy defaults."""

from __future__ import annotations

import argparse
import dataclasses
import unittest
import warnings
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
    _build_evaluation_identity,
    _build_training_identity,
    _cuda_memory_snapshot,
    _exception_summary,
    _release_cuda_probe_memory,
    _resume_auto_batch_fallback,
    _train_mask_numpy_from_data,
    build_benchmark_config_inputs,
    build_config,
    build_runtime_config_inputs,
    build_runtime_model,
    normalize_benchmark_config_overrides,
)
from scripts.query_results import _format_scoremix
from src.data.canonical import CanonicalInteractions
from src.data.graph_builder import build_graph
from src.losses.loss_suite import LossSuite
from src.models.baselines.dice import PaperGCNDICE
from src.models.baselines.lightgcn import PaperLightGCN
from src.profiling.gpu_profiler import (
    GPUProfiler,
    TrainingResourceStats,
    sample_gpu_resource_snapshot,
)
from src.training.mini_batch_trainer import MiniBatchTrainer
from src.utils.config import UCaGNNConfig
from src.utils.experiment_naming import build_canonical_experiment_name
from src.utils.reproducibility import build_torch_generator
from src.utils.trainer_runtime import TrainerRuntime
from torch.nn import functional


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


def _tiny_canonical() -> CanonicalInteractions:
    """Return a small split-safe canonical interaction table for model factory tests."""
    return CanonicalInteractions(
        user_id=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        item_id=np.array([0, 1, 1, 2, 2, 3], dtype=np.int64),
        label=np.ones(6, dtype=np.float32),
        timestamp=np.arange(1, 7, dtype=np.int64),
        sign=np.ones(6, dtype=np.float32),
        popularity=np.ones(4, dtype=np.float32),
        n_users=3,
        n_items=4,
        user_map={0: 0, 1: 1, 2: 2},
        item_map={0: 0, 1: 1, 2: 2, 3: 3},
        train_mask=np.array([True, True, True, True, False, False], dtype=bool),
        val_mask=np.array([False, False, False, False, True, False], dtype=bool),
        test_mask=np.array([False, False, False, False, False, True], dtype=bool),
    )


def _positive_dcor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the positive DICE distance-correlation discrepancy."""
    distance_x = torch.cdist(x.float(), x.float(), p=2)
    distance_y = torch.cdist(y.float(), y.float(), p=2)
    centered_x = (
        distance_x
        - distance_x.mean(dim=0, keepdim=True)
        - distance_x.mean(dim=1, keepdim=True)
        + distance_x.mean()
    )
    centered_y = (
        distance_y
        - distance_y.mean(dim=0, keepdim=True)
        - distance_y.mean(dim=1, keepdim=True)
        + distance_y.mean()
    )
    n = float(x.size(0) * x.size(0))
    dcov_xy = (centered_x * centered_y).sum() / n
    dcov_xx = (centered_x * centered_x).sum() / n
    dcov_yy = (centered_y * centered_y).sum() / n
    denominator = torch.sqrt(
        dcov_xx.clamp_min(1e-12).sqrt() * dcov_yy.clamp_min(1e-12).sqrt(),
    )
    return torch.sqrt(dcov_xy.clamp_min(1e-12)) / denominator.clamp_min(1e-12)


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
        self.assertFalse(hasattr(config, "scoring_weight_mode"))

    def test_build_config_resolves_kuairec_default_preprocessing_preset(self) -> None:
        """Default config assembly should pin the causal-ready KuaiRec view."""
        config = build_config(_experiment_args(dataset="kuairec_v2"))

        self.assertEqual(config.preprocessing_preset, "kuairec_watchratio")
        self.assertIn(
            "ppresetkuairec_watchratio",
            build_canonical_experiment_name(config, None, None),
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
        canonical = build_canonical_experiment_name(config, "ucagnn", None)

        self.assertIn("lr-cosine", canonical)

    def test_query_results_reuses_runtime_canonical_name_contract(self) -> None:
        """Stored configs should render the same canonical label as live runs."""
        config = build_config(
            _experiment_args(
                preset="ucagnn",
                graph_policy="cagra_augmented",
                sample_interactions=500,
                loader_max_rows=1000,
            ),
        )
        stored_config = dataclasses.asdict(config)

        self.assertEqual(
            build_canonical_experiment_name(stored_config, "ucagnn", None),
            build_canonical_experiment_name(config, "ucagnn", None),
        )

    def test_query_results_scoremix_uses_current_config_field(self) -> None:
        """Result tables should not label learned UCaGNN score fusion as fixed."""
        learned = dataclasses.asdict(build_config(_experiment_args(preset="ucagnn")))
        fixed = learned | {"use_learned_score_mix": False}

        self.assertEqual(_format_scoremix(learned), "learned")
        self.assertEqual(_format_scoremix(fixed), "fixed")
        self.assertEqual(_format_scoremix({"scoring_weight_mode": "learned"}), "learned")

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

    def test_exception_summary_keeps_type_and_first_line(self) -> None:
        """OOM diagnostics should keep the useful exception identity."""
        summary = _exception_summary(RuntimeError("CUDA out of memory.\nsecond line"))

        self.assertEqual(summary, "RuntimeError: CUDA out of memory.")

    def test_cuda_memory_snapshot_reports_allocated_and_reserved_memory(self) -> None:
        """OOM diagnostics should reconcile allocator and resource-monitor views."""
        mib = 1024**2
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=1 * mib),
            patch("torch.cuda.memory_reserved", return_value=2 * mib),
            patch("torch.cuda.max_memory_allocated", return_value=3 * mib),
            patch("torch.cuda.max_memory_reserved", return_value=4 * mib),
        ):
            snapshot = _cuda_memory_snapshot()

        self.assertEqual(
            snapshot,
            ("cuda_memory=allocated=1MB reserved=2MB peak_allocated=3MB peak_reserved=4MB"),
        )

    def test_gpu_resource_snapshot_parses_utilization_and_memory_used(self) -> None:
        """Training telemetry should capture nvidia-smi utilization and memory used."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("subprocess.check_output", return_value="42, 1536\n"),
        ):
            snapshot = sample_gpu_resource_snapshot(torch.device("cuda"))

        assert snapshot is not None
        self.assertEqual(snapshot.utilization_pct, 42.0)
        self.assertEqual(snapshot.memory_used_mb, 1536.0)

    def test_log_epoch_to_sqlite_uses_training_resource_stats(self) -> None:
        """Epoch resource rows should come from training-window samples."""
        runtime = TrainerRuntime.__new__(TrainerRuntime)
        runtime.device = torch.device("cuda")
        runtime.experiment_logger = Mock()
        runtime.exp_id = 9
        runtime.mlflow_module = None
        runtime.model = Mock()
        resource_stats = TrainingResourceStats(
            pytorch_peak_allocated_mb=1024.0,
            pytorch_peak_reserved_mb=2048.0,
            nvidia_peak_memory_used_mb=3072.0,
            avg_gpu_utilization_pct=33.0,
            max_gpu_utilization_pct=66.0,
        )

        runtime._log_epoch_to_sqlite(
            epoch=3,
            avg_loss=0.25,
            epoch_time_s=12.0,
            val_metrics={"NDCG@40": 0.7},
            resource_stats=resource_stats,
        )

        metric_calls = runtime.experiment_logger.log_metric.call_args_list
        logged = {(call.args[1], call.args[2]) for call in metric_calls}
        self.assertIn(("train_peak_vram_allocated_mb", 1024.0), logged)
        self.assertIn(("train_peak_vram_reserved_mb", 2048.0), logged)
        self.assertIn(("train_peak_gpu_memory_used_mb", 3072.0), logged)
        self.assertIn(("gpu_utilization_pct", 33.0), logged)
        self.assertIn(("max_gpu_utilization_pct", 66.0), logged)
        self.assertIn(("peak_vram_mb", 3072.0), logged)

    def test_ucagnn_preset_applies_fused_scoring_defaults(self) -> None:
        """The ucagnn preset should target the fused-score contract."""
        config = build_config(_experiment_args(preset="ucagnn"))

        self.assertFalse(hasattr(config, "scoring_weight_mode"))
        self.assertFalse(hasattr(config, "train_scoring_mode"))
        self.assertFalse(hasattr(config, "eval_scoring_mode"))
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
        self.assertEqual(config.negative_sampling_strategy, "dice")
        self.assertEqual(config.n_negatives, 1)
        self.assertEqual(config.dice_branch_margin, config.dice_sampler_margin)
        self.assertFalse(config.dice_adaptive_decay)

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

    def test_lightgcn_paper_preset_uses_full_graph_training(self) -> None:
        """Paper LightGCN must not use sampled-neighborhood training."""
        config = build_config(_experiment_args(preset="lightgcn_paper"))

        self.assertFalse(config.use_dual_branch)
        self.assertFalse(config.use_features)
        self.assertFalse(config.use_learned_score_mix)
        self.assertEqual(config.training_graph_mode, "full")
        self.assertEqual(config.baseline_family, "lightgcn_paper")
        self.assertEqual(config.single_branch_gnn_layers, 3)
        self.assertEqual(config.dropout, 0.0)
        self.assertEqual(config.lr, 0.001)
        self.assertEqual(config.lr_scheduler, "none")
        self.assertEqual(config.weight_decay, 1e-4)
        self.assertEqual(config.batch_size, 2048)
        self.assertFalse(config.auto_batch_size)

    def test_lightgcn_paper_ignores_shared_dropout_override(self) -> None:
        """Paper LightGCN should keep architecture and optimizer paper defaults."""
        config = build_config(
            _experiment_args(
                preset="lightgcn_paper",
                dropout=0.25,
                graph_policy="cagra_augmented",
                lr=0.01,
                lr_scheduler="cosine",
                weight_decay=1e-2,
                batch_size=8192,
                auto_batch_size=True,
            ),
        )

        self.assertEqual(config.dropout, 0.0)
        self.assertEqual(config.graph_policy, "observed")
        self.assertEqual(config.lr, 0.001)
        self.assertEqual(config.lr_scheduler, "none")
        self.assertEqual(config.weight_decay, 1e-4)
        self.assertEqual(config.batch_size, 2048)
        self.assertFalse(config.auto_batch_size)

    def test_lightgcn_paper_propagation_has_no_self_loops(self) -> None:
        """Paper LightGCN should average ego and neighbor layers without self-loops."""
        config = UCaGNNConfig(device="cpu", embed_dim=1).preset_lightgcn_paper()
        config.single_branch_gnn_layers = 1
        model = PaperLightGCN(n_users=1, n_items=1, config=config)
        with torch.no_grad():
            model.user_embedding.weight.fill_(2.0)
            model.item_embedding.weight.fill_(10.0)

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_norm = torch.ones(edge_index.size(1), dtype=torch.float32)

        propagated = model.get_propagated_for_eval(edge_index, edge_norm=edge_norm)

        expected = torch.tensor([[6.0]])
        torch.testing.assert_close(propagated["user"], expected)
        torch.testing.assert_close(propagated["item"], expected)

    def test_dice_paper_preset_matches_dice_gcn_training_contract(self) -> None:
        """DICE paper baseline should expose the external GCN-DICE sampler/loss defaults."""
        config = build_config(_experiment_args(preset="dice_paper"))

        self.assertTrue(config.use_dual_branch)
        self.assertFalse(config.use_features)
        self.assertFalse(config.use_learned_score_mix)
        self.assertEqual(config.training_graph_mode, "full")
        self.assertEqual(config.baseline_family, "dice_paper")
        self.assertEqual(config.branch_loss_mode, "dice")
        self.assertEqual(config.negative_sampling_strategy, "dice")
        self.assertEqual(config.n_negatives, 4)
        self.assertEqual(config.single_branch_gnn_layers, 2)
        self.assertEqual(config.interest_gnn_layers, 2)
        self.assertEqual(config.conformity_gnn_layers, 2)
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.dice_branch_margin, config.dice_sampler_margin)
        self.assertEqual(config.graph_policy, "observed")
        self.assertEqual(config.lr, 0.001)
        self.assertEqual(config.lr_scheduler, "none")
        self.assertEqual(config.batch_size, 128)

    def test_dice_paper_ignores_shared_runtime_overrides(self) -> None:
        """Paper GCN-DICE should keep external-code optimizer and sampler defaults."""
        config = build_config(
            _experiment_args(
                preset="dice_paper",
                dropout=0.05,
                graph_policy="cagra_augmented",
                lr=0.01,
                lr_scheduler="cosine",
                batch_size=8192,
                auto_batch_size=True,
            ),
        )

        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.graph_policy, "observed")
        self.assertEqual(config.lr, 0.001)
        self.assertEqual(config.lr_scheduler, "none")
        self.assertEqual(config.weight_decay, 5e-8)
        self.assertEqual(config.batch_size, 128)
        self.assertFalse(config.auto_batch_size)
        self.assertEqual(config.negative_sampling_strategy, "dice")
        self.assertEqual(config.n_negatives, 4)

    def test_build_runtime_model_uses_explicit_paper_baseline_classes(self) -> None:
        """Paper baselines should not be hidden as UCaGNN config variants."""
        canonical = _tiny_canonical()

        lightgcn_config = build_config(_experiment_args(preset="lightgcn_paper", device="cpu"))
        lightgcn_graph = build_graph(canonical, lightgcn_config)
        lightgcn_model = build_runtime_model(lightgcn_config, canonical, lightgcn_graph)

        dice_config = build_config(_experiment_args(preset="dice_paper", device="cpu"))
        dice_graph = build_graph(canonical, dice_config)
        dice_model = build_runtime_model(dice_config, canonical, dice_graph)

        self.assertIsInstance(lightgcn_model, PaperLightGCN)
        self.assertIsInstance(dice_model, PaperGCNDICE)

        dice_propagated = dice_model.get_propagated_for_eval(
            dice_graph.edge_index,
            edge_norm=dice_graph.edge_norm,
        )
        dice_scores = dice_model.get_score_components_from_propagated(
            dice_propagated,
            torch.tensor([0], dtype=torch.long),
        )
        self.assertTrue(
            torch.equal(
                dice_scores["score_mix_weights"],
                torch.tensor([[1.0, 1.0, 0.0]]),
            ),
        )
        self.assertTrue(
            torch.allclose(
                dice_scores["final_score"],
                dice_scores["interest_score"] + dice_scores["conformity_score"],
            ),
        )

    def test_paper_baselines_use_paper_optimizer_families(self) -> None:
        """Paper adapters should not inherit U-CaGNN's AdamW optimizer."""
        canonical = _tiny_canonical()

        lightgcn_config = build_config(_experiment_args(preset="lightgcn_paper", device="cpu"))
        lightgcn_graph = build_graph(canonical, lightgcn_config)
        lightgcn_trainer = MiniBatchTrainer(
            model=build_runtime_model(lightgcn_config, canonical, lightgcn_graph),
            loss_suite=LossSuite(lightgcn_config),
            data=lightgcn_graph,
            config=lightgcn_config,
        )

        dice_config = build_config(_experiment_args(preset="dice_paper", device="cpu"))
        dice_graph = build_graph(canonical, dice_config)
        dice_trainer = MiniBatchTrainer(
            model=build_runtime_model(dice_config, canonical, dice_graph),
            loss_suite=LossSuite(dice_config),
            data=dice_graph,
            config=dice_config,
        )

        ucagnn_config = build_config(_experiment_args(preset="ucagnn", device="cpu"))
        ucagnn_graph = build_graph(canonical, ucagnn_config)
        ucagnn_trainer = MiniBatchTrainer(
            model=build_runtime_model(ucagnn_config, canonical, ucagnn_graph),
            loss_suite=LossSuite(ucagnn_config),
            data=ucagnn_graph,
            config=ucagnn_config,
        )

        self.assertIsInstance(lightgcn_trainer.optimizer, torch.optim.Adam)
        self.assertNotIsInstance(lightgcn_trainer.optimizer, torch.optim.AdamW)
        self.assertEqual(lightgcn_trainer.optimizer.param_groups[0]["weight_decay"], 0.0)

        self.assertIsInstance(dice_trainer.optimizer, torch.optim.Adam)
        self.assertNotIsInstance(dice_trainer.optimizer, torch.optim.AdamW)
        self.assertEqual(dice_trainer.optimizer.defaults["betas"], (0.5, 0.99))
        self.assertTrue(dice_trainer.optimizer.defaults["amsgrad"])
        self.assertEqual(dice_trainer.optimizer.param_groups[0]["weight_decay"], 5e-8)

        self.assertIsInstance(ucagnn_trainer.optimizer, torch.optim.AdamW)

    def test_lightgcn_paper_loss_includes_embedding_l2_regularization(self) -> None:
        """Paper LightGCN should use explicit ego-embedding L2 regularization."""
        config = UCaGNNConfig(device="cpu", embed_dim=2).preset_lightgcn_paper()
        config.weight_decay = 0.1
        loss_suite = LossSuite(config)
        pos_scores = {
            "final_score": torch.tensor([2.0]),
            "interest_score": torch.tensor([2.0]),
            "conformity_score": torch.zeros(1),
            "context_score": torch.zeros(1),
        }
        neg_scores = {
            "final_score": torch.tensor([0.0]),
            "interest_score": torch.tensor([0.0]),
            "conformity_score": torch.zeros(1),
            "context_score": torch.zeros(1),
        }
        model_output = {
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "embeddings": {
                "user": torch.tensor([[1.0, 2.0]]),
                "item": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
            },
            "propagated": {},
            "ipw_weights": torch.ones(1),
            "loss_user_ids": torch.tensor([0], dtype=torch.long),
            "loss_neg_item_ids": torch.tensor([1], dtype=torch.long),
        }

        losses = loss_suite(
            model_output,
            item_popularity=torch.ones(2),
            pos_item_ids=torch.tensor([0], dtype=torch.long),
            epoch=0,
        )

        expected_reg = 0.5 * (5.0 + 25.0 + 61.0)
        self.assertAlmostEqual(losses["embedding_reg"].item(), expected_reg, places=6)
        self.assertAlmostEqual(
            losses["total"].item(),
            losses["rec"].item() + 0.1 * expected_reg,
            places=6,
        )

    def test_dice_paper_loss_matches_external_branch_objective(self) -> None:
        """Paper DICE should train on summed branches and masked DICE auxiliaries."""
        config = UCaGNNConfig(device="cpu", embed_dim=2).preset_dice_paper()
        loss_suite = LossSuite(config)
        pos_interest = torch.tensor([1.8, 0.2])
        neg_interest = torch.tensor([0.1, 1.0])
        pos_conformity = torch.tensor([0.3, 1.5])
        neg_conformity = torch.tensor([1.2, 0.4])
        user_interest = torch.tensor([[1.0, 0.1], [0.2, 1.1]])
        user_conformity = torch.tensor([[0.4, 1.2], [1.3, 0.3]])
        item_interest = torch.tensor([[1.5, 0.2], [0.4, 1.0], [1.0, 1.0]])
        item_conformity = torch.tensor([[0.1, 1.4], [1.2, 0.5], [0.7, 1.1]])
        model_output = {
            "pos_scores": {
                "final_score": torch.tensor([-100.0, -100.0]),
                "interest_score": pos_interest,
                "conformity_score": pos_conformity,
                "context_score": torch.zeros(2),
            },
            "neg_scores": {
                "final_score": torch.tensor([100.0, 100.0]),
                "interest_score": neg_interest,
                "conformity_score": neg_conformity,
                "context_score": torch.zeros(2),
            },
            "propagated": {
                "user_interest": user_interest,
                "user_conformity": user_conformity,
                "item_interest": item_interest,
                "item_conformity": item_conformity,
            },
            "ipw_weights": torch.ones(2),
            "loss_user_ids": torch.tensor([0, 1], dtype=torch.long),
            "loss_neg_item_ids": torch.tensor([2, 0], dtype=torch.long),
            "dice_negative_mask": torch.tensor([False, True]),
        }

        losses = loss_suite(
            model_output,
            item_popularity=torch.tensor([0.1, 0.6, 1.0]),
            branch_item_popularity=torch.tensor([10.0, 60.0, 100.0]),
            pos_item_ids=torch.tensor([0, 1], dtype=torch.long),
            epoch=0,
        )

        threshold_mask = torch.tensor([True, False])
        mask = torch.tensor([False, True])
        self.assertFalse(torch.equal(mask, threshold_mask))
        expected_rec = -functional.logsigmoid(
            (pos_interest + pos_conformity) - (neg_interest + neg_conformity),
        ).mean()
        expected_interest = -(
            mask.float() * functional.logsigmoid(pos_interest - neg_interest)
        ).mean()
        expected_conformity = (
            -(mask.float() * functional.logsigmoid(neg_conformity - pos_conformity)).mean()
            - ((~mask).float() * functional.logsigmoid(pos_conformity - neg_conformity)).mean()
        )
        expected_independence = _positive_dcor(user_interest, user_conformity) + _positive_dcor(
            item_interest,
            item_conformity,
        )
        expected_total = (
            expected_rec
            + 0.1 * expected_interest
            + 0.1 * expected_conformity
            + 0.01 * expected_independence
        )

        self.assertAlmostEqual(losses["rec"].item(), expected_rec.item(), places=6)
        self.assertAlmostEqual(
            losses["interest_bpr"].item(),
            expected_interest.item(),
            places=6,
        )
        self.assertAlmostEqual(
            losses["conformity_bpr"].item(),
            expected_conformity.item(),
            places=6,
        )
        self.assertAlmostEqual(
            losses["independence"].item(),
            expected_independence.item(),
            places=6,
        )
        self.assertAlmostEqual(losses["total"].item(), expected_total.item(), places=6)

    def test_dice_paper_discrepancy_uses_unique_users_like_external_code(self) -> None:
        """DICE discrepancy should not overweight users repeated by multiple negatives."""
        config = UCaGNNConfig(device="cpu", embed_dim=2).preset_dice_paper()
        config.loss_weight_recommendation = 0.0
        config.loss_weight_interest_bpr = 0.0
        config.loss_weight_conformity_bpr = 0.0
        config.loss_weight_independence = 1.0
        loss_suite = LossSuite(config)
        loss_user_ids = torch.tensor([0, 0, 0, 1, 2], dtype=torch.long)
        pos_item_ids = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
        neg_item_ids = torch.tensor([1, 2, 2, 0, 0], dtype=torch.long)
        user_interest = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        user_conformity = torch.tensor([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0]])
        item_interest = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
        item_conformity = torch.tensor([[0.0, 1.0], [2.0, 2.0], [1.0, 4.0]])
        score_zeros = torch.zeros(loss_user_ids.numel())
        model_output = {
            "pos_scores": {
                "final_score": score_zeros,
                "interest_score": score_zeros,
                "conformity_score": score_zeros,
                "context_score": score_zeros,
            },
            "neg_scores": {
                "final_score": score_zeros,
                "interest_score": score_zeros,
                "conformity_score": score_zeros,
                "context_score": score_zeros,
            },
            "propagated": {
                "user_interest": user_interest,
                "user_conformity": user_conformity,
                "item_interest": item_interest,
                "item_conformity": item_conformity,
            },
            "ipw_weights": torch.ones(loss_user_ids.numel()),
            "loss_user_ids": loss_user_ids,
            "loss_neg_item_ids": neg_item_ids,
            "dice_negative_mask": torch.zeros(loss_user_ids.numel(), dtype=torch.bool),
        }

        losses = loss_suite(
            model_output,
            item_popularity=torch.ones(3),
            branch_item_popularity=torch.ones(3),
            pos_item_ids=pos_item_ids,
            epoch=0,
        )

        unique_user_ids = torch.unique(loss_user_ids)
        unique_item_ids = torch.unique(torch.cat([pos_item_ids, neg_item_ids]))
        expected = _positive_dcor(
            user_interest[unique_user_ids],
            user_conformity[unique_user_ids],
        ) + _positive_dcor(item_interest[unique_item_ids], item_conformity[unique_item_ids])
        duplicated_user_value = _positive_dcor(
            user_interest[loss_user_ids],
            user_conformity[loss_user_ids],
        ) + _positive_dcor(item_interest[unique_item_ids], item_conformity[unique_item_ids])

        self.assertGreater(abs(expected.item() - duplicated_user_value.item()), 1e-4)
        self.assertAlmostEqual(losses["independence"].item(), expected.item(), places=6)
        self.assertAlmostEqual(losses["total"].item(), expected.item(), places=6)

    def test_ucagnn_preset_keeps_causal_branches_active(self) -> None:
        """Main U-CaGNN should train causal branches instead of allowing collapse."""
        config = build_config(_experiment_args(preset="ucagnn"))

        self.assertEqual(config.baseline_family, "ucagnn")
        self.assertEqual(config.branch_loss_mode, "dice")
        self.assertGreater(config.score_mix_min_weight, 0.0)
        self.assertGreater(config.loss_weight_interest_bpr, 0.0)
        self.assertGreater(config.loss_weight_conformity_bpr, 0.0)
        self.assertEqual(config.negative_sampling_strategy, "dice")
        self.assertEqual(config.dice_branch_margin, config.dice_sampler_margin)

    def test_ucagnn_linear_ramp_keeps_branch_bpr_active_at_epoch_zero(self) -> None:
        """DICE branch BPR is primary causal supervision, not ramped auxiliary loss."""
        config = UCaGNNConfig(device="cpu", embed_dim=2).preset_full()
        config.loss_weight_recommendation = 0.0
        config.loss_weight_independence = 0.0
        config.loss_weight_contrastive = 0.0
        config.loss_weight_align = 0.0
        config.loss_weight_uniform = 0.0
        config.loss_weight_popularity = 0.0
        config.loss_weight_propensity_calibration = 0.0
        loss_suite = LossSuite(config)
        pos_interest = torch.tensor([2.0, 1.0])
        neg_interest = torch.tensor([0.0, 0.0])
        pos_conformity = torch.tensor([0.0, 2.0])
        neg_conformity = torch.tensor([2.0, 0.0])
        model_output = {
            "pos_scores": {
                "final_score": torch.zeros(2),
                "interest_score": pos_interest,
                "conformity_score": pos_conformity,
                "context_score": torch.zeros(2),
            },
            "neg_scores": {
                "final_score": torch.zeros(2),
                "interest_score": neg_interest,
                "conformity_score": neg_conformity,
                "context_score": torch.zeros(2),
            },
            "propagated": {},
            "ipw_weights": torch.ones(2),
            "loss_user_ids": torch.tensor([0, 1], dtype=torch.long),
            "loss_neg_item_ids": torch.tensor([2, 0], dtype=torch.long),
            "dice_negative_mask": torch.tensor([True, False]),
        }

        losses = loss_suite(
            model_output,
            item_popularity=torch.tensor([0.1, 0.6, 1.0]),
            branch_item_popularity=torch.tensor([10.0, 60.0, 100.0]),
            pos_item_ids=torch.tensor([0, 1], dtype=torch.long),
            epoch=0,
        )

        mask = torch.tensor([True, False])
        expected_interest = -(
            mask.float() * functional.logsigmoid(pos_interest - neg_interest)
        ).mean()
        expected_conformity = (
            -(mask.float() * functional.logsigmoid(neg_conformity - pos_conformity)).mean()
            - ((~mask).float() * functional.logsigmoid(pos_conformity - neg_conformity)).mean()
        )
        expected_total = (
            config.loss_weight_interest_bpr * expected_interest
            + config.loss_weight_conformity_bpr * expected_conformity
        )

        self.assertEqual(config.auxiliary_loss_schedule, "linear_ramp")
        self.assertGreater(config.loss_weight_interest_bpr, 0.0)
        self.assertGreater(config.loss_weight_conformity_bpr, 0.0)
        self.assertGreater(losses["total"].item(), 0.0)
        self.assertAlmostEqual(losses["interest_bpr"].item(), expected_interest.item(), places=6)
        self.assertAlmostEqual(
            losses["conformity_bpr"].item(),
            expected_conformity.item(),
            places=6,
        )
        self.assertAlmostEqual(losses["total"].item(), expected_total.item(), places=6)

    def test_ucagnn_sampler_margin_does_not_decay_without_adaptive_dice(self) -> None:
        """U-CaGNN should keep a stable DICE margin unless adaptive decay is explicit."""
        canonical = _tiny_canonical()
        config = UCaGNNConfig(device="cpu").preset_full()
        graph = build_graph(canonical, config)
        model = build_runtime_model(config, canonical, graph)
        trainer = MiniBatchTrainer(
            model=model,
            loss_suite=LossSuite(config),
            data=graph,
            config=config,
        )

        self.assertEqual(config.negative_sampling_strategy, "dice")
        self.assertFalse(config.dice_adaptive_decay)
        self.assertEqual(trainer.sampler.dice_margin_decay, 1.0)

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
        self.assertFalse(hasattr(config, "scoring_weight_mode"))
        self.assertFalse(hasattr(config, "train_scoring_mode"))
        self.assertFalse(hasattr(config, "eval_scoring_mode"))
        self.assertEqual(config.interest_gnn_layers, 2)
        self.assertEqual(config.conformity_gnn_layers, 2)
        self.assertEqual(config.max_gnn_layers, 2)

    def test_training_identity_ignores_eval_only_overrides(self) -> None:
        """Resume compatibility should ignore non-training config changes."""
        base = build_config(_experiment_args(preset="ucagnn"))
        eval_override = build_config(_experiment_args(preset="ucagnn"))
        eval_override.eval_ks = [5, 10]

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

    def test_training_identity_changes_when_score_mix_behavior_changes(self) -> None:
        """Resume compatibility should change when score mixing switches modes."""
        learned_mix = build_config(_experiment_args(preset="ucagnn"))
        fixed_mix = dataclasses.replace(learned_mix, use_learned_score_mix=False)

        learned_identity, learned_hash = _build_training_identity(
            learned_mix,
            "ucagnn",
            None,
        )
        fixed_identity, fixed_hash = _build_training_identity(fixed_mix, "ucagnn", None)

        self.assertTrue(learned_mix.use_learned_score_mix)
        self.assertFalse(fixed_mix.use_learned_score_mix)
        self.assertNotEqual(learned_hash, fixed_hash)
        self.assertNotEqual(learned_identity["config"], fixed_identity["config"])

    def test_training_identity_changes_when_distance_correlation_cap_changes(self) -> None:
        """Resume compatibility should track DICE discrepancy estimator changes."""
        base = build_config(_experiment_args(preset="ucagnn"))
        changed = dataclasses.replace(
            base,
            distance_correlation_max_pairs=base.distance_correlation_max_pairs * 2,
        )

        _, base_hash = _build_training_identity(base, "ucagnn", None)
        _, changed_hash = _build_training_identity(changed, "ucagnn", None)

        self.assertNotEqual(base_hash, changed_hash)

    def test_training_identity_changes_when_uniformity_cap_changes(self) -> None:
        """Resume compatibility should track DirectAU uniformity estimator changes."""
        base = build_config(_experiment_args(preset="ucagnn"))
        changed = dataclasses.replace(
            base,
            uniformity_max_pairs=base.uniformity_max_pairs * 2,
        )

        _, base_hash = _build_training_identity(base, "ucagnn", None)
        _, changed_hash = _build_training_identity(changed, "ucagnn", None)

        self.assertNotEqual(base_hash, changed_hash)

    def test_default_formal_profile_is_core_ucagnn_mainline(self) -> None:
        """The default formal profile should target the thesis mainline."""
        profile_name = default_formal_profile_name()
        profile = get_formal_profile(profile_name)
        benchmark_args = formal_main._build_new_run_args(
            SimpleNamespace(overwrite_checkpoint=False),
            profile_name,
        )

        self.assertTrue(profile["config_overrides"]["use_early_stopping"])
        self.assertEqual(profile["config_overrides"]["patience"], 10)
        self.assertEqual(profile["id"], "core-ucagnn-mainline")
        self.assertEqual(
            profile["matrix"]["datasets"],
            ["amazonbook", "movielens1m", "kuairec_v2", "kuairand1k"],
        )
        self.assertNotIn("taobao", profile["matrix"]["datasets"])
        self.assertNotIn("movielens20m", profile["matrix"]["datasets"])
        self.assertEqual(profile["matrix"]["presets"], ["ucagnn"])
        self.assertNotIn("scoring_weight_modes", profile["matrix"])
        self.assertNotIn("batch_size", profile["config_overrides"])
        self.assertNotIn("auto_batch_size", profile["config_overrides"])
        self.assertEqual(
            profile["config_overrides"]["batch_size_candidates"],
            [32768, 16384, 8192, 4096, 2048, 1024, 512, 256],
        )
        self.assertEqual(profile["config_overrides"]["single_branch_gnn_layers"], 2)
        self.assertEqual(profile["config_overrides"]["interest_gnn_layers"], 1)
        self.assertEqual(profile["config_overrides"]["conformity_gnn_layers"], 2)
        self.assertEqual(profile["config_overrides"]["dropout"], 0.1)
        self.assertEqual(
            profile["config_overrides"]["num_neighbors"],
            {
                "small": [[6, 3], [4, 2]],
                "medium": [[10, 5], [16, 8]],
            },
        )
        self.assertNotIn("hard_negative_ratio", profile["config_overrides"])
        self.assertNotIn("loss_schedule", profile["config_overrides"])
        self.assertTrue(benchmark_args["use_early_stopping"])
        self.assertEqual(benchmark_args["patience"], 10)
        self.assertEqual(
            benchmark_args["datasets"],
            ["amazonbook", "movielens1m", "kuairec_v2", "kuairand1k"],
        )
        self.assertEqual(benchmark_args["presets"], ["ucagnn"])
        self.assertNotIn("scoring_weight_modes", benchmark_args)
        self.assertEqual(benchmark_args["batch_size"], 4096)
        self.assertTrue(benchmark_args["auto_batch_size"])
        self.assertEqual(
            benchmark_args["batch_size_candidates"],
            [32768, 16384, 8192, 4096, 2048, 1024, 512, 256],
        )
        self.assertEqual(benchmark_args["single_branch_gnn_layers"], 2)
        self.assertEqual(benchmark_args["interest_gnn_layers"], 1)
        self.assertEqual(benchmark_args["conformity_gnn_layers"], 2)
        self.assertEqual(benchmark_args["dropout"], 0.1)
        self.assertEqual(
            benchmark_args["num_neighbors"],
            {
                "small": [[6, 3], [4, 2]],
                "medium": [[10, 5], [16, 8]],
            },
        )
        self.assertEqual(benchmark_args["graph_policy"], "observed")
        self.assertIsNone(benchmark_args["graph_policy_options"])
        self.assertEqual(benchmark_args["hard_negative_ratio"], 0.0)
        self.assertIsNone(benchmark_args["loss_schedule"])
        self.assertEqual(benchmark_args["auxiliary_losses_start_epoch"], 15)
        self.assertEqual(benchmark_args["popularity_supervision_start_epoch"], 30)

    def test_deeper_comparison_profile_keeps_original_num_neighbors_sweep(self) -> None:
        """The deeper comparison profile should stay on its original fan-out sweep."""
        profile = get_formal_profile("core-deeper-comparison-i2-c3")
        benchmark_args = formal_main._build_new_run_args(
            SimpleNamespace(overwrite_checkpoint=False),
            "core-deeper-comparison-i2-c3",
        )

        expected_neighbors = [[10, 5, 3], [5, 3, 2]]
        self.assertEqual(profile["config_overrides"]["num_neighbors"], expected_neighbors)
        self.assertEqual(benchmark_args["num_neighbors"], expected_neighbors)

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
        self.assertTrue(config.use_early_stopping)
        self.assertEqual(config.patience, 10)
        self.assertEqual(config.graph_policy, "cagra_augmented")
        self.assertEqual(config.num_neighbors, [10, 5])

    def test_benchmark_config_inputs_preserve_auxiliary_loss_overrides(self) -> None:
        """Formal profiles should be able to run causal auxiliary-loss ablations."""
        benchmark_args = normalize_benchmark_config_overrides(
            {
                "loss_weight_contrastive": 0.03,
                "loss_weight_propensity_calibration": 0.04,
                "use_ipw": True,
                "auxiliary_loss_schedule": "linear_ramp",
                "auxiliary_ramp_rate": 0.002,
                "contrastive_max_pairs": 64,
                "contrastive_temperature": 0.15,
            },
        )
        config_inputs = build_benchmark_config_inputs(
            benchmark_args,
            dataset="kuairand1k",
            preset="ucagnn",
            lr_scheduler="cosine",
            num_neighbors=[10, 5],
        )

        config = build_config(config_inputs)

        self.assertEqual(config.loss_weight_contrastive, 0.03)
        self.assertEqual(config.loss_weight_propensity_calibration, 0.04)
        self.assertTrue(config.use_ipw)
        self.assertEqual(config.auxiliary_loss_schedule, "linear_ramp")
        self.assertEqual(config.auxiliary_ramp_rate, 0.002)
        self.assertEqual(config.contrastive_max_pairs, 64)
        self.assertEqual(config.contrastive_temperature, 0.15)

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
        self.assertFalse(hasattr(config, "eval_scoring_mode"))
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
        self.assertEqual(default_profile["id"], "core-ucagnn-mainline")
        self.assertIn("abauto", default_profile["name"])
        self.assertEqual(get_formal_profile("latest")["id"], "core-ucagnn-mainline")
        self.assertEqual(get_formal_profile("development")["id"], "dev-ucagnn")
        self.assertEqual(get_formal_profile("dev-ucagnn")["id"], "dev-ucagnn")

    def test_second_formal_profile_is_ucagnn_mainline_only(self) -> None:
        """The main thesis profile should not rerun fixed paper baselines."""
        profile_name = formal_profile_names()[1]
        profile = get_formal_profile(profile_name)
        benchmark_args = formal_main._build_new_run_args(
            SimpleNamespace(overwrite_checkpoint=False),
            profile_name,
        )

        self.assertEqual(
            profile["matrix"]["presets"],
            ["ucagnn"],
        )
        self.assertEqual(
            profile["matrix"]["datasets"],
            ["amazonbook", "movielens1m", "kuairec_v2", "kuairand1k"],
        )
        self.assertEqual(
            profile["id"],
            "core-ucagnn-mainline",
        )
        self.assertEqual(
            get_formal_profile("core-paper-architecture-comparison")["id"],
            "core-ucagnn-mainline",
        )
        self.assertEqual(
            get_formal_profile("paper-lightgcn-small-baselines")["matrix"]["presets"],
            ["lightgcn_paper"],
        )
        self.assertEqual(
            get_formal_profile("paper-dice-all-runtime-probes")["matrix"]["presets"],
            ["dice_paper"],
        )
        self.assertNotIn("scoring_weight_modes", profile["matrix"])
        self.assertEqual(profile["config_overrides"]["single_branch_gnn_layers"], 2)
        self.assertNotIn("batch_size", profile["config_overrides"])
        self.assertNotIn("auto_batch_size", profile["config_overrides"])
        self.assertEqual(profile["config_overrides"]["interest_gnn_layers"], 1)
        self.assertEqual(profile["config_overrides"]["conformity_gnn_layers"], 2)
        self.assertEqual(
            profile["config_overrides"]["num_neighbors"],
            {
                "small": [[6, 3], [4, 2]],
                "medium": [[10, 5], [16, 8]],
            },
        )
        self.assertNotIn("scoring_weight_modes", benchmark_args)
        self.assertEqual(
            benchmark_args["datasets"],
            ["amazonbook", "movielens1m", "kuairec_v2", "kuairand1k"],
        )
        self.assertTrue(benchmark_args["auto_batch_size"])
        self.assertEqual(benchmark_args["single_branch_gnn_layers"], 2)
        self.assertEqual(benchmark_args["interest_gnn_layers"], 1)
        self.assertEqual(benchmark_args["conformity_gnn_layers"], 2)
        self.assertEqual(
            benchmark_args["num_neighbors"],
            {
                "small": [[6, 3], [4, 2]],
                "medium": [[10, 5], [16, 8]],
            },
        )
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

    def test_requested_profile_ignores_removed_saved_profile_state(self) -> None:
        """Explicit profiles should start fresh when old saved state references a removed id."""
        requested_profile_name = "paper-dice-all-runtime-probes"
        cli_args = SimpleNamespace(profile=requested_profile_name, overwrite_checkpoint=True)

        with (
            patch.object(
                formal_main,
                "_load_saved_formal_state",
                side_effect=ValueError(
                    "The saved formal-run state references a profile that is no longer defined.",
                ),
            ),
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

    def test_resolve_benchmark_args_requires_profile_to_resume_non_default_state(
        self,
    ) -> None:
        """Unspecified profiles should not silently resume non-default saved runs."""
        cli_args = SimpleNamespace(profile=None, overwrite_checkpoint=False)
        saved_state = {
            "profile_name": "paper-dice-all-runtime-probes",
            "profile_slug": "runtime-probe",
            "benchmark_args": {"datasets": ["kuairec_v2"], "presets": ["dice_paper"]},
        }

        with (
            patch.object(formal_main, "_load_saved_formal_state", return_value=saved_state),
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
        self.assertEqual(normalized["use_early_stopping"], UCaGNNConfig().use_early_stopping)
        self.assertEqual(normalized["patience"], UCaGNNConfig().patience)
        self.assertIsNone(normalized["graph_policy"])
        self.assertIsNone(normalized["graph_policy_options"])
        self.assertEqual(normalized["hard_negative_ratio"], 0.25)

    def test_normalize_benchmark_config_overrides_supports_dataset_keyed_neighbor_sweeps(self) -> None:
        """Benchmark payload normalization should preserve dataset-keyed neighbor sweeps."""
        normalized = normalize_benchmark_config_overrides(
            {
                "num_neighbors": {
                    "small": [[6, 3], [4, 2]],
                    "medium": [[10, 5], [16, 8]],
                },
            },
        )

        self.assertEqual(
            normalized["num_neighbors"],
            {
                "small": [[6, 3], [4, 2]],
                "medium": [[10, 5], [16, 8]],
            },
        )

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

    def test_step_scheduler_uses_metric_only_for_plateau(self) -> None:
        """ReduceLROnPlateau should consume the validation metric; others should not."""
        runtime = TrainerRuntime.__new__(TrainerRuntime)
        runtime.config = SimpleNamespace(
            auxiliary_losses_start_epoch=0,
            popularity_supervision_start_epoch=0,
        )

        plateau_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1)
        runtime.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(plateau_optimizer)
        runtime._step_scheduler(metric_value=0.5, epoch=0)
        self.assertEqual(runtime.scheduler.best, 0.5)

        step_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1)
        runtime.scheduler = torch.optim.lr_scheduler.StepLR(step_optimizer, step_size=1)
        step_optimizer.step()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            runtime._step_scheduler(metric_value=0.5, epoch=0)

        self.assertEqual(runtime.scheduler.last_epoch, 1)

    def test_validation_eval_skips_refined_diagnostics(self) -> None:
        """Validation should keep only the cheap thesis-primary metrics."""
        runtime = TrainerRuntime.__new__(TrainerRuntime)
        runtime.device = torch.device("cpu")
        runtime.config = SimpleNamespace(use_amp=False)
        runtime.use_amp = False
        runtime.amp_dtype = torch.float16
        runtime.ema_model = None
        runtime.model = object()
        runtime.data = SimpleNamespace(val_mask=torch.tensor([True, False]))
        runtime.evaluator = Mock()
        runtime.evaluator.evaluate.return_value = {"NDCG@40": 0.5}

        metrics = runtime._evaluate_validation_metrics()

        self.assertEqual(metrics, {"NDCG@40": 0.5})
        runtime.evaluator.evaluate.assert_called_once()
        self.assertFalse(
            runtime.evaluator.evaluate.call_args.kwargs["include_refined_diagnostics"],
        )

    def test_split_eval_defaults_to_primary_metrics_only(self) -> None:
        """Refined diagnostics should be opt-in so only final test evaluation pays for them."""
        runtime = TrainerRuntime.__new__(TrainerRuntime)
        runtime.device = torch.device("cpu")
        runtime.config = SimpleNamespace(use_amp=False)
        runtime.use_amp = False
        runtime.amp_dtype = torch.float16
        runtime.ema_model = None
        runtime.model = object()
        runtime.data = object()
        runtime.evaluator = Mock()
        runtime.evaluator.evaluate.return_value = {"NDCG@40": 0.5}

        metrics = runtime._evaluate_split_metrics("test-mask")

        self.assertEqual(metrics, {"NDCG@40": 0.5})
        self.assertFalse(
            runtime.evaluator.evaluate.call_args.kwargs["include_refined_diagnostics"],
        )

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

    def test_auto_batch_fallback_resumes_from_last_completed_checkpoint(self) -> None:
        """Late auto-batch OOM fallback should preserve completed epoch progress."""
        checkpoint_path = Path("/tmp/auto-batch-recovery.pt")
        checkpoint_path.touch()
        self.addCleanup(checkpoint_path.unlink, missing_ok=True)
        history = {
            "train_loss": [1.0, 0.5],
            "val_metrics": [{"NDCG@40": 0.1}, {"NDCG@40": 0.2}],
        }
        trainer = Mock(completed_epoch=1, resume_history=history)

        start_epoch, resumed_history = _resume_auto_batch_fallback(
            trainer,
            checkpoint_path,
        )

        trainer.load_checkpoint.assert_called_once_with(checkpoint_path)
        self.assertEqual(start_epoch, 2)
        self.assertIs(resumed_history, history)

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
                epoch=3,
            )

        self.assertIs(actual_batch, expected_batch)
        self.assertTrue(trainer._force_cpu_sampler)
        self.assertEqual(trainer.sampler_device, torch.device("cpu"))
        trainer._build_subgraph_sampler.assert_called_once_with(trainer.data)
        self.assertEqual(trainer._prepare_batch_on_sampler_device.call_count, 2)
        self.assertEqual(
            trainer._prepare_batch_on_sampler_device.call_args_list[-1].args[-1],
            3,
        )
        empty_cache.assert_called()

    def test_prepare_batch_falls_back_after_runtime_cuda_oom(self) -> None:
        """CUDA OOM RuntimeError messages should use the same CPU sampler fallback."""
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
            side_effect=[
                RuntimeError("CUDA out of memory. Tried to allocate 4.49 GiB."),
                expected_batch,
            ],
        )

        with patch("torch.cuda.empty_cache"):
            actual_batch = trainer._prepare_batch(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                random_seed=13,
                epoch=3,
            )

        self.assertIs(actual_batch, expected_batch)
        self.assertTrue(trainer._force_cpu_sampler)
        self.assertEqual(trainer.sampler_device, torch.device("cpu"))

    def test_sampled_batch_preparation_passes_epoch_to_negative_sampler(self) -> None:
        """Sampled U-CaGNN should apply epoch-aware DICE sampling like full-graph DICE."""

        class RecordingSampler:
            def __init__(self) -> None:
                self.epoch: int | None = None

            def sample_with_metadata(
                self,
                batch_size: int,
                positive_items: torch.Tensor,
                *,
                user_ids: torch.Tensor,
                device: torch.device,
                generator: torch.Generator,
                epoch: int,
            ) -> tuple[torch.Tensor, None]:
                del positive_items, user_ids, device, generator
                self.epoch = epoch
                return torch.ones((batch_size, 1), dtype=torch.long), None

        class RecordingSubgraphSampler:
            def __init__(self) -> None:
                self.neg_items: torch.Tensor | None = None

            def sample(
                self,
                batch_users: torch.Tensor,
                batch_pos_items: torch.Tensor,
                batch_neg_items: torch.Tensor,
                *,
                generator: torch.Generator,
                dice_negative_mask: torch.Tensor | None = None,
            ) -> object:
                del batch_users, batch_pos_items, generator, dice_negative_mask
                self.neg_items = batch_neg_items
                return object()

        trainer = MiniBatchTrainer.__new__(MiniBatchTrainer)
        trainer.config = UCaGNNConfig(device="cpu", n_negatives=1)
        trainer.sampler_device = torch.device("cpu")
        trainer.device = torch.device("cpu")
        trainer.sampler = RecordingSampler()
        trainer.subgraph_sampler = RecordingSubgraphSampler()

        actual_batch = trainer._prepare_batch_on_sampler_device(
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([2, 3], dtype=torch.long),
            random_seed=13,
            epoch=7,
        )

        self.assertIsNotNone(actual_batch)
        self.assertEqual(trainer.sampler.epoch, 7)
        self.assertTrue(torch.equal(trainer.subgraph_sampler.neg_items, torch.ones(2).long()))

    def test_full_graph_training_tensors_are_cached_per_device(self) -> None:
        """Full-graph paper baselines should not restage graph tensors every batch."""
        trainer = MiniBatchTrainer.__new__(MiniBatchTrainer)
        trainer.data = object()
        trainer.device = torch.device("cpu")
        trainer._full_graph_tensors = None
        trainer._full_graph_tensor_device = None
        staged = (
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            torch.ones(2),
            torch.ones(2),
        )

        with patch(
            "src.training.mini_batch_trainer.stage_graph_tensors_for_device",
            return_value=staged,
        ) as stage:
            first = trainer._get_full_graph_training_tensors()
            second = trainer._get_full_graph_training_tensors()

        self.assertIs(first, staged)
        self.assertIs(second, staged)
        stage.assert_called_once_with(trainer.data, torch.device("cpu"))

    def test_full_graph_cuda_cache_is_released_before_eval(self) -> None:
        """Full-graph CUDA training cache should not duplicate evaluator graph memory."""
        trainer = MiniBatchTrainer.__new__(MiniBatchTrainer)
        trainer.device = torch.device("cuda")
        trainer._full_graph_tensor_device = torch.device("cuda")
        trainer._full_graph_tensors = (
            torch.tensor([[0], [1]], dtype=torch.long),
            None,
            None,
        )

        with patch("src.training.mini_batch_trainer.empty_cuda_cache") as empty_cache:
            released = trainer._release_full_graph_cache_for_eval()

        self.assertTrue(released)
        self.assertIsNone(trainer._full_graph_tensors)
        self.assertIsNone(trainer._full_graph_tensor_device)
        empty_cache.assert_called_once_with(torch.device("cuda"))


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

        self.assertEqual(args.presets, ["ucagnn", "lightgcn_paper", "dice_paper"])
        self.assertFalse(hasattr(args, "scoring_weight_modes"))

    def test_benchmark_parser_exposes_only_canonical_preset_names(self) -> None:
        """Benchmark CLI choices should match the canonical preset list."""
        parser = build_benchmark_parser()
        preset_action = next(action for action in parser._actions if action.dest == "presets")

        self.assertEqual(
            list(preset_action.choices),
            [
                "ucagnn",
                "lightgcn",
                "lightgcn_paper",
                "dice_paper",
                "dice_like",
                "dice_like_ablation",
            ],
        )

    def test_benchmark_plan_relies_on_preset_owned_score_mix_defaults(self) -> None:
        """Formal sweeps should no longer expose score-mix modes as a matrix axis."""
        args = formal_main._normalize_benchmark_args(
            SimpleNamespace(
                datasets=["movielens1m"],
                presets=["ucagnn", "lightgcn_paper", "dice_paper"],
                num_neighbors=[10, 5],
            ),
        )

        self.assertNotIn("scoring_weight_modes", args)
        self.assertEqual(
            build_benchmark_plan(args),
            [
                ("movielens1m", "ucagnn", "plateau", None, "observed", (10, 5)),
                ("movielens1m", "lightgcn_paper", "none", None, "observed", (10, 5)),
                ("movielens1m", "dice_paper", "none", None, "observed", (10, 5)),
            ],
        )

    def test_build_benchmark_plan_keeps_paper_baselines_on_constant_lr(self) -> None:
        """Paper baselines should not expand LR-scheduler sweeps into duplicate runs."""
        args = SimpleNamespace(
            datasets=["movielens1m"],
            presets=["lightgcn_paper", "dice_paper"],
            num_neighbors=[10, 5],
            lr_scheduler="all",
        )

        self.assertEqual(
            build_benchmark_plan(args),
            [
                ("movielens1m", "lightgcn_paper", "none", None, "observed", (10, 5)),
                ("movielens1m", "dice_paper", "none", None, "observed", (10, 5)),
            ],
        )

    def test_normalize_benchmark_args_strips_removed_scoring_weight_modes_field(self) -> None:
        """Legacy saved payloads should drop removed score-mix mode metadata."""
        args = formal_main._normalize_benchmark_args(
            {
                "datasets": ["movielens1m"],
                "presets": ["ucagnn"],
                "num_neighbors": [10, 5],
                "scoring_weight_modes": ["fixed", "learned"],
            },
        )

        self.assertNotIn("scoring_weight_modes", args)
        self.assertEqual(args["presets"], ["ucagnn"])
        self.assertEqual(args["num_neighbors"], [10, 5])

    def test_plan_sweeps_datasets_within_each_method_combo(self) -> None:
        """Datasets should be the innermost loop of the execution plan."""
        from types import SimpleNamespace

        args = SimpleNamespace(
            datasets=["small"],
            presets=["ucagnn", "lightgcn"],
            num_neighbors=[[10, 5], [5, 3]],
        )

        plan = build_benchmark_plan(args)

        expected_prefix = [
            ("amazonbook", "ucagnn", "plateau", None, "observed", (10, 5)),
            ("amazonbook", "ucagnn", "plateau", None, "observed", (5, 3)),
            ("movielens1m", "ucagnn", "plateau", None, "observed", (10, 5)),
            ("movielens1m", "ucagnn", "plateau", None, "observed", (5, 3)),
            ("amazonbook", "lightgcn", "plateau", None, "observed", (10, 5)),
            ("amazonbook", "lightgcn", "plateau", None, "observed", (5, 3)),
        ]

        self.assertEqual(plan[: len(expected_prefix)], expected_prefix)
        self.assertEqual(len(plan), 8)

    def test_build_benchmark_plan_sweeps_graph_policies(self) -> None:
        """Benchmark planning should expand graph-policy sweeps into separate runs."""
        args = SimpleNamespace(
            datasets=["movielens1m"],
            presets=["ucagnn"],
            graph_policy="observed",
            graph_policy_options=["observed", "cagra_augmented"],
            num_neighbors=[10, 5],
            lr_scheduler="plateau",
        )

        plan = build_benchmark_plan(args)

        self.assertEqual(
            plan,
            [
                ("movielens1m", "ucagnn", "plateau", None, "observed", (10, 5)),
                (
                    "movielens1m",
                    "ucagnn",
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
                    "plateau",
                    "kuairec_fullobs",
                    "observed",
                    (10, 5),
                ),
                (
                    "kuairec_v2",
                    "ucagnn",
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
            num_neighbors=[10, 5],
            lr_scheduler="cosine",
        )

        plan = build_benchmark_plan(args)

        self.assertIn(
            ("movielens1m", "ucagnn", "cosine", None, "observed", (10, 5)),
            plan,
        )
        self.assertIn(
            ("amazonbook", "ucagnn", "cosine", None, "observed", (10, 5)),
            plan,
        )

    def test_build_benchmark_plan_resolves_dataset_keyed_num_neighbors(self) -> None:
        """Benchmark planning should resolve per-tier neighbor sweeps before expansion."""
        args = SimpleNamespace(
            datasets=["small", "medium"],
            presets=["ucagnn"],
            num_neighbors={
                "small": [[6, 3], [4, 2]],
                "medium": [[10, 5], [16, 8]],
            },
            lr_scheduler="plateau",
        )

        plan = build_benchmark_plan(args)

        self.assertEqual(
            plan[:4],
            [
                ("amazonbook", "ucagnn", "plateau", None, "observed", (6, 3)),
                ("amazonbook", "ucagnn", "plateau", None, "observed", (4, 2)),
                ("movielens1m", "ucagnn", "plateau", None, "observed", (6, 3)),
                ("movielens1m", "ucagnn", "plateau", None, "observed", (4, 2)),
            ],
        )
        self.assertIn(
            ("kuairec_v2", "ucagnn", "plateau", None, "observed", (10, 5)),
            plan,
        )
        self.assertIn(
            ("kuairand1k", "ucagnn", "plateau", None, "observed", (16, 8)),
            plan,
        )
        self.assertEqual(len(plan), 8)

    def test_build_benchmark_plan_resolves_all_lr_schedulers(self) -> None:
        """The lr_scheduler='all' shorthand should expand to all supported schedulers."""
        args = SimpleNamespace(
            datasets=["movielens1m"],
            presets=["ucagnn"],
            num_neighbors=[10, 5],
            lr_scheduler="all",
        )

        plan = build_benchmark_plan(args)
        schedulers = {entry[2] for entry in plan}

        self.assertEqual(schedulers, set(formal_main.SUPPORTED_LR_SCHEDULERS))

    def test_run_benchmark_reuses_pre_normalized_payload_for_dry_run(self) -> None:
        """Dry-run benchmark execution should not renormalize an internal payload."""
        normalized_args = formal_main._normalize_benchmark_args(
            SimpleNamespace(
                datasets=["movielens1m"],
                presets=["ucagnn"],
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

    def test_parse_formal_profile_sequence_accepts_comma_separated_names(self) -> None:
        """formal-run should accept a comma-separated profile queue."""
        profile_names = formal_main._parse_formal_profile_sequence(
            "paper-lightgcn-small-baselines, paper-dice-all-runtime-probes",
        )

        self.assertEqual(
            profile_names,
            ["paper-lightgcn-small-baselines", "paper-dice-all-runtime-probes"],
        )

    def test_formal_main_runs_comma_separated_profiles_in_order(self) -> None:
        """Multiple formal profiles should execute sequentially and report aggregate failure."""
        cli_args = SimpleNamespace(
            profile="paper-lightgcn-small-baselines,paper-dice-all-runtime-probes",
            list_profiles=False,
            overwrite_checkpoint=False,
        )

        with (
            patch.object(formal_main, "build_formal_run_parser") as build_parser,
            patch.object(
                formal_main,
                "_run_single_formal_profile",
                side_effect=[0, 1],
            ) as run_single_profile,
        ):
            build_parser.return_value.parse_args.return_value = cli_args
            exit_code = formal_main.formal_main()

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            [call.args[0] for call in run_single_profile.call_args_list],
            ["paper-lightgcn-small-baselines", "paper-dice-all-runtime-probes"],
        )

    def test_runtime_probe_estimate_scales_observed_epoch_to_target_epochs(self) -> None:
        """Probe estimates should scale measured training throughput to the target run."""
        estimate = formal_main._build_runtime_probe_estimate(
            target_epochs=200,
            observed_training_time_s=610.0,
            observed_epochs=1,
            train_batches_per_epoch=1745,
        )

        self.assertAlmostEqual(estimate["runtime_probe_seconds_per_epoch"], 610.0)
        self.assertAlmostEqual(
            estimate["runtime_probe_observed_batches_per_second"],
            1745.0 / 610.0,
        )
        self.assertAlmostEqual(estimate["runtime_probe_estimated_train_time_s"], 122000.0)
        self.assertAlmostEqual(
            estimate["runtime_probe_estimated_remaining_train_time_s"],
            121390.0,
        )

    def test_runtime_probe_metadata_belongs_only_to_one_epoch_probe_profiles(self) -> None:
        """Run-once small baselines should not be labeled as runtime approximations."""
        probe_profile_ids = {
            profile_name
            for profile_name in formal_profile_names()
            if get_formal_profile(profile_name).get("runtime_probe") is not None
        }

        self.assertEqual(
            probe_profile_ids,
            {
                "paper-lightgcn-large-runtime-probes",
                "paper-dice-all-runtime-probes",
            },
        )
        for profile_name in probe_profile_ids:
            profile = get_formal_profile(profile_name)
            self.assertEqual(profile["config_overrides"]["epochs"], 1)
            self.assertEqual(profile["runtime_probe"]["target_epochs"], 200)


if __name__ == "__main__":
    unittest.main()
