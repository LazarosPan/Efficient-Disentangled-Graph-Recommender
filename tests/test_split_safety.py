"""Regression coverage for split safety and runtime causal contracts."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from experiments.run_experiment import _apply_item_universe_policy, build_runtime_model
from src.data.canonical import CanonicalInteractions, sample_canonical_interactions
from src.data.graph_builder import build_graph
from src.data.subgraph_sampler import SubgraphBatch
from src.losses.loss_suite import LossSuite, _bpr_loss
from src.models.edgrec import EDGRec
from src.models.lightgcn import DualBranchGCN, LightGCNBranch
from src.training.evaluator import (
    THESIS_PRIMARY_METRICS,
    Evaluator,
    _EvaluatorDiagnosticsAccumulator,
    _rowwise_spearman,
)
from src.training.mini_batch_trainer import MiniBatchTrainer
from src.utils.config import EDGRecConfig
from src.utils.interaction_indexing import remap_interaction_ids
from src.utils.trainer_runtime import TrainerRuntime, stage_graph_tensors_for_device
from torch_geometric.data import Data


class _RankingModel(torch.nn.Module):
    """Return deterministic scores for evaluator regression tests."""

    def __init__(self, base_scores: torch.Tensor) -> None:
        """Store one fixed score vector for all evaluated users."""
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "_base_scores",
            base_scores.float(),
            persistent=False,
        )

    def get_all_scores(
        self,
        edge_index: torch.Tensor,
        batch_users: torch.Tensor,
        edge_sign: torch.Tensor | None,
        scoring_mode: str = "default",
    ) -> torch.Tensor:
        """Return the same score vector for every user in the batch."""
        del edge_index, edge_sign, scoring_mode
        return (
            self._base_scores.to(batch_users.device)
            .unsqueeze(0)
            .expand(
                batch_users.size(0),
                -1,
            )
        )

    def get_propagated_for_eval(
        self,
        edge_index: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
        edge_norm: torch.Tensor | None = None,
        embedding_dtype: torch.dtype | None = None,
    ) -> dict:
        """Stub: no real propagation; return empty dict sentinel."""
        del edge_index, edge_sign, edge_norm, embedding_dtype
        return {}

    def score_users_from_propagated(
        self,
        propagated: dict,
        user_ids: torch.Tensor,
        scoring_mode: str | None = None,
    ) -> torch.Tensor:
        """Return the same fixed scores for every user."""
        del propagated, scoring_mode
        return self._base_scores.to(user_ids.device).unsqueeze(0).expand(user_ids.size(0), -1)


class _ComponentRankingModel(torch.nn.Module):
    """Return deterministic refined score components for evaluator tests."""

    def __init__(
        self,
        interest_scores: torch.Tensor,
        conformity_scores: torch.Tensor,
        context_scores: torch.Tensor,
        score_mix_weights: torch.Tensor,
        user_interest_emb: torch.Tensor,
        user_conformity_emb: torch.Tensor,
        branch_interest_scores: torch.Tensor | None = None,
        branch_conformity_scores: torch.Tensor | None = None,
        expand_context_scores: bool = False,
    ) -> None:
        """Store full-catalog refined scorer components."""
        super().__init__()
        self.expand_context_scores = expand_context_scores
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("interest_scores", interest_scores.float(), persistent=False)
        self.register_buffer("conformity_scores", conformity_scores.float(), persistent=False)
        self.register_buffer("context_scores", context_scores.float(), persistent=False)
        self.register_buffer(
            "branch_interest_scores",
            (interest_scores if branch_interest_scores is None else branch_interest_scores).float(),
            persistent=False,
        )
        self.register_buffer(
            "branch_conformity_scores",
            (
                conformity_scores if branch_conformity_scores is None else branch_conformity_scores
            ).float(),
            persistent=False,
        )
        self.register_buffer("score_mix_weights", score_mix_weights.float(), persistent=False)
        self.register_buffer("user_interest_emb", user_interest_emb.float(), persistent=False)
        self.register_buffer("user_conformity_emb", user_conformity_emb.float(), persistent=False)
        self.register_buffer(
            "final_scores",
            (
                score_mix_weights[:, 0].unsqueeze(1) * interest_scores
                + score_mix_weights[:, 1].unsqueeze(1) * conformity_scores
                + score_mix_weights[:, 2].unsqueeze(1) * context_scores
            ).to(torch.float32),
            persistent=False,
        )

    def get_propagated_for_eval(
        self,
        edge_index: torch.Tensor,
        edge_sign: torch.Tensor | None = None,
        edge_norm: torch.Tensor | None = None,
        embedding_dtype: torch.dtype | None = None,
    ) -> dict:
        """Stub: no real propagation; return empty dict sentinel."""
        del edge_index, edge_sign, edge_norm, embedding_dtype
        return {}

    def score_users_from_propagated(
        self,
        propagated: dict,
        user_ids: torch.Tensor,
        scoring_mode: str | None = None,
    ) -> torch.Tensor:
        """Return deterministic final scores."""
        del propagated, scoring_mode
        return self.final_scores.index_select(0, user_ids.cpu()).to(device=user_ids.device)

    def get_score_components_from_propagated(
        self,
        propagated: dict,
        user_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return deterministic refined score components."""
        del propagated
        batch_users = user_ids.cpu()
        device = user_ids.device
        if self.expand_context_scores:
            context_scores = (
                self.context_scores[0]
                .to(device=device)
                .unsqueeze(0)
                .expand(
                    user_ids.size(0),
                    -1,
                )
            )
        else:
            context_scores = self.context_scores.index_select(0, batch_users).to(device=device)
        return {
            "interest_score": self.interest_scores.index_select(0, batch_users).to(device=device),
            "conformity_score": self.conformity_scores.index_select(0, batch_users).to(
                device=device
            ),
            "context_score": context_scores,
            "branch_interest_score": self.branch_interest_scores.index_select(0, batch_users).to(
                device=device
            ),
            "branch_conformity_score": self.branch_conformity_scores.index_select(
                0,
                batch_users,
            ).to(device=device),
            "score_mix_weights": self.score_mix_weights.index_select(0, batch_users).to(
                device=device
            ),
            "final_score": self.final_scores.index_select(0, batch_users).to(device=device),
            "user_interest_emb": self.user_interest_emb.index_select(0, batch_users).to(
                device=device
            ),
            "user_conformity_emb": self.user_conformity_emb.index_select(
                0,
                batch_users,
            ).to(device=device),
        }


class _GatherBeforeFloatGuard:
    """Raise if evaluator diagnostics cast the full matrix before gathering top-k."""

    def __init__(self, scores: torch.Tensor) -> None:
        """Store the native-dtype score matrix."""
        self._scores = scores

    def gather(self, dim: int, index: torch.Tensor) -> torch.Tensor:
        """Gather directly from the native-dtype score matrix."""
        return self._scores.gather(dim, index)

    def size(self, dim: int) -> int:
        """Return one dimension without exposing full-matrix casting."""
        return self._scores.size(dim)

    def topk(self, k: int, dim: int):
        """Run top-k on the native-dtype score matrix."""
        return self._scores.topk(k, dim=dim)

    def float(self) -> torch.Tensor:
        """Reject full-matrix float casts so tests catch the regression."""
        msg = "full score matrix cast to float before top-k gather"
        raise AssertionError(msg)


def _small_memory_canonical() -> CanonicalInteractions:
    """Return a tiny predefined-split canonical graph for memory-contract tests."""
    return CanonicalInteractions(
        user_id=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        item_id=np.array([0, 1, 1, 2, 2, 3], dtype=np.int64),
        label=np.ones(6, dtype=np.float32),
        timestamp=np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        sign=np.ones(6, dtype=np.float32),
        popularity=np.ones(4, dtype=np.float32),
        n_users=3,
        n_items=4,
        user_map={100: 0, 101: 1, 102: 2},
        item_map={200: 0, 201: 1, 202: 2, 203: 3},
        train_mask=np.array([True, True, True, True, False, False]),
        val_mask=np.array([False, False, False, False, True, False]),
        test_mask=np.array([False, False, False, False, False, True]),
    )


class SplitSafetyTests(unittest.TestCase):
    """Pin the thesis-critical split boundary behavior."""

    def test_thesis_primary_metrics_match_runtime_contract(self) -> None:
        """Runtime evaluation should log only the thesis-facing metric set."""
        self.assertEqual(
            THESIS_PRIMARY_METRICS,
            (
                "NDCG@20",
                "Recall@20",
                "AveragePopularity@20",
                "HitRatio@20",
                "Personalization@20",
                "NDCG@40",
                "Recall@40",
                "AveragePopularity@40",
                "HitRatio@40",
                "Personalization@40",
            ),
        )

    def test_random_exposure_item_universe_compacts_items_and_targets(self) -> None:
        """KuaiRand compaction should remap interaction, feature, and target arrays."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0, 1, 1], dtype=np.int64),
            item_id=np.array([0, 1, 2, 3], dtype=np.int64),
            label=np.ones(4, dtype=np.float32),
            timestamp=np.array([1, 2, 3, 4], dtype=np.int64),
            sign=np.ones(4, dtype=np.float32),
            popularity=np.ones(4, dtype=np.float32),
            n_users=2,
            n_items=4,
            user_map={10: 0, 11: 1},
            item_map={100: 0, 101: 1, 102: 2, 103: 3},
            item_features=np.arange(8, dtype=np.float32).reshape(4, 2),
            exposure_flag=np.array([True, False, True, False]),
            item_propensity_targets=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            train_mask=np.array([True, True, False, False]),
            val_mask=np.array([False, False, True, False]),
            test_mask=np.array([False, False, False, True]),
        )
        config = EDGRecConfig(
            dataset="kuairand1k",
            device="cpu",
            item_universe_policy="random_exposure_items_only",
        )

        compacted = _apply_item_universe_policy(canonical, config)

        np.testing.assert_array_equal(compacted.user_id, np.array([0, 1]))
        np.testing.assert_array_equal(compacted.item_id, np.array([0, 1]))
        np.testing.assert_array_equal(
            compacted.item_features,
            np.array([[0.0, 1.0], [4.0, 5.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            compacted.item_propensity_targets,
            np.array([0.1, 0.3], dtype=np.float32),
        )
        np.testing.assert_array_equal(compacted.train_mask, np.array([True, False]))
        np.testing.assert_array_equal(compacted.val_mask, np.array([False, True]))
        self.assertEqual(compacted.n_items, 2)
        self.assertEqual(
            compacted.metadata["item_universe_original_n_items"],
            4,
        )

    def test_train_edge_dropout_affects_graph_only_not_split_masks(self) -> None:
        """Observed-edge dropout must not remove labels or split assignments."""
        canonical = _small_memory_canonical()
        config = EDGRecConfig(
            device="cpu",
            train_edge_keep_prob=0.5,
            seed=7,
            embed_dim=4,
            num_neighbors=[4, 2],
        )

        data = build_graph(canonical, config)

        self.assertEqual(int(data.train_mask.sum().item()), 4)
        self.assertEqual(int(data.val_mask.sum().item()), 1)
        self.assertEqual(int(data.test_mask.sum().item()), 1)
        self.assertEqual(data.labels.numel(), len(canonical))
        self.assertLess(data.edge_index.size(1), 8)
        self.assertGreater(data.edge_index.size(1), 0)
        self.assertEqual(data.edge_index.size(1) % 2, 0)
        self.assertEqual(data.train_edge_keep_prob, 0.5)

    def test_new_runtime_config_fields_validate_values(self) -> None:
        """New optional runtime fields should reject unsupported values early."""
        with self.assertRaisesRegex(ValueError, "validation_every_n_epochs"):
            EDGRecConfig(device="cpu", validation_every_n_epochs=0)
        with self.assertRaisesRegex(ValueError, "label_mode"):
            EDGRecConfig(device="cpu", label_mode="bad")
        with self.assertRaisesRegex(ValueError, "sampler_residency_policy"):
            EDGRecConfig(device="cpu", sampler_residency_policy="bad")
        with self.assertRaisesRegex(ValueError, "propagation_backend"):
            EDGRecConfig(device="cpu", propagation_backend="bad")
        EDGRecConfig(device="cpu", propagation_backend="cuda_sparse_adjacency")
        EDGRecConfig(device="cpu", propagation_backend="chunked_edge_index_aggregation")
        sparse_alias_config = EDGRecConfig(device="cpu", embedding_sparse_optimizer=True)
        self.assertEqual(sparse_alias_config.embedding_optimizer, "sparseadam")
        sparse_string_config = EDGRecConfig(device="cpu", embedding_optimizer="sparseadam")
        self.assertTrue(sparse_string_config.embedding_sparse_optimizer)
        with self.assertRaisesRegex(ValueError, "Use either embedding_sparse_optimizer"):
            EDGRecConfig(
                device="cpu",
                embedding_sparse_optimizer=True,
                embedding_optimizer="sgd",
            )
        with self.assertRaisesRegex(ValueError, "KuaiRand-only"):
            EDGRecConfig(device="cpu", label_mode="strict_like_follow")

    def test_sparse_embedding_optimizer_splits_sparse_tables(self) -> None:
        """Sparse embedding modes should not put giant tables in AdamW."""
        canonical = _small_memory_canonical()
        config = EDGRecConfig(
            device="cpu",
            embed_dim=4,
            embedding_sparse_optimizer=True,
            use_features=False,
            use_popularity_emb=False,
            use_popularity_head=False,
            loss_weight_contrastive=0.0,
            loss_weight_align=0.0,
            loss_weight_uniform=0.0,
            num_neighbors=[4, 2],
        )
        data = build_graph(canonical, config)
        model = build_runtime_model(config, canonical, data)
        trainer = MiniBatchTrainer(
            model=model,
            loss_suite=LossSuite(config),
            data=data,
            config=config,
        )

        self.assertTrue(model.embedding.item_embed.sparse)
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)
        self.assertIsInstance(trainer.embedding_optimizer, torch.optim.SparseAdam)

        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.embedding_optimizer.zero_grad(set_to_none=True)
        model.embedding.item_embed(torch.tensor([0, 1], dtype=torch.long)).sum().backward()
        grad = model.embedding.item_embed.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(grad.is_sparse)

    def test_sparse_embedding_optimizer_falls_back_on_dense_table_gradients(self) -> None:
        """Sparse optimizer mode should return to AdamW if tables get dense grads."""
        canonical = _small_memory_canonical()
        config = EDGRecConfig(
            device="cpu",
            embed_dim=4,
            embedding_sparse_optimizer=True,
            use_features=False,
            use_popularity_emb=False,
            use_popularity_head=False,
            num_neighbors=[4, 2],
        )
        data = build_graph(canonical, config)
        model = build_runtime_model(config, canonical, data)
        trainer = MiniBatchTrainer(
            model=model,
            loss_suite=LossSuite(config),
            data=data,
            config=config,
        )

        self.assertIsInstance(trainer.embedding_optimizer, torch.optim.SparseAdam)
        loss = model.embedding.item_embed.weight.sum()
        loss = loss + model.embedding.user_interest(torch.tensor([0, 1], dtype=torch.long)).sum()
        trainer._apply_optimization_step(loss)

        self.assertIsNone(trainer.embedding_optimizer)
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)
        self.assertFalse(model.embedding.item_embed.sparse)
        for parameter in model.parameters():
            self.assertIsNone(parameter.grad)

    def test_validation_cadence_skips_early_stopping_on_non_validation_epochs(self) -> None:
        """Skipped validation epochs should not advance scheduler or early stopping."""
        canonical = _small_memory_canonical()
        config = EDGRecConfig(
            device="cpu",
            embed_dim=4,
            epochs=2,
            batch_size=2,
            validation_every_n_epochs=2,
            sampler_residency_policy="keep_resident",
            lr_scheduler="none",
            show_progress_bar=False,
            use_features=False,
            use_popularity_emb=False,
            use_popularity_head=False,
            loss_weight_contrastive=0.0,
            loss_weight_align=0.0,
            loss_weight_uniform=0.0,
            loss_weight_popularity=0.0,
            num_neighbors=[4, 2],
        )
        data = build_graph(canonical, config)
        model = build_runtime_model(config, canonical, data)
        trainer = MiniBatchTrainer(
            model=model,
            loss_suite=LossSuite(config),
            data=data,
            config=config,
        )

        with (
            patch.object(
                trainer,
                "_evaluate_validation_metrics",
                return_value={"NDCG@40": 0.5},
            ) as evaluate_validation,
            patch.object(
                trainer,
                "_update_early_stopping",
                wraps=trainer._update_early_stopping,
            ) as update_early_stopping,
            patch.object(
                trainer,
                "_release_cuda_sampler_for_eval",
                wraps=trainer._release_cuda_sampler_for_eval,
            ) as release_sampler,
        ):
            history = trainer.train(checkpoint_every=0)

        self.assertEqual(evaluate_validation.call_count, 1)
        self.assertEqual(update_early_stopping.call_count, 1)
        self.assertEqual(release_sampler.call_count, 0)
        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_metrics"]), 1)

    def test_get_splits_keeps_predefined_test_mask_intact(self) -> None:
        """Validation must be carved from train without touching test rows."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0, 0, 0], dtype=np.int64),
            item_id=np.array([0, 1, 2, 3], dtype=np.int64),
            label=np.ones(4, dtype=np.float32),
            timestamp=np.array([1, 2, 3, 4], dtype=np.int64),
            sign=np.ones(4, dtype=np.float32),
            popularity=np.ones(4, dtype=np.float32),
            n_users=1,
            n_items=4,
            user_map={0: 0},
            item_map={0: 0, 1: 1, 2: 2, 3: 3},
            train_mask=np.array([True, True, True, False]),
            test_mask=np.array([False, False, False, True]),
        )

        train_mask, val_mask, test_mask = canonical.get_splits(
            train_ratio=0.8,
            val_ratio=0.1,
        )

        np.testing.assert_array_equal(test_mask, canonical.test_mask)
        np.testing.assert_array_equal(train_mask | val_mask, canonical.train_mask)
        self.assertFalse(np.any(train_mask & val_mask))
        self.assertFalse(np.any(train_mask & test_mask))
        self.assertFalse(np.any(val_mask & test_mask))

    def test_build_graph_uses_train_only_edges_and_popularity(self) -> None:
        """Graph edges and popularity must be derived from train rows only."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0, 0], dtype=np.int64),
            item_id=np.array([0, 1, 1], dtype=np.int64),
            label=np.ones(3, dtype=np.float32),
            timestamp=np.array([1, 2, 3], dtype=np.int64),
            sign=np.ones(3, dtype=np.float32),
            popularity=np.array([0.5, 1.0], dtype=np.float32),
            n_users=1,
            n_items=2,
            user_map={0: 0},
            item_map={0: 0, 1: 1},
            train_mask=np.array([True, False, False]),
            val_mask=np.array([False, True, False]),
            test_mask=np.array([False, False, True]),
        )
        config = EDGRecConfig(device="cuda")

        data = build_graph(canonical, config)

        expected_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        expected_popularity = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)

        self.assertTrue(torch.equal(data.edge_index.cpu(), expected_edge_index))
        self.assertTrue(torch.equal(data.popularity.cpu(), expected_popularity))

    def test_per_user_temporal_split_is_default_for_derived_splits(self) -> None:
        """Derived splits should preserve chronology within each user history."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0, 1, 1], dtype=np.int64),
            item_id=np.array([0, 1, 2, 3], dtype=np.int64),
            label=np.ones(4, dtype=np.float32),
            timestamp=np.array([1, 2, 100, 101], dtype=np.int64),
            sign=np.ones(4, dtype=np.float32),
            popularity=np.ones(4, dtype=np.float32),
            n_users=2,
            n_items=4,
            user_map={0: 0, 1: 1},
            item_map={0: 0, 1: 1, 2: 2, 3: 3},
        )

        train_mask, val_mask, test_mask = canonical.get_splits(
            train_ratio=0.5,
            val_ratio=0.0,
        )

        np.testing.assert_array_equal(
            train_mask,
            np.array([True, False, True, False]),
        )
        np.testing.assert_array_equal(val_mask, np.zeros(4, dtype=bool))
        np.testing.assert_array_equal(
            test_mask,
            np.array([False, True, False, True]),
        )

    def test_global_temporal_split_remains_available(self) -> None:
        """Callers must still be able to opt into global temporal splitting."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0, 1, 1], dtype=np.int64),
            item_id=np.array([0, 1, 2, 3], dtype=np.int64),
            label=np.ones(4, dtype=np.float32),
            timestamp=np.array([1, 2, 100, 101], dtype=np.int64),
            sign=np.ones(4, dtype=np.float32),
            popularity=np.ones(4, dtype=np.float32),
            n_users=2,
            n_items=4,
            user_map={0: 0, 1: 1},
            item_map={0: 0, 1: 1, 2: 2, 3: 3},
        )

        train_mask, _, test_mask = canonical.get_splits(
            train_ratio=0.5,
            val_ratio=0.0,
            derived_split_mode="global_temporal",
        )

        np.testing.assert_array_equal(
            train_mask,
            np.array([True, True, False, False]),
        )
        np.testing.assert_array_equal(
            test_mask,
            np.array([False, False, True, True]),
        )

    def test_build_graph_uses_positive_train_popularity(self) -> None:
        """Popularity should be computed from positive train interactions only."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0, 0], dtype=np.int64),
            item_id=np.array([0, 0, 1], dtype=np.int64),
            label=np.ones(3, dtype=np.float32),
            timestamp=np.array([1, 5, 100], dtype=np.int64),
            sign=np.ones(3, dtype=np.float32),
            popularity=np.ones(2, dtype=np.float32),
            n_users=1,
            n_items=2,
            user_map={0: 0},
            item_map={0: 0, 1: 1},
            train_mask=np.array([True, True, True]),
            val_mask=np.array([False, False, False]),
            test_mask=np.array([False, False, False]),
        )
        config = EDGRecConfig(
            device="cuda",
        )

        data = build_graph(canonical, config)

        expected_popularity = torch.tensor([1.0, 0.5], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(data.popularity.cpu(), expected_popularity))

    def test_build_graph_carries_causal_fields(self) -> None:
        """Graph data should retain the extended canonical causal descriptors."""
        metadata = {"repeat_collapse": {"applied": True, "dropped_rows": 1}}
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0], dtype=np.int64),
            item_id=np.array([0, 1], dtype=np.int64),
            label=np.ones(2, dtype=np.float32),
            timestamp=np.array([1, 2], dtype=np.int64),
            sign=np.ones(2, dtype=np.float32),
            popularity=np.ones(2, dtype=np.float32),
            user_features=np.array([[1.0], [2.0]], dtype=np.float32),
            item_features=np.array([[0.5], [1.5]], dtype=np.float32),
            raw_target=np.array([3.0, 4.0], dtype=np.float32),
            behavior_type=np.array(["fav", "buy"]),
            exposure_flag=np.array([False, True]),
            source_domain=np.array(["standard", "random"]),
            repeat_count=np.array([2, 1], dtype=np.uint8),
            repeat_mean_target=np.array([1.5, 4.0], dtype=np.float32),
            repeat_max_target=np.array([3.0, 4.0], dtype=np.float32),
            repeat_latest_target=np.array([2.0, 4.0], dtype=np.float32),
            repeat_first_timestamp=np.array([1, 2], dtype=np.uint8),
            repeat_last_timestamp=np.array([3, 4], dtype=np.uint8),
            repeat_behavior_counts=np.array([[1, 1], [0, 1]], dtype=np.uint8),
            repeat_behavior_labels=np.array(["pv", "buy"]),
            feedback_type="multi-behavior",
            preprocessing_preset="taobao_multibehavior",
            n_users=1,
            n_items=2,
            user_map={0: 0},
            item_map={0: 0, 1: 1},
            train_mask=np.array([True, False]),
            val_mask=np.array([False, False]),
            test_mask=np.array([False, True]),
            metadata=metadata,
        )
        config = EDGRecConfig(device="cuda")

        data = build_graph(canonical, config)

        self.assertTrue(torch.equal(data.user_features.cpu(), torch.tensor([[1.0], [2.0]])))
        self.assertTrue(torch.equal(data.item_features.cpu(), torch.tensor([[0.5], [1.5]])))
        self.assertTrue(torch.equal(data.raw_target.cpu(), torch.tensor([3.0, 4.0])))
        self.assertTrue(
            torch.equal(data.exposure_flag.cpu(), torch.tensor([False, True])),
        )
        np.testing.assert_array_equal(data.behavior_type, np.array(["fav", "buy"]))
        np.testing.assert_array_equal(
            data.source_domain,
            np.array(["standard", "random"]),
        )
        self.assertTrue(torch.equal(data.repeat_count.cpu(), torch.tensor([2, 1])))
        self.assertTrue(
            torch.equal(
                data.repeat_mean_target.cpu(),
                torch.tensor([1.5, 4.0]),
            ),
        )
        self.assertTrue(
            torch.equal(
                data.repeat_max_target.cpu(),
                torch.tensor([3.0, 4.0]),
            ),
        )
        self.assertTrue(
            torch.equal(
                data.repeat_latest_target.cpu(),
                torch.tensor([2.0, 4.0]),
            ),
        )
        self.assertTrue(
            torch.equal(data.repeat_first_timestamp.cpu(), torch.tensor([1, 2])),
        )
        self.assertTrue(
            torch.equal(data.repeat_last_timestamp.cpu(), torch.tensor([3, 4])),
        )
        self.assertTrue(
            torch.equal(
                data.repeat_behavior_counts.cpu(),
                torch.tensor([[1, 1], [0, 1]]),
            ),
        )
        np.testing.assert_array_equal(data.repeat_behavior_labels, np.array(["pv", "buy"]))
        self.assertEqual(data.feedback_type, "multi-behavior")
        self.assertEqual(data.preprocessing_preset, "taobao_multibehavior")
        self.assertEqual(data.metadata, metadata)

    def test_sample_canonical_interactions_slices_extended_fields(self) -> None:
        """Tiny-run sampling should keep canonical field shapes internally aligned."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
            item_id=np.array([0, 1, 1, 2, 2, 3], dtype=np.int64),
            label=np.ones(6, dtype=np.float32),
            timestamp=np.arange(1, 7, dtype=np.int64),
            sign=np.ones(6, dtype=np.float32),
            popularity=np.ones(4, dtype=np.float32),
            user_features=np.arange(6, dtype=np.float32).reshape(3, 2),
            item_features=np.arange(8, dtype=np.float32).reshape(4, 2),
            raw_target=np.arange(6, dtype=np.float32),
            behavior_type=np.array(["a", "b", "c", "d", "e", "f"]),
            exposure_flag=np.array([True, False, True, False, True, False]),
            source_domain=np.array(["s", "s", "r", "r", "s", "r"]),
            repeat_count=np.arange(10, 16, dtype=np.uint8),
            repeat_mean_target=np.arange(20, 26, dtype=np.float32),
            repeat_max_target=np.arange(30, 36, dtype=np.float32),
            repeat_latest_target=np.arange(40, 46, dtype=np.float32),
            repeat_first_timestamp=np.arange(50, 56, dtype=np.int64),
            repeat_last_timestamp=np.arange(60, 66, dtype=np.int64),
            repeat_behavior_counts=np.arange(12, dtype=np.uint8).reshape(6, 2),
            repeat_behavior_labels=np.array(["view", "buy"]),
            item_propensity_targets=np.linspace(0.1, 0.4, 4, dtype=np.float32),
            feedback_type="synthetic",
            preprocessing_preset="unit_test",
            n_users=3,
            n_items=4,
            user_map={10: 0, 11: 1, 12: 2},
            item_map={20: 0, 21: 1, 22: 2, 23: 3},
            train_mask=np.array([True, True, False, False, False, False]),
            val_mask=np.array([False, False, True, True, False, False]),
            test_mask=np.array([False, False, False, False, True, True]),
            metadata={
                "interaction_values": np.arange(6),
                "user_values": np.arange(3),
                "item_values": np.arange(4),
                "scalar_value": np.array(1),
                "unchanged": "keep",
            },
        )

        sampled = sample_canonical_interactions(
            canonical,
            sample_interactions=3,
            seed=13,
            train_ratio=0.8,
            val_ratio=0.1,
        )

        self.assertEqual(len(sampled), 3)
        self.assertEqual(int(sampled.train_mask.sum()), 1)
        self.assertEqual(int(sampled.val_mask.sum()), 1)
        self.assertEqual(int(sampled.test_mask.sum()), 1)
        for field_name in (
            "raw_target",
            "behavior_type",
            "exposure_flag",
            "source_domain",
            "repeat_count",
            "repeat_mean_target",
            "repeat_max_target",
            "repeat_latest_target",
            "repeat_first_timestamp",
            "repeat_last_timestamp",
            "repeat_behavior_counts",
        ):
            field = getattr(sampled, field_name)
            assert field is not None
            self.assertEqual(field.shape[0], len(sampled), field_name)
        assert sampled.user_features is not None
        assert sampled.item_features is not None
        assert sampled.item_propensity_targets is not None
        self.assertEqual(sampled.user_features.shape[0], sampled.n_users)
        self.assertEqual(sampled.item_features.shape[0], sampled.n_items)
        self.assertEqual(sampled.popularity.shape[0], sampled.n_items)
        self.assertEqual(sampled.item_propensity_targets.shape[0], sampled.n_items)
        self.assertEqual(sampled.feedback_type, "synthetic")
        self.assertEqual(sampled.preprocessing_preset, "unit_test")
        np.testing.assert_array_equal(sampled.repeat_behavior_labels, np.array(["view", "buy"]))
        assert sampled.metadata is not None
        self.assertEqual(sampled.metadata["interaction_values"].shape[0], len(sampled))
        self.assertEqual(sampled.metadata["user_values"].shape[0], sampled.n_users)
        self.assertEqual(sampled.metadata["item_values"].shape[0], sampled.n_items)
        np.testing.assert_array_equal(sampled.metadata["scalar_value"], np.array(1))
        self.assertEqual(sampled.metadata["unchanged"], "keep")

    def test_build_graph_copies_read_only_numpy_fields_before_torch_conversion(self) -> None:
        """Graph boundary conversion should not alias read-only NumPy payloads."""
        raw_target = np.array([3.0, 4.0], dtype=np.float32)
        raw_target.setflags(write=False)
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0], dtype=np.int64),
            item_id=np.array([0, 1], dtype=np.int64),
            label=np.ones(2, dtype=np.float32),
            timestamp=np.array([1, 2], dtype=np.int64),
            sign=np.ones(2, dtype=np.float32),
            popularity=np.ones(2, dtype=np.float32),
            raw_target=raw_target,
            n_users=1,
            n_items=2,
            user_map={0: 0},
            item_map={0: 0, 1: 1},
            train_mask=np.array([True, False]),
            val_mask=np.array([False, False]),
            test_mask=np.array([False, True]),
        )
        config = EDGRecConfig(device="cuda")

        data = build_graph(canonical, config)
        data.raw_target[0] = 9.0

        np.testing.assert_array_equal(raw_target, np.array([3.0, 4.0], dtype=np.float32))
        self.assertEqual(float(data.raw_target[0]), 9.0)

    def test_remapped_ids_use_narrow_integer_storage(self) -> None:
        """Contiguous interaction IDs should downcast to the smallest safe dtype."""
        indexed = remap_interaction_ids(
            np.array([10, 11, 12], dtype=np.int64),
            np.array([100, 101, 100], dtype=np.int64),
        )

        self.assertEqual(indexed.user_id.dtype, np.dtype(np.uint8))
        self.assertEqual(indexed.item_id.dtype, np.dtype(np.uint8))

    def test_compute_item_recency_uses_train_mask_and_normalizes(self) -> None:
        """Per-item recency should use only the provided split mask and normalize."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0, 0, 0], dtype=np.int64),
            item_id=np.array([0, 0, 1, 1], dtype=np.int64),
            label=np.ones(4, dtype=np.float32),
            timestamp=np.array([10, 20, 30, 40], dtype=np.int64),
            sign=np.ones(4, dtype=np.float32),
            popularity=np.ones(2, dtype=np.float32),
            n_users=1,
            n_items=2,
            user_map={0: 0},
            item_map={0: 0, 1: 1},
        )

        recency = canonical.compute_item_recency(
            np.array([True, True, True, False], dtype=bool),
        )

        np.testing.assert_allclose(
            recency,
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_evaluator_masks_observed_non_target_items_before_ranking(self) -> None:
        """Held-out targets must rank after masking previously observed items."""
        config = EDGRecConfig(device="cuda")
        evaluator = Evaluator(config)

        data = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=51)
        data.edge_sign = torch.empty((0,), dtype=torch.bfloat16)
        data.train_mask = torch.tensor([True, False], dtype=torch.bool)
        data.val_mask = torch.tensor([False, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, True], dtype=torch.bool)
        data.user_nodes = torch.tensor([0, 0], dtype=torch.long)
        data.item_nodes = torch.tensor([1, 2], dtype=torch.long)
        data.n_users = 1
        data.n_items = 50
        data.popularity = torch.linspace(1.0, 0.1, 50)

        base_scores = torch.zeros(50, dtype=torch.bfloat16)
        base_scores[0] = 10.0
        base_scores[1] = 9.0
        model = _RankingModel(base_scores)

        metrics = evaluator.evaluate(model, data, data.test_mask, batch_size=16)

        self.assertAlmostEqual(metrics["Recall@20"], 1.0, places=6)
        self.assertAlmostEqual(metrics["Recall@40"], 1.0, places=6)
        self.assertAlmostEqual(metrics["NDCG@20"], 1.0, places=6)
        self.assertAlmostEqual(metrics["NDCG@40"], 1.0, places=6)
        self.assertAlmostEqual(metrics["HitRatio@20"], 1.0, places=6)
        self.assertAlmostEqual(metrics["HitRatio@40"], 1.0, places=6)
        self.assertAlmostEqual(metrics["Personalization@20"], 0.0, places=6)
        self.assertAlmostEqual(metrics["Personalization@40"], 0.0, places=6)

    def test_evaluator_uses_only_positive_labels_as_ground_truth(self) -> None:
        """Negative or weak held-out interactions must not count as relevant."""
        evaluator = Evaluator(EDGRecConfig(device="cpu"))

        data = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=51)
        data.edge_sign = torch.empty((0,), dtype=torch.bfloat16)
        data.train_mask = torch.tensor([True, False, False], dtype=torch.bool)
        data.val_mask = torch.tensor([False, False, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, True, True], dtype=torch.bool)
        data.labels = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
        data.user_nodes = torch.tensor([0, 0, 0], dtype=torch.long)
        data.item_nodes = torch.tensor([1, 2, 3], dtype=torch.long)
        data.n_users = 1
        data.n_items = 50
        data.popularity = torch.linspace(1.0, 0.1, 50)

        base_scores = torch.linspace(50.0, 1.0, 50, dtype=torch.float32)
        base_scores[1] = 100.0
        base_scores[2] = -100.0
        model = _RankingModel(base_scores)

        metrics = evaluator.evaluate(model, data, data.test_mask, batch_size=16)

        self.assertAlmostEqual(metrics["Recall@20"], 0.0, places=6)
        self.assertAlmostEqual(metrics["HitRatio@20"], 0.0, places=6)

    def test_evaluator_appends_refined_score_diagnostics_from_exported_components(self) -> None:
        """Evaluator diagnostics should be computed from exported refined score components."""
        evaluator = Evaluator(EDGRecConfig(device="cpu"))

        data = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=52)
        data.edge_sign = torch.empty((0,), dtype=torch.bfloat16)
        data.train_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
        data.val_mask = torch.tensor([False, False, False, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, True, False, True], dtype=torch.bool)
        data.user_nodes = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        data.item_nodes = torch.tensor([1, 50, 3, 2], dtype=torch.long)
        data.n_users = 2
        data.n_items = 50
        data.popularity = torch.arange(50, dtype=torch.float32)

        base = torch.arange(50, dtype=torch.float32).unsqueeze(0).expand(2, -1)
        model = _ComponentRankingModel(
            interest_scores=-base,
            conformity_scores=base,
            context_scores=base,
            score_mix_weights=torch.tensor(
                [[0.25, 0.25, 0.50], [0.75, 0.10, 0.15]],
                dtype=torch.float32,
            ),
            user_interest_emb=torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
            user_conformity_emb=torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        )

        metrics = evaluator.evaluate(
            model,
            data,
            data.test_mask,
            batch_size=2,
            include_refined_diagnostics=True,
        )

        self.assertAlmostEqual(metrics["score_mix_interest_mean"], 0.5, places=6)
        self.assertAlmostEqual(metrics["score_mix_interest_std"], 0.25, places=6)
        self.assertAlmostEqual(metrics["score_mix_conformity_mean"], 0.175, places=6)
        self.assertAlmostEqual(metrics["score_mix_conformity_std"], 0.075, places=6)
        self.assertAlmostEqual(metrics["score_mix_context_mean"], 0.325, places=6)
        self.assertAlmostEqual(metrics["score_mix_context_std"], 0.175, places=6)
        self.assertAlmostEqual(metrics["interest_conformity_cosine_mean"], 0.5, places=6)
        self.assertAlmostEqual(metrics["interest_conformity_cosine_std"], 0.5, places=6)
        self.assertLess(metrics["interest_contribution@20"], 0.0)
        self.assertGreater(metrics["conformity_contribution@20"], 0.0)
        self.assertGreater(metrics["context_contribution@40"], 0.0)
        self.assertLess(metrics["interest_popularity_spearman@20"], 0.0)
        self.assertGreater(metrics["context_popularity_spearman@20"], 0.0)

        diagnostic_keys = (
            "score_mix_interest_mean",
            "score_mix_interest_std",
            "score_mix_conformity_mean",
            "score_mix_conformity_std",
            "score_mix_context_mean",
            "score_mix_context_std",
            "interest_contribution@20",
            "interest_contribution@40",
            "conformity_contribution@20",
            "conformity_contribution@40",
            "context_contribution@20",
            "context_contribution@40",
            "interest_conformity_cosine_mean",
            "interest_conformity_cosine_std",
            "interest_popularity_spearman@20",
            "interest_popularity_spearman@40",
            "conformity_popularity_spearman@20",
            "conformity_popularity_spearman@40",
            "context_popularity_spearman@20",
            "context_popularity_spearman@40",
            "final_popularity_spearman@20",
            "final_popularity_spearman@40",
            "interest_branch_NDCG@20",
            "interest_branch_Recall@20",
            "interest_branch_AveragePopularity@20",
            "conformity_branch_NDCG@20",
            "conformity_branch_Recall@20",
            "conformity_branch_AveragePopularity@20",
        )
        for key in diagnostic_keys:
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]), msg=f"{key} should be finite")

    def test_evaluator_keeps_expanded_context_diagnostics_finite_after_seen_masking(
        self,
    ) -> None:
        """Item-only expanded context scores must not be mutated by seen-item masking."""
        evaluator = Evaluator(EDGRecConfig(device="cpu"))

        user_nodes = [0] * 11 + [1] * 11
        item_ids = [*list(range(30, 40)), 0, *list(range(40, 50)), 1]
        data = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=52)
        data.edge_sign = torch.empty((0,), dtype=torch.bfloat16)
        data.train_mask = torch.tensor([True] * 10 + [False] + [True] * 10 + [False])
        data.val_mask = torch.zeros(22, dtype=torch.bool)
        data.test_mask = ~data.train_mask
        data.labels = torch.ones(22, dtype=torch.float32)
        data.user_nodes = torch.tensor(user_nodes, dtype=torch.long)
        data.item_nodes = torch.tensor([item_id + 2 for item_id in item_ids], dtype=torch.long)
        data.n_users = 2
        data.n_items = 50
        data.popularity = torch.arange(50, dtype=torch.float32)

        base = torch.arange(50, dtype=torch.float32).unsqueeze(0).expand(2, -1)
        context = torch.arange(50, dtype=torch.float32).unsqueeze(0)
        model = _ComponentRankingModel(
            interest_scores=base,
            conformity_scores=torch.zeros_like(base),
            context_scores=context,
            score_mix_weights=torch.tensor(
                [[0.75, 0.0, 0.25], [0.75, 0.0, 0.25]],
                dtype=torch.float32,
            ),
            user_interest_emb=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            user_conformity_emb=torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
            expand_context_scores=True,
        )

        metrics = evaluator.evaluate(
            model,
            data,
            data.test_mask,
            batch_size=2,
            include_refined_diagnostics=True,
        )

        self.assertTrue(math.isfinite(metrics["context_contribution@20"]))
        self.assertTrue(math.isfinite(metrics["context_contribution@40"]))
        self.assertGreater(metrics["context_contribution@20"], 0.0)
        self.assertGreater(metrics["context_contribution@40"], 0.0)

    def test_evaluator_branch_ranking_diagnostics_use_raw_branch_scores(self) -> None:
        """Branch ranking diagnostics should evaluate branch-local BPR score surfaces."""
        evaluator = Evaluator(EDGRecConfig(device="cpu"))

        data = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=51)
        data.edge_sign = torch.empty((0,), dtype=torch.bfloat16)
        data.train_mask = torch.tensor([True, False], dtype=torch.bool)
        data.val_mask = torch.tensor([False, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, True], dtype=torch.bool)
        data.user_nodes = torch.tensor([0, 0], dtype=torch.long)
        data.item_nodes = torch.tensor([1, 26], dtype=torch.long)
        data.n_users = 1
        data.n_items = 50
        data.popularity = torch.arange(50, dtype=torch.float32)

        component_scores = torch.arange(50, dtype=torch.float32).unsqueeze(0)
        component_scores[:, 25] = -100.0
        branch_scores = torch.zeros((1, 50), dtype=torch.float32)
        branch_scores[:, 25] = 100.0
        model = _ComponentRankingModel(
            interest_scores=component_scores,
            conformity_scores=torch.zeros_like(component_scores),
            context_scores=torch.zeros_like(component_scores),
            score_mix_weights=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            user_interest_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            user_conformity_emb=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            branch_interest_scores=branch_scores,
        )

        metrics = evaluator.evaluate(
            model,
            data,
            data.test_mask,
            batch_size=1,
            include_refined_diagnostics=True,
        )

        self.assertEqual(metrics["interest_branch_Recall@20"], 1.0)

    def test_evaluator_skips_refined_score_diagnostics_when_disabled(self) -> None:
        """Validation-style evaluation should omit refined diagnostics when disabled."""
        evaluator = Evaluator(EDGRecConfig(device="cpu"))

        data = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=52)
        data.edge_sign = torch.empty((0,), dtype=torch.bfloat16)
        data.train_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
        data.val_mask = torch.tensor([False, False, False, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, True, False, True], dtype=torch.bool)
        data.user_nodes = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        data.item_nodes = torch.tensor([1, 50, 3, 2], dtype=torch.long)
        data.n_users = 2
        data.n_items = 50
        data.popularity = torch.arange(50, dtype=torch.float32)

        base = torch.arange(50, dtype=torch.float32).unsqueeze(0).expand(2, -1)
        model = _ComponentRankingModel(
            interest_scores=-base,
            conformity_scores=base,
            context_scores=base,
            score_mix_weights=torch.tensor(
                [[0.25, 0.25, 0.50], [0.75, 0.10, 0.15]],
                dtype=torch.float32,
            ),
            user_interest_emb=torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
            user_conformity_emb=torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        )

        metrics = evaluator.evaluate(
            model,
            data,
            data.test_mask,
            batch_size=2,
            include_refined_diagnostics=False,
        )

        self.assertIn("NDCG@20", metrics)
        self.assertIn("Recall@40", metrics)
        for key in (
            "score_mix_interest_mean",
            "score_mix_interest_std",
            "score_mix_conformity_mean",
            "score_mix_conformity_std",
            "score_mix_context_mean",
            "score_mix_context_std",
            "interest_contribution@20",
            "interest_contribution@40",
            "conformity_contribution@20",
            "conformity_contribution@40",
            "context_contribution@20",
            "context_contribution@40",
            "interest_conformity_cosine_mean",
            "interest_conformity_cosine_std",
            "interest_popularity_spearman@20",
            "interest_popularity_spearman@40",
            "conformity_popularity_spearman@20",
            "conformity_popularity_spearman@40",
            "context_popularity_spearman@20",
            "context_popularity_spearman@40",
            "final_popularity_spearman@20",
            "final_popularity_spearman@40",
            "interest_branch_NDCG@20",
            "interest_branch_Recall@20",
            "interest_branch_AveragePopularity@20",
            "conformity_branch_NDCG@20",
            "conformity_branch_Recall@20",
            "conformity_branch_AveragePopularity@20",
        ):
            self.assertNotIn(key, metrics)

    def test_evaluator_omits_branch_ranking_diagnostics_for_single_branch_model(self) -> None:
        """Single-branch exporters should not report misleading branch rank metrics."""
        evaluator = Evaluator(EDGRecConfig(device="cpu", use_dual_branch=False))

        data = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=52)
        data.edge_sign = torch.empty((0,), dtype=torch.bfloat16)
        data.train_mask = torch.tensor([True, False], dtype=torch.bool)
        data.val_mask = torch.tensor([False, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, True], dtype=torch.bool)
        data.user_nodes = torch.tensor([0, 0], dtype=torch.long)
        data.item_nodes = torch.tensor([1, 26], dtype=torch.long)
        data.n_users = 1
        data.n_items = 50
        data.popularity = torch.arange(50, dtype=torch.float32)

        scores = torch.arange(50, dtype=torch.float32).unsqueeze(0)
        scores[:, 25] = 100.0
        model = _ComponentRankingModel(
            interest_scores=scores,
            conformity_scores=torch.zeros_like(scores),
            context_scores=torch.zeros_like(scores),
            score_mix_weights=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            user_interest_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            user_conformity_emb=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        )

        metrics = evaluator.evaluate(
            model,
            data,
            data.test_mask,
            batch_size=1,
            include_refined_diagnostics=True,
        )

        self.assertEqual(metrics["Recall@20"], 1.0)
        self.assertNotIn("interest_branch_Recall@20", metrics)
        self.assertNotIn("conformity_branch_Recall@20", metrics)

    def test_evaluator_diagnostics_gather_score_components_before_float_cast(self) -> None:
        """Diagnostics should gather native-dtype top-k slices before float math."""
        pred_index_mat = torch.tensor([[3, 1, 0], [0, 2, 3]], dtype=torch.long)
        popularity = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        interest_scores = torch.tensor(
            [[1.0, 4.0, 2.0, 5.0], [5.0, 2.0, 4.0, 1.0]],
            dtype=torch.bfloat16,
        )

        expected = _EvaluatorDiagnosticsAccumulator((2,))
        expected.update({"interest_score": interest_scores}, pred_index_mat, popularity)

        guarded = _EvaluatorDiagnosticsAccumulator((2,))
        guarded.update(
            {"interest_score": _GatherBeforeFloatGuard(interest_scores)},
            pred_index_mat,
            popularity,
        )

        expected_metrics = expected.compute()
        guarded_metrics = guarded.compute()
        self.assertEqual(set(guarded_metrics), set(expected_metrics))
        for key, value in expected_metrics.items():
            self.assertAlmostEqual(guarded_metrics[key], value, places=6)

    def test_evaluator_batch_budget_accounts_for_component_export_path(self) -> None:
        """Diagnostics-enabled evaluation should budget for all full score matrices."""
        evaluator = Evaluator(EDGRecConfig(device="cpu"))

        self.assertEqual(
            evaluator._effective_eval_batch_size(
                requested_batch_size=2,
                n_items=40_000_000,
                export_score_components=False,
            ),
            2,
        )
        self.assertEqual(
            evaluator._effective_eval_batch_size(
                requested_batch_size=2,
                n_items=40_000_000,
                export_score_components=True,
            ),
            1,
        )

    def test_rowwise_spearman_uses_average_ranks_for_ties(self) -> None:
        """Tie groups should use average ranks rather than stable-order tie breaking."""
        correlation = _rowwise_spearman(
            torch.tensor([[3.0, 3.0, 2.0, 1.0]], dtype=torch.float32),
            torch.tensor([[1.0, 2.0, 2.0, 3.0]], dtype=torch.float32),
        )

        self.assertAlmostEqual(correlation.item(), -5.0 / 6.0, places=6)

    def test_evaluator_diagnostics_zero_weight_rows_have_zero_popularity_spearman(self) -> None:
        """Zero-weighted component rows should contribute zero popularity correlation."""
        accumulator = _EvaluatorDiagnosticsAccumulator((4,))
        accumulator.update(
            {
                "interest_score": torch.tensor(
                    [[5.0, 4.0, 3.0, 2.0], [1.0, 2.0, 3.0, 4.0]],
                    dtype=torch.bfloat16,
                ),
                "score_mix_weights": torch.tensor(
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                ),
            },
            torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
            torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32),
        )

        diagnostics = accumulator.compute()

        self.assertIn("interest_popularity_spearman@4", diagnostics)
        self.assertAlmostEqual(diagnostics["interest_popularity_spearman@4"], 0.0, places=6)

    def test_evaluator_diagnostics_report_only_current_popularity_diagnostics(self) -> None:
        """Component diagnostics should stay limited to the current popularity correlation."""
        accumulator = _EvaluatorDiagnosticsAccumulator((2,))
        accumulator.update(
            {
                "interest_score": torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
                "conformity_score": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                "final_score": torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
            },
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([1.0, 0.8, 0.2, 0.0], dtype=torch.float32),
        )

        diagnostics = accumulator.compute()

        self.assertEqual(
            set(diagnostics),
            {
                "conformity_contribution@2",
                "conformity_popularity_spearman@2",
                "final_popularity_spearman@2",
                "interest_contribution@2",
                "interest_popularity_spearman@2",
            },
        )

    def test_stage_graph_tensors_for_device_keeps_optional_edge_fields(self) -> None:
        """Shared graph staging should preserve optional edge tensors."""
        data = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), num_nodes=2)
        data.edge_sign = torch.tensor([1.0, -1.0], dtype=torch.float32)
        data.edge_norm = torch.tensor([0.5, 0.5], dtype=torch.float32)

        edge_index, edge_sign, edge_norm = stage_graph_tensors_for_device(
            data,
            torch.device("cpu"),
        )

        self.assertTrue(torch.equal(edge_index, data.edge_index))
        assert edge_sign is not None
        assert edge_norm is not None
        self.assertTrue(torch.equal(edge_sign, data.edge_sign))
        self.assertTrue(torch.equal(edge_norm, data.edge_norm))

    def test_val_evaluation_does_not_exclude_test_items_from_pool(self) -> None:
        """Test items must NOT be masked during validation evaluation.

        Excluding test items from the val ranking pool would allow test-set
        knowledge to influence early stopping and best-model selection,
        violating the information barrier between training and test phases.
        """
        evaluator = Evaluator(EDGRecConfig(device="cuda"))

        # 3 interactions: one per split, same user
        data = Data(num_nodes=4)
        data.train_mask = torch.tensor([True, False, False], dtype=torch.bool)
        data.val_mask = torch.tensor([False, True, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, False, True], dtype=torch.bool)

        exclude = evaluator._observed_non_target_mask(data, data.val_mask)

        self.assertTrue(
            exclude[0].item(),
            "train interaction must be excluded from val pool",
        )
        self.assertFalse(
            exclude[1].item(),
            "val interaction (target) must not self-exclude",
        )
        self.assertFalse(
            exclude[2].item(),
            "test interaction must NOT be excluded from val pool",
        )

    def test_test_evaluation_excludes_train_and_val(self) -> None:
        """Both train and val interactions must be excluded from the test ranking pool.

        The test evaluation pool should only contain items the model has never
        encountered during training or been selected against during validation.
        """
        evaluator = Evaluator(EDGRecConfig(device="cuda"))

        data = Data(num_nodes=4)
        data.train_mask = torch.tensor([True, False, False], dtype=torch.bool)
        data.val_mask = torch.tensor([False, True, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, False, True], dtype=torch.bool)

        exclude = evaluator._observed_non_target_mask(data, data.test_mask)

        self.assertTrue(
            exclude[0].item(),
            "train interaction must be excluded from test pool",
        )
        self.assertTrue(
            exclude[1].item(),
            "val interaction must be excluded from test pool",
        )
        self.assertFalse(
            exclude[2].item(),
            "test interaction (target) must not self-exclude",
        )

    def test_test_evaluation_excludes_val_for_copied_test_mask(self) -> None:
        """Equivalent test masks must keep the same train+val exclusion contract."""
        evaluator = Evaluator(EDGRecConfig(device="cuda"))

        data = Data(num_nodes=4)
        data.train_mask = torch.tensor([True, False, False], dtype=torch.bool)
        data.val_mask = torch.tensor([False, True, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, False, True], dtype=torch.bool)

        exclude = evaluator._observed_non_target_mask(data, data.test_mask.clone())

        self.assertTrue(
            exclude[0].item(),
            "train interaction must be excluded from test pool",
        )
        self.assertTrue(
            exclude[1].item(),
            "val interaction must be excluded from copied test pool",
        )
        self.assertFalse(
            exclude[2].item(),
            "test interaction (target) must not self-exclude",
        )

    def test_overlapping_split_masks_raise(self) -> None:
        """get_splits must raise if train and test masks share any index."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0], dtype=np.int64),
            item_id=np.array([0, 1], dtype=np.int64),
            label=np.ones(2, dtype=np.float32),
            timestamp=np.array([1, 2], dtype=np.int64),
            sign=np.zeros(2, dtype=np.float32),
            popularity=np.ones(2, dtype=np.float32),
            n_users=1,
            n_items=2,
            user_map={0: 0},
            item_map={0: 0, 1: 1},
            # Row 0 appears in both train and test — intentional bad data
            train_mask=np.array([True, False]),
            val_mask=np.array([False, False]),
            test_mask=np.array([True, True]),
        )
        with self.assertRaises(
            ValueError,
            msg="Overlapping train/test masks must raise ValueError",
        ):
            canonical.get_splits()

    def test_graph_and_training_batches_use_only_positive_train_labels(self) -> None:
        """Graph edges and BPR positives should ignore train rows with label 0."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0, 1], dtype=np.int64),
            item_id=np.array([0, 1, 2], dtype=np.int64),
            label=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            timestamp=np.array([1, 2, 3], dtype=np.int64),
            sign=np.array([1.0, -1.0, 1.0], dtype=np.float32),
            popularity=np.ones(3, dtype=np.float32),
            n_users=2,
            n_items=3,
            user_map={0: 0, 1: 1},
            item_map={0: 0, 1: 1, 2: 2},
            train_mask=np.array([True, True, True]),
            val_mask=np.array([False, False, False]),
            test_mask=np.array([False, False, False]),
        )
        data = build_graph(canonical, EDGRecConfig(device="cpu"))

        expected_positive_mask = torch.tensor([True, False, True], dtype=torch.bool)
        self.assertTrue(torch.equal(data.train_positive_mask, expected_positive_mask))
        self.assertFalse(torch.any(data.edge_index == 3))
        torch.testing.assert_close(
            data.popularity,
            torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32),
        )

        runtime = object.__new__(TrainerRuntime)
        runtime.data = data
        train_users, train_items = TrainerRuntime._get_train_interactions(runtime)

        self.assertTrue(torch.equal(train_users, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(train_items, torch.tensor([0, 2], dtype=torch.long)))


class CausalTrainingContractTests(unittest.TestCase):
    """Pin the intended dual-branch scoring and sign-aware behavior."""

    @staticmethod
    def _build_dual_branch_config() -> EDGRecConfig:
        """Return a small CUDA-tagged config suitable for unit tests."""
        config = EDGRecConfig(device="cuda")
        config.use_torch_compile = False
        config.use_ipw = False
        config.use_dual_branch = True
        return config

    @staticmethod
    def _patch_user_dependent_score_mix(model: EDGRec) -> None:
        """Make the learned score-mix head depend on user embeddings."""
        with torch.no_grad():
            first_layer = model.scoring.alpha_mlp[0]
            second_layer = model.scoring.alpha_mlp[2]
            first_layer.weight.zero_()
            first_layer.bias.zero_()
            first_layer.weight[0, 0] = 1.0
            first_layer.weight[1, model.config.embed_dim] = 1.0
            second_layer.weight.zero_()
            second_layer.bias.zero_()
            second_layer.weight[0, 0] = 5.0
            second_layer.weight[1, 1] = 5.0

    @staticmethod
    def _tiny_dual_branch_scoring_config(*, separate_items: bool = False) -> EDGRecConfig:
        """Return a tiny deterministic dual-branch scorer config."""
        config = EDGRecConfig(device="cpu", embed_dim=4)
        config.use_dual_branch = True
        config.use_features = False
        config.use_popularity_head = False
        config.use_learned_score_mix = False
        config.score_weight_interest = 0.5
        config.score_weight_conformity = 0.5
        config.score_weight_popularity = 0.0
        config.score_mix_min_weight = 0.0
        config.interest_gnn_layers = 1
        config.conformity_gnn_layers = 1
        config.num_neighbors = [8]
        config.separate_item_branch_embeddings = separate_items
        return config

    @staticmethod
    def _loss_suite_model_output() -> tuple[
        dict[str, torch.Tensor | dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return a finite synthetic LossSuite payload with active auxiliaries."""
        user_interest = torch.tensor(
            [[0.1, 0.4], [0.2, 0.3], [0.3, 0.2]],
            dtype=torch.float32,
        )
        user_conformity = torch.tensor(
            [[0.4, 0.1], [0.3, 0.2], [0.2, 0.3]],
            dtype=torch.float32,
        )
        item_interest = torch.tensor(
            [[0.5, 0.1], [0.1, 0.5], [0.3, 0.3], [0.2, 0.4]],
            dtype=torch.float32,
        )
        item_conformity = torch.tensor(
            [[0.2, 0.4], [0.4, 0.2], [0.3, 0.1], [0.5, 0.2]],
            dtype=torch.float32,
        )
        pos_item_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        neg_item_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        pos_scores = {
            "final_score": torch.tensor([0.8, 0.7, 0.6], dtype=torch.float32),
            "interest_score": torch.tensor([0.7, 0.6, 0.5], dtype=torch.float32),
            "conformity_score": torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32),
            "context_score": torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32),
            "branch_interest_score": torch.tensor([0.75, 0.65, 0.55], dtype=torch.float32),
            "branch_conformity_score": torch.tensor([0.45, 0.55, 0.65], dtype=torch.float32),
            "raw_context_score": torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32),
            "score_mix_weights": torch.tensor(
                [[0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.4, 0.4, 0.2]],
                dtype=torch.float32,
            ),
        }
        neg_scores = {
            "final_score": torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32),
            "interest_score": torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32),
            "conformity_score": torch.tensor([0.6, 0.5, 0.4], dtype=torch.float32),
            "context_score": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            "branch_interest_score": torch.tensor([0.25, 0.35, 0.45], dtype=torch.float32),
            "branch_conformity_score": torch.tensor([0.65, 0.55, 0.45], dtype=torch.float32),
        }
        propagated = {
            "user_interest": user_interest,
            "user_conformity": user_conformity,
            "item_interest": item_interest,
            "item_conformity": item_conformity,
        }
        model_output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "propagated": propagated,
            "embeddings": {"user": user_interest, "item": item_interest},
            "ipw_weights": torch.ones(3, dtype=torch.float32),
            "loss_user_ids": torch.tensor([0, 1, 2], dtype=torch.long),
            "loss_neg_item_ids": neg_item_ids,
            "propensity_scores": torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32),
        }
        item_popularity = torch.tensor([0.1, 0.3, 0.6, 0.9], dtype=torch.float32)
        propensity_targets = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32)
        return model_output, item_popularity, pos_item_ids, propensity_targets

    def test_presets_pin_expected_scoring_contracts(self) -> None:
        """Preset helpers should expose the thesis scoring contract directly."""
        lightgcn = EDGRecConfig(device="cuda").preset_lightgcn()
        dice_like = EDGRecConfig(device="cuda").preset_dice_like()
        edgrec = EDGRecConfig(device="cuda").preset_full()

        self.assertFalse(hasattr(lightgcn, "train_scoring_mode"))
        self.assertFalse(hasattr(lightgcn, "eval_scoring_mode"))
        self.assertFalse(lightgcn.use_features)
        self.assertFalse(dice_like.use_sign_aware)
        self.assertFalse(dice_like.use_features)
        self.assertFalse(hasattr(dice_like, "scoring_weight_mode"))
        self.assertFalse(hasattr(dice_like, "train_scoring_mode"))
        self.assertFalse(hasattr(dice_like, "eval_scoring_mode"))
        self.assertFalse(hasattr(edgrec, "train_scoring_mode"))
        self.assertFalse(hasattr(edgrec, "eval_scoring_mode"))
        self.assertFalse(edgrec.use_ipw)

    def test_dual_branch_shared_item_embeddings_forward_and_full_catalog_shape(self) -> None:
        """Shared item embeddings should keep the default branch-scoring path intact."""
        config = self._tiny_dual_branch_scoring_config(separate_items=False)
        model = EDGRec(n_users=2, n_items=3, config=config)
        edge_index = torch.tensor([[0, 2, 1, 3], [2, 0, 3, 1]], dtype=torch.long)
        user_ids = torch.tensor([0, 1], dtype=torch.long)
        pos_item_ids = torch.tensor([0, 1], dtype=torch.long)
        neg_item_ids = torch.tensor([1, 2], dtype=torch.long)

        embeddings = model.embedding.get_embeddings()
        output = model(edge_index, user_ids, pos_item_ids, neg_item_ids)
        propagated = output["propagated"]
        assert isinstance(propagated, dict)
        scores = model.score_users_from_propagated(propagated, user_ids)

        self.assertFalse(config.separate_item_branch_embeddings)
        self.assertEqual(tuple(embeddings["item"].shape), (3, 4))
        self.assertNotIn("item_interest", embeddings)
        self.assertEqual(tuple(propagated["item"].shape), (3, 4))
        self.assertEqual(tuple(propagated["item_interest"].shape), (3, 4))
        self.assertEqual(tuple(propagated["item_conformity"].shape), (3, 4))
        self.assertEqual(tuple(scores.shape), (2, 3))

    def test_dual_branch_separate_item_embeddings_forward_and_full_catalog_shape(self) -> None:
        """Separate item branch embeddings should expose explicit branch tensors."""
        config = self._tiny_dual_branch_scoring_config(separate_items=True)
        model = EDGRec(n_users=2, n_items=3, config=config)
        edge_index = torch.tensor([[0, 2, 1, 3], [2, 0, 3, 1]], dtype=torch.long)
        user_ids = torch.tensor([0, 1], dtype=torch.long)
        pos_item_ids = torch.tensor([0, 1], dtype=torch.long)
        neg_item_ids = torch.tensor([1, 2], dtype=torch.long)

        embeddings = model.embedding.get_embeddings()
        output = model(edge_index, user_ids, pos_item_ids, neg_item_ids)
        propagated = output["propagated"]
        assert isinstance(propagated, dict)
        scores = model.score_users_from_propagated(propagated, user_ids)

        self.assertTrue(config.separate_item_branch_embeddings)
        self.assertEqual(tuple(embeddings["item"].shape), (3, 4))
        self.assertEqual(tuple(embeddings["item_interest"].shape), (3, 4))
        self.assertEqual(tuple(embeddings["item_conformity"].shape), (3, 4))
        self.assertEqual(tuple(model.embedding.get_stacked_embeddings().shape), (5, 4))
        self.assertEqual(tuple(propagated["item"].shape), (3, 4))
        self.assertEqual(tuple(propagated["item_interest"].shape), (3, 4))
        self.assertEqual(tuple(propagated["item_conformity"].shape), (3, 4))
        self.assertEqual(tuple(scores.shape), (2, 3))

    def test_loss_normalization_none_preserves_raw_weighted_total(self) -> None:
        """The default loss path should keep the historical weighted raw formula."""
        config = EDGRecConfig(device="cpu", embed_dim=2)
        config.use_dual_branch = True
        config.branch_loss_mode = "symmetric_bpr"
        config.loss_weight_interest_bpr = 0.2
        config.loss_weight_conformity_bpr = 0.3
        config.loss_weight_independence = 0.4
        config.loss_weight_contrastive = 0.0
        config.loss_weight_align = 0.0
        config.loss_weight_uniform = 0.0
        config.loss_weight_popularity = 0.0
        config.loss_weight_propensity_calibration = 0.0
        config.loss_normalization = "none"
        config.auxiliary_losses_start_epoch = 0
        config.popularity_supervision_start_epoch = 0
        model_output, item_popularity, pos_item_ids, propensity_targets = (
            self._loss_suite_model_output()
        )

        losses = LossSuite(config)(
            model_output,
            item_popularity=item_popularity,
            pos_item_ids=pos_item_ids,
            epoch=1,
            propensity_targets=propensity_targets,
        )

        expected = (
            config.loss_weight_recommendation * losses["rec"]
            + config.loss_weight_interest_bpr * losses["interest_bpr"]
            + config.loss_weight_conformity_bpr * losses["conformity_bpr"]
            + config.loss_weight_independence * losses["independence"]
        )
        torch.testing.assert_close(losses["total"], expected)
        torch.testing.assert_close(losses["raw_rec_loss"], losses["rec"])
        self.assertNotIn("normalized_interest_bpr", losses)

    def test_loss_normalization_ema_aux_logs_normalized_losses_without_eval_update(
        self,
    ) -> None:
        """EMA normalization should affect auxiliaries only and freeze in eval."""
        config = EDGRecConfig(
            device="cpu",
            embed_dim=2,
            use_ipw=True,
            loss_weight_propensity_calibration=0.1,
        )
        config.use_dual_branch = True
        config.branch_loss_mode = "symmetric_bpr"
        config.use_popularity_head = True
        config.loss_weight_interest_bpr = 0.2
        config.loss_weight_conformity_bpr = 0.3
        config.loss_weight_independence = 0.4
        config.loss_weight_contrastive = 0.1
        config.loss_weight_align = 0.1
        config.loss_weight_uniform = 0.1
        config.loss_weight_popularity = 0.1
        config.loss_normalization = "ema_aux"
        config.auxiliary_losses_start_epoch = 0
        config.popularity_supervision_start_epoch = 0
        model_output, item_popularity, pos_item_ids, propensity_targets = (
            self._loss_suite_model_output()
        )
        loss_suite = LossSuite(config)

        losses = loss_suite(
            model_output,
            item_popularity=item_popularity,
            pos_item_ids=pos_item_ids,
            epoch=1,
            propensity_targets=propensity_targets,
        )
        ema_after_train = loss_suite._aux_loss_ema.detach().clone()
        loss_suite.eval()
        eval_losses = loss_suite(
            model_output,
            item_popularity=item_popularity,
            pos_item_ids=pos_item_ids,
            epoch=1,
            propensity_targets=propensity_targets,
        )

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.isfinite(eval_losses["total"]))
        self.assertIn("normalized_interest_bpr", losses)
        self.assertIn("normalized_propensity_calibration", losses)
        self.assertIn("weighted_interest_bpr", losses)
        torch.testing.assert_close(losses["raw_rec_loss"], losses["rec"])
        torch.testing.assert_close(loss_suite._aux_loss_ema, ema_after_train)

    def test_ipw_requires_calibrated_propensity_objective(self) -> None:
        """IPW should not use random unsupervised propensity estimates."""
        with self.assertRaisesRegex(ValueError, "use_ipw requires"):
            EDGRecConfig(use_ipw=True, loss_weight_propensity_calibration=0.0)

        config = EDGRecConfig(
            use_ipw=True,
            loss_weight_propensity_calibration=0.1,
        )

        self.assertTrue(config.use_ipw)

    def test_presets_reset_preset_owned_fields_when_switched(self) -> None:
        """Switching presets on one config should not preserve stale preset state."""
        edgrec = EDGRecConfig(device="cuda").preset_dice_like().preset_full()
        baseline = EDGRecConfig(device="cuda").preset_full().preset_dice_like()

        self.assertTrue(edgrec.use_features)
        self.assertFalse(hasattr(edgrec, "scoring_weight_mode"))
        self.assertEqual(edgrec.auxiliary_losses_start_epoch, 15)
        self.assertEqual(edgrec.popularity_supervision_start_epoch, 30)
        self.assertEqual(edgrec.propensity_clip_min, 0.1)

        self.assertFalse(baseline.use_ipw)
        self.assertFalse(baseline.use_features)
        self.assertEqual(baseline.propensity_clip_min, 0.01)
        self.assertEqual(baseline.auxiliary_losses_start_epoch, 0)
        self.assertEqual(baseline.popularity_supervision_start_epoch, 0)

    def test_short_term_interest_uses_recent_train_items_when_temporal_order_exists(self) -> None:
        """The refined scorer should derive short-term interest from recent train items."""
        config = EDGRecConfig(device="cpu", embed_dim=2)
        config.use_dual_branch = True
        config.use_ipw = False
        config.use_learned_score_mix = False
        config.score_weight_interest = 1.0
        config.score_weight_conformity = 0.0
        config.score_weight_popularity = 0.0
        model = EDGRec(
            n_users=1,
            n_items=2,
            config=config,
            recent_train_items=torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long),
            recent_train_mask=torch.tensor(
                [[True, False, False, False, False, False, False, False, False, False]],
                dtype=torch.bool,
            ),
        )

        with torch.no_grad():
            model.scoring.interest_gate_mlp[0].weight.zero_()
            model.scoring.interest_gate_mlp[0].bias.zero_()
            model.scoring.interest_gate_mlp[2].weight.zero_()
            model.scoring.interest_gate_mlp[2].bias.fill_(10.0)

        propagated = {
            "user_interest": torch.tensor([[1.0, 0.0]]),
            "item_interest": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "user_conformity": torch.tensor([[1.0, 0.0]]),
            "item_conformity": torch.zeros(2, 2),
        }

        scores = model.score_users_from_propagated(
            propagated,
            user_ids=torch.tensor([0]),
        )
        self.assertTrue(
            torch.allclose(
                scores,
                torch.tensor([[0.0, 1.0]]),
                atol=1e-4,
            ),
        )

    def test_forward_subgraph_uses_global_recent_train_items_without_local_index_errors(
        self,
    ) -> None:
        """Subgraph scoring should not index global history items into local item tables."""
        config = EDGRecConfig(device="cpu", embed_dim=2).preset_full()
        config.use_ipw = False
        model = EDGRec(
            n_users=1,
            n_items=3,
            config=config,
            recent_train_items=torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long),
            recent_train_mask=torch.tensor(
                [[True, False, False, False, False, False, False, False, False, False]],
                dtype=torch.bool,
            ),
        )

        sub_batch = SubgraphBatch(
            sub_edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long),
            sub_edge_sign=None,
            sub_edge_norm=None,
            user_global_ids=torch.tensor([0], dtype=torch.long),
            item_global_ids=torch.tensor([0, 1], dtype=torch.long),
            n_sub_users=1,
            n_sub_items=2,
            batch_user_local=torch.tensor([0], dtype=torch.long),
            batch_pos_local=torch.tensor([0], dtype=torch.long),
            batch_neg_local=torch.tensor([1], dtype=torch.long),
        )

        output = model.forward_subgraph(sub_batch)

        pos_scores = output["pos_scores"]
        neg_scores = output["neg_scores"]
        assert isinstance(pos_scores, dict)
        assert isinstance(neg_scores, dict)
        self.assertIn("final_score", pos_scores)
        self.assertIn("final_score", neg_scores)
        self.assertEqual(tuple(pos_scores["final_score"].shape), (1,))
        self.assertEqual(tuple(neg_scores["final_score"].shape), (1,))

    def test_model_parameter_count_does_not_require_context_head_lazy_init(self) -> None:
        """Model parameter counting should work before any scorer forward pass."""
        model = EDGRec(
            n_users=1,
            n_items=2,
            config=EDGRecConfig(device="cpu", embed_dim=4).preset_full(),
        )

        n_params = sum(parameter.numel() for parameter in model.parameters())

        self.assertGreater(n_params, 0)

    def test_bpr_loss_is_scale_invariant_under_ipw_weights(self) -> None:
        """IPW reweighting should not change the loss scale under uniform scaling."""
        pos_scores = torch.tensor([2.0, 0.5, -1.0], dtype=torch.float32)
        neg_scores = torch.zeros(3, dtype=torch.float32)
        weights = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)

        base_loss = _bpr_loss(pos_scores, neg_scores, weights)
        scaled_loss = _bpr_loss(pos_scores, neg_scores, weights * 2.0)

        self.assertAlmostEqual(base_loss.item(), scaled_loss.item(), places=6)

    def test_bpr_loss_detaches_ipw_weights_from_autograd(self) -> None:
        """Ranking loss should not backpropagate into the IPW sample weights."""
        pos_scores = torch.tensor([2.0, 0.5, -1.0], dtype=torch.float32, requires_grad=True)
        neg_scores = torch.zeros(3, dtype=torch.float32, requires_grad=True)
        weights = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32, requires_grad=True)

        loss = _bpr_loss(pos_scores, neg_scores, weights)
        loss.backward()

        self.assertIsNone(weights.grad)
        self.assertIsNotNone(pos_scores.grad)
        self.assertIsNotNone(neg_scores.grad)

    def test_runtime_model_converts_numpy_propensity_targets_to_tensor(self) -> None:
        """Runtime model construction should convert canonical propensity arrays to tensors."""
        canonical = CanonicalInteractions(
            user_id=np.array([0], dtype=np.int64),
            item_id=np.array([0], dtype=np.int64),
            label=np.ones(1, dtype=np.float32),
            timestamp=np.array([1], dtype=np.int64),
            sign=np.ones(1, dtype=np.float32),
            popularity=np.ones(1, dtype=np.float32),
            n_users=1,
            n_items=1,
            user_map={0: 0},
            item_map={0: 0},
            train_mask=np.array([True]),
            val_mask=np.array([False]),
            test_mask=np.array([False]),
            item_propensity_targets=np.array([0.75], dtype=np.float32),
        )
        config = self._build_dual_branch_config()

        model = build_runtime_model(config, canonical, build_graph(canonical, config))

        self.assertTrue(
            torch.equal(
                model.embedding.item_propensity_targets,
                torch.tensor([0.75], dtype=torch.bfloat16),
            ),
        )

    def test_runtime_model_recent_train_history_is_split_safe(self) -> None:
        """Held-out interactions should not change recent training history buffers."""
        canonical_a = CanonicalInteractions(
            user_id=np.array([0, 0, 0, 1], dtype=np.int64),
            item_id=np.array([0, 1, 2, 3], dtype=np.int64),
            label=np.ones(4, dtype=np.float32),
            timestamp=np.array([1, 2, 50, 60], dtype=np.int64),
            sign=np.ones(4, dtype=np.float32),
            popularity=np.zeros(4, dtype=np.float32),
            n_users=2,
            n_items=4,
            user_map={0: 0, 1: 1},
            item_map={0: 0, 1: 1, 2: 2, 3: 3},
            train_mask=np.array([True, True, False, False]),
            val_mask=np.array([False, False, True, False]),
            test_mask=np.array([False, False, False, True]),
        )
        canonical_b = CanonicalInteractions(
            user_id=np.array([0, 0, 0, 1], dtype=np.int64),
            item_id=np.array([0, 1, 3, 2], dtype=np.int64),
            label=np.ones(4, dtype=np.float32),
            timestamp=np.array([1, 2, 500, 600], dtype=np.int64),
            sign=np.ones(4, dtype=np.float32),
            popularity=np.zeros(4, dtype=np.float32),
            n_users=2,
            n_items=4,
            user_map={0: 0, 1: 1},
            item_map={0: 0, 1: 1, 2: 2, 3: 3},
            train_mask=np.array([True, True, False, False]),
            val_mask=np.array([False, False, True, False]),
            test_mask=np.array([False, False, False, True]),
        )
        config = self._build_dual_branch_config()
        model_a = build_runtime_model(config, canonical_a, build_graph(canonical_a, config))
        model_b = build_runtime_model(config, canonical_b, build_graph(canonical_b, config))

        expected_items = torch.tensor(
            [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.long,
        )
        expected_mask = torch.tensor(
            [[True, True, False, False, False, False, False, False, False, False], [False] * 10],
            dtype=torch.bool,
        )

        self.assertTrue(torch.equal(model_a.embedding.recent_train_items, expected_items))
        self.assertTrue(torch.equal(model_a.embedding.recent_train_mask, expected_mask))
        self.assertTrue(
            torch.equal(
                model_a.embedding.recent_train_items,
                model_b.embedding.recent_train_items,
            ),
        )
        self.assertTrue(
            torch.equal(
                model_a.embedding.recent_train_mask,
                model_b.embedding.recent_train_mask,
            ),
        )

    def test_scoring_exports_context_score_mix_weights_and_item_only_context(self) -> None:
        """The refined scorer should expose context diagnostics and keep context item-only."""
        config = EDGRecConfig(device="cpu", embed_dim=2).preset_full()
        model = EDGRec(
            n_users=1,
            n_items=2,
            config=config,
            item_popularity=torch.tensor([0.4, 0.4]),
            item_recency=torch.tensor([0.7, 0.7]),
            item_propensity_targets=torch.tensor([0.1, 0.1]),
            recent_train_items=torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long),
            recent_train_mask=torch.tensor(
                [[False, False, False, False, False, False, False, False, False, False]],
                dtype=torch.bool,
            ),
        )
        propagated = {
            "user_interest": torch.tensor([[1.0, 0.0]]),
            "item_interest": torch.tensor([[2.0, 0.0], [1.0, 0.0]]),
            "user_conformity": torch.tensor([[0.0, 1.0]]),
            "item_conformity": torch.tensor([[0.0, 2.0], [0.0, 1.0]]),
            "item_popularity": torch.tensor([0.4, 0.4]),
            "item_recency": torch.tensor([0.7, 0.7]),
            "item_propensity_targets": torch.tensor([0.1, 0.1]),
        }

        scores = model.scoring.score_all_items(
            propagated=propagated,
            user_ids=torch.tensor([0]),
        )

        self.assertIn("context_score", scores)
        self.assertIn("score_mix_weights", scores)
        self.assertIn("interest_score", scores)
        self.assertIn("conformity_score", scores)
        self.assertEqual(tuple(scores["score_mix_weights"].shape), (1, 3))
        self.assertAlmostEqual(scores["score_mix_weights"].sum().item(), 1.0, places=6)
        self.assertAlmostEqual(
            scores["context_score"][0, 0].item(),
            scores["context_score"][0, 1].item(),
            places=6,
        )

    def test_default_context_scorer_ignores_uncalibrated_propensity_targets(self) -> None:
        """Post-treatment exposure proxies should not affect default recommendations."""
        config = EDGRecConfig(device="cpu", embed_dim=2).preset_full()
        config.loss_weight_propensity_calibration = 0.0
        model = EDGRec(
            n_users=1,
            n_items=2,
            config=config,
            item_popularity=torch.tensor([0.5, 0.5]),
            item_recency=torch.tensor([0.3, 0.3]),
            item_propensity_targets=torch.tensor([0.0, 1.0]),
        )
        with torch.no_grad():
            first_layer = model.scoring.context_head[0]
            second_layer = model.scoring.context_head[2]
            first_layer.weight.zero_()
            first_layer.bias.zero_()
            first_layer.weight[0, 2] = 10.0
            second_layer.weight.zero_()
            second_layer.bias.zero_()
            second_layer.weight[0, 0] = 1.0
        propagated = {
            "user_interest": torch.tensor([[1.0, 0.0]]),
            "item_interest": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "user_conformity": torch.tensor([[0.0, 1.0]]),
            "item_conformity": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            "item_popularity": torch.tensor([0.5, 0.5]),
            "item_recency": torch.tensor([0.3, 0.3]),
            "item_propensity_targets": torch.tensor([0.0, 1.0]),
        }

        scores = model.scoring.score_all_items(
            propagated=propagated,
            user_ids=torch.tensor([0]),
        )

        self.assertAlmostEqual(
            scores["context_score"][0, 0].item(),
            scores["context_score"][0, 1].item(),
            places=6,
        )

    def test_uncalibrated_propensity_targets_do_not_activate_context_mix(self) -> None:
        """Ignored exposure proxies should not reserve score-mix mass for context."""
        config = EDGRecConfig(device="cpu", embed_dim=2).preset_full()
        config.score_mix_min_weight = 0.1
        model = EDGRec(
            n_users=1,
            n_items=2,
            config=config,
            item_propensity_targets=torch.tensor([0.0, 1.0]),
        )
        propagated = {
            "user_interest": torch.tensor([[1.0, 0.0]]),
            "item_interest": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "user_conformity": torch.tensor([[0.0, 1.0]]),
            "item_conformity": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            "item_propensity_targets": torch.tensor([0.0, 1.0]),
        }

        scores = model.scoring.score_all_items(
            propagated=propagated,
            user_ids=torch.tensor([0]),
        )

        self.assertTrue(torch.allclose(scores["context_score"], torch.zeros((1, 2))))
        self.assertEqual(scores["score_mix_weights"][0, 2].item(), 0.0)
        self.assertGreaterEqual(scores["score_mix_weights"][0, 1].item(), 0.1)

    def test_calibrated_context_scorer_can_use_propensity_targets(self) -> None:
        """Explicit propensity calibration can opt into the exposure-proxy context slot."""
        config = EDGRecConfig(device="cpu", embed_dim=2).preset_full()
        config.use_ipw = True
        config.loss_weight_propensity_calibration = 0.1
        model = EDGRec(
            n_users=1,
            n_items=2,
            config=config,
            item_popularity=torch.tensor([0.5, 0.5]),
            item_recency=torch.tensor([0.3, 0.3]),
            item_propensity_targets=torch.tensor([0.0, 1.0]),
        )
        with torch.no_grad():
            first_layer = model.scoring.context_head[0]
            second_layer = model.scoring.context_head[2]
            first_layer.weight.zero_()
            first_layer.bias.zero_()
            first_layer.weight[0, 2] = 10.0
            second_layer.weight.zero_()
            second_layer.bias.zero_()
            second_layer.weight[0, 0] = 1.0
        propagated = {
            "user_interest": torch.tensor([[1.0, 0.0]]),
            "item_interest": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "user_conformity": torch.tensor([[0.0, 1.0]]),
            "item_conformity": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            "item_popularity": torch.tensor([0.5, 0.5]),
            "item_recency": torch.tensor([0.3, 0.3]),
            "item_propensity_targets": torch.tensor([0.0, 1.0]),
        }

        scores = model.scoring.score_all_items(
            propagated=propagated,
            user_ids=torch.tensor([0]),
        )

        self.assertGreater(
            scores["context_score"][0, 1].item(),
            scores["context_score"][0, 0].item(),
        )

    def test_baseline_presets_keep_fixed_score_mix_weights_while_edgrec_learns_them(
        self,
    ) -> None:
        """Baseline presets should keep preset-owned fixed mixing while EDGRec stays learned."""
        user_ids = torch.tensor([0, 1], dtype=torch.long)

        lightgcn = EDGRec(
            n_users=2,
            n_items=2,
            config=EDGRecConfig(device="cpu", embed_dim=2).preset_lightgcn(),
        )
        lightgcn_scores = lightgcn.scoring.score_all_items(
            propagated={
                "user": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "item": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            },
            user_ids=user_ids,
        )
        self.assertTrue(
            torch.equal(
                lightgcn_scores["score_mix_weights"],
                torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            ),
        )

        dice_like = EDGRec(
            n_users=2,
            n_items=2,
            config=EDGRecConfig(device="cpu", embed_dim=2).preset_dice_like(),
        )
        self._patch_user_dependent_score_mix(dice_like)
        dice_scores = dice_like.scoring.score_all_items(
            propagated={
                "user_interest": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "item_interest": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "user_conformity": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                "item_conformity": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            },
            user_ids=user_ids,
        )
        self.assertTrue(
            torch.allclose(
                dice_scores["score_mix_weights"],
                torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]),
                atol=1e-6,
            ),
        )

        edgrec = EDGRec(
            n_users=2,
            n_items=2,
            config=EDGRecConfig(device="cpu", embed_dim=2).preset_full(),
            item_popularity=torch.tensor([0.4, 0.7]),
            item_recency=torch.tensor([0.2, 0.9]),
            item_propensity_targets=torch.tensor([0.1, 0.3]),
        )
        self._patch_user_dependent_score_mix(edgrec)
        edgrec_scores = edgrec.scoring.score_all_items(
            propagated={
                "user_interest": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "item_interest": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "user_conformity": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                "item_conformity": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                "item_popularity": torch.tensor([0.4, 0.7]),
                "item_recency": torch.tensor([0.2, 0.9]),
                "item_propensity_targets": torch.tensor([0.1, 0.3]),
            },
            user_ids=user_ids,
        )
        self.assertFalse(
            torch.allclose(
                edgrec_scores["score_mix_weights"][0],
                edgrec_scores["score_mix_weights"][1],
                atol=1e-6,
            ),
        )

    def test_no_popularity_head_ablation_keeps_learned_user_specific_non_context_mix(self) -> None:
        """Removing the context head must not collapse learned interest-conformity mixing."""
        config = EDGRecConfig(device="cpu", embed_dim=2).preset_full()
        config.use_popularity_head = False
        config.score_weight_popularity = 0.0
        model = EDGRec(
            n_users=2,
            n_items=2,
            config=config,
        )
        self._patch_user_dependent_score_mix(model)

        scores = model.scoring.score_all_items(
            propagated={
                "user_interest": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "item_interest": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "user_conformity": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                "item_conformity": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            },
            user_ids=torch.tensor([0, 1], dtype=torch.long),
        )

        self.assertTrue(
            torch.allclose(
                scores["score_mix_weights"][:, 2],
                torch.zeros(2),
                atol=1e-6,
            ),
        )
        self.assertFalse(
            torch.allclose(
                scores["score_mix_weights"][0, :2],
                scores["score_mix_weights"][1, :2],
                atol=1e-6,
            ),
        )

    def test_edgrec_score_mix_floor_uses_component_availability(self) -> None:
        """Available causal components should keep floor mass even when scores are zero."""
        config = EDGRecConfig(device="cpu", embed_dim=2).preset_full()
        config.score_mix_min_weight = 0.1
        model = EDGRec(
            n_users=1,
            n_items=1,
            config=config,
            item_popularity=torch.zeros(1),
            item_recency=torch.zeros(1),
            item_propensity_targets=torch.zeros(1),
        )
        with torch.no_grad():
            first_layer = model.scoring.alpha_mlp[0]
            second_layer = model.scoring.alpha_mlp[2]
            first_layer.weight.zero_()
            first_layer.bias.zero_()
            second_layer.weight.zero_()
            second_layer.bias.copy_(torch.tensor([20.0, -20.0, -20.0]))
            if model.scoring.context_head is not None:
                for parameter in model.scoring.context_head.parameters():
                    parameter.zero_()

        scores = model.scoring.score_all_items(
            propagated={
                "user_interest": torch.tensor([[1.0, 0.0]]),
                "item_interest": torch.tensor([[1.0, 0.0]]),
                "user_conformity": torch.zeros(1, 2),
                "item_conformity": torch.zeros(1, 2),
                "item_popularity": torch.zeros(1),
                "item_recency": torch.zeros(1),
                "item_propensity_targets": torch.zeros(1),
            },
            user_ids=torch.tensor([0], dtype=torch.long),
        )

        expected_floor = torch.tensor(0.1)
        self.assertGreaterEqual(scores["score_mix_weights"][0, 1], expected_floor)
        self.assertGreaterEqual(scores["score_mix_weights"][0, 2], expected_floor)
        self.assertAlmostEqual(scores["score_mix_weights"].sum().item(), 1.0, places=6)

    def test_edgrec_score_mix_is_not_overridden_by_raw_conformity_scale(self) -> None:
        """Small conformity weight should not beat interest via raw embedding norm."""
        config = EDGRecConfig(device="cpu", embed_dim=2).preset_full()
        config.use_learned_score_mix = False
        config.use_popularity_head = False
        config.score_mix_min_weight = 0.0
        config.score_weight_interest = 0.95
        config.score_weight_conformity = 0.05
        config.score_weight_popularity = 0.0
        model = EDGRec(n_users=1, n_items=2, config=config)

        scores = model.scoring.score_all_items(
            propagated={
                "user_interest": torch.tensor([[1.0, 0.0]]),
                "item_interest": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "user_conformity": torch.tensor([[100.0, 0.0]]),
                "item_conformity": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            },
            user_ids=torch.tensor([0], dtype=torch.long),
        )

        self.assertGreater(
            scores["final_score"][0, 0].item(),
            scores["final_score"][0, 1].item(),
        )

    def test_embedding_feature_projection_accepts_cpu_fallback_inputs(self) -> None:
        """CPU fallback paths should project bf16 feature buffers without dtype errors."""
        config = EDGRecConfig(device="cpu", embed_dim=4).preset_full()
        model = EDGRec(
            n_users=1,
            n_items=2,
            config=config,
            item_features=torch.tensor(
                [[1000.0, 0.0, -5.0], [0.0, 2000.0, 5.0]],
                dtype=torch.float32,
            ),
            item_popularity=torch.tensor([0.2, 0.8], dtype=torch.float32),
            item_recency=torch.tensor([0.1, 0.9], dtype=torch.float32),
        )

        embeddings = model.embedding.get_embeddings()

        self.assertEqual(embeddings["item_interest"].dtype, torch.float32)
        self.assertEqual(embeddings["item_conformity"].dtype, torch.float32)
        self.assertGreaterEqual(float(embeddings["item_safe_features"].float().min()), 0.0)
        self.assertLessEqual(float(embeddings["item_safe_features"].float().max()), 1.0)

    def test_scoring_accepts_bf16_inputs_on_cpu_without_autocast(self) -> None:
        """The scorer should handle bf16 propagated tensors on CPU fallback paths."""
        config = EDGRecConfig(device="cpu", embed_dim=2).preset_full()
        model = EDGRec(
            n_users=1,
            n_items=2,
            config=config,
            item_popularity=torch.tensor([0.2, 0.8], dtype=torch.float32),
            item_recency=torch.tensor([0.1, 0.9], dtype=torch.float32),
            recent_train_items=torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long),
            recent_train_mask=torch.tensor(
                [[False, False, False, False, False, False, False, False, False, False]],
                dtype=torch.bool,
            ),
        )
        propagated = {
            "user_interest": torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
            "item_interest": torch.tensor([[2.0, 0.0], [1.0, 0.0]], dtype=torch.bfloat16),
            "user_conformity": torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16),
            "item_conformity": torch.tensor([[0.0, 2.0], [0.0, 1.0]], dtype=torch.bfloat16),
            "item_popularity": torch.tensor([0.2, 0.8], dtype=torch.bfloat16),
            "item_recency": torch.tensor([0.1, 0.9], dtype=torch.bfloat16),
        }

        scores = model.scoring(
            propagated,
            user_ids=torch.tensor([0]),
            item_ids=torch.tensor([0]),
        )

        self.assertIn("final_score", scores)
        self.assertTrue(torch.isfinite(scores["final_score"]).all())

    def test_scoring_hot_path_does_not_call_tensor_item_for_active_components(self) -> None:
        """Active-component masking should not synchronize tensors to Python scalars."""
        config = EDGRecConfig(device="cpu", embed_dim=2).preset_full()
        model = EDGRec(n_users=1, n_items=2, config=config)
        propagated = {
            "user_interest": torch.tensor([[1.0, 0.0]]),
            "item_interest": torch.tensor([[2.0, 0.0], [1.0, 0.0]]),
            "user_conformity": torch.tensor([[0.0, 1.0]]),
            "item_conformity": torch.tensor([[0.0, 2.0], [0.0, 1.0]]),
            "item_popularity": torch.tensor([0.2, 0.8]),
            "item_recency": torch.tensor([0.1, 0.9]),
        }

        with patch.object(torch.Tensor, "item", side_effect=AssertionError("tensor.item sync")):
            scores = model.scoring.score_all_items(
                propagated,
                user_ids=torch.tensor([0]),
            )

        self.assertIn("final_score", scores)

    def test_sign_aware_weighting_falls_back_without_mixed_signs(self) -> None:
        """One-sided graphs should keep the plain LightGCN unit baseline."""
        config = EDGRecConfig(device="cuda", use_dual_branch=False, use_sign_aware=True)
        model = DualBranchGCN(config)

        weights = model._compute_edge_weights_impl(torch.tensor([1.0, 1.0, 0.0]))

        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertTrue(torch.equal(weights, torch.ones(3)))

    def test_sign_aware_weighting_keeps_positive_edges_at_least_neutral(self) -> None:
        """Observed positive interactions must not be weaker than neutral ANN edges."""
        config = EDGRecConfig(device="cuda", use_dual_branch=False, use_sign_aware=True)
        model = DualBranchGCN(config)

        weights = model._compute_edge_weights_impl(torch.tensor([1.0, 0.0, -1.0]))

        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertGreaterEqual(weights[0].item(), weights[1].item())
        self.assertLess(weights[2].item(), weights[1].item())

    def test_sign_aware_weighting_is_constant_without_negative_edges(self) -> None:
        """All-positive observed graphs should not request sparse edge-value gradients."""
        config = EDGRecConfig(device="cuda", use_dual_branch=False, use_sign_aware=True)
        model = DualBranchGCN(config)

        weights = model._compute_edge_weights_impl(torch.tensor([1.0, 1.0, 0.0]))

        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertTrue(torch.equal(weights, torch.ones(3)))
        self.assertFalse(weights.requires_grad)

    def test_lightgcn_branch_matches_sparse_adjacency_matmul(self) -> None:
        """LightGCNBranch should equal repeated sparse adjacency matmuls."""
        branch = LightGCNBranch(n_layers=2)
        node_embeddings = torch.tensor(
            [[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]],
            dtype=torch.float32,
        )
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]],
            dtype=torch.long,
        )
        edge_weight = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        sparse_adjacency = DualBranchGCN._build_sparse_adjacency(
            edge_index,
            edge_weight,
            num_nodes=3,
            dtype=node_embeddings.dtype,
        )

        first_layer_embeddings = torch.sparse.mm(sparse_adjacency, node_embeddings)
        second_layer_embeddings = torch.sparse.mm(sparse_adjacency, first_layer_embeddings)
        expected = (node_embeddings + first_layer_embeddings + second_layer_embeddings) / 3.0

        actual = branch(node_embeddings, sparse_adjacency)

        self.assertTrue(torch.allclose(actual, expected))

    def test_lightgcn_edge_propagation_matches_sparse_adjacency_matmul(self) -> None:
        """Chunked edge-list propagation should preserve LightGCN math."""
        branch = LightGCNBranch(n_layers=2)
        node_embeddings = torch.tensor(
            [[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]],
            dtype=torch.float32,
        )
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]],
            dtype=torch.long,
        )
        edge_weight = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        sparse_adjacency = DualBranchGCN._build_sparse_adjacency(
            edge_index,
            edge_weight,
            num_nodes=3,
            dtype=node_embeddings.dtype,
        )

        expected = branch(node_embeddings, sparse_adjacency)
        actual = branch.forward_edges(
            node_embeddings,
            edge_index,
            edge_weight,
            num_nodes=3,
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_edge_propagation_backprops_through_sign_weights(self) -> None:
        """Chunked LightGCN propagation should preserve gradients for sign-aware weights."""
        config = EDGRecConfig(device="cuda", use_dual_branch=False, use_sign_aware=True)
        model = DualBranchGCN(config)
        embeddings = {
            "user": torch.tensor([[1.0, 0.0]], dtype=torch.float32, requires_grad=True),
            "item": torch.tensor([[0.0, 1.0]], dtype=torch.float32, requires_grad=True),
        }
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_sign = torch.tensor([1.0, -1.0], dtype=torch.float32)
        edge_norm = torch.ones(2, dtype=torch.float32)

        propagated = model(
            embeddings,
            edge_index,
            edge_sign=edge_sign,
            n_users=1,
            n_items=1,
            edge_norm=edge_norm,
        )
        loss = propagated["user"].sum() + propagated["item"].sum()
        loss.backward()

        self.assertIsNotNone(model.alpha_pos.grad)
        self.assertIsNotNone(model.alpha_neg.grad)

    def test_loss_suite_supports_popularity_aware_branch_contrastive_mainline(self) -> None:
        """LossSuite should expose DCCL-style interest and conformity contrastive losses."""
        config = EDGRecConfig(device="cuda", embed_dim=3)
        config.use_dual_branch = True
        config.use_ipw = False
        config.use_popularity_head = False
        config.loss_weight_recommendation = 0.0
        config.loss_weight_interest_bpr = 0.0
        config.loss_weight_conformity_bpr = 0.0
        config.loss_weight_independence = 0.0
        config.loss_weight_align = 0.0
        config.loss_weight_uniform = 0.0
        config.loss_weight_popularity = 0.0
        config.loss_weight_contrastive = 1.0
        config.auxiliary_loss_schedule = "linear_ramp"
        config.auxiliary_ramp_rate = 1.0
        config.contrastive_temperature = 1.0
        config.contrastive_max_pairs = 3
        loss_suite = LossSuite(config)

        model_output = {
            "pos_scores": {
                "final_score": torch.zeros(3),
                "interest_score": torch.zeros(3),
                "conformity_score": torch.zeros(3),
                "context_score": torch.zeros(3),
            },
            "neg_scores": {
                "final_score": torch.zeros(3),
                "interest_score": torch.zeros(3),
                "conformity_score": torch.zeros(3),
                "context_score": torch.zeros(3),
            },
            "propagated": {
                "user_interest": torch.eye(3),
                "item_interest": torch.eye(3),
                "user_conformity": torch.eye(3),
                "item_conformity": torch.eye(3),
            },
            "ipw_weights": torch.ones(3),
            "loss_user_ids": torch.tensor([0, 1, 2]),
        }

        losses = loss_suite(
            model_output,
            item_popularity=torch.tensor([0.1, 0.4, 0.9]),
            pos_item_ids=torch.tensor([0, 1, 2]),
            epoch=1,
        )

        pos_logit = 1.0
        interest_log_denom = math.log(math.exp(pos_logit) + 2.0)
        expected_interest = interest_log_denom - pos_logit
        conformity_loss_0 = math.log(math.exp(pos_logit) + 2.0) - pos_logit
        conformity_loss_1 = math.log(math.exp(pos_logit) + 1.0) - pos_logit
        conformity_weight_0 = 1.0 - math.exp(-0.1)
        conformity_weight_1 = 1.0 - math.exp(-0.4)
        expected_conformity = (
            conformity_weight_0 * conformity_loss_0 + conformity_weight_1 * conformity_loss_1
        ) / (conformity_weight_0 + conformity_weight_1)

        self.assertIn("interest_contrastive", losses)
        self.assertIn("conformity_contrastive", losses)
        self.assertIn("contrastive", losses)
        self.assertAlmostEqual(
            losses["interest_contrastive"].item(),
            expected_interest,
            places=6,
        )
        self.assertAlmostEqual(
            losses["conformity_contrastive"].item(),
            expected_conformity,
            places=6,
        )
        self.assertAlmostEqual(
            losses["contrastive"].item(),
            losses["interest_contrastive"].item() + losses["conformity_contrastive"].item(),
            places=6,
        )
        self.assertAlmostEqual(
            losses["total"].item(),
            losses["contrastive"].item(),
            places=6,
        )

    def test_loss_suite_dice_branch_mode_reverses_popularity_bpr_for_popular_negatives(
        self,
    ) -> None:
        """DICE-style branch loss should train conformity as popularity, not interest."""
        config = EDGRecConfig(device="cpu", embed_dim=2)
        config.use_dual_branch = True
        config.use_popularity_head = False
        config.branch_loss_mode = "dice"
        config.loss_weight_recommendation = 0.0
        config.loss_weight_interest_bpr = 1.0
        config.loss_weight_conformity_bpr = 1.0
        config.loss_weight_independence = 0.0
        config.loss_weight_contrastive = 0.0
        config.loss_weight_align = 0.0
        config.loss_weight_uniform = 0.0
        config.loss_weight_popularity = 0.0
        loss_suite = LossSuite(config)

        model_output = {
            "pos_scores": {
                "final_score": torch.zeros(2),
                "interest_score": torch.tensor([2.0, 2.0]),
                "conformity_score": torch.tensor([0.0, 2.0]),
                "context_score": torch.zeros(2),
            },
            "neg_scores": {
                "final_score": torch.zeros(2),
                "interest_score": torch.tensor([0.0, 0.0]),
                "conformity_score": torch.tensor([2.0, 0.0]),
                "context_score": torch.zeros(2),
            },
            "propagated": {
                "user_interest": torch.eye(2),
                "item_interest": torch.eye(3, 2),
                "user_conformity": torch.eye(2),
                "item_conformity": torch.eye(3, 2),
            },
            "ipw_weights": torch.ones(2),
            "loss_user_ids": torch.tensor([0, 1]),
            "loss_neg_item_ids": torch.tensor([2, 0]),
        }

        losses = loss_suite(
            model_output,
            item_popularity=torch.tensor([0.1, 0.8, 0.9]),
            pos_item_ids=torch.tensor([0, 1]),
            epoch=0,
        )

        expected_interest = torch.nn.functional.softplus(torch.tensor(-2.0)).item() / 2.0
        expected_conformity = torch.nn.functional.softplus(torch.tensor(-2.0)).item()

        self.assertAlmostEqual(losses["interest_bpr"].item(), expected_interest, places=6)
        self.assertAlmostEqual(losses["conformity_bpr"].item(), expected_conformity, places=6)
        self.assertAlmostEqual(
            losses["total"].item(),
            expected_interest + expected_conformity,
            places=6,
        )

    def test_loss_suite_popularity_head_prefers_raw_context_for_supervision(
        self,
    ) -> None:
        """Context regression should not train against the calibrated fusion score."""
        config = EDGRecConfig(device="cpu", embed_dim=2)
        config.use_dual_branch = True
        config.use_popularity_head = True
        config.loss_weight_recommendation = 0.0
        config.loss_weight_interest_bpr = 0.0
        config.loss_weight_conformity_bpr = 0.0
        config.loss_weight_independence = 0.0
        config.loss_weight_contrastive = 0.0
        config.loss_weight_align = 0.0
        config.loss_weight_uniform = 0.0
        config.loss_weight_popularity = 1.0
        config.auxiliary_loss_schedule = "phased"
        config.popularity_supervision_start_epoch = 0
        loss_suite = LossSuite(config)

        pos_item_ids = torch.tensor([0, 1])
        pop_target = torch.tensor([0.2, 0.8])
        score_zeros = torch.zeros(2)
        model_output = {
            "pos_scores": {
                "final_score": score_zeros,
                "interest_score": score_zeros,
                "conformity_score": score_zeros,
                "context_score": torch.zeros(2),
                "raw_context_score": pop_target.clone(),
            },
            "neg_scores": {
                "final_score": score_zeros,
                "interest_score": score_zeros,
                "conformity_score": score_zeros,
                "context_score": score_zeros,
            },
            "propagated": {
                "user_interest": torch.eye(2),
                "item_interest": torch.eye(2),
                "user_conformity": torch.eye(2),
                "item_conformity": torch.eye(2),
            },
            "ipw_weights": torch.ones(2),
            "loss_user_ids": torch.tensor([0, 1]),
        }

        losses = loss_suite(
            model_output,
            item_popularity=pop_target,
            pos_item_ids=pos_item_ids,
            epoch=0,
        )

        self.assertAlmostEqual(losses["pop"].item(), 0.0, places=7)
        self.assertAlmostEqual(losses["total"].item(), 0.0, places=7)

    def test_loss_suite_caps_dice_discrepancy_pairwise_entities(self) -> None:
        """DICE discrepancy should not allocate quadratic matrices over huge batches."""
        config = EDGRecConfig(device="cpu", embed_dim=2)
        config.use_dual_branch = True
        config.branch_loss_mode = "dice"
        config.loss_weight_recommendation = 0.0
        config.loss_weight_interest_bpr = 0.0
        config.loss_weight_conformity_bpr = 0.0
        config.loss_weight_independence = 1.0
        config.loss_weight_contrastive = 0.0
        config.loss_weight_align = 0.0
        config.loss_weight_uniform = 0.0
        config.loss_weight_popularity = 0.0
        config.auxiliary_losses_start_epoch = 0
        config.contrastive_max_pairs = 3
        config.distance_correlation_max_pairs = 4
        loss_suite = LossSuite(config)

        batch_size = 12
        n_items = 14
        score_zeros = torch.zeros(batch_size)
        loss_user_ids = torch.arange(batch_size, dtype=torch.long)
        pos_item_ids = torch.arange(batch_size, dtype=torch.long)
        neg_item_ids = torch.arange(1, batch_size + 1, dtype=torch.long)
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
                "user_interest": torch.randn(batch_size, 2),
                "item_interest": torch.randn(n_items, 2),
                "user_conformity": torch.randn(batch_size, 2),
                "item_conformity": torch.randn(n_items, 2),
            },
            "ipw_weights": torch.ones(batch_size),
            "loss_user_ids": loss_user_ids,
            "loss_neg_item_ids": neg_item_ids,
            "dice_negative_mask": torch.zeros(batch_size, dtype=torch.bool),
        }
        seen_sizes: list[tuple[int, int]] = []

        def fake_distance_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            seen_sizes.append((x.size(0), y.size(0)))
            return x.sum() * 0.0 + y.sum() * 0.0

        with patch(
            "src.losses.loss_suite._distance_correlation_loss",
            side_effect=fake_distance_correlation,
        ):
            losses = loss_suite(
                model_output,
                item_popularity=torch.ones(n_items),
                branch_item_popularity=torch.ones(n_items),
                pos_item_ids=pos_item_ids,
                epoch=0,
            )

        self.assertEqual(seen_sizes, [(4, 4), (4, 4)])
        self.assertTrue(torch.isfinite(losses["independence"]))

    def test_loss_suite_caps_directau_uniformity_pairwise_rows(self) -> None:
        """DirectAU uniformity should not run pdist over the full training batch."""
        config = EDGRecConfig(device="cpu", embed_dim=2)
        config.use_dual_branch = True
        config.loss_weight_recommendation = 0.0
        config.loss_weight_interest_bpr = 0.0
        config.loss_weight_conformity_bpr = 0.0
        config.loss_weight_independence = 0.0
        config.loss_weight_contrastive = 0.0
        config.loss_weight_align = 0.0
        config.loss_weight_uniform = 1.0
        config.loss_weight_popularity = 0.0
        config.auxiliary_losses_start_epoch = 0
        config.uniformity_max_pairs = 5
        loss_suite = LossSuite(config)

        batch_size = 12
        score_zeros = torch.zeros(batch_size)
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
                "user_interest": torch.randn(batch_size, 2),
                "item_interest": torch.randn(batch_size, 2),
                "user_conformity": torch.randn(batch_size, 2),
                "item_conformity": torch.randn(batch_size, 2),
            },
            "ipw_weights": torch.ones(batch_size),
            "loss_user_ids": torch.arange(batch_size, dtype=torch.long),
        }
        seen_rows: list[int] = []
        original_pdist = torch.pdist

        def recording_pdist(x: torch.Tensor, p: float = 2) -> torch.Tensor:
            seen_rows.append(int(x.size(0)))
            return original_pdist(x, p=p)

        with patch("src.losses.loss_suite.torch.pdist", side_effect=recording_pdist):
            losses = loss_suite(
                model_output,
                item_popularity=torch.ones(batch_size),
                pos_item_ids=torch.arange(batch_size, dtype=torch.long),
                epoch=0,
            )

        self.assertEqual(seen_rows, [5, 5])
        self.assertTrue(torch.isfinite(losses["uniform"]))

    def test_loss_suite_dice_discrepancy_has_finite_tiny_batch_gradients(self) -> None:
        """DICE distance-correlation discrepancy should stay finite on smoke batches."""
        config = EDGRecConfig(device="cpu", embed_dim=2)
        config.use_dual_branch = True
        config.branch_loss_mode = "dice"
        config.loss_weight_recommendation = 0.0
        config.loss_weight_interest_bpr = 0.0
        config.loss_weight_conformity_bpr = 0.0
        config.loss_weight_independence = 1.0
        config.loss_weight_contrastive = 0.0
        config.loss_weight_align = 0.0
        config.loss_weight_uniform = 0.0
        config.loss_weight_popularity = 0.0
        config.auxiliary_losses_start_epoch = 0
        loss_suite = LossSuite(config)
        user_interest = torch.zeros((2, 2), requires_grad=True)
        user_conformity = torch.zeros((2, 2), requires_grad=True)
        item_interest = torch.zeros((3, 2), requires_grad=True)
        item_conformity = torch.zeros((3, 2), requires_grad=True)

        model_output = {
            "pos_scores": {
                "final_score": torch.zeros(2),
                "interest_score": torch.zeros(2),
                "conformity_score": torch.zeros(2),
                "context_score": torch.zeros(2),
            },
            "neg_scores": {
                "final_score": torch.zeros(2),
                "interest_score": torch.zeros(2),
                "conformity_score": torch.zeros(2),
                "context_score": torch.zeros(2),
            },
            "propagated": {
                "user_interest": user_interest,
                "item_interest": item_interest,
                "user_conformity": user_conformity,
                "item_conformity": item_conformity,
            },
            "ipw_weights": torch.ones(2),
            "loss_user_ids": torch.tensor([0, 1]),
            "loss_neg_item_ids": torch.tensor([2, 0]),
        }

        losses = loss_suite(
            model_output,
            item_popularity=torch.tensor([0.1, 0.8, 0.9]),
            pos_item_ids=torch.tensor([0, 1]),
            epoch=0,
        )
        losses["total"].backward()

        self.assertTrue(torch.isfinite(losses["independence"]))
        for tensor in (user_interest, user_conformity, item_interest, item_conformity):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
