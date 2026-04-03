"""Regression coverage for split safety and runtime causal contracts."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.canonical import CanonicalInteractions
from src.data.graph_builder import build_graph
from src.losses.loss_suite import LossSuite
from src.models.lightgcn import DualBranchGCN
from src.models.ucagnn import UCaGNN
from src.training.evaluator import Evaluator
from src.utils.config import UCaGNNConfig
from src.utils.interaction_indexing import remap_interaction_ids


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
    ) -> dict:
        """Stub: no real propagation; return empty dict sentinel."""
        return {}

    def score_users_from_propagated(
        self,
        propagated: dict,
        user_ids: torch.Tensor,
        scoring_mode: str | None = None,
    ) -> torch.Tensor:
        """Return the same fixed scores for every user."""
        del propagated, scoring_mode
        return (
            self._base_scores.to(user_ids.device)
            .unsqueeze(0)
            .expand(user_ids.size(0), -1)
        )


class SplitSafetyTests(unittest.TestCase):
    """Pin the thesis-critical split boundary behavior."""

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
        config = UCaGNNConfig(graph_method="knn", device="cuda")

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

    def test_build_graph_can_use_time_windowed_train_popularity(self) -> None:
        """Popularity windows should be computed from the recent train slice only."""
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
        config = UCaGNNConfig(
            graph_method="knn",
            device="cuda",
            popularity_window_seconds=10,
        )

        data = build_graph(canonical, config)

        expected_popularity = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(data.popularity.cpu(), expected_popularity))

    def test_build_graph_carries_causal_fields(self) -> None:
        """Graph data should retain the extended canonical causal descriptors."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0], dtype=np.int64),
            item_id=np.array([0, 1], dtype=np.int64),
            label=np.ones(2, dtype=np.float32),
            timestamp=np.array([1, 2], dtype=np.int64),
            sign=np.ones(2, dtype=np.float32),
            popularity=np.ones(2, dtype=np.float32),
            raw_target=np.array([3.0, 4.0], dtype=np.float32),
            behavior_type=np.array(["fav", "buy"]),
            exposure_flag=np.array([False, True]),
            source_domain=np.array(["standard", "random"]),
            feedback_type="multi-behavior",
            preprocessing_preset="taobao_multibehavior",
            n_users=1,
            n_items=2,
            user_map={0: 0},
            item_map={0: 0, 1: 1},
            train_mask=np.array([True, False]),
            val_mask=np.array([False, False]),
            test_mask=np.array([False, True]),
        )
        config = UCaGNNConfig(graph_method="knn", device="cuda")

        data = build_graph(canonical, config)

        self.assertTrue(torch.equal(data.raw_target.cpu(), torch.tensor([3.0, 4.0])))
        self.assertTrue(
            torch.equal(data.exposure_flag.cpu(), torch.tensor([False, True]))
        )
        np.testing.assert_array_equal(data.behavior_type, np.array(["fav", "buy"]))
        np.testing.assert_array_equal(
            data.source_domain,
            np.array(["standard", "random"]),
        )
        self.assertEqual(data.feedback_type, "multi-behavior")
        self.assertEqual(data.preprocessing_preset, "taobao_multibehavior")

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
            np.array([True, True, True, False], dtype=bool)
        )

        np.testing.assert_allclose(
            recency,
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_evaluator_masks_observed_non_target_items_before_ranking(self) -> None:
        """Held-out targets must rank after masking previously observed items."""
        config = UCaGNNConfig(device="cuda")
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

    def test_val_evaluation_does_not_exclude_test_items_from_pool(self) -> None:
        """Test items must NOT be masked during validation evaluation.

        Excluding test items from the val ranking pool would allow test-set
        knowledge to influence early stopping and best-model selection,
        violating the information barrier between training and test phases.
        """
        evaluator = Evaluator(UCaGNNConfig(device="cuda"))

        # 3 interactions: one per split, same user
        data = Data(num_nodes=4)
        data.train_mask = torch.tensor([True, False, False], dtype=torch.bool)
        data.val_mask = torch.tensor([False, True, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, False, True], dtype=torch.bool)

        exclude = evaluator._observed_non_target_mask(data, data.val_mask)

        self.assertTrue(
            exclude[0].item(), "train interaction must be excluded from val pool"
        )
        self.assertFalse(
            exclude[1].item(), "val interaction (target) must not self-exclude"
        )
        self.assertFalse(
            exclude[2].item(), "test interaction must NOT be excluded from val pool"
        )

    def test_test_evaluation_excludes_train_and_val(self) -> None:
        """Both train and val interactions must be excluded from the test ranking pool.

        The test evaluation pool should only contain items the model has never
        encountered during training or been selected against during validation.
        """
        evaluator = Evaluator(UCaGNNConfig(device="cuda"))

        data = Data(num_nodes=4)
        data.train_mask = torch.tensor([True, False, False], dtype=torch.bool)
        data.val_mask = torch.tensor([False, True, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, False, True], dtype=torch.bool)

        exclude = evaluator._observed_non_target_mask(data, data.test_mask)

        self.assertTrue(
            exclude[0].item(), "train interaction must be excluded from test pool"
        )
        self.assertTrue(
            exclude[1].item(), "val interaction must be excluded from test pool"
        )
        self.assertFalse(
            exclude[2].item(), "test interaction (target) must not self-exclude"
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
            ValueError, msg="Overlapping train/test masks must raise ValueError"
        ):
            canonical.get_splits()


class CausalTrainingContractTests(unittest.TestCase):
    """Pin the intended dual-branch scoring and sign-aware behavior."""

    @staticmethod
    def _build_dual_branch_config() -> UCaGNNConfig:
        """Return a small CUDA-tagged config suitable for unit tests."""
        config = UCaGNNConfig(device="cuda")
        config.use_torch_compile = False
        config.use_ipw = False
        config.use_dual_branch = True
        config.use_counterfactual = True
        config.use_popularity_head = False
        config.train_scoring_mode = "interest_only"
        return config

    def test_build_training_output_respects_train_scoring_mode(self) -> None:
        """Dual-branch training should optimize the configured training score path."""
        config = self._build_dual_branch_config()
        model = UCaGNN(n_users=1, n_items=2, config=config)

        propagated = {
            "user_interest": torch.tensor([[1.0, 0.0]]),
            "item_interest": torch.tensor([[2.0, 0.0], [1.0, 0.0]]),
            "user_conformity": torch.tensor([[1.0, 0.0]]),
            "item_conformity": torch.tensor([[5.0, 0.0], [4.0, 0.0]]),
        }

        output = model.build_training_output(
            embeddings={},
            propagated=propagated,
            user_ids=torch.tensor([0]),
            pos_item_ids=torch.tensor([0]),
            neg_item_ids=torch.tensor([1]),
        )
        pos_scores = output["pos_scores"]
        neg_scores = output["neg_scores"]

        self.assertAlmostEqual(
            pos_scores["final_score"].item(),
            pos_scores["interest_score"].item(),
            places=6,
        )
        self.assertAlmostEqual(
            neg_scores["final_score"].item(),
            neg_scores["interest_score"].item(),
            places=6,
        )

    def test_wave1_scoring_exports_popularity_score_and_simplex_gate(self) -> None:
        """The fused scorer should expose popularity scores and simplex gate weights."""
        config = UCaGNNConfig(device="cuda", embed_dim=2).preset_full()
        model = UCaGNN(
            n_users=1,
            n_items=2,
            config=config,
            item_popularity=torch.tensor([0.2, 0.8]),
        )
        propagated = {
            "user_interest": torch.tensor([[1.0, 0.0]]),
            "item_interest": torch.tensor([[2.0, 0.0], [1.0, 0.0]]),
            "user_conformity": torch.tensor([[0.0, 1.0]]),
            "item_conformity": torch.tensor([[0.0, 2.0], [0.0, 1.0]]),
            "item_popularity": torch.tensor([0.2, 0.8]),
            "item_pop": torch.zeros(2, config.pop_embed_dim),
        }

        scores = model.scoring(
            propagated,
            user_ids=torch.tensor([0]),
            item_ids=torch.tensor([0]),
            scoring_mode="default",
        )

        self.assertIn("popularity_score", scores)
        self.assertIn("gate_weights", scores)
        self.assertEqual(tuple(scores["gate_weights"].shape), (1, 3))
        self.assertAlmostEqual(scores["gate_weights"].sum().item(), 1.0, places=6)
        self.assertAlmostEqual(
            scores["counterfactual_score"].item(),
            scores["interest_score"].item() - scores["conformity_score"].item(),
            places=6,
        )

    def test_sign_aware_weighting_falls_back_without_mixed_signs(self) -> None:
        """One-sided graphs should keep the plain LightGCN unit baseline."""
        config = UCaGNNConfig(device="cuda", use_dual_branch=False, use_sign_aware=True)
        model = DualBranchGCN(config)

        weights = model._compute_edge_weights_impl(torch.tensor([1.0, 1.0, 0.0]))

        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertTrue(torch.equal(weights, torch.ones(3)))

    def test_sign_aware_weighting_keeps_positive_edges_at_least_neutral(self) -> None:
        """Observed positive interactions must not be weaker than neutral ANN edges."""
        config = UCaGNNConfig(device="cuda", use_dual_branch=False, use_sign_aware=True)
        model = DualBranchGCN(config)

        weights = model._compute_edge_weights_impl(torch.tensor([1.0, 0.0, -1.0]))

        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertGreaterEqual(weights[0].item(), weights[1].item())
        self.assertLess(weights[2].item(), weights[1].item())

    def test_loss_suite_supports_within_branch_contrastive_mainline(self) -> None:
        """LossSuite should expose a ramped within-branch contrastive auxiliary."""
        config = UCaGNNConfig(device="cuda", embed_dim=2)
        config.use_dual_branch = True
        config.use_ipw = False
        config.use_popularity_head = False
        config.lambda_rec = 0.0
        config.lambda_interest_bpr = 0.0
        config.lambda_conformity_bpr = 0.0
        config.lambda_independence = 0.0
        config.lambda_align = 0.0
        config.lambda_uniform = 0.0
        config.lambda_pop = 0.0
        config.lambda_contrastive = 1.0
        config.auxiliary_loss_schedule = "linear_ramp"
        config.auxiliary_ramp_rate = 1.0
        config.contrastive_max_pairs = 2
        loss_suite = LossSuite(config)

        model_output = {
            "pos_scores": {
                "final_score": torch.zeros(2),
                "interest_score": torch.zeros(2),
                "conformity_score": torch.zeros(2),
                "popularity_score": torch.zeros(2),
            },
            "neg_scores": {
                "final_score": torch.zeros(2),
                "interest_score": torch.zeros(2),
                "conformity_score": torch.zeros(2),
                "popularity_score": torch.zeros(2),
            },
            "propagated": {
                "user_interest": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "item_interest": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "user_conformity": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "item_conformity": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            },
            "ipw_weights": torch.ones(2),
            "loss_user_ids": torch.tensor([0, 1]),
        }

        losses = loss_suite(
            model_output,
            item_popularity=torch.zeros(2),
            pos_item_ids=torch.tensor([0, 1]),
            epoch=1,
        )

        self.assertIn("contrastive", losses)
        self.assertGreater(losses["contrastive"].item(), 0.0)
        self.assertAlmostEqual(
            losses["total"].item(),
            losses["contrastive"].item(),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
