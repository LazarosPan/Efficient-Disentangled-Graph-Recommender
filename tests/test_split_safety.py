"""Regression coverage for split safety and runtime causal contracts."""

from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch
from src.data.canonical import CanonicalInteractions
from src.data.graph_builder import build_graph
from src.losses.loss_suite import LossSuite
from src.models.lightgcn import DualBranchGCN, LightGCNBranch
from src.models.ucagnn import UCaGNN
from src.training.evaluator import THESIS_PRIMARY_METRICS, Evaluator
from src.utils.config import UCaGNNConfig
from src.utils.interaction_indexing import remap_interaction_ids
from src.utils.trainer_runtime import stage_graph_tensors_for_device
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
        config = UCaGNNConfig(device="cuda")

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
            device="cuda",
            popularity_window_seconds=10,
        )

        data = build_graph(canonical, config)

        expected_popularity = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
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
        config = UCaGNNConfig(device="cuda")

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
        config = UCaGNNConfig(device="cuda")

        data = build_graph(canonical, config)
        data.raw_target[0] = 9.0

        np.testing.assert_array_equal(raw_target, np.array([3.0, 4.0], dtype=np.float32))
        self.assertEqual(float(data.raw_target[0]), 9.0)

    def test_cagra_runtime_failure_falls_back_to_interaction_graph(self) -> None:
        """CAGRA runtime failures should warn once and keep the train-interaction graph."""
        canonical = CanonicalInteractions(
            user_id=np.array([0], dtype=np.int64),
            item_id=np.array([0], dtype=np.int64),
            label=np.ones(1, dtype=np.float32),
            timestamp=np.array([1], dtype=np.int64),
            sign=np.ones(1, dtype=np.float32),
            popularity=np.ones(2, dtype=np.float32),
            n_users=1,
            n_items=2,
            user_map={0: 0},
            item_map={0: 0, 1: 1},
            train_mask=np.array([True]),
            val_mask=np.array([False]),
            test_mask=np.array([False]),
        )
        config = UCaGNNConfig(device="cuda", seed=17)
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            dtype=torch.float32,
        )

        class _FakeCupyArray:
            """Minimal CuPy-like wrapper for graph-builder tests."""

            def __init__(self, array: np.ndarray) -> None:
                self.array = np.asarray(array, dtype=np.float32)

        fake_cp = ModuleType("cupy")
        fake_cp.asarray = lambda array: _FakeCupyArray(array)
        fake_cp.asnumpy = lambda array: array.array

        cagra_module = ModuleType("cagra")
        cagra_module.IndexParams = lambda **kwargs: kwargs
        cagra_module.SearchParams = lambda **kwargs: kwargs

        def _raise_on_build(*args: object, **kwargs: object) -> object:
            raise RuntimeError("boom")

        cagra_module.build = _raise_on_build

        neighbors_module = ModuleType("cuvs.neighbors")
        neighbors_module.cagra = cagra_module
        cuvs_module = ModuleType("cuvs")
        cuvs_module.neighbors = neighbors_module

        with (
            patch.dict(
                sys.modules,
                {
                    "cupy": fake_cp,
                    "cuvs": cuvs_module,
                    "cuvs.neighbors": neighbors_module,
                },
            ),
            self.assertWarnsRegex(
                RuntimeWarning,
                "using train-interaction graph",
            ),
        ):
            data = build_graph(canonical, config, embeddings=embeddings)

        edge_signs = {
            (src, dst): float(sign)
            for (src, dst), sign in zip(
                data.edge_index.t().cpu().tolist(),
                data.edge_sign.cpu().tolist(),
                strict=True,
            )
        }
        self.assertEqual(edge_signs[(0, 1)], 1.0)
        self.assertEqual(edge_signs[(1, 0)], 1.0)
        self.assertEqual(len(edge_signs), 2)

    def test_cagra_uses_seeded_search_params(self) -> None:
        """CAGRA search should thread the config seed into its search params."""
        canonical = CanonicalInteractions(
            user_id=np.array([0], dtype=np.int64),
            item_id=np.array([0], dtype=np.int64),
            label=np.ones(1, dtype=np.float32),
            timestamp=np.array([1], dtype=np.int64),
            sign=np.ones(1, dtype=np.float32),
            popularity=np.ones(2, dtype=np.float32),
            n_users=1,
            n_items=2,
            user_map={0: 0},
            item_map={0: 0, 1: 1},
            train_mask=np.array([True]),
            val_mask=np.array([False]),
            test_mask=np.array([False]),
        )
        config = UCaGNNConfig(device="cuda", seed=29, cagra_k=1)
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            dtype=torch.float32,
        )

        class _FakeCupyArray:
            """Minimal CuPy-like wrapper for graph-builder tests."""

            def __init__(self, array: np.ndarray) -> None:
                self.array = np.asarray(array, dtype=np.float32)

        fake_cp = ModuleType("cupy")
        fake_cp.asarray = lambda array: _FakeCupyArray(array)
        fake_cp.asnumpy = lambda array: array.array

        seen: dict[str, object] = {}

        cagra_module = ModuleType("cagra")

        class _IndexParams:
            def __init__(self, **kwargs: object) -> None:
                seen["index_params"] = kwargs

        class _SearchParams:
            def __init__(self, **kwargs: object) -> None:
                seen["search_params"] = kwargs

        def _build(index_params: object, dataset: object) -> object:
            del index_params
            seen["build_dataset"] = dataset
            return object()

        def _search(
            search_params: object,
            index: object,
            queries: object,
            k: int,
        ) -> tuple[None, _FakeCupyArray]:
            del search_params, index, k
            seen["queries"] = queries
            return None, _FakeCupyArray(np.array([[2], [0], [1]], dtype=np.int64))

        cagra_module.IndexParams = _IndexParams
        cagra_module.SearchParams = _SearchParams
        cagra_module.build = _build
        cagra_module.search = _search

        neighbors_module = ModuleType("cuvs.neighbors")
        neighbors_module.cagra = cagra_module
        cuvs_module = ModuleType("cuvs")
        cuvs_module.neighbors = neighbors_module

        with patch.dict(
            sys.modules,
            {
                "cupy": fake_cp,
                "cuvs": cuvs_module,
                "cuvs.neighbors": neighbors_module,
            },
        ):
            data = build_graph(canonical, config, embeddings=embeddings)

        self.assertEqual(
            seen["search_params"],
            {"team_size": 0, "rand_xor_mask": 29, "itopk_size": 64},
        )
        self.assertIsInstance(seen["build_dataset"], _FakeCupyArray)
        self.assertIs(seen["build_dataset"], seen["queries"])
        edge_set = {tuple(edge) for edge in data.edge_index.t().cpu().tolist()}
        self.assertIn((0, 2), edge_set)
        self.assertIn((2, 0), edge_set)

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
        self.assertAlmostEqual(metrics["HitRatio@20"], 1.0, places=6)
        self.assertAlmostEqual(metrics["HitRatio@40"], 1.0, places=6)
        self.assertAlmostEqual(metrics["Personalization@20"], 0.0, places=6)
        self.assertAlmostEqual(metrics["Personalization@40"], 0.0, places=6)

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
        evaluator = Evaluator(UCaGNNConfig(device="cuda"))

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
        evaluator = Evaluator(UCaGNNConfig(device="cuda"))

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

    def test_presets_pin_expected_scoring_contracts(self) -> None:
        """Preset helpers should expose the thesis scoring contract directly."""
        lightgcn = UCaGNNConfig(device="cuda").preset_lightgcn()
        dice_like = UCaGNNConfig(device="cuda").preset_dice_like()
        ucagnn = UCaGNNConfig(device="cuda").preset_full()

        self.assertEqual(lightgcn.train_scoring_mode, "default")
        self.assertEqual(lightgcn.eval_scoring_mode, "default")
        self.assertFalse(lightgcn.use_features)
        self.assertFalse(dice_like.use_sign_aware)
        self.assertFalse(dice_like.use_features)
        self.assertEqual(dice_like.scoring_weight_mode, "fixed")
        self.assertEqual(dice_like.alpha_interest, 1.0)
        self.assertEqual(dice_like.beta_conformity, 1.0)
        self.assertEqual(dice_like.train_scoring_mode, "default")
        self.assertEqual(dice_like.eval_scoring_mode, "default")
        self.assertEqual(ucagnn.train_scoring_mode, "default")
        self.assertEqual(ucagnn.eval_scoring_mode, "default")

    def test_presets_reset_preset_owned_fields_when_switched(self) -> None:
        """Switching presets on one config should not preserve stale preset state."""
        ucagnn = UCaGNNConfig(device="cuda").preset_dice_like().preset_full()
        baseline = UCaGNNConfig(device="cuda").preset_full().preset_dice_like()

        self.assertTrue(ucagnn.use_features)
        self.assertEqual(ucagnn.scoring_weight_mode, "learned")
        self.assertEqual(ucagnn.alpha_interest, 0.5)
        self.assertEqual(ucagnn.beta_conformity, 0.3)
        self.assertEqual(ucagnn.gamma_popularity, 0.2)
        self.assertEqual(ucagnn.auxiliary_losses_start_epoch, 15)
        self.assertEqual(ucagnn.popularity_supervision_start_epoch, 30)
        self.assertEqual(ucagnn.propensity_clip_min, 0.1)

        self.assertFalse(baseline.use_ipw)
        self.assertFalse(baseline.use_features)
        self.assertEqual(baseline.propensity_clip_min, 0.01)
        self.assertEqual(baseline.auxiliary_losses_start_epoch, 0)
        self.assertEqual(baseline.popularity_supervision_start_epoch, 0)

    def test_same_checkpoint_eval_can_override_score_view(self) -> None:
        """Evaluation should support alternate score views on one checkpoint."""
        config = UCaGNNConfig(device="cuda", embed_dim=2)
        config.use_dual_branch = True
        config.use_ipw = False
        config.use_counterfactual = True
        config.use_popularity_head = False
        config.scoring_weight_mode = "fixed"
        model = UCaGNN(n_users=1, n_items=2, config=config)

        propagated = {
            "user_interest": torch.tensor([[1.0, 0.0]]),
            "item_interest": torch.tensor([[2.0, 0.0], [1.0, 0.0]]),
            "user_conformity": torch.tensor([[1.0, 0.0]]),
            "item_conformity": torch.tensor([[5.0, 0.0], [0.0, 0.0]]),
        }

        default_scores = model.score_users_from_propagated(
            propagated,
            user_ids=torch.tensor([0]),
            scoring_mode="default",
        )
        interest_only_scores = model.score_users_from_propagated(
            propagated,
            user_ids=torch.tensor([0]),
            scoring_mode="interest_only",
        )

        self.assertFalse(torch.allclose(default_scores, interest_only_scores))
        self.assertTrue(
            torch.allclose(
                interest_only_scores,
                torch.tensor([[2.0, 1.0]]),
            ),
        )

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

    def test_scoring_exports_popularity_score_and_simplex_gate(self) -> None:
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

    def test_embedding_feature_projection_accepts_cpu_fallback_inputs(self) -> None:
        """CPU fallback paths should project bf16 feature buffers without dtype errors."""
        config = UCaGNNConfig(device="cpu", embed_dim=4).preset_full()
        model = UCaGNN(
            n_users=1,
            n_items=2,
            config=config,
            item_features=torch.tensor(
                [[1.0, 0.0, 0.5], [0.0, 1.0, 0.25]],
                dtype=torch.float32,
            ),
            item_popularity=torch.tensor([0.2, 0.8], dtype=torch.float32),
            item_recency=torch.tensor([0.1, 0.9], dtype=torch.float32),
        )

        embeddings = model.embedding.get_embeddings()

        self.assertEqual(embeddings["item_interest"].dtype, torch.float32)
        self.assertEqual(embeddings["item_conformity"].dtype, torch.float32)

    def test_scoring_accepts_bf16_inputs_on_cpu_without_autocast(self) -> None:
        """The scorer should handle bf16 propagated tensors on CPU fallback paths."""
        config = UCaGNNConfig(device="cpu", embed_dim=2).preset_full()
        model = UCaGNN(
            n_users=1,
            n_items=2,
            config=config,
            item_popularity=torch.tensor([0.2, 0.8], dtype=torch.float32),
            item_recency=torch.tensor([0.1, 0.9], dtype=torch.float32),
        )
        propagated = {
            "user_interest": torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
            "item_interest": torch.tensor([[2.0, 0.0], [1.0, 0.0]], dtype=torch.bfloat16),
            "user_conformity": torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16),
            "item_conformity": torch.tensor([[0.0, 2.0], [0.0, 1.0]], dtype=torch.bfloat16),
            "item_popularity": torch.tensor([0.2, 0.8], dtype=torch.bfloat16),
            "item_recency": torch.tensor([0.1, 0.9], dtype=torch.bfloat16),
            "item_pop": torch.zeros(2, config.pop_embed_dim, dtype=torch.bfloat16),
        }

        scores = model.scoring(
            propagated,
            user_ids=torch.tensor([0]),
            item_ids=torch.tensor([0]),
            scoring_mode="default",
        )

        self.assertIn("final_score", scores)
        self.assertTrue(torch.isfinite(scores["final_score"]).all())

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

    def test_lightgcn_branch_matches_sparse_adjacency_matmul(self) -> None:
        """LightGCNBranch should equal repeated sparse adjacency matmuls."""
        branch = LightGCNBranch(n_layers=2)
        x = torch.tensor(
            [[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]],
            dtype=torch.float32,
        )
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]],
            dtype=torch.long,
        )
        edge_weight = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        adj = DualBranchGCN._build_sparse_adjacency(
            edge_index,
            edge_weight,
            num_nodes=3,
            dtype=x.dtype,
        )

        x_1 = torch.sparse.mm(adj, x)
        x_2 = torch.sparse.mm(adj, x_1)
        expected = (x + x_1 + x_2) / 3.0

        actual = branch(x, adj)

        self.assertTrue(torch.allclose(actual, expected))

    def test_sparse_propagation_backprops_through_sign_weights(self) -> None:
        """Sparse LightGCN propagation should preserve gradients for sign-aware weights."""
        config = UCaGNNConfig(device="cuda", use_dual_branch=False, use_sign_aware=True)
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
