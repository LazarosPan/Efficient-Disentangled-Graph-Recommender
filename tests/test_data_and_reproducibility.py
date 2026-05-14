"""Regression coverage for data semantics and reproducibility helpers."""

from __future__ import annotations

import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from src.data.canonical import (
    CanonicalInteractions,
    build_indexed_canonical_interactions,
)
from src.data.feature_policy import (
    enabled_feature_sources,
    registered_feature_sources,
    thesis_default_columns,
)
from src.data.loaders import load_dataset
from src.data.loaders.kuairand1k import load_kuairand1k
from src.data.loaders.kuairec_v2 import load_kuairec_v2
from src.data.loaders.movielens1m import load_movielens1m
from src.data.loaders.movielens20m import load_movielens20m
from src.data.loaders.taobao import load_taobao
from src.data.subgraph_sampler import SubgraphSampler
from src.data_exploration.data_exploration import _derive_ucagnn_requirements
from src.data_exploration.explore_all_datasets import (
    build_dataset_summary_payload,
)
from src.data_exploration.explore_all_datasets import (
    summarize_dataset as summarize_visual_dataset,
)
from src.utils.csv_features import (
    PolicyCsvFeatureSpec,
    load_csv_features,
    load_policy_csv_feature_blocks,
    load_policy_csv_features,
    stack_feature_blocks,
)
from src.utils.dataset_loader_utils import downcast_numeric_array
from src.utils.interaction_indexing import (
    collapse_pairwise_max_priority_rows,
    remap_interaction_ids,
    select_pairwise_max_priority_indices,
    summarize_pairwise_max_priority_collapse,
)
from src.utils.reproducibility import build_torch_generator, seed_everything


class DataContractTests(unittest.TestCase):
    """Pin dataset-loader semantics that affect training behavior."""

    def test_build_indexed_canonical_interactions_derives_popularity(self) -> None:
        """Shared canonical assembly should preserve indexing and derive popularity."""
        indexed = remap_interaction_ids(
            np.array([10, 10, 20], dtype=np.int64),
            np.array([100, 200, 100], dtype=np.int64),
        )

        canonical = build_indexed_canonical_interactions(
            indexed,
            label=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            timestamp=np.array([1, 2, 3], dtype=np.int64),
            sign=np.array([0.5, -0.5, 1.0], dtype=np.float32),
            feedback_type="implicit",
            preprocessing_preset="test",
        )

        np.testing.assert_array_equal(canonical.user_id, indexed.user_id)
        np.testing.assert_array_equal(canonical.item_id, indexed.item_id)
        np.testing.assert_allclose(
            canonical.popularity,
            np.array([1.0, 0.5], dtype=np.float32),
        )
        self.assertEqual(canonical.n_users, indexed.n_users)
        self.assertEqual(canonical.n_items, indexed.n_items)

    def test_kuairand_comment_rows_keep_neutral_sign(self) -> None:
        """Comment-only interactions should stay neutral until sentiment is known."""
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "KuaiRand-1K" / "data"
            data_root.mkdir(parents=True)
            (data_root / "log_standard_comment.csv").write_text(
                "user_id,video_id,is_comment,is_rand,time_ms\n1,10,1,0,1000\n",
                encoding="utf-8",
            )

            canonical = load_kuairand1k(
                data_dir=tmp_dir,
                include_optional_features=False,
            )

        np.testing.assert_array_equal(
            canonical.label,
            np.array([0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(canonical.sign, np.array([0.0], dtype=np.float32))
        np.testing.assert_array_equal(canonical.behavior_type, np.array(["comment"]))
        np.testing.assert_array_equal(canonical.exposure_flag, np.array([False]))
        np.testing.assert_array_equal(canonical.source_domain, np.array(["standard"]))

    def test_kuairand_loader_skips_bad_core_rows_and_nonfinite_optional_floats(
        self,
    ) -> None:
        """Malformed core rows should be dropped and NaN optional floats zeroed."""
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "KuaiRand-1K" / "data"
            data_root.mkdir(parents=True)
            (data_root / "log_standard_quality.csv").write_text(
                (
                    "user_id,video_id,play_time_ms,duration_ms,is_click,is_rand,time_ms\n1,10,nan,20,0,0,1000\nbad,11,3,10,1,0,1001\n"
                ),
                encoding="utf-8",
            )

            canonical = load_kuairand1k(
                data_dir=tmp_dir,
                include_optional_features=False,
            )

        self.assertEqual(len(canonical), 1)
        np.testing.assert_array_equal(canonical.raw_target, np.array([0.0], dtype=np.float32))
        np.testing.assert_array_equal(
            canonical.item_id,
            np.array([0], dtype=canonical.item_id.dtype),
        )

    def test_kuairec_loader_skips_nonfinite_watch_ratio_rows(self) -> None:
        """Rows with NaN watch ratios should not enter the canonical dataset."""
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "KuaiRec_v2" / "data"
            data_root.mkdir(parents=True)
            (data_root / "small_matrix.csv").write_text(
                ("user_id,video_id,watch_ratio,timestamp\n1,10,nan,100\n1,11,0.8,200\n"),
                encoding="utf-8",
            )

            canonical = load_kuairec_v2(
                data_dir=tmp_dir,
                matrix_variant="small_matrix",
                include_optional_features=False,
            )

        self.assertEqual(len(canonical), 1)
        np.testing.assert_array_equal(canonical.raw_target, np.array([0.8], dtype=np.float32))
        np.testing.assert_array_equal(canonical.timestamp, np.array([200]))

    def test_taobao_loader_skips_unknown_behaviors(self) -> None:
        """Unknown Taobao behavior labels should be excluded from training data."""
        with TemporaryDirectory() as tmp_dir:
            raw_root = Path(tmp_dir) / "Taobao" / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "UserBehavior.csv").write_text(
                "1,10,5,buy,100\n1,11,6,share,101\n",
                encoding="utf-8",
            )

            canonical = load_taobao(
                data_dir=tmp_dir,
                include_optional_features=False,
            )

        self.assertEqual(len(canonical), 1)
        np.testing.assert_array_equal(canonical.behavior_type, np.array(["buy"]))

    def test_stack_feature_blocks_skips_missing_blocks(self) -> None:
        """Optional feature matrices should combine without caller-side branching."""
        first = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        second = np.array([[0.5], [1.5]], dtype=np.float32)

        stacked = stack_feature_blocks(first, None, second)

        self.assertIsNotNone(stacked)
        assert stacked is not None
        self.assertEqual(stacked.shape, (2, 3))
        np.testing.assert_allclose(
            stacked,
            np.array([[1.0, 2.0, 0.5], [3.0, 4.0, 1.5]], dtype=np.float32),
        )

    def test_downcast_numeric_array_uses_signed_integer_ranges(self) -> None:
        """Signed integer narrowing should use actual min/max bounds."""
        narrowed = downcast_numeric_array(np.array([-1, 127], dtype=np.int64))
        self.assertEqual(narrowed.dtype, np.dtype(np.int8))

    def test_downcast_numeric_array_uses_unsigned_integer_ranges(self) -> None:
        """Non-negative integer arrays should narrow through unsigned widths."""
        narrowed = downcast_numeric_array(np.array([0, 256], dtype=np.int64))
        self.assertEqual(narrowed.dtype, np.dtype(np.uint16))

    def test_downcast_numeric_array_respects_float_widths(self) -> None:
        """Float narrowing should keep float64 only when float32 is not safe."""
        small_values = np.array([0.5, 1.5], dtype=np.float64)
        large_values = np.array(
            [float(np.finfo(np.float32).max) * 2.0],
            dtype=np.float64,
        )

        self.assertEqual(
            downcast_numeric_array(small_values, allow_float16=True).dtype,
            np.dtype(np.float16),
        )
        self.assertEqual(
            downcast_numeric_array(small_values, allow_float16=False).dtype,
            np.dtype(np.float32),
        )
        self.assertEqual(
            downcast_numeric_array(large_values, allow_float16=True).dtype,
            np.dtype(np.float64),
        )

    def test_downcast_numeric_array_leaves_bool_and_empty_arrays_unchanged(
        self,
    ) -> None:
        """Non-numeric edge cases should not be coerced."""
        empty = np.array([], dtype=np.int64)
        bool_values = np.array([True, False], dtype=bool)

        self.assertEqual(downcast_numeric_array(empty).dtype, empty.dtype)
        self.assertEqual(downcast_numeric_array(bool_values).dtype, bool_values.dtype)

    def test_subgraph_sampler_reuses_sorted_local_indices_for_item_mappings(self) -> None:
        """Positive and negative item remaps should stay aligned to one item index space."""
        edge_index = torch.tensor(
            [
                [0, 2, 0, 3, 1, 3, 1, 4, 2, 0, 3, 0, 3, 1, 4, 1],
                [2, 0, 3, 0, 3, 1, 4, 1, 0, 2, 0, 3, 1, 3, 1, 4],
            ],
            dtype=torch.long,
        )
        sampler = SubgraphSampler(
            edge_index=edge_index,
            edge_sign=None,
            edge_norm=None,
            n_users=2,
            n_items=3,
            num_hops=1,
            max_neighbors_per_hop=None,
        )

        subgraph = sampler.sample(
            batch_users=torch.tensor([0, 1], dtype=torch.long),
            batch_pos_items=torch.tensor([0, 1], dtype=torch.long),
            batch_neg_items=torch.tensor([1, 2], dtype=torch.long),
        )

        self.assertTrue(
            torch.equal(
                subgraph.user_global_ids[subgraph.batch_user_local],
                torch.tensor([0, 1], dtype=torch.long),
            ),
        )
        self.assertTrue(
            torch.equal(
                subgraph.item_global_ids[subgraph.batch_pos_local],
                torch.tensor([0, 1], dtype=torch.long),
            ),
        )
        self.assertTrue(
            torch.equal(
                subgraph.item_global_ids[subgraph.batch_neg_local],
                torch.tensor([1, 2], dtype=torch.long),
            ),
        )

    def test_load_policy_csv_features_respects_feature_policy(self) -> None:
        """Policy-gated CSV helpers should load only enabled feature sources."""
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "user_features_1k.csv"
            path.write_text("user_id,age\n1,7\n", encoding="utf-8")

            disabled = load_policy_csv_features(
                path,
                feature_policy="thesis_default",
                dataset_name="kuairand1k",
                aspect="user_features",
                relative_path="data/user_features_1k.csv",
                id_col="user_id",
                id_map={1: 0},
                n_entities=1,
            )
            enabled = load_policy_csv_features(
                path,
                feature_policy="all_optional",
                dataset_name="kuairand1k",
                aspect="user_features",
                relative_path="data/user_features_1k.csv",
                id_col="user_id",
                id_map={1: 0},
                n_entities=1,
            )

        self.assertIsNone(disabled)
        self.assertIsNotNone(enabled)
        assert enabled is not None
        self.assertEqual(enabled.dtype, np.dtype(np.float16))
        np.testing.assert_allclose(enabled, np.array([[7.0]], dtype=np.float16))

    def test_load_csv_features_encodes_temporal_and_categorical_columns(self) -> None:
        """Mixed-type CSV features should keep temporal and categorical signal."""
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "video_features_basic_1k.csv"
            path.write_text(
                (
                    "video_id,video_type,upload_dt,server_width,music_id\n10,NORMAL,2022-04-10,720,77\n11,AD,2022-04-11,1080,88\n"
                ),
                encoding="utf-8",
            )
            features = load_csv_features(
                path,
                id_col="video_id",
                id_map={10: 0, 11: 1},
                n_entities=2,
                include_columns=("video_type", "upload_dt", "server_width", "music_id"),
            )

        self.assertIsNotNone(features)
        assert features is not None
        np.testing.assert_allclose(
            features,
            np.array(
                [
                    [2.0, 1649548800.0, 720.0, 1.0],
                    [1.0, 1649635200.0, 1080.0, 2.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_feature_policy_lists_enabled_and_registered_sources(self) -> None:
        """Feature-policy source introspection should mirror thesis-default gating."""
        self.assertEqual(
            registered_feature_sources("kuairand1k", "item_features"),
            (
                "data/video_features_basic_1k.csv",
                "data/video_features_statistic_1k.csv",
            ),
        )
        self.assertEqual(
            enabled_feature_sources("thesis_default", "kuairand1k", "user_features"),
            (),
        )
        self.assertEqual(
            enabled_feature_sources("all_optional", "kuairand1k", "user_features"),
            ("data/user_features_1k.csv",),
        )
        self.assertEqual(
            enabled_feature_sources("thesis_default", "movielens20m", "item_features"),
            ("raw/movies.csv",),
        )

    def test_load_policy_csv_feature_blocks_stacks_only_enabled_sources(self) -> None:
        """Stacked policy-gated CSV helpers should encode mixed-type columns consistently."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            basic_path = root / "video_features_basic_1k.csv"
            stats_path = root / "video_features_statistic_1k.csv"
            basic_path.write_text(
                (
                    "video_id,author_id,video_type,upload_dt,upload_type,visible_status,server_width,server_height,music_id,music_type\n10,33,NORMAL,2022-04-10,Web,1,720,1280,77,4\n11,44,AD,2022-04-11,LongImport,3,1080,1920,88,7\n"
                ),
                encoding="utf-8",
            )
            stats_path.write_text(
                "video_id,show_cnt\n10,99\n11,101\n",
                encoding="utf-8",
            )

            thesis_default = load_policy_csv_feature_blocks(
                feature_policy="thesis_default",
                dataset_name="kuairand1k",
                aspect="item_features",
                id_map={10: 0, 11: 1},
                n_entities=2,
                sources=(
                    PolicyCsvFeatureSpec(
                        path=basic_path,
                        relative_path="data/video_features_basic_1k.csv",
                        id_col="video_id",
                    ),
                    PolicyCsvFeatureSpec(
                        path=stats_path,
                        relative_path="data/video_features_statistic_1k.csv",
                        id_col="video_id",
                    ),
                ),
            )
            all_optional = load_policy_csv_feature_blocks(
                feature_policy="all_optional",
                dataset_name="kuairand1k",
                aspect="item_features",
                id_map={10: 0, 11: 1},
                n_entities=2,
                sources=(
                    PolicyCsvFeatureSpec(
                        path=basic_path,
                        relative_path="data/video_features_basic_1k.csv",
                        id_col="video_id",
                    ),
                    PolicyCsvFeatureSpec(
                        path=stats_path,
                        relative_path="data/video_features_statistic_1k.csv",
                        id_col="video_id",
                    ),
                ),
            )

        self.assertIsNotNone(thesis_default)
        self.assertIsNotNone(all_optional)
        assert thesis_default is not None
        assert all_optional is not None
        self.assertEqual(thesis_default.shape, (2, 9))
        self.assertEqual(all_optional.shape, (2, 10))
        np.testing.assert_allclose(
            thesis_default,
            np.array(
                [
                    [1.0, 2.0, 1649548800.0, 2.0, 1.0, 720.0, 1280.0, 1.0, 1.0],
                    [2.0, 1.0, 1649635200.0, 1.0, 2.0, 1080.0, 1920.0, 2.0, 2.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            all_optional,
            np.array(
                [
                    [1.0, 2.0, 1649548800.0, 2.0, 1.0, 720.0, 1280.0, 1.0, 1.0, 99.0],
                    [2.0, 1.0, 1649635200.0, 1.0, 2.0, 1080.0, 1920.0, 2.0, 2.0, 101.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_thesis_default_columns_keep_selected_mixed_type_fields(self) -> None:
        """Thesis-default item features should keep the audited mixed-type subset."""
        self.assertEqual(
            thesis_default_columns(
                "kuairec_v2",
                "item_features",
                "data/item_daily_features.csv",
            ),
            (
                "author_id",
                "music_id",
                "video_type",
                "upload_dt",
                "upload_type",
                "visible_status",
            ),
        )
        self.assertEqual(
            thesis_default_columns(
                "kuairand1k",
                "item_features",
                "data/video_features_basic_1k.csv",
            ),
            (
                "author_id",
                "video_type",
                "upload_dt",
                "upload_type",
                "visible_status",
                "server_width",
                "server_height",
                "music_id",
                "music_type",
            ),
        )

    def test_kuairec_loader_keeps_nonzero_mixed_type_item_features(self) -> None:
        """KuaiRec item features should preserve selected string/date columns."""
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "KuaiRec_v2" / "data"
            data_root.mkdir(parents=True)
            (data_root / "small_matrix.csv").write_text(
                ("user_id,video_id,watch_ratio,timestamp\n1,10,0.8,100\n1,11,0.9,200\n"),
                encoding="utf-8",
            )
            (data_root / "item_daily_features.csv").write_text(
                (
                    "video_id,author_id,music_id,video_type,upload_dt,upload_type,visible_status\n10,33,77,NORMAL,2020-03-30,Web,public\n11,44,88,AD,2020-03-31,LongImport,private\n"
                ),
                encoding="utf-8",
            )

            canonical = load_kuairec_v2(
                data_dir=tmp_dir,
                matrix_variant="small_matrix",
                include_optional_features=True,
            )

        self.assertIsNotNone(canonical.item_features)
        assert canonical.item_features is not None
        self.assertEqual(canonical.item_features.shape, (2, 6))
        np.testing.assert_allclose(
            canonical.item_features,
            np.array(
                [
                    [1.0, 1.0, 2.0, 1585526400.0, 2.0, 2.0],
                    [2.0, 2.0, 1.0, 1585612800.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_kuairec_thesis_default_excludes_post_treatment_item_daily_counts(self) -> None:
        """Thesis-default KuaiRec item features should exclude engagement aggregates."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            item_daily_path = root / "item_daily_features.csv"
            item_daily_path.write_text(
                (
                    "video_id,author_id,music_id,video_type,upload_dt,upload_type,visible_status,show_cnt,play_cnt,like_cnt,share_cnt\n10,33,77,NORMAL,2020-03-30,Web,public,90,80,70,60\n11,44,88,AD,2020-03-31,LongImport,private,50,40,30,20\n"
                ),
                encoding="utf-8",
            )

            thesis_default = load_policy_csv_feature_blocks(
                feature_policy="thesis_default",
                dataset_name="kuairec_v2",
                aspect="item_features",
                id_map={10: 0, 11: 1},
                n_entities=2,
                sources=(
                    PolicyCsvFeatureSpec(
                        path=item_daily_path,
                        relative_path="data/item_daily_features.csv",
                        id_col="video_id",
                    ),
                ),
            )
            all_optional = load_policy_csv_feature_blocks(
                feature_policy="all_optional",
                dataset_name="kuairec_v2",
                aspect="item_features",
                id_map={10: 0, 11: 1},
                n_entities=2,
                sources=(
                    PolicyCsvFeatureSpec(
                        path=item_daily_path,
                        relative_path="data/item_daily_features.csv",
                        id_col="video_id",
                    ),
                ),
            )

        self.assertIsNotNone(thesis_default)
        self.assertIsNotNone(all_optional)
        assert thesis_default is not None
        assert all_optional is not None
        self.assertEqual(thesis_default.shape, (2, 6))
        self.assertEqual(all_optional.shape, (2, 10))
        np.testing.assert_allclose(
            all_optional[:, -4:],
            np.array(
                [
                    [90.0, 80.0, 70.0, 60.0],
                    [50.0, 40.0, 30.0, 20.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_ucagnn_requirements_track_predefined_splits_separately(self) -> None:
        """Train/test files should not be mistaken for real timestamp support."""
        requirements = _derive_ucagnn_requirements(
            {
                "name": "AmazonBook",
                "kind": "interaction_lists",
                "present_files": ["train.txt", "test.txt"],
                "files": [
                    {
                        "columns": ["user_id", "item_ids..."],
                    },
                ],
            },
        )

        self.assertTrue(requirements["supports_pairwise_triplets"])
        self.assertTrue(requirements["supports_predefined_split"])
        self.assertFalse(requirements["supports_timestamp_split"])
        self.assertTrue(requirements["supports_popularity_signal"])

    def test_visual_summary_reports_repeated_pairs_and_exposure_share(self) -> None:
        """Benchmark summaries should expose repeated-pair and exposure balance signals."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0, 1], dtype=np.int64),
            item_id=np.array([0, 0, 1], dtype=np.int64),
            label=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            timestamp=np.array([1, 2, 3], dtype=np.int64),
            sign=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            popularity=np.array([1.0, 1.0], dtype=np.float32),
            exposure_flag=np.array([True, True, False]),
            n_users=2,
            n_items=2,
            user_map={0: 0, 1: 1},
            item_map={0: 0, 1: 1},
        )

        summary = summarize_visual_dataset("kuairand1k", canonical)

        self.assertEqual(summary.unique_pair_count, 2)
        self.assertAlmostEqual(summary.repeated_pair_share, 1.0 / 3.0)
        self.assertAlmostEqual(summary.randomized_exposure_share or 0.0, 2.0 / 3.0)

    def test_visual_summary_payload_exports_context_as_text_stats(self) -> None:
        """Visualization payloads should expose interpretable context statistics."""
        canonical = CanonicalInteractions(
            user_id=np.array([0, 0, 1, 1], dtype=np.int64),
            item_id=np.array([0, 0, 1, 2], dtype=np.int64),
            label=np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            timestamp=np.array([1, 2, 3, 4], dtype=np.int64),
            sign=np.array([1.0, 0.0, 1.0, -1.0], dtype=np.float32),
            popularity=np.array([1.0, 1.0, 0.5], dtype=np.float32),
            raw_target=np.array([0.9, 0.1, 0.7, 0.2], dtype=np.float32),
            behavior_type=np.array(["like", "click", "like", "hate"]),
            exposure_flag=np.array([True, False, True, False]),
            source_domain=np.array(["random", "standard", "random", "standard"]),
            n_users=2,
            n_items=3,
            user_map={0: 0, 1: 1},
            item_map={0: 0, 1: 1, 2: 2},
        )

        summary = summarize_visual_dataset("kuairand1k", canonical)
        payload = build_dataset_summary_payload(canonical, summary)

        self.assertEqual(payload["split_counts"]["train"], 2)
        self.assertEqual(payload["response_signal"]["name"], "raw_target")
        self.assertEqual(payload["behavior_mix_top"]["like"], 2)
        self.assertAlmostEqual(payload["exposure_policy"]["randomized_share"], 0.5)
        self.assertAlmostEqual(
            payload["source_domain_positive_rate"]["random"]["positive_rate"],
            1.0,
        )

    def test_pairwise_priority_collapse_keeps_one_row_per_pair(self) -> None:
        """Pairwise collapse should keep the strongest row for each user-item pair."""
        keep_idx = select_pairwise_max_priority_indices(
            np.array([1, 1, 1, 2], dtype=np.int64),
            np.array([10, 10, 11, 10], dtype=np.int64),
            np.array([0.0, 3.0, 1.0, 2.0], dtype=np.float32),
            np.array([100, 200, 150, 120], dtype=np.int64),
        )

        np.testing.assert_array_equal(keep_idx, np.array([1, 2, 3], dtype=np.int64))

    def test_pairwise_priority_collapse_summarizes_repeat_stats(self) -> None:
        """Repeat-aware collapse should preserve count and priority summaries."""
        summary = summarize_pairwise_max_priority_collapse(
            np.array([1, 1, 2], dtype=np.int64),
            np.array([10, 10, 11], dtype=np.int64),
            np.array([1.0, 3.0, 2.0], dtype=np.float32),
            np.array([100, 200, 150], dtype=np.int64),
        )

        np.testing.assert_array_equal(summary.keep_idx, np.array([1, 2], dtype=np.int64))
        np.testing.assert_array_equal(summary.repeat_count, np.array([2, 1], dtype=np.int64))
        np.testing.assert_allclose(
            summary.priority_mean,
            np.array([2.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            summary.priority_max,
            np.array([3.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            summary.latest_priority,
            np.array([3.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            summary.first_timestamp,
            np.array([100, 150], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            summary.last_timestamp,
            np.array([200, 150], dtype=np.int64),
        )

    def test_pairwise_priority_collapse_rows_reuses_kept_indices_for_all_arrays(self) -> None:
        """Plain collapse should slice all aligned arrays with the retained rows."""
        collapsed_arrays, dropped_rows = collapse_pairwise_max_priority_rows(
            np.array([1, 1, 2], dtype=np.int64),
            np.array([10, 10, 11], dtype=np.int64),
            np.array([1.0, 3.0, 2.0], dtype=np.float32),
            np.array([100, 200, 150], dtype=np.int64),
            np.array([0.1, 0.3, 0.2], dtype=np.float32),
            timestamp=np.array([100, 200, 150], dtype=np.int64),
        )

        self.assertEqual(dropped_rows, 1)
        self.assertEqual(len(collapsed_arrays), 4)
        np.testing.assert_array_equal(
            collapsed_arrays[0],
            np.array([1, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            collapsed_arrays[1],
            np.array([10, 11], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            collapsed_arrays[2],
            np.array([200, 150], dtype=np.int64),
        )
        np.testing.assert_allclose(
            collapsed_arrays[3],
            np.array([0.3, 0.2], dtype=np.float32),
        )

    def test_taobao_default_preset_collapses_repeated_pairs(self) -> None:
        """Taobao thesis-default preprocessing should collapse repeated user-item events."""
        with TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "Taobao" / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "UserBehavior.csv").write_text(
                "1,10,100,buy,1\n1,10,100,pv,2\n2,11,101,pv,3\n",
                encoding="utf-8",
            )

            canonical = load_taobao(data_dir=tmp_dir)
            raw_canonical = load_taobao(
                data_dir=tmp_dir,
                preprocessing_preset="taobao_multibehavior_raw",
            )

        self.assertEqual(len(canonical), 2)
        self.assertEqual(len(raw_canonical), 3)
        np.testing.assert_allclose(canonical.raw_target, np.array([3.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(canonical.repeat_count, np.array([2, 1], dtype=np.uint8))
        np.testing.assert_allclose(
            canonical.repeat_mean_target,
            np.array([1.5, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            canonical.repeat_max_target,
            np.array([3.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            canonical.repeat_latest_target,
            np.array([0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            canonical.repeat_behavior_labels,
            np.array(["pv", "fav", "cart", "buy"]),
        )
        np.testing.assert_array_equal(
            canonical.repeat_behavior_counts,
            np.array([[1, 0, 0, 1], [1, 0, 0, 0]], dtype=np.uint8),
        )
        self.assertEqual(canonical.metadata["repeat_collapse"]["dropped_rows"], 1)
        self.assertTrue(canonical.metadata["repeat_collapse"]["preserves_repeat_stats"])
        self.assertEqual(canonical.metadata["repeat_collapse"]["stage"], "pre_split")
        self.assertIn(
            "cannot span train/val/test",
            canonical.metadata["repeat_collapse"]["reason"],
        )

    def test_kuairec_defaults_prefer_small_matrix_fullobs(self) -> None:
        """KuaiRec defaults should prefer the nearly fully observed small matrix."""
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "KuaiRec_v2" / "data"
            data_root.mkdir(parents=True)
            (data_root / "small_matrix.csv").write_text(
                ("user_id,video_id,watch_ratio,timestamp\n1,10,573.4572,100\n2,11,0.8,300\n"),
                encoding="utf-8",
            )

            direct_canonical = load_kuairec_v2(
                data_dir=tmp_dir,
                include_optional_features=False,
            )
            registry_canonical = load_dataset(
                "kuairec_v2",
                data_dir=tmp_dir,
                include_optional_features=False,
            )

        for canonical in (direct_canonical, registry_canonical):
            self.assertEqual(canonical.preprocessing_preset, "kuairec_fullobs")
            self.assertEqual(canonical.metadata["matrix_variant"], "small_matrix")
            self.assertEqual(canonical.metadata["watch_ratio_policy"], "raw")
            np.testing.assert_array_equal(
                canonical.source_domain,
                np.array(["small_matrix", "small_matrix"]),
            )
            np.testing.assert_allclose(
                canonical.raw_target,
                np.array([573.4572, 0.8], dtype=np.float32),
            )
            self.assertFalse(canonical.metadata["repeat_collapse"]["applied"])

    def test_kuairec_watchratio_preset_clips_targets_and_collapses_pairs(self) -> None:
        """Explicit KuaiRec watch-ratio runs should keep the clipped big-matrix path."""
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "KuaiRec_v2" / "data"
            data_root.mkdir(parents=True)
            (data_root / "big_matrix.csv").write_text(
                (
                    "user_id,video_id,watch_ratio,timestamp\n1,10,573.4572,100\n1,10,0.2,200\n2,11,0.8,300\n"
                ),
                encoding="utf-8",
            )

            canonical = load_kuairec_v2(
                data_dir=tmp_dir,
                matrix_variant="big_matrix",
                preprocessing_preset="kuairec_watchratio",
            )
            raw_canonical = load_kuairec_v2(
                data_dir=tmp_dir,
                matrix_variant="big_matrix",
                preprocessing_preset="kuairec_watchratio_raw",
            )

        self.assertEqual(len(canonical), 2)
        self.assertEqual(len(raw_canonical), 3)
        self.assertEqual(canonical.preprocessing_preset, "kuairec_watchratio")
        np.testing.assert_allclose(canonical.raw_target, np.array([5.0, 0.8], dtype=np.float32))
        np.testing.assert_array_equal(canonical.repeat_count, np.array([2, 1], dtype=np.uint8))
        np.testing.assert_allclose(
            canonical.repeat_mean_target,
            np.array([2.6, 0.8], dtype=np.float32),
        )
        np.testing.assert_allclose(
            canonical.repeat_max_target,
            np.array([5.0, 0.8], dtype=np.float32),
        )
        np.testing.assert_allclose(
            canonical.repeat_latest_target,
            np.array([0.2, 0.8], dtype=np.float32),
        )
        self.assertEqual(canonical.metadata["watch_ratio_policy"], "clipped_to_5")
        self.assertEqual(canonical.metadata["repeat_collapse"]["dropped_rows"], 1)
        self.assertTrue(canonical.metadata["repeat_collapse"]["preserves_repeat_stats"])
        self.assertEqual(canonical.metadata["repeat_collapse"]["stage"], "pre_split")
        self.assertIn(
            "cannot span train/val/test",
            canonical.metadata["repeat_collapse"]["reason"],
        )

    def test_kuairand_default_preset_collapses_repeated_pairs_before_split(self) -> None:
        """KuaiRand collapse metadata should document the pre-split leakage guard."""
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "KuaiRand-1K" / "data"
            data_root.mkdir(parents=True)
            (data_root / "log_standard_pairs.csv").write_text(
                (
                    "user_id,video_id,is_click,is_like,is_hate,is_follow,is_comment,long_view,play_time_ms,duration_ms,is_rand,time_ms\n1,10,1,1,0,0,0,1,5000,1000,0,10\n1,10,1,0,0,0,0,0,1000,1000,0,20\n"
                ),
                encoding="utf-8",
            )

            canonical = load_kuairand1k(data_dir=tmp_dir, include_optional_features=False)

        self.assertEqual(len(canonical), 1)
        np.testing.assert_array_equal(canonical.repeat_count, np.array([2], dtype=np.uint8))
        np.testing.assert_allclose(canonical.repeat_mean_target, np.array([3.0], dtype=np.float32))
        np.testing.assert_allclose(canonical.repeat_max_target, np.array([5.0], dtype=np.float32))
        np.testing.assert_allclose(
            canonical.repeat_latest_target, np.array([1.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            canonical.repeat_behavior_labels,
            np.array(["click", "like", "follow", "comment", "hate", "long_view"]),
        )
        np.testing.assert_array_equal(
            canonical.repeat_behavior_counts,
            np.array([[2, 1, 0, 0, 0, 1]], dtype=np.uint8),
        )
        self.assertEqual(canonical.metadata["repeat_collapse"]["dropped_rows"], 1)
        self.assertTrue(canonical.metadata["repeat_collapse"]["preserves_repeat_stats"])
        self.assertEqual(canonical.metadata["repeat_collapse"]["stage"], "pre_split")
        self.assertIn(
            "cannot span train/val/test",
            canonical.metadata["repeat_collapse"]["reason"],
        )

    def test_kuairand_random_only_preset_filters_standard_rows(self) -> None:
        """KuaiRand should expose a random-only causal view through preprocessing presets."""
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "KuaiRand-1K" / "data"
            data_root.mkdir(parents=True)
            (data_root / "log_standard_4_08_to_4_21_1k.csv").write_text(
                (
                    "user_id,video_id,is_click,is_like,is_hate,is_follow,is_comment,long_view,play_time_ms,duration_ms,is_rand,time_ms\n1,10,1,0,0,0,0,1,1000,1000,0,10\n"
                ),
                encoding="utf-8",
            )
            (data_root / "log_random_4_22_to_5_08_1k.csv").write_text(
                (
                    "user_id,video_id,is_click,is_like,is_hate,is_follow,is_comment,long_view,play_time_ms,duration_ms,is_rand,time_ms\n2,11,1,1,0,0,0,1,1000,1000,1,20\n"
                ),
                encoding="utf-8",
            )

            canonical = load_kuairand1k(
                data_dir=tmp_dir,
                preprocessing_preset="kuairand_random_only",
            )

        self.assertEqual(len(canonical), 1)
        self.assertTrue(bool(canonical.exposure_flag[0]))
        self.assertEqual(canonical.source_domain.tolist(), ["random"])
        self.assertEqual(canonical.metadata["exposure_summary"]["standard_count"], 0)

    def test_movielens20m_dense_genome_moves_behind_all_optional(self) -> None:
        """MovieLens20M thesis-default features should stay narrower than all-optional."""
        with TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "MovieLens20M" / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "ratings.csv").write_text(
                "userId,movieId,rating,timestamp\n1,10,4.0,100\n2,11,5.0,200\n",
                encoding="utf-8",
            )
            (raw_dir / "movies.csv").write_text(
                "movieId,title,genres\n10,Movie A,Action|Comedy\n11,Movie B,Drama\n",
                encoding="utf-8",
            )
            (raw_dir / "genome-scores.csv").write_text(
                "movieId,tagId,relevance\n10,1,0.2\n10,2,0.9\n11,1,0.1\n11,2,0.3\n",
                encoding="utf-8",
            )

            thesis_default = load_movielens20m(data_dir=tmp_dir)
            all_optional = load_movielens20m(data_dir=tmp_dir, feature_policy="all_optional")

        assert thesis_default.item_features is not None
        assert all_optional.item_features is not None
        self.assertEqual(thesis_default.item_features.shape[1], 3)
        self.assertEqual(all_optional.item_features.shape[1], 5)

    def test_movielens20m_single_row_input_still_loads(self) -> None:
        """Single-row ML-20M fixtures should still parse into a 2D numeric table."""
        with TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "MovieLens20M" / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "ratings.csv").write_text(
                "userId,movieId,rating,timestamp\n1,10,4.5,100\n",
                encoding="utf-8",
            )
            (raw_dir / "movies.csv").write_text(
                "movieId,title,genres\n10,Movie A,Action\n",
                encoding="utf-8",
            )

            canonical = load_movielens20m(data_dir=tmp_dir)

        self.assertEqual(len(canonical), 1)
        np.testing.assert_array_equal(
            canonical.user_id,
            np.array([0], dtype=canonical.user_id.dtype),
        )
        np.testing.assert_array_equal(
            canonical.item_id,
            np.array([0], dtype=canonical.item_id.dtype),
        )
        np.testing.assert_allclose(canonical.raw_target, np.array([4.5], dtype=np.float32))
        self.assertIsNotNone(canonical.item_features)
        assert canonical.item_features is not None
        self.assertEqual(canonical.item_features.shape, (1, 1))

    def test_movielens1m_single_row_input_uses_shared_explicit_contract(self) -> None:
        """ML-1M should keep the same explicit canonical contract after deduplication."""
        with TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "MovieLens1M" / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "ratings.dat").write_text(
                "1::10::5::100\n",
                encoding="latin-1",
            )
            (raw_dir / "users.dat").write_text(
                "1::F::25::3::00000\n",
                encoding="latin-1",
            )
            (raw_dir / "movies.dat").write_text(
                "10::Movie A (2000)::Action|Comedy\n",
                encoding="latin-1",
            )

            canonical = load_movielens1m(data_dir=tmp_dir)

        self.assertEqual(len(canonical), 1)
        np.testing.assert_array_equal(canonical.label, np.array([1.0], dtype=np.float32))
        np.testing.assert_array_equal(canonical.sign, np.array([1.0], dtype=np.float32))
        np.testing.assert_allclose(canonical.raw_target, np.array([5.0], dtype=np.float32))
        self.assertEqual(canonical.preprocessing_preset, "movielens_explicit")
        self.assertIsNotNone(canonical.user_features)
        self.assertIsNotNone(canonical.item_features)

    def test_kuairec_loader_parses_quoted_category_csv_fields(self) -> None:
        """KuaiRec auxiliary category files should survive quoted commas in CSV text fields."""
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "KuaiRec_v2" / "data"
            data_root.mkdir(parents=True)
            (data_root / "small_matrix.csv").write_text(
                ("user_id,video_id,watch_ratio,timestamp\n1,10,0.8,100\n1,11,0.2,200\n"),
                encoding="utf-8",
            )
            (data_root / "item_daily_features.csv").write_text(
                (
                    "video_id,date,author_id,video_type,upload_dt,upload_type,visible_status,music_id\n10,20200705,33,NORMAL,2020-03-30,ShortImport,public,77\n11,20200705,44,NORMAL,2020-03-30,ShortImport,private,88\n"
                ),
                encoding="utf-8",
            )
            (data_root / "item_categories.csv").write_text(
                'video_id,feat\n10,"[27, 9]"\n11,[8]\n',
                encoding="utf-8",
            )
            (data_root / "kuairec_caption_category.csv").write_text(
                (
                    "video_id,manual_cover_text,caption,topic_tag,"
                    "first_level_category_id,first_level_category_name,"
                    "second_level_category_id,second_level_category_name,"
                    "third_level_category_id,third_level_category_name\n"
                    '10,UNKNOWN,"caption, with comma",[],27,HighTech,124,Sub,9,Leaf\n'
                    "11,UNKNOWN,,[],8,Face,673,Sub,-124,UNKNOWN\n"
                ),
                encoding="utf-8",
            )

            canonical = load_kuairec_v2(data_dir=tmp_dir, matrix_variant="small_matrix")

        self.assertIsNotNone(canonical.item_features)
        assert canonical.item_features is not None
        self.assertEqual(canonical.item_features.shape, (2, 12))
        np.testing.assert_allclose(
            canonical.item_features,
            np.array(
                [
                    [1.0, 1.0, 1.0, 1585526400.0, 1.0, 2.0, 0.0, 1.0, 1.0, 27.0, 124.0, 9.0],
                    [2.0, 2.0, 1.0, 1585526400.0, 1.0, 1.0, 1.0, 0.0, 0.0, 8.0, 673.0, -124.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_kuairec_loader_skips_missing_caption_category_values(self) -> None:
        """KuaiRec caption-category parsing should ignore missing ordinal fields."""
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "KuaiRec_v2" / "data"
            data_root.mkdir(parents=True)
            (data_root / "small_matrix.csv").write_text(
                ("user_id,video_id,watch_ratio,timestamp\n1,10,0.8,100\n"),
                encoding="utf-8",
            )
            (data_root / "item_daily_features.csv").write_text(
                (
                    "video_id,date,author_id,video_type,upload_dt,upload_type,visible_status,music_id\n10,20200705,33,NORMAL,2020-03-30,ShortImport,public,77\n"
                ),
                encoding="utf-8",
            )
            (data_root / "item_categories.csv").write_text(
                'video_id,feat\n10,"[27, 9]"\n',
                encoding="utf-8",
            )
            (data_root / "kuairec_caption_category.csv").write_text(
                (
                    "video_id,manual_cover_text,caption,topic_tag,first_level_category_id,first_level_category_name,second_level_category_id,second_level_category_name,third_level_category_id,third_level_category_name\n10,UNKNOWN,,[],27,HighTech,124,Sub\n"
                ),
                encoding="utf-8",
            )

            canonical = load_kuairec_v2(data_dir=tmp_dir, matrix_variant="small_matrix")

        self.assertIsNotNone(canonical.item_features)
        assert canonical.item_features is not None
        self.assertEqual(canonical.item_features.shape[0], 1)
        self.assertGreaterEqual(canonical.item_features.shape[1], 1)
        self.assertTrue(np.isfinite(canonical.item_features).all())
        self.assertEqual(canonical.item_features[0, -1], 0.0)


class ReproducibilityHelperTests(unittest.TestCase):
    """Pin the shared RNG helper behavior used by the training runtime."""

    def test_seed_everything_resets_python_numpy_and_torch(self) -> None:
        """Reapplying the same seed should reproduce the same random draws."""
        seed_everything(17)
        first_python = random.random()
        first_numpy = float(np.random.random())
        first_torch = torch.rand(4)

        seed_everything(17)
        second_python = random.random()
        second_numpy = float(np.random.random())
        second_torch = torch.rand(4)

        self.assertEqual(first_python, second_python)
        self.assertEqual(first_numpy, second_numpy)
        self.assertTrue(torch.equal(first_torch, second_torch))

    def test_build_torch_generator_repeats_randperm(self) -> None:
        """Seeded generators should make epoch permutations repeatable."""
        first_perm = torch.randperm(
            16,
            generator=build_torch_generator(13, "cpu"),
            device="cpu",
        )
        second_perm = torch.randperm(
            16,
            generator=build_torch_generator(13, "cpu"),
            device="cpu",
        )
        different_perm = torch.randperm(
            16,
            generator=build_torch_generator(14, "cpu"),
            device="cpu",
        )

        self.assertTrue(torch.equal(first_perm, second_perm))
        self.assertFalse(torch.equal(first_perm, different_perm))
