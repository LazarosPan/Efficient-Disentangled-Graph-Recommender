"""Regression coverage for the formal CRRU utility definition."""

from __future__ import annotations

import unittest

from src.utils.crru import (
    compute_validation_crru_for_k,
    compute_validation_crru_objective,
)


def _valid_metrics() -> dict[str, float]:
    return {
        "NDCG@20": 0.30,
        "Recall@20": 0.40,
        "HitRatio@20": 0.50,
        "Personalization@20": 0.60,
        "AveragePopularity@20": 2.0,
        "NDCG@40": 0.35,
        "Recall@40": 0.45,
        "HitRatio@40": 0.55,
        "Personalization@40": 0.65,
        "AveragePopularity@40": 2.5,
    }


class FormalCRRUTests(unittest.TestCase):
    """Pin CRRU as an absolute weighted geometric utility."""

    def test_validation_crru_is_bounded_and_aggregates_per_k_mean(self) -> None:
        """CRRU@20, CRRU@40, and their arithmetic mean stay in [0, 1]."""
        metrics = _valid_metrics()

        crru_20 = compute_validation_crru_for_k(
            metrics,
            k=20,
            peak_vram_mb=2048.0,
            epoch_time_s=2.0,
            largest_training_item_interaction_count=10.0,
        )
        crru_40 = compute_validation_crru_for_k(
            metrics,
            k=40,
            peak_vram_mb=2048.0,
            epoch_time_s=2.0,
            largest_training_item_interaction_count=10.0,
        )
        aggregate = compute_validation_crru_objective(
            metrics,
            peak_vram_mb=2048.0,
            epoch_time_s=2.0,
            largest_training_item_interaction_count=10.0,
        )

        for value in (crru_20, crru_40, aggregate):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(aggregate, (crru_20 + crru_40) / 2.0)

    def test_report_crru_is_independent_of_other_rows(self) -> None:
        """Adding unrelated rows must not change a run's already computed CRRU."""
        base_score = compute_validation_crru_for_k(
            _valid_metrics(),
            k=20,
            peak_vram_mb=1000.0,
            epoch_time_s=10.0,
            largest_training_item_interaction_count=10.0,
        )
        unrelated_metrics = _valid_metrics()
        unrelated_metrics["NDCG@20"] = 1.0
        _ = compute_validation_crru_for_k(
            unrelated_metrics,
            k=20,
            peak_vram_mb=1.0,
            epoch_time_s=0.1,
            largest_training_item_interaction_count=10.0,
        )
        repeated_score = compute_validation_crru_for_k(
            _valid_metrics(),
            k=20,
            peak_vram_mb=1000.0,
            epoch_time_s=10.0,
            largest_training_item_interaction_count=10.0,
        )

        self.assertAlmostEqual(base_score, repeated_score)

    def test_crru_monotonicity(self) -> None:
        """Improving benefits or reducing costs must not hurt CRRU."""
        metrics = _valid_metrics()
        base = compute_validation_crru_for_k(
            metrics,
            k=20,
            peak_vram_mb=2048.0,
            epoch_time_s=2.0,
            largest_training_item_interaction_count=10.0,
        )

        for metric_name in ("NDCG@20", "Recall@20", "HitRatio@20", "Personalization@20"):
            improved = dict(metrics)
            improved[metric_name] = min(1.0, metrics[metric_name] + 0.10)
            self.assertGreaterEqual(
                compute_validation_crru_for_k(
                    improved,
                    k=20,
                    peak_vram_mb=2048.0,
                    epoch_time_s=2.0,
                    largest_training_item_interaction_count=10.0,
                ),
                base,
            )

        higher_popularity = dict(metrics)
        higher_popularity["AveragePopularity@20"] = 8.0
        self.assertLessEqual(
            compute_validation_crru_for_k(
                higher_popularity,
                k=20,
                peak_vram_mb=2048.0,
                epoch_time_s=2.0,
                largest_training_item_interaction_count=10.0,
            ),
            base,
        )
        self.assertLessEqual(
            compute_validation_crru_for_k(
                metrics,
                k=20,
                peak_vram_mb=4096.0,
                epoch_time_s=2.0,
                largest_training_item_interaction_count=10.0,
            ),
            base,
        )
        self.assertLessEqual(
            compute_validation_crru_for_k(
                metrics,
                k=20,
                peak_vram_mb=2048.0,
                epoch_time_s=4.0,
                largest_training_item_interaction_count=10.0,
            ),
            base,
        )

    def test_crru_rejects_missing_or_invalid_inputs(self) -> None:
        """Missing, non-finite, or out-of-domain inputs must not receive utility."""
        metrics = _valid_metrics()
        for metric_name, bad_value in (
            ("NDCG@20", 1.2),
            ("Recall@20", -0.1),
            ("HitRatio@20", float("nan")),
            ("Personalization@20", float("inf")),
            ("AveragePopularity@20", -0.1),
        ):
            invalid = dict(metrics)
            invalid[metric_name] = bad_value
            with self.assertRaises(ValueError):
                compute_validation_crru_for_k(
                    invalid,
                    k=20,
                    peak_vram_mb=2048.0,
                    epoch_time_s=2.0,
                    largest_training_item_interaction_count=10.0,
                )

        invalid_popularity_scale = dict(metrics)
        invalid_popularity_scale["AveragePopularity@20"] = 11.0
        with self.assertRaises(ValueError):
            compute_validation_crru_for_k(
                invalid_popularity_scale,
                k=20,
                peak_vram_mb=2048.0,
                epoch_time_s=2.0,
                largest_training_item_interaction_count=10.0,
            )

        missing = dict(metrics)
        del missing["AveragePopularity@20"]
        with self.assertRaises(ValueError):
            compute_validation_crru_for_k(
                missing,
                k=20,
                peak_vram_mb=2048.0,
                epoch_time_s=2.0,
                largest_training_item_interaction_count=10.0,
            )
        with self.assertRaises(ValueError):
            compute_validation_crru_for_k(
                metrics,
                k=20,
                peak_vram_mb=2048.0,
                epoch_time_s=2.0,
                largest_training_item_interaction_count=None,
            )
        with self.assertRaises(ValueError):
            compute_validation_crru_for_k(
                metrics,
                k=20,
                peak_vram_mb=None,
                epoch_time_s=2.0,
                largest_training_item_interaction_count=10.0,
            )
        with self.assertRaises(ValueError):
            compute_validation_crru_for_k(
                metrics,
                k=20,
                peak_vram_mb=2048.0,
                epoch_time_s=0.0,
                largest_training_item_interaction_count=10.0,
            )


if __name__ == "__main__":
    unittest.main()
