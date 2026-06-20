"""Regression coverage for checkpoint retention pruning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prune_checkpoints import build_prune_plan, execute_plan
from src.utils.experiment_logger import ExperimentLogger


class CheckpointPruneTests(unittest.TestCase):
    """Pin SQLite-backed checkpoint retention decisions."""

    def _write_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        dataset: str,
        preset: str,
        training_hash: str,
    ) -> Path:
        path = checkpoint_dir / f"{dataset}_{preset}_seed13_train-{training_hash}.pt"
        path.write_bytes((training_hash + "\n").encode("utf-8"))
        return path

    def _log_run(
        self,
        logger: ExperimentLogger,
        *,
        dataset: str,
        preset: str,
        training_hash: str,
        score: float,
        status: str = "completed",
        batch_id: str | None = "formal-test",
        profile_name: str | None = "formal-profile",
    ) -> None:
        exp_id = logger.log_experiment(
            dataset,
            {"seed": 13},
            preset=preset,
            training_mode="mini_batch",
            status=status,
            batch_id=batch_id,
            profile_name=profile_name,
            training_hash=training_hash,
        )
        logger.log_metric(exp_id, "NDCG@40", score, split="val")

    def test_prune_plan_keeps_top_three_per_dataset_family(self) -> None:
        """The retention plan should keep only the best mapped family checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            db_path = root / "experiments.sqlite"
            logger = ExperimentLogger(db_path=str(db_path))
            try:
                hashes = [
                    "0000000000000001",
                    "0000000000000002",
                    "0000000000000003",
                    "0000000000000004",
                    "0000000000000005",
                ]
                for index, training_hash in enumerate(hashes, start=1):
                    self._write_checkpoint(
                        checkpoint_dir,
                        dataset="amazonbook",
                        preset="edgrec",
                        training_hash=training_hash,
                    )
                    self._log_run(
                        logger,
                        dataset="amazonbook",
                        preset="edgrec",
                        training_hash=training_hash,
                        score=index / 10,
                    )
                self._write_checkpoint(
                    checkpoint_dir,
                    dataset="amazonbook",
                    preset="edgrec",
                    training_hash="ffffffffffffffff",
                )
                self._log_run(
                    logger,
                    dataset="amazonbook",
                    preset="edgrec",
                    training_hash="ffffffffffffffff",
                    score=0.99,
                    batch_id="optuna-study-trial-0",
                    profile_name="edgrec-core-optimization",
                )
            finally:
                logger.close()

            unmatched = checkpoint_dir / "legacy_seed13.pt"
            unmatched.write_bytes(b"legacy")

            plan = build_prune_plan(
                checkpoint_dir=checkpoint_dir,
                db_path=db_path,
                keep=3,
            )

            deleted_hashes = {decision.training_hash for decision in plan.delete}
            kept_hashes = {decision.training_hash for decision in plan.keep}
            self.assertIn("0000000000000001", deleted_hashes)
            self.assertIn("0000000000000002", deleted_hashes)
            self.assertIn("ffffffffffffffff", deleted_hashes)
            self.assertIn("0000000000000003", kept_hashes)
            self.assertIn("0000000000000004", kept_hashes)
            self.assertIn("0000000000000005", kept_hashes)
            self.assertTrue(unmatched.exists())

            execute_plan(plan)

            self.assertFalse(
                (checkpoint_dir / "amazonbook_edgrec_seed13_train-0000000000000001.pt").exists(),
            )
            self.assertFalse(
                (checkpoint_dir / "amazonbook_edgrec_seed13_train-0000000000000002.pt").exists(),
            )
            self.assertFalse(
                (checkpoint_dir / "amazonbook_edgrec_seed13_train-ffffffffffffffff.pt").exists(),
            )
            self.assertTrue(
                (checkpoint_dir / "amazonbook_edgrec_seed13_train-0000000000000005.pt").exists(),
            )
            self.assertTrue(unmatched.exists())


if __name__ == "__main__":
    unittest.main()
