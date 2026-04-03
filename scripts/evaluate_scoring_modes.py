#!/usr/bin/env python
"""Evaluate one trained checkpoint under multiple scoring modes.

This script is the thesis-facing mechanism-validation path: it reloads a single
trained checkpoint, rebuilds the matching dataset/graph, and reports the six
headline metrics under multiple evaluation-time scoring modes without
retraining.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from experiments.run_experiment import (
    build_runtime_model,
    load_checkpoint_payload,
    load_runtime_data,
)
from src.models.ucagnn import UCaGNN
from src.training.evaluator import Evaluator, THESIS_PRIMARY_METRICS
from src.utils.config import UCaGNNConfig
from scripts._workflow_helpers import write_json_report

DEFAULT_MODES = ("default", "interest_only", "conformity_suppressed")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for same-checkpoint scoring-mode evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a single checkpoint under multiple scoring modes"
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a completed training checkpoint produced by run_experiment.py",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=list(DEFAULT_MODES),
        choices=[
            "default",
            "interest_only",
            "conformity_only",
            "counterfactual_only",
            "conformity_suppressed",
        ],
        help="Evaluation-time scoring modes to compare",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "both"],
        default="test",
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Evaluation batch size for full-catalog scoring",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override; defaults to the checkpoint config device",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path for the collected metric table",
    )
    return parser.parse_args()


def _resolve_device(config: UCaGNNConfig, override: str | None) -> str:
    """Resolve the runtime device, falling back to CPU if CUDA is unavailable."""
    requested = override or config.device
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _mask_for_split(data, split: str):
    """Return the requested split mask from the graph data object."""
    if split == "val":
        return data.val_mask
    if split == "test":
        return data.test_mask
    raise ValueError(f"Unsupported split: {split}")


def _collect_mode_metrics(
    model: UCaGNN,
    data,
    config: UCaGNNConfig,
    modes: list[str],
    split: str,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """Evaluate one split under all requested scoring modes."""
    results: dict[str, dict[str, float]] = {}
    mask = _mask_for_split(data, split)
    for mode in modes:
        eval_config = dataclasses.replace(config, eval_scoring_mode=mode)
        evaluator = Evaluator(eval_config)
        metrics = evaluator.evaluate(model, data, mask, batch_size=batch_size)
        results[mode] = {
            metric_name: float(metrics.get(metric_name, 0.0))
            for metric_name in THESIS_PRIMARY_METRICS
        }
    return results


def _print_table(split: str, results: dict[str, dict[str, float]]) -> None:
    """Render a compact text table for the six thesis metrics."""
    print(f"\nSCORING MODE EVALUATION ({split})")
    print("Note: AveragePopularity is lower-is-better.")
    print(
        f"{'Mode':<24} | {'NDCG@20':>8} | {'Recall@20':>10} | {'AvgPop@20':>10} | {'NDCG@40':>8} | {'Recall@40':>10} | {'AvgPop@40':>10}"
    )
    print("-" * 101)
    for mode, metrics in results.items():
        print(
            f"{mode:<24} | {metrics['NDCG@20']:>8.4f} | {metrics['Recall@20']:>10.4f} | {metrics['AveragePopularity@20']:>10.4f} | "
            f"{metrics['NDCG@40']:>8.4f} | {metrics['Recall@40']:>10.4f} | {metrics['AveragePopularity@40']:>10.4f}"
        )


def main() -> int:
    """Run same-checkpoint evaluation under multiple scoring modes."""
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    probe_device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = load_checkpoint_payload(
        checkpoint_path,
        probe_device,
        require_runtime_keys=True,
        require_config=True,
    )
    config: UCaGNNConfig = checkpoint["config"]
    resolved_device = _resolve_device(config, args.device)
    config.device = resolved_device

    canonical, data = load_runtime_data(config)
    model = build_runtime_model(config, canonical, data)
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolved_device)
    model.eval()

    split_names = ["val", "test"] if args.split == "both" else [args.split]
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "dataset": config.dataset,
        "canonical_name": checkpoint.get("canonical_name"),
        "splits": {},
    }

    for split_name in split_names:
        split_results = _collect_mode_metrics(
            model,
            data,
            config,
            args.modes,
            split_name,
            args.batch_size,
        )
        payload["splits"][split_name] = split_results
        _print_table(split_name, split_results)

    if args.output_json is not None:
        output_path = write_json_report(args.output_json, payload)
        print(f"\nSaved JSON: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
