#!/usr/bin/env python
"""Run tiny feature-utility and feature-policy probes for thesis datasets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_experiment import build_config, run_experiment
from src.data.feature_policy import (
    datasets_with_feature_utility,
    datasets_with_policy_ablation,
)


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CATEGORIES = ("utility", "policy")
DEFAULT_UTILITY_DATASETS = list(datasets_with_feature_utility())
DEFAULT_POLICY_DATASETS = list(datasets_with_policy_ablation())
DEFAULT_METRIC_KEYS = (
    "Precision@50",
    "Recall@50",
    "F1@50",
    "NDCG@50",
    "HitRatio@50",
    "MAP@50",
    "MRR@50",
    "Coverage@50",
    "AveragePopularity@50",
)
TINY_SAMPLE_INTERACTIONS = {
    "movielens1m": 100,
    "movielens20m": 100,
    "taobao": 100,
    "kuairec_v2": 100,
    "kuairand1k": 100,
}
TINY_LOADER_MAX_ROWS = {
    "movielens1m": 100,
    "movielens20m": 100,
    "taobao": 100,
    "kuairec_v2": 100,
    "kuairand1k": 100,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiny feature-utility and feature-policy probes for thesis datasets"
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        choices=DEFAULT_CATEGORIES,
        default=list(DEFAULT_CATEGORIES),
        help="Probe categories to run",
    )
    parser.add_argument(
        "--utility-datasets",
        nargs="*",
        default=DEFAULT_UTILITY_DATASETS,
        help="Datasets for ID-only vs thesis_default utility probes",
    )
    parser.add_argument(
        "--policy-datasets",
        nargs="*",
        default=DEFAULT_POLICY_DATASETS,
        help="Datasets for thesis_default vs all_optional policy probes",
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--device", default="cuda", help="Execution device")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per probe")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size per probe"
    )
    parser.add_argument(
        "--output-json",
        default="results/feature_policy_probes.json",
        help="Optional JSON output path for probe results",
    )
    parser.add_argument(
        "--enable-mlflow", action="store_true", help="Enable MLflow tracking for probes"
    )
    return parser.parse_args()


def _sample_interactions(dataset: str) -> int:
    return TINY_SAMPLE_INTERACTIONS.get(dataset, 100)


def _loader_max_rows(dataset: str) -> int:
    return TINY_LOADER_MAX_ROWS.get(dataset, 100)


def _make_namespace(
    args: argparse.Namespace,
    dataset: str,
    *,
    use_features: bool,
    feature_policy: str = "thesis_default",
):
    return argparse.Namespace(
        dataset=dataset,
        recipe="full_full_graph_knn",
        preset=None,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=None,
        n_gnn_layers=None,
        interest_gnn_layers=None,
        conformity_gnn_layers=None,
        lr=None,
        eval_scoring_mode=None,
        scoring_weight_mode=None,
        use_features=use_features,
        feature_policy=feature_policy,
        graph_method=None,
        training_mode=None,
        num_neighbors=None,
        sample_interactions=_sample_interactions(dataset),
        loader_max_rows=_loader_max_rows(dataset),
        device=args.device,
        data_dir=args.data_dir,
    )


def _tracked_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    return {key: metrics.get(key) for key in DEFAULT_METRIC_KEYS}


def _rank_delta(
    candidate: dict[str, float], baseline: dict[str, float], key: str
) -> float:
    return float(candidate.get(key, 0.0) - baseline.get(key, 0.0))


def _promotion_gate(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    candidate_name: str,
    comparison_kind: str,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    ndcg_delta = _rank_delta(candidate, baseline, "NDCG@50")
    recall_delta = _rank_delta(candidate, baseline, "Recall@50")
    map_delta = _rank_delta(candidate, baseline, "MAP@50")
    mrr_delta = _rank_delta(candidate, baseline, "MRR@50")
    hitratio_delta = _rank_delta(candidate, baseline, "HitRatio@50")
    coverage_delta = _rank_delta(candidate, baseline, "Coverage@50")
    avg_pop_delta = _rank_delta(candidate, baseline, "AveragePopularity@50")

    if ndcg_delta > 0.0:
        reasons.append(f"NDCG@50 improved by {ndcg_delta:.4f}")
    if recall_delta > 0.0:
        reasons.append(f"Recall@50 improved by {recall_delta:.4f}")
    if map_delta > 0.0:
        reasons.append(f"MAP@50 improved by {map_delta:.4f}")
    if mrr_delta > 0.0:
        reasons.append(f"MRR@50 improved by {mrr_delta:.4f}")
    if hitratio_delta > 0.0:
        reasons.append(f"HitRatio@50 improved by {hitratio_delta:.4f}")

    diversity_ok = True
    if coverage_delta < -1e-4:
        diversity_ok = False
        reasons.append("Coverage@50 regressed")
    if avg_pop_delta > 1e-4:
        diversity_ok = False
        reasons.append("AveragePopularity@50 increased (worse)")

    if comparison_kind == "policy":
        if (
            ndcg_delta > 1e-4
            or recall_delta > 1e-4
            or map_delta > 1e-4
            or mrr_delta > 1e-4
        ) and diversity_ok:
            return f"promote_{candidate_name}", reasons or [
                "ranking improved without PyG-metric regressions"
            ]
        if ndcg_delta <= 1e-4 and recall_delta <= 1e-4:
            return "keep_baseline", reasons or [
                "broader optional scans did not earn a ranking gain"
            ]
        return "review", reasons or ["mixed PyG metric signal"]

    if (
        ndcg_delta >= 0.0
        and recall_delta >= 0.0
        and map_delta >= 0.0
        and mrr_delta >= 0.0
        and diversity_ok
    ):
        return f"promote_{candidate_name}", reasons or [
            "ranking held or improved without PyG-metric regressions"
        ]
    if ndcg_delta <= -0.01 and recall_delta <= -0.01:
        return "keep_baseline", reasons or ["ranking regressed materially"]
    return "review", reasons or ["mixed PyG metric signal"]


def _run_probe_case(
    args: argparse.Namespace,
    dataset: str,
    *,
    use_features: bool,
    feature_policy: str,
    label: str,
) -> tuple[dict, float]:
    namespace = _make_namespace(
        args,
        dataset,
        use_features=use_features,
        feature_policy=feature_policy,
    )
    config = build_config(namespace)
    config.patience = 1
    config.enable_profiling = False

    started = time.perf_counter()
    result = run_experiment(
        config,
        preset="full",
        intervention=f"feature_probe_{label}",
        save_checkpoint=False,
        enable_mlflow=args.enable_mlflow,
        mlflow_experiment_name="ucagnn-feature-policy-probes",
        recipe_name="full_full_graph_knn",
        auto_resume=False,
    )
    return result, time.perf_counter() - started


def _print_case(
    dataset: str, label: str, elapsed: float, metrics: dict[str, float | None]
) -> None:
    summary = ", ".join(
        f"{name}={value:.4f}"
        for name, value in metrics.items()
        if value is not None
        and name in {"Precision@50", "Recall@50", "NDCG@50", "MAP@50", "Coverage@50"}
    )
    print(f"OK   {dataset:<12} {label:<28} {elapsed:>7.2f}s | {summary}")


def _run_utility_probes(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    print(
        f"Utility probes: {len(args.utility_datasets)} datasets (id_only vs thesis_default)"
    )
    for dataset in args.utility_datasets:
        baseline_result, baseline_elapsed = _run_probe_case(
            args,
            dataset,
            use_features=False,
            feature_policy="thesis_default",
            label=f"{dataset}_id_only",
        )
        candidate_result, candidate_elapsed = _run_probe_case(
            args,
            dataset,
            use_features=True,
            feature_policy="thesis_default",
            label=f"{dataset}_thesis_default",
        )

        baseline_metrics = _tracked_metrics(baseline_result["test_metrics"])
        candidate_metrics = _tracked_metrics(candidate_result["test_metrics"])
        decision, reasons = _promotion_gate(
            candidate_result["test_metrics"],
            baseline_result["test_metrics"],
            candidate_name="thesis_default",
            comparison_kind="utility",
        )

        _print_case(dataset, "id_only", baseline_elapsed, baseline_metrics)
        _print_case(dataset, "thesis_default", candidate_elapsed, candidate_metrics)
        print(f"GATE {dataset:<12} utility -> {decision} | {'; '.join(reasons)}")

        rows.append(
            {
                "category": "utility",
                "dataset": dataset,
                "baseline": {"label": "id_only", "metrics": baseline_metrics},
                "candidate": {"label": "thesis_default", "metrics": candidate_metrics},
                "decision": decision,
                "reasons": reasons,
            }
        )
    return rows


def _run_policy_probes(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    print(
        f"Policy probes: {len(args.policy_datasets)} datasets (thesis_default vs all_optional)"
    )
    for dataset in args.policy_datasets:
        baseline_result, baseline_elapsed = _run_probe_case(
            args,
            dataset,
            use_features=True,
            feature_policy="thesis_default",
            label=f"{dataset}_thesis_default",
        )
        candidate_result, candidate_elapsed = _run_probe_case(
            args,
            dataset,
            use_features=True,
            feature_policy="all_optional",
            label=f"{dataset}_all_optional",
        )

        baseline_metrics = _tracked_metrics(baseline_result["test_metrics"])
        candidate_metrics = _tracked_metrics(candidate_result["test_metrics"])
        decision, reasons = _promotion_gate(
            candidate_result["test_metrics"],
            baseline_result["test_metrics"],
            candidate_name="all_optional",
            comparison_kind="policy",
        )

        _print_case(dataset, "thesis_default", baseline_elapsed, baseline_metrics)
        _print_case(dataset, "all_optional", candidate_elapsed, candidate_metrics)
        print(f"GATE {dataset:<12} policy  -> {decision} | {'; '.join(reasons)}")

        rows.append(
            {
                "category": "policy",
                "dataset": dataset,
                "baseline": {"label": "thesis_default", "metrics": baseline_metrics},
                "candidate": {"label": "all_optional", "metrics": candidate_metrics},
                "decision": decision,
                "reasons": reasons,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    rows: list[dict[str, object]] = []

    print("=" * 78)
    print("FEATURE POLICY PROBES")
    print("=" * 78)

    if "utility" in args.categories:
        rows.extend(_run_utility_probes(args))
    if "policy" in args.categories:
        rows.extend(_run_policy_probes(args))

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "generated_at_seconds": time.time(),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("-" * 78)
    print(f"Wrote probe summary to {output_path}")
    print(f"Total time: {time.perf_counter() - started:.2f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
