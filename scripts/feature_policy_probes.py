#!/usr/bin/env python
"""Run tiny feature-utility and feature-policy probes for thesis datasets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_experiment import build_config
from src.data.feature_policy import (
    datasets_with_feature_utility,
    datasets_with_policy_ablation,
)
from src.training.evaluator import THESIS_PRIMARY_METRICS
from scripts._workflow_helpers import (
    dataset_limit,
    timed_run_experiment,
    write_json_report,
)


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CATEGORIES = ("utility", "policy")
DEFAULT_UTILITY_DATASETS = list(datasets_with_feature_utility())
DEFAULT_POLICY_DATASETS = list(datasets_with_policy_ablation())
DEFAULT_METRIC_KEYS = THESIS_PRIMARY_METRICS
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
    return dataset_limit(dataset, TINY_SAMPLE_INTERACTIONS, default=100)


def _loader_max_rows(dataset: str) -> int:
    return dataset_limit(dataset, TINY_LOADER_MAX_ROWS, default=100)


def _make_namespace(
    args: argparse.Namespace,
    dataset: str,
    *,
    use_features: bool,
    feature_policy: str = "thesis_default",
) -> argparse.Namespace:
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
        intervention=None,
    )


def _build_runtime_config(namespace: argparse.Namespace):
    config = build_config(namespace)
    config.patience = 1
    config.enable_profiling = False
    return config


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
    ndcg_20_delta = _rank_delta(candidate, baseline, "NDCG@20")
    recall_20_delta = _rank_delta(candidate, baseline, "Recall@20")
    avg_pop_20_delta = _rank_delta(candidate, baseline, "AveragePopularity@20")
    ndcg_delta = _rank_delta(candidate, baseline, "NDCG@50")
    recall_delta = _rank_delta(candidate, baseline, "Recall@50")
    avg_pop_delta = _rank_delta(candidate, baseline, "AveragePopularity@50")

    if ndcg_20_delta > 0.0:
        reasons.append(f"NDCG@20 improved by {ndcg_20_delta:.4f}")
    if recall_20_delta > 0.0:
        reasons.append(f"Recall@20 improved by {recall_20_delta:.4f}")
    if ndcg_delta > 0.0:
        reasons.append(f"NDCG@50 improved by {ndcg_delta:.4f}")
    if recall_delta > 0.0:
        reasons.append(f"Recall@50 improved by {recall_delta:.4f}")

    popularity_ok = True
    if avg_pop_20_delta > 1e-4:
        popularity_ok = False
        reasons.append("AveragePopularity@20 increased (worse)")
    if avg_pop_delta > 1e-4:
        popularity_ok = False
        reasons.append("AveragePopularity@50 increased (worse)")

    if comparison_kind == "policy":
        if (
            ndcg_delta > 1e-4
            or recall_delta > 1e-4
            or ndcg_20_delta > 1e-4
            or recall_20_delta > 1e-4
        ) and popularity_ok:
            return f"promote_{candidate_name}", reasons or [
                "thesis metrics improved without popularity regressions"
            ]
        if (
            ndcg_20_delta <= 1e-4
            and recall_20_delta <= 1e-4
            and ndcg_delta <= 1e-4
            and recall_delta <= 1e-4
        ):
            return "keep_baseline", reasons or [
                "broader optional scans did not earn a thesis-metric gain"
            ]
        return "review", reasons or ["mixed thesis-metric signal"]

    if (
        ndcg_20_delta >= 0.0
        and recall_20_delta >= 0.0
        and ndcg_delta >= 0.0
        and recall_delta >= 0.0
        and popularity_ok
    ):
        return f"promote_{candidate_name}", reasons or [
            "thesis metrics held or improved without popularity regressions"
        ]
    if (
        ndcg_20_delta <= -0.01
        and recall_20_delta <= -0.01
        and ndcg_delta <= -0.01
        and recall_delta <= -0.01
    ):
        return "keep_baseline", reasons or ["thesis metrics regressed materially"]
    return "review", reasons or ["mixed thesis-metric signal"]


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
    config = _build_runtime_config(namespace)

    return timed_run_experiment(
        config,
        preset="full",
        intervention=f"feature_probe_{label}",
        save_checkpoint=False,
        enable_mlflow=args.enable_mlflow,
        mlflow_experiment_name="ucagnn-feature-policy-probes",
        recipe_name="full_full_graph_knn",
        auto_resume=False,
    )


def _print_case(
    dataset: str, label: str, elapsed: float, metrics: dict[str, float | None]
) -> None:
    summary = ", ".join(
        f"{name}={value:.4f}" for name, value in metrics.items() if value is not None
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

    output_path = write_json_report(
        args.output_json,
        {
            "generated_at_seconds": time.time(),
            "rows": rows,
        },
        root=PROJECT_ROOT,
    )

    print("-" * 78)
    print(f"Wrote probe summary to {output_path}")
    print(f"Total time: {time.perf_counter() - started:.2f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
