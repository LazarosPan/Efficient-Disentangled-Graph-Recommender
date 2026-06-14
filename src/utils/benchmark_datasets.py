"""Benchmark dataset selector helpers shared across CLIs and runners."""

from __future__ import annotations

BENCHMARK_DATASETS = [
    "amazonbook",
    "movielens1m",
    "movielens20m",
    "kuairec_v2",
    "taobao",
    "kuairand1k",
]
BENCHMARK_DATASET_TIERS: dict[str, list[str]] = {
    "small": ["amazonbook", "movielens1m"],
    "medium": ["kuairec_v2", "kuairand1k"],
    "large": ["movielens20m", "taobao"],
}
BENCHMARK_DATASET_TIERS["all"] = BENCHMARK_DATASETS
BENCHMARK_TIER_CHOICES = list(BENCHMARK_DATASET_TIERS)


def normalize_benchmark_datasets_arg(raw: object) -> list[str]:
    """Normalize a benchmark dataset selector field to a list of selectors."""
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return ["all"]


def resolve_benchmark_datasets(selectors: list[str] | str) -> list[str]:
    """Expand tier selectors or explicit datasets to a deduplicated dataset list."""
    if isinstance(selectors, str):
        selectors = [selectors]
    if "all" in selectors:
        return list(BENCHMARK_DATASET_TIERS["all"])

    known_datasets = set(BENCHMARK_DATASET_TIERS["all"])
    seen: dict[str, None] = {}
    for selector in selectors:
        if selector in BENCHMARK_DATASET_TIERS:
            for dataset_name in BENCHMARK_DATASET_TIERS[selector]:
                seen[dataset_name] = None
        elif selector in known_datasets:
            seen[selector] = None
        else:
            choices = sorted([*BENCHMARK_DATASET_TIERS, *known_datasets])
            raise ValueError(
                f"Unknown dataset or tier {selector!r}. Expected one of {choices}.",
            )
    return list(seen)


def benchmark_dataset_lookup_keys(dataset: str) -> list[str]:
    """Return exact, tier, and broad fallback lookup keys for one dataset."""
    keys = [dataset]
    for tier_name, tier_datasets in BENCHMARK_DATASET_TIERS.items():
        if tier_name == "all":
            continue
        if dataset in tier_datasets and tier_name not in keys:
            keys.append(tier_name)
    if "all" not in keys:
        keys.append("all")
    return keys


__all__ = [
    "BENCHMARK_DATASETS",
    "BENCHMARK_DATASET_TIERS",
    "BENCHMARK_TIER_CHOICES",
    "benchmark_dataset_lookup_keys",
    "normalize_benchmark_datasets_arg",
    "resolve_benchmark_datasets",
]
