#!/usr/bin/env python
"""Visualize the six benchmark datasets through the canonical loader path.

The plots are intentionally built from ``CanonicalInteractions`` rather than
raw files or learned embeddings so every dataset is shown on the same footing.
This keeps the figures thesis-friendly: they compare scale, sparsity,
long-tail behavior, temporal coverage, response signals, and dataset-specific
exposure/context signals without coupling the output to a trained model checkpoint.
Rerunning the script rewrites the same fixed PNG outputs in place.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from src.utils.benchmark_datasets import BENCHMARK_DATASETS
from src.utils.cli_parsers import build_explore_all_datasets_parser

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from src.data.canonical import CanonicalInteractions


def load_dataset_api() -> tuple[dict[str, Any], Any]:
    """Load dataset helpers after the repo-root path bootstrap.

    Args:
        None.

    Returns:
        Loader registry and loader function used by this script.

    """
    module = importlib.import_module("src.data.loaders")
    return module.LOADERS, module.load_dataset


LOADERS, load_dataset = load_dataset_api()

DISPLAY_NAMES = {
    "amazonbook": "Amazon-Book",
    "movielens1m": "MovieLens 1M",
    "movielens20m": "MovieLens 20M",
    "kuairec_v2": "KuaiRec v2",
    "taobao": "Taobao",
    "kuairand1k": "KuaiRand-1K",
}

# Dataset identity colors are intentionally kept away from the interaction-sign
# palette: no green for positive, red for negative, slate gray for neutral, or
# blue/yellow used by user/item topology figures.
DATASET_COLORS = dict(
    zip(
        BENCHMARK_DATASETS,
        (
            "#6a3d9a",  # AmazonBook: deep purple
            "#9c179e",  # MovieLens 1M: magenta-purple
            "#b07aa1",  # MovieLens 20M: muted mauve
            "#8c6d31",  # KuaiRec v2: umber
            "#a66f2e",  # Taobao: copper-brown
            "#7f4f24",  # KuaiRand 1K: dark bronze
        ),
        strict=False,
    ),
)
USER_COLOR = "#1f77b4"
ITEM_COLOR = "#f2c230"
ROW_COLOR = "#64748b"
STRUCTURE_COLOR = ROW_COLOR
RAW_RESPONSE_COLOR = ROW_COLOR
USER_FEATURE_COLOR = USER_COLOR
ITEM_FEATURE_COLOR = ITEM_COLOR
LABEL_POSITIVE_COLOR = "#2ca25f"
LABEL_NONPOSITIVE_COLOR = "#cbd5e1"
SIGN_POSITIVE_COLOR = "#00897b"
SIGN_ZERO_COLOR = "#6b7280"
SIGN_NEGATIVE_COLOR = "#de2d26"
EXPOSURE_RANDOM_COLOR = "#7c3aed"
EXPOSURE_STANDARD_COLOR = "#475569"
FEATURE_COLORS = (USER_FEATURE_COLOR, ITEM_FEATURE_COLOR)


@dataclass(frozen=True)
class DatasetSummary:
    """Compact dataset summary used for overview plots and console output."""

    name: str
    display_name: str
    n_users: int
    n_items: int
    n_interactions: int
    density: float
    pos_rate: float
    mean_sign: float
    positive_label_rate: float
    nonpositive_label_rate: float
    positive_sign_rate: float
    zero_sign_rate: float
    negative_sign_rate: float
    train_size: int
    val_size: int
    test_size: int
    split_source: str
    split_label: str
    timestamp_coverage: float
    unique_pair_count: int
    repeated_pair_share: float
    user_feature_dim: int
    item_feature_dim: int
    feedback_type: str | None
    feedback_description: str
    preprocessing_preset: str | None
    median_user_activity: float
    median_item_popularity: float
    modalities_label: str
    randomized_exposure_share: float | None


@dataclass(frozen=True)
class ResponseSignalSpec:
    """Plot metadata for one canonical response signal.

    Args:
        label: Plot title for the response distribution panel.
        axis_label: X-axis label for the plotted signal.
        value_labels: Optional categorical labels keyed by numeric value.

    """

    label: str
    axis_label: str
    value_labels: dict[float, str] | None = None


RAW_TARGET_RESPONSE_SPECS = {
    "movielens1m": ResponseSignalSpec("Rating distribution", "rating"),
    "movielens20m": ResponseSignalSpec("Rating distribution", "rating"),
    "kuairec_v2": ResponseSignalSpec("Watch-ratio distribution", "watch ratio"),
    "kuairand1k": ResponseSignalSpec(
        "Normalized watch-time distribution",
        "play time / video duration",
    ),
    "taobao": ResponseSignalSpec(
        "Interaction-type distribution",
        "interaction type",
        {
            0.0: "page view",
            1.0: "favorite",
            2.0: "cart",
            3.0: "purchase",
        },
    ),
}
DEFAULT_RAW_TARGET_RESPONSE_SPEC = ResponseSignalSpec(
    "Response-value distribution",
    "response value",
)
AMAZONBOOK_RESPONSE_SPEC = ResponseSignalSpec(
    "Observed interaction distribution",
    "interaction outcome",
    {1.0: "observed interaction"},
)
SIGNED_RESPONSE_SPEC = ResponseSignalSpec(
    "Signed-feedback distribution",
    "signed feedback score",
)
LABEL_RESPONSE_SPEC = ResponseSignalSpec(
    "Observed interaction distribution",
    "interaction outcome",
)


def display_name(name: str) -> str:
    """Return a thesis-friendly dataset label.

    Args:
        name: Internal loader registry name.

    Returns:
        Human-readable dataset label.

    """
    return DISPLAY_NAMES.get(name, name)


def describe_split_strategy(split_source: str) -> str:
    """Return a thesis-facing split description.

    Args:
        split_source: Internal split-source code.

    Returns:
        Human-readable split description.

    """
    descriptions = {
        "predefined": "provided train / validation / test split",
        "train/test+derived-val": "provided train / test split with validation carved from train",
        "derived:per_user_temporal": "per-user temporal split derived from timestamps",
        "derived:per_user_loader_order": "per-user loader-order split (no timestamps)",
    }
    return descriptions.get(split_source, split_source.replace("_", " "))


def describe_feedback_semantics(name: str) -> str:
    """Return an audience-facing description of the dataset response semantics.

    Args:
        name: Internal loader registry name.

    Returns:
        Human-readable feedback description.

    """
    descriptions = {
        "amazonbook": "implicit observed interactions only",
        "movielens1m": "explicit 1-to-5 star ratings",
        "movielens20m": "explicit 1-to-5 star ratings",
        "kuairec_v2": "watch-ratio feedback from short-video viewing",
        "taobao": "multi-behavior shopping interactions",
        "kuairand1k": "short-video engagement logs with randomized exposure",
    }
    return descriptions.get(name, "observed user-item interactions")


def describe_plotted_context(name: str) -> str:
    """Return the extra context emphasized in the per-dataset panels.

    Args:
        name: Internal loader registry name.

    Returns:
        Human-readable description of the dataset-specific context.

    """
    descriptions = {
        "amazonbook": "implicit interaction graph and split structure",
        "movielens1m": "ratings over time plus user and item metadata",
        "movielens20m": "ratings over time plus rich item metadata",
        "kuairec_v2": "watch ratio plus item-side content descriptors",
        "taobao": "interaction types from page view to purchase",
        "kuairand1k": "watch time, engagement labels, and exposure policy",
    }
    return descriptions.get(name, "interaction structure and auxiliary context")


def compact_feedback_label(name: str) -> str:
    """Return a short profile-card label for dataset feedback semantics."""
    labels = {
        "amazonbook": "implicit graph",
        "movielens1m": "explicit ratings",
        "movielens20m": "explicit ratings",
        "kuairec_v2": "watch ratio",
        "taobao": "shopping behavior",
        "kuairand1k": "video engagement + exposure",
    }
    return labels.get(name, "observed interactions")


def compact_split_label(split_source_value: str) -> str:
    """Return a short profile-card split label."""
    labels = {
        "predefined": "provided train/val/test",
        "train/test+derived-val": "train/test + derived val",
        "derived:per_user_temporal": "per-user temporal",
        "derived:per_user_loader_order": "per-user loader order",
    }
    return labels.get(split_source_value, split_source_value.replace("_", " "))


def split_source(canonical: CanonicalInteractions) -> str:
    """Describe where the train/val/test split came from.

    Args:
        canonical: Loaded canonical dataset.

    Returns:
        Short split-source label.

    """
    if (
        canonical.train_mask is not None
        and canonical.val_mask is not None
        and canonical.test_mask is not None
    ):
        return "predefined"
    if canonical.train_mask is not None and canonical.test_mask is not None:
        return "train/test+derived-val"
    if not np.any(canonical.timestamp > 0):
        return "derived:per_user_loader_order"
    return "derived:per_user_temporal"


def compute_entity_counts(entity_ids: np.ndarray, n_entities: int) -> np.ndarray:
    """Count interactions per user or item.

    Args:
        entity_ids: Reindexed user or item IDs for each interaction.
        n_entities: Total number of entities for the chosen axis.

    Returns:
        Count array aligned to entity IDs.

    """
    return np.bincount(entity_ids, minlength=n_entities)


def count_unique_user_item_pairs(
    user_id: np.ndarray,
    item_id: np.ndarray,
    n_items: int,
) -> int:
    """Return the number of distinct user-item interaction pairs.

    Args:
        user_id: Reindexed user IDs aligned to the interactions.
        item_id: Reindexed item IDs aligned to the interactions.
        n_items: Total number of items used to build a stable pair key.

    Returns:
        Number of unique user-item pairs represented in the interaction table.

    """
    if user_id.size == 0:
        return 0
    pair_keys = user_id.astype(np.int64, copy=False) * int(n_items) + item_id.astype(
        np.int64,
        copy=False,
    )
    return int(np.unique(pair_keys).size)


def top_category_counts(
    values: np.ndarray,
    top_k: int = 6,
) -> tuple[list[str], np.ndarray]:
    """Return the most common categorical values with an optional ``other`` bin.

    Args:
        values: Categorical array to summarize.
        top_k: Maximum number of bars to emit.

    Returns:
        Tuple of labels and aligned counts.

    """
    if values.size == 0:
        return [], np.array([], dtype=np.int64)

    labels, counts = np.unique(values.astype(str, copy=False), return_counts=True)
    order = np.argsort(counts)[::-1]
    labels = labels[order]
    counts = counts[order]

    if labels.size <= top_k:
        return labels.tolist(), counts

    head_labels = [*labels[: top_k - 1].tolist(), "other"]
    head_counts = np.concatenate(
        [counts[: top_k - 1], np.array([counts[top_k - 1 :].sum()], dtype=counts.dtype)],
    )
    return head_labels, head_counts


def summarize_dataset(name: str, canonical: CanonicalInteractions) -> DatasetSummary:
    """Compute scalar summary statistics for one dataset.

    Args:
        name: Loader registry name.
        canonical: Loaded canonical interactions.

    Returns:
        Scalar dataset summary used for printing and overview charts.

    """
    train_mask, val_mask, test_mask = canonical.get_splits()
    user_counts = compute_entity_counts(canonical.user_id, canonical.n_users)
    item_counts = compute_entity_counts(canonical.item_id, canonical.n_items)
    active_user_counts = user_counts[user_counts > 0]
    active_item_counts = item_counts[item_counts > 0]
    unique_pair_count = count_unique_user_item_pairs(
        canonical.user_id,
        canonical.item_id,
        canonical.n_items,
    )

    resolved_split_source = split_source(canonical)
    feedback_description = describe_feedback_semantics(name)
    plotted_context = describe_plotted_context(name)

    return DatasetSummary(
        name=name,
        display_name=display_name(name),
        n_users=canonical.n_users,
        n_items=canonical.n_items,
        n_interactions=len(canonical),
        density=(
            len(canonical) / float(canonical.n_users * canonical.n_items)
            if canonical.n_users > 0 and canonical.n_items > 0
            else 0.0
        ),
        pos_rate=float(np.mean(canonical.label)),
        mean_sign=float(np.mean(canonical.sign)),
        positive_label_rate=float(np.mean(canonical.label > 0)),
        nonpositive_label_rate=float(np.mean(canonical.label <= 0)),
        positive_sign_rate=float(np.mean(canonical.sign > 0)),
        zero_sign_rate=float(np.mean(canonical.sign == 0)),
        negative_sign_rate=float(np.mean(canonical.sign < 0)),
        train_size=int(train_mask.sum()),
        val_size=int(val_mask.sum()),
        test_size=int(test_mask.sum()),
        split_source=resolved_split_source,
        split_label=describe_split_strategy(resolved_split_source),
        timestamp_coverage=float(np.mean(canonical.timestamp > 0)),
        unique_pair_count=unique_pair_count,
        repeated_pair_share=(
            1.0 - (unique_pair_count / float(len(canonical))) if len(canonical) > 0 else 0.0
        ),
        user_feature_dim=(
            int(canonical.user_features.shape[1]) if canonical.user_features is not None else 0
        ),
        item_feature_dim=(
            int(canonical.item_features.shape[1]) if canonical.item_features is not None else 0
        ),
        feedback_type=canonical.feedback_type,
        feedback_description=feedback_description,
        preprocessing_preset=canonical.preprocessing_preset,
        median_user_activity=(
            float(np.median(active_user_counts)) if active_user_counts.size > 0 else 0.0
        ),
        median_item_popularity=(
            float(np.median(active_item_counts)) if active_item_counts.size > 0 else 0.0
        ),
        modalities_label=plotted_context,
        randomized_exposure_share=(
            float(np.mean(canonical.exposure_flag))
            if canonical.exposure_flag is not None and len(canonical.exposure_flag) > 0
            else None
        ),
    )


def summarize_numeric_signal(values: np.ndarray) -> dict[str, float | int] | None:
    """Return compact descriptive statistics for one numeric response signal.

    Args:
        values: Numeric values to summarize.

    Returns:
        Summary dictionary over finite values or ``None`` when none exist.

    """
    finite_values = values[np.isfinite(values)].astype(np.float64, copy=False)
    if finite_values.size == 0:
        return None
    return {
        "count": int(finite_values.size),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
    }


def build_dataset_summary_payload(
    canonical: CanonicalInteractions,
    summary: DatasetSummary,
) -> dict[str, Any]:
    """Build a text-friendly dataset summary payload from canonical data.

    Args:
        canonical: Loaded canonical interactions.
        summary: Scalar summary for the same dataset.

    Returns:
        JSON-serializable payload capturing the main statistics behind the plots.

    """
    payload: dict[str, Any] = asdict(summary)
    payload["split_counts"] = {
        "train": summary.train_size,
        "val": summary.val_size,
        "test": summary.test_size,
    }

    if canonical.raw_target is not None:
        payload["response_signal"] = {
            "name": "raw_target",
            "summary": summarize_numeric_signal(np.asarray(canonical.raw_target)),
        }
    elif np.unique(canonical.sign).size > 1:
        payload["response_signal"] = {
            "name": "sign",
            "summary": summarize_numeric_signal(np.asarray(canonical.sign)),
        }
    else:
        payload["response_signal"] = {
            "name": "label",
            "summary": summarize_numeric_signal(np.asarray(canonical.label)),
        }

    denominator = float(len(canonical)) if len(canonical) else 1.0
    positive_label_mask = canonical.label > 0
    nonpositive_label_mask = canonical.label <= 0
    positive_sign_mask = canonical.sign > 0
    zero_sign_mask = canonical.sign == 0
    negative_sign_mask = canonical.sign < 0
    label_counts = {
        "positive_label": int(positive_label_mask.sum()),
        "nonpositive_label": int(nonpositive_label_mask.sum()),
    }
    sign_counts = {
        "positive_sign": int(positive_sign_mask.sum()),
        "zero_sign": int(zero_sign_mask.sum()),
        "negative_sign": int(negative_sign_mask.sum()),
    }
    payload["label_distribution"] = {
        "counts": label_counts,
        "shares": {key: count / denominator for key, count in label_counts.items()},
        "note": "Binary label defines recommendation relevance and positive graph edges.",
    }
    payload["sign_distribution"] = {
        "counts": sign_counts,
        "shares": {key: count / denominator for key, count in sign_counts.items()},
        "note": "Sign is a graded feedback score stored separately from the binary label.",
    }
    payload["label_sign_overlap"] = {
        "positive_label_positive_sign": int((positive_label_mask & positive_sign_mask).sum()),
        "positive_label_zero_sign": int((positive_label_mask & zero_sign_mask).sum()),
        "positive_label_negative_sign": int((positive_label_mask & negative_sign_mask).sum()),
        "nonpositive_label_negative_sign": int(
            (nonpositive_label_mask & negative_sign_mask).sum(),
        ),
    }

    if canonical.exposure_flag is not None and canonical.exposure_flag.size > 0:
        randomized_mask = np.asarray(canonical.exposure_flag, dtype=bool)
        standard_mask = ~randomized_mask
        payload["exposure_policy"] = {
            "randomized_count": int(randomized_mask.sum()),
            "standard_count": int(standard_mask.sum()),
            "randomized_share": float(np.mean(randomized_mask)),
            "randomized_positive_rate": (
                float(np.mean(canonical.label[randomized_mask]))
                if np.any(randomized_mask)
                else None
            ),
            "standard_positive_rate": (
                float(np.mean(canonical.label[standard_mask])) if np.any(standard_mask) else None
            ),
        }

    if canonical.behavior_type is not None and np.unique(canonical.behavior_type).size > 1:
        labels, counts = top_category_counts(canonical.behavior_type)
        payload["behavior_mix_top"] = {
            label: int(count) for label, count in zip(labels, counts.tolist(), strict=True)
        }

    if canonical.source_domain is not None and np.unique(canonical.source_domain).size > 1:
        domain_summary: dict[str, dict[str, float | int]] = {}
        for label in np.unique(canonical.source_domain.astype(str, copy=False)).tolist():
            mask = canonical.source_domain.astype(str, copy=False) == label
            domain_summary[label] = {
                "count": int(mask.sum()),
                "positive_rate": float(np.mean(canonical.label[mask])) if np.any(mask) else 0.0,
            }
        payload["source_domain_positive_rate"] = domain_summary

    return payload


def render_summary_markdown(dataset_payloads: list[dict[str, Any]]) -> str:
    """Render a concise markdown summary for the benchmark datasets.

    Args:
        dataset_payloads: Per-dataset summary payloads.

    Returns:
        Markdown text that mirrors the machine-readable summary export.

    """
    lines = [
        "# Benchmark Dataset Summary",
        "",
        "Generated from `src/data_exploration/explore_all_datasets.py` using the same",
        "canonical statistics that power the benchmark figures.",
        "",
        "## How to Read the Figures",
        "",
        (
            "- `benchmark_overview.png` is the cross-dataset scale and feedback-semantics "
            "figure. Dataset colors identify datasets only. Green is reserved for "
            "`label > 0` graph/relevance rows; teal/red/slate-gray show `sign > 0`, "
            "`sign < 0`, and `sign = 0` feedback rows."
        ),
        (
            "- Each `*_profile.png` uses the same evidence order: entity/row scale, user "
            "long tail, item long tail, binary label versus graded sign, response signal "
            "or split, and the most relevant dataset-specific context."
        ),
        (
            "- `label > 0` is the binary relevance signal used for positive graph edges and "
            "top-K relevance. `sign` is a separate graded feedback descriptor, drawn in "
            "a different color family. They often overlap, but KuaiRec and KuaiRand show "
            "why they should not be treated as identical."
        ),
        (
            "- Constant or unavailable signals are not forced into one-bar charts. Exact "
            "counts and omitted-signal explanations are kept here and in "
            "`benchmark_summary.json`."
        ),
        "",
        "## Committee Questions Covered",
        "",
        "| Likely question | Evidence to cite |",
        "| --- | --- |",
        (
            "| How large and sparse is each dataset? | Overview scale/density panels plus the "
            "profile scale panel. |"
        ),
        (
            "| Are the datasets long-tailed? | User-activity and item-popularity CCDF panels "
            "in each profile. |"
        ),
        (
            "| What is a positive edge for the GCN? | Binary label row in each profile and the "
            "Chapter 3 graph-construction description. |"
        ),
        (
            "| Are negative and neutral feedback available? | Sign row in each profile and the "
            "sign panel in the overview. |"
        ),
        (
            "| Are label and sign the same thing? | Label/sign stacked bars and the overlap "
            "counts below; mismatches are explicitly reported as `label > 0 AND sign < 0`. |"
        ),
        (
            "| Why are side features used only for some datasets? | Feature availability/context "
            "panels and the safe-feature policy in Chapter 3. |"
        ),
        (
            "| Which plots are descriptive rather than performance evidence? | All figures in "
            "this directory are setup evidence; ranking claims require the results tables. |"
        ),
        "",
        "## Dataset Table",
        "",
        (
            "| Dataset | Interactions | Pair reuse | Positive share | Timestamp coverage | "
            "Randomized share | User feat | Item feat | Split |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for payload in dataset_payloads:
        randomized_share = payload.get("randomized_exposure_share")
        randomized_text = f"{randomized_share:.1%}" if randomized_share is not None else "-"
        lines.append(
            f"| {payload['display_name']} | {payload['n_interactions']:,} | "
            f"{payload['repeated_pair_share']:.2%} | {payload['pos_rate']:.2%} | "
            f"{payload['timestamp_coverage']:.1%} | {randomized_text} | "
            f"{payload['user_feature_dim']} | {payload['item_feature_dim']} | "
            f"{payload['split_label']} |"
        )
    lines.append("")

    for payload in dataset_payloads:
        split_counts = payload["split_counts"]
        lines.extend(
            [
                f"## {payload['display_name']}",
                "",
                f"- Interaction semantics: {payload['feedback_description']}",
                f"- Plotted context: {payload['modalities_label']}",
                f"- Distinct user-item pairs: {payload['unique_pair_count']:,}",
                f"- Repeated-pair share: {payload['repeated_pair_share']:.2%}",
                (
                    f"- Split counts: train={split_counts['train']:,}, "
                    f"val={split_counts['val']:,}, test={split_counts['test']:,}"
                ),
            ],
        )
        response_signal = payload.get("response_signal", {})
        response_summary = response_signal.get("summary")
        if response_summary is not None:
            lines.append(
                (
                f"- Response summary ({response_signal['name']}): mean="
                f"{response_summary['mean']:.4f}, std={response_summary['std']:.4f}, "
                f"min={response_summary['min']:.4f}, max={response_summary['max']:.4f}"
                ),
            )
        label_distribution = payload.get("label_distribution", {})
        label_counts = label_distribution.get("counts", {})
        if isinstance(label_counts, dict) and label_counts:
            lines.append(
                "- Label distribution: "
                f"label > 0={label_counts.get('positive_label', 0):,}, "
                f"label <= 0={label_counts.get('nonpositive_label', 0):,}.",
            )
        sign_distribution = payload.get("sign_distribution", {})
        sign_counts = sign_distribution.get("counts", {})
        if isinstance(sign_counts, dict) and sign_counts:
            lines.append(
                "- Sign distribution: "
                f"sign > 0={sign_counts.get('positive_sign', 0):,}, "
                f"sign = 0={sign_counts.get('zero_sign', 0):,}, "
                f"sign < 0={sign_counts.get('negative_sign', 0):,}.",
            )
            overlap = payload.get("label_sign_overlap", {}).get(
                "positive_label_negative_sign",
            )
            if overlap:
                lines.append(
                    f"- Overlap rows (`label > 0` and `sign < 0`): {int(overlap):,}.",
                )
        exposure_policy = payload.get("exposure_policy")
        if exposure_policy is not None:
            randomized_positive_rate = exposure_policy["randomized_positive_rate"]
            standard_positive_rate = exposure_policy["standard_positive_rate"]
            randomized_positive_text = (
                f"{randomized_positive_rate:.2%}" if randomized_positive_rate is not None else "-"
            )
            standard_positive_text = (
                f"{standard_positive_rate:.2%}" if standard_positive_rate is not None else "-"
            )
            lines.append(
                ""
                f"- Exposure policy: randomized_share={exposure_policy['randomized_share']:.2%}, "
                f"positive_rate_randomized={randomized_positive_text}, "
                f"positive_rate_standard={standard_positive_text}"
            )
        behavior_mix = payload.get("behavior_mix_top")
        if behavior_mix:
            mix_text = ", ".join(f"{label}={count:,}" for label, count in behavior_mix.items())
            lines.append(f"- Top behavior mix: {mix_text}")
        domain_rates = payload.get("source_domain_positive_rate")
        if domain_rates:
            domain_text = ", ".join(
                f"{label}: n={stats['count']:,}, pos={stats['positive_rate']:.2%}"
                for label, stats in domain_rates.items()
            )
            lines.append(f"- Source domains: {domain_text}")
        lines.append("")

    return "\n".join(lines)


def save_summary_exports(
    dataset_payloads: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write machine-readable and markdown summary exports beside the figures.

    Args:
        dataset_payloads: Per-dataset benchmark summary payloads.
        output_dir: Directory where the exports should be written.

    Returns:
        Paths to the JSON and markdown summary files.

    """
    json_path = output_dir / "benchmark_summary.json"
    markdown_path = output_dir / "benchmark_summary.md"
    json_path.write_text(
        json.dumps({"datasets": dataset_payloads}, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_summary_markdown(dataset_payloads),
        encoding="utf-8",
    )
    return json_path, markdown_path


def format_scalar(value: float) -> str:
    """Format a scalar for axis labels or dataset cards.

    Args:
        value: Numeric value to format.

    Returns:
        Compact string representation.

    """
    rounded = round(float(value))
    if np.isclose(value, rounded):
        return str(int(rounded))
    return f"{float(value):.2f}"


def plot_tail_ccdf(
    ax: Axes,
    counts: np.ndarray,
    title: str,
    xlabel: str,
    population_label: str,
    color: str,
) -> None:
    """Plot a log-log complementary CDF for long-tail activity.

    Args:
        ax: Matplotlib axes that receive the plot.
        counts: Per-user or per-item interaction counts.
        title: Axes title.
        xlabel: X-axis label.
        population_label: Name of the population shown on the y-axis.
        color: Line color.

    Returns:
        None. The axes are modified in place.

    """
    positive_counts = counts[counts > 0]
    if positive_counts.size == 0:
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No positive counts", ha="center", va="center")
        return

    values, frequencies = np.unique(positive_counts, return_counts=True)
    ccdf = np.cumsum(frequencies[::-1])[::-1] / positive_counts.size
    ax.step(values, ccdf, where="post", color=color, linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"share of {population_label} with at least x interactions")
    ax.grid(True, which="both", alpha=0.25)


def plot_text_summary_panel(
    ax: Axes,
    title: str,
    lines: list[str],
    color: str,
) -> None:
    """Render an explanatory panel when a chart would be visually meaningless."""
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.05,
        0.72,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        color="#111827",
        linespacing=1.35,
    )


def plot_scale_summary_panel(ax: Axes, summary: DatasetSummary) -> None:
    """Plot core entity and row scale for one dataset profile."""
    labels = ["users", "items", "rows"]
    values = [summary.n_users, summary.n_items, summary.n_interactions]
    colors = [USER_COLOR, ITEM_COLOR, ROW_COLOR]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_title("Entity and row scale")
    ax.set_ylabel("count (log scale)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.margins(y=0.18)
    ax.text(
        0.02,
        0.96,
        f"density {summary.density * 100:.5f}%\n{compact_split_label(summary.split_source)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def plot_relative_time_histogram(ax: Axes, timestamps: np.ndarray, color: str) -> None:
    """Plot interaction density over relative dataset time.

    Args:
        ax: Matplotlib axes that receive the plot.
        timestamps: Interaction timestamps aligned to the canonical rows.
        color: Histogram color.

    Returns:
        None. The axes are modified in place.

    """
    valid_timestamps = timestamps[timestamps > 0].astype(np.float64, copy=False)
    ax.set_title("Interaction volume over relative time")

    if valid_timestamps.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No usable timestamps", ha="center", va="center")
        return

    start_timestamp = float(valid_timestamps.min())
    end_timestamp = float(valid_timestamps.max())
    if np.isclose(start_timestamp, end_timestamp):
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Timestamps collapse to one instant",
            ha="center",
            va="center",
        )
        return

    relative_time = (valid_timestamps - start_timestamp) / (end_timestamp - start_timestamp)
    bin_edges = np.linspace(0.0, 1.0, 25)
    bin_counts, _ = np.histogram(relative_time, bins=bin_edges)
    occupied_bins = np.flatnonzero(bin_counts > 0)
    dominant_bin_share = float(bin_counts.max() / max(1, bin_counts.sum()))
    if occupied_bins.size <= 3 or dominant_bin_share >= 0.90:
        if dominant_bin_share >= 0.999:
            concentration_line = ">99.9% of rows fall in one time bin"
        elif dominant_bin_share >= 0.90:
            concentration_line = f"{dominant_bin_share:.1%} of rows fall in one time bin"
        else:
            concentration_line = (
                f"only {occupied_bins.size}/{bin_counts.size} relative-time bins are populated"
            )
        plot_text_summary_panel(
            ax,
            "Interaction time coverage",
            [
                f"{valid_timestamps.size:,} timestamped rows",
                concentration_line,
                "histogram omitted: temporal signal is concentrated",
            ],
            color,
        )
        return

    ax.hist(
        relative_time,
        bins=bin_edges,
        color=color,
        edgecolor="white",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("relative time within dataset")
    ax.set_ylabel("number of interactions")
    ax.grid(True, alpha=0.2)


def response_signal_for_plot(
    canonical: CanonicalInteractions,
    summary: DatasetSummary,
) -> tuple[np.ndarray, ResponseSignalSpec]:
    """Return response values and plot metadata for one dataset.

    Args:
        canonical: Loaded canonical dataset.
        summary: Scalar dataset summary for label selection.

    Returns:
        Tuple of numeric values and response plot metadata.

    """
    if summary.name == "amazonbook":
        return np.asarray(canonical.label), AMAZONBOOK_RESPONSE_SPEC

    if canonical.raw_target is not None:
        spec = RAW_TARGET_RESPONSE_SPECS.get(summary.name, DEFAULT_RAW_TARGET_RESPONSE_SPEC)
        return np.asarray(canonical.raw_target), spec

    if np.any(canonical.sign != canonical.sign[0]):
        return np.asarray(canonical.sign), SIGNED_RESPONSE_SPEC

    return np.asarray(canonical.label), LABEL_RESPONSE_SPEC


def _annotate_stacked_bar_segments(
    ax: Axes,
    left_edges: np.ndarray,
    widths: np.ndarray,
    y_position: float,
) -> None:
    """Annotate readable stacked-bar segments with their row share."""
    for left, width in zip(left_edges, widths, strict=True):
        if width < 0.075:
            continue
        ax.text(
            left + width / 2.0,
            y_position,
            f"{width:.0%}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if width > 0.30 else "black",
        )


def plot_label_sign_summary(ax: Axes, canonical: CanonicalInteractions) -> None:
    """Plot binary label and graded sign distributions as separate row summaries."""
    denominator = max(1, len(canonical))
    positive_label_mask = canonical.label > 0
    nonpositive_label_mask = canonical.label <= 0
    positive_sign_mask = canonical.sign > 0
    zero_sign_mask = canonical.sign == 0
    negative_sign_mask = canonical.sign < 0
    label_shares = np.asarray(
        [
            positive_label_mask.sum() / denominator,
            nonpositive_label_mask.sum() / denominator,
        ],
        dtype=np.float64,
    )
    sign_shares = np.asarray(
        [
            positive_sign_mask.sum() / denominator,
            zero_sign_mask.sum() / denominator,
            negative_sign_mask.sum() / denominator,
        ],
        dtype=np.float64,
    )

    label_left = np.concatenate(([0.0], np.cumsum(label_shares[:-1])))
    sign_left = np.concatenate(([0.0], np.cumsum(sign_shares[:-1])))
    ax.barh(
        [1.0] * 2,
        label_shares,
        left=label_left,
        height=0.32,
        color=[LABEL_POSITIVE_COLOR, LABEL_NONPOSITIVE_COLOR],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.barh(
        [0.0] * 3,
        sign_shares,
        left=sign_left,
        height=0.32,
        color=[SIGN_POSITIVE_COLOR, SIGN_ZERO_COLOR, SIGN_NEGATIVE_COLOR],
        edgecolor="white",
        linewidth=0.6,
    )
    _annotate_stacked_bar_segments(ax, label_left, label_shares, 1.0)
    _annotate_stacked_bar_segments(ax, sign_left, sign_shares, 0.0)

    overlap = int((positive_label_mask & negative_sign_mask).sum())
    if overlap:
        ax.text(
            0.01,
            0.05,
            f"overlap: label > 0 and sign < 0 = {overlap:,}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2.5},
        )
    elif int(positive_sign_mask.sum() + negative_sign_mask.sum()) == 0:
        ax.text(
            0.01,
            0.05,
            "no graded sign; sign stored as 0",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2.5},
        )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=LABEL_POSITIVE_COLOR,
            label="label > 0 (GCN edge)",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=LABEL_NONPOSITIVE_COLOR,
            label="label <= 0",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=SIGN_POSITIVE_COLOR,
            label="sign > 0",
        ),
        plt.Rectangle((0, 0), 1, 1, color=SIGN_ZERO_COLOR, label="sign = 0"),
        plt.Rectangle((0, 0), 1, 1, color=SIGN_NEGATIVE_COLOR, label="sign < 0"),
    ]
    ax.set_title("Binary label vs feedback sign")
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["label", "sign"])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("share of canonical rows")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=8)


def plot_response_distribution(
    ax: Axes,
    canonical: CanonicalInteractions,
    summary: DatasetSummary,
    color: str,
) -> None:
    """Plot the richest available response signal for one dataset.

    Args:
        ax: Matplotlib axes that receive the plot.
        canonical: Loaded canonical dataset.
        summary: Scalar dataset summary for label selection.
        color: Plot color.

    Returns:
        None. The axes are modified in place.

    """
    response_values, response_spec = response_signal_for_plot(canonical, summary)

    finite_values = response_values[np.isfinite(response_values)]
    ax.set_title(response_spec.label)

    if finite_values.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
        return

    sample_size = min(finite_values.size, 50_000)
    sample_indices = np.linspace(0, finite_values.size - 1, sample_size, dtype=np.int64)
    sample_values = finite_values[sample_indices]

    if np.unique(sample_values).size <= 12:
        unique_values, counts = np.unique(finite_values, return_counts=True)
        tick_labels = [
            response_spec.value_labels.get(float(value), format_scalar(value))
            if response_spec.value_labels is not None
            else format_scalar(value)
            for value in unique_values
        ]
        if unique_values.size == 1:
            plot_text_summary_panel(
                ax,
                response_spec.label,
                [
                    f"{finite_values.size:,} rows share one response value",
                    f"value: {tick_labels[0]}",
                    "bar chart omitted because there is no distribution",
                ],
                color,
            )
            return

        ax.bar(
            tick_labels,
            counts,
            color=color,
            edgecolor=STRUCTURE_COLOR,
            linewidth=0.4,
        )
        ax.set_xlabel(response_spec.axis_label)
        ax.set_ylabel("number of interactions")
    else:
        ax.hist(finite_values, bins=40, color=color, edgecolor="white")
        ax.set_xlabel(response_spec.axis_label)
        ax.set_ylabel("number of interactions")
    ax.grid(True, axis="y", alpha=0.2)


def response_signal_has_variation(
    canonical: CanonicalInteractions,
    summary: DatasetSummary,
) -> bool:
    """Return whether the selected response signal is worth plotting as a distribution."""
    response_values, _response_spec = response_signal_for_plot(canonical, summary)
    finite_values = response_values[np.isfinite(response_values)]
    if finite_values.size == 0:
        return False
    sample_size = min(finite_values.size, 50_000)
    sample_indices = np.linspace(0, finite_values.size - 1, sample_size, dtype=np.int64)
    return bool(np.unique(finite_values[sample_indices]).size > 1)


def plot_context_panel(
    ax: Axes,
    canonical: CanonicalInteractions,
    summary: DatasetSummary,
    accent_color: str,
) -> None:
    """Plot dataset-specific context or side-feature availability.

    Args:
        ax: Matplotlib axes that receive the plot.
        canonical: Loaded canonical dataset.
        summary: Scalar summary for the same dataset.
        accent_color: Dataset identity color for non-semantic descriptive bars.

    Returns:
        None. The axes are modified in place.

    """
    if canonical.exposure_flag is not None:
        plot_group_positive_rate(
            ax=ax,
            group_labels=["randomized", "standard"],
            group_masks=[
                np.asarray(canonical.exposure_flag, dtype=bool),
                ~np.asarray(canonical.exposure_flag, dtype=bool),
            ],
            labels=canonical.label,
            title="Share labeled positive by exposure policy",
        )
        return

    if summary.name == "taobao" and (summary.user_feature_dim > 0 or summary.item_feature_dim > 0):
        if np.count_nonzero([summary.user_feature_dim, summary.item_feature_dim]) == 1:
            plot_text_summary_panel(
                ax,
                "Available side-feature dimensions",
                [
                    f"user feature columns: {summary.user_feature_dim}",
                    f"item feature columns: {summary.item_feature_dim}",
                    "bar chart omitted",
                    "only item-side feature columns exist",
                ],
                RAW_RESPONSE_COLOR,
            )
            return

        ax.bar(
            ["user features", "item features"],
            [summary.user_feature_dim, summary.item_feature_dim],
            color=FEATURE_COLORS,
        )
        ax.set_title("Available side-feature dimensions")
        ax.set_ylabel("columns")
        ax.grid(True, axis="y", alpha=0.2)
        return

    if canonical.behavior_type is not None and np.unique(canonical.behavior_type).size > 1:
        labels, _counts = top_category_counts(canonical.behavior_type)
        title = "Interaction-type mix"
    elif canonical.source_domain is not None and np.unique(canonical.source_domain).size > 1:
        source_labels = np.unique(
            canonical.source_domain.astype(str, copy=False),
        ).tolist()
        plot_group_positive_rate(
            ax=ax,
            group_labels=source_labels,
            group_masks=[
                canonical.source_domain.astype(str, copy=False) == label for label in source_labels
            ],
            labels=canonical.label,
            title="Share labeled positive by source domain",
        )
        return
    elif (
        canonical.raw_target is not None
        and summary.feedback_type == "explicit"
        and summary.timestamp_coverage > 0.0
    ):
        plot_mean_raw_target_over_time(
            ax=ax,
            raw_target=canonical.raw_target,
            timestamps=canonical.timestamp,
            color=accent_color,
        )
        return
    else:
        labels, counts, title = [], np.array([], dtype=np.int64), ""

    if counts.size > 0:
        if np.count_nonzero(counts) == 1:
            nonzero_index = int(np.flatnonzero(counts > 0)[0])
            plot_text_summary_panel(
                ax,
                title,
                [
                    f"{counts[nonzero_index]:,} rows in one category",
                    f"category: {labels[nonzero_index]}",
                    "bar chart omitted because there is no category mix",
                ],
                accent_color,
            )
            return

        ax.bar(labels, counts, color=accent_color, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("interaction type")
        ax.set_ylabel("number of interactions")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.2)
        return

    if summary.user_feature_dim > 0 or summary.item_feature_dim > 0:
        if np.count_nonzero([summary.user_feature_dim, summary.item_feature_dim]) == 1:
            plot_text_summary_panel(
                ax,
                "Available side-feature dimensions",
                [
                    f"user feature columns: {summary.user_feature_dim}",
                    f"item feature columns: {summary.item_feature_dim}",
                    "bar chart omitted",
                    "only one side has feature columns",
                ],
                accent_color,
            )
            return

        ax.bar(
            ["user features", "item features"],
            [summary.user_feature_dim, summary.item_feature_dim],
            color=FEATURE_COLORS,
        )
        ax.set_title("Available side-feature dimensions")
        ax.set_ylabel("columns")
        ax.grid(True, axis="y", alpha=0.2)
        return

    plot_split_composition(ax, summary)


def plot_feature_availability_panel(ax: Axes, summary: DatasetSummary) -> None:
    """Plot split-safe feature-column availability without a misleading one-bar chart."""
    labels = ["user features", "item features"]
    values = [summary.user_feature_dim, summary.item_feature_dim]
    bars = ax.barh(labels, values, color=FEATURE_COLORS, edgecolor="white")
    ax.set_title("Available side-feature dimensions")
    ax.set_xlabel("columns")
    ax.set_xlim(0.0, max(1.0, max(values) * 1.15))
    ax.grid(True, axis="x", alpha=0.2)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + max(1.0, max(values) * 0.03),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value}",
            ha="left",
            va="center",
            fontsize=8.5,
        )


def plot_protocol_availability_panel(ax: Axes, summary: DatasetSummary) -> None:
    """Plot which auxiliary evidence sources exist for a dataset."""
    labels = ["timestamps", "user features", "item features", "random exposure"]
    values = np.asarray(
        [
            summary.timestamp_coverage,
            float(summary.user_feature_dim > 0),
            float(summary.item_feature_dim > 0),
            summary.randomized_exposure_share or 0.0,
        ],
        dtype=np.float64,
    )
    colors = [STRUCTURE_COLOR, USER_FEATURE_COLOR, ITEM_FEATURE_COLOR, EXPOSURE_RANDOM_COLOR]
    if not bool(values.any()):
        ax.axis("off")
        ax.set_title("Auxiliary evidence availability")
        ax.text(
            0.5,
            0.52,
            "No timestamp, side-feature,\nor randomized-exposure fields\nin this canonical view.",
            ha="center",
            va="center",
            fontsize=10,
            color="#111827",
        )
        return

    ax.barh(labels, values, color=colors, edgecolor="white")
    ax.set_title("Auxiliary evidence availability")
    ax.set_xlabel("share of rows or binary availability")
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="x", alpha=0.2)


def plot_profile_context_panel(
    ax: Axes,
    canonical: CanonicalInteractions,
    summary: DatasetSummary,
    color: str,
) -> None:
    """Choose the most informative sixth profile panel for the dataset."""
    if canonical.exposure_flag is not None:
        plot_group_positive_rate(
            ax=ax,
            group_labels=["randomized", "standard"],
            group_masks=[
                np.asarray(canonical.exposure_flag, dtype=bool),
                ~np.asarray(canonical.exposure_flag, dtype=bool),
            ],
            labels=canonical.label,
            title="Share labeled positive by exposure policy",
        )
        return

    if canonical.behavior_type is not None and np.unique(canonical.behavior_type).size > 1:
        labels, _counts = top_category_counts(canonical.behavior_type)
        behavior_values = canonical.behavior_type.astype(str, copy=False)
        plot_group_positive_rate(
            ax=ax,
            group_labels=labels,
            group_masks=[behavior_values == label for label in labels],
            labels=canonical.label,
            title="Share labeled positive by interaction type",
            annotate_counts=False,
        )
        ax.set_ylim(0.0, 1.05)
        return

    if (
        canonical.raw_target is not None
        and summary.feedback_type == "explicit"
        and summary.timestamp_coverage > 0.0
    ):
        plot_mean_raw_target_over_time(
            ax=ax,
            raw_target=canonical.raw_target,
            timestamps=canonical.timestamp,
            color=color,
        )
        return

    if summary.timestamp_coverage > 0.0:
        plot_relative_time_histogram(ax, canonical.timestamp, color)
        return

    if summary.user_feature_dim > 0 or summary.item_feature_dim > 0:
        plot_feature_availability_panel(ax, summary)
        return

    plot_protocol_availability_panel(ax, summary)


def plot_group_positive_rate(
    ax: Axes,
    group_labels: list[str],
    group_masks: list[np.ndarray],
    labels: np.ndarray,
    title: str,
    annotate_counts: bool = True,
) -> None:
    """Plot positive rate for a small set of dataset-specific groups.

    Args:
        ax: Matplotlib axes that receive the plot.
        group_labels: Display labels for the groups.
        group_masks: Boolean masks selecting each group.
        labels: Binary interaction labels used to compute positive rates.
        title: Axes title.

    Returns:
        None. The axes are modified in place.

    """
    counts = np.array([int(mask.sum()) for mask in group_masks], dtype=np.int64)
    rates = np.array(
        [float(np.mean(labels[mask])) if np.any(mask) else 0.0 for mask in group_masks],
        dtype=np.float64,
    )
    if group_labels == ["randomized", "standard"]:
        colors = [EXPOSURE_RANDOM_COLOR, EXPOSURE_STANDARD_COLOR]
    else:
        colors = [RAW_RESPONSE_COLOR] * len(group_labels)

    bars = ax.bar(group_labels, rates, color=colors, edgecolor=STRUCTURE_COLOR, linewidth=0.4)
    ax.set_title(title)
    ax.set_ylabel("share labeled positive")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, max(1.0, float(rates.max()) * 1.15))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.2)

    if annotate_counts:
        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={count:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def plot_mean_raw_target_over_time(
    ax: Axes,
    raw_target: np.ndarray,
    timestamps: np.ndarray,
    color: str,
) -> None:
    """Plot average rating across relative time bins.

    Args:
        ax: Matplotlib axes that receive the plot.
        raw_target: Rating values aligned with the timestamps.
        timestamps: Interaction timestamps aligned with ``raw_target``.
        color: Plot color.

    Returns:
        None. The axes are modified in place.

    """
    valid_mask = (timestamps > 0) & np.isfinite(raw_target)
    valid_timestamps = timestamps[valid_mask].astype(np.float64, copy=False)
    valid_target = raw_target[valid_mask].astype(np.float64, copy=False)

    if valid_timestamps.size < 2:
        ax.axis("off")
        ax.set_title("Average rating over relative time")
        ax.text(0.5, 0.5, "Not enough timestamped targets", ha="center", va="center")
        return

    start_timestamp = float(valid_timestamps.min())
    end_timestamp = float(valid_timestamps.max())
    if np.isclose(start_timestamp, end_timestamp):
        ax.axis("off")
        ax.set_title("Average rating over relative time")
        ax.text(
            0.5,
            0.5,
            "Timestamps collapse to one instant",
            ha="center",
            va="center",
        )
        return

    bin_edges = np.linspace(start_timestamp, end_timestamp, 11)
    bin_centers: list[float] = []
    bin_means: list[float] = []

    for idx in range(bin_edges.size - 1):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        in_bin = (
            (valid_timestamps >= lower) & (valid_timestamps <= upper)
            if idx == bin_edges.size - 2
            else (valid_timestamps >= lower) & (valid_timestamps < upper)
        )
        if np.any(in_bin):
            bin_centers.append((idx + 0.5) / (bin_edges.size - 1))
            bin_means.append(float(np.mean(valid_target[in_bin])))

    if not bin_centers:
        ax.axis("off")
        ax.set_title("Average rating over relative time")
        ax.text(0.5, 0.5, "No populated time bins", ha="center", va="center")
        return

    ax.plot(bin_centers, bin_means, marker="o", linewidth=2.0, color=color)
    ax.fill_between(bin_centers, bin_means, alpha=0.15, color=color)
    ax.set_title("Average rating over relative time")
    ax.set_xlabel("relative time within dataset")
    ax.set_ylabel("average rating")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.2)


def plot_split_composition(ax: Axes, summary: DatasetSummary) -> None:
    """Plot train/validation/test interaction counts.

    Args:
        ax: Matplotlib axes that receive the plot.
        summary: Scalar dataset summary with split counts.

    Returns:
        None. The axes are modified in place.

    """
    split_labels = ["train", "val", "test"]
    split_counts = [summary.train_size, summary.val_size, summary.test_size]
    split_colors = ["#4C72B0", "#DD8452", "#55A868"]

    ax.bar(split_labels, split_counts, color=split_colors, edgecolor="white")
    ax.set_title("Train / validation / test split sizes")
    ax.set_ylabel("number of interactions")
    ax.grid(True, axis="y", alpha=0.2)


def save_figure(fig: Figure, output_path: Path, dpi: int) -> None:
    """Persist a figure to disk and release the Matplotlib handle.

    Args:
        fig: Figure object to save.
        output_path: Target image path.
        dpi: Figure resolution.

    Returns:
        None. The figure is saved and closed.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_dataset_profile(
    canonical: CanonicalInteractions,
    summary: DatasetSummary,
    output_dir: Path,
    dpi: int,
) -> Path:
    """Create and save one per-dataset profile figure.

    Args:
        canonical: Loaded canonical dataset.
        summary: Scalar dataset summary.
        output_dir: Directory where the image should be written.
        dpi: Figure resolution.

    Returns:
        Output path of the saved profile figure.

    """
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.8), constrained_layout=True)
    scale_ax, user_tail_ax, item_tail_ax, label_sign_ax, response_ax, context_ax = axes.ravel()
    plot_scale_summary_panel(scale_ax, summary)
    user_counts = compute_entity_counts(canonical.user_id, canonical.n_users)
    item_counts = compute_entity_counts(canonical.item_id, canonical.n_items)
    plot_tail_ccdf(
        user_tail_ax,
        user_counts,
        "User activity long tail",
        "interactions per user",
        "users",
        USER_COLOR,
    )
    plot_tail_ccdf(
        item_tail_ax,
        item_counts,
        "Item popularity long tail",
        "interactions per item",
        "items",
        ITEM_COLOR,
    )
    plot_label_sign_summary(label_sign_ax, canonical)
    if response_signal_has_variation(canonical, summary):
        plot_response_distribution(response_ax, canonical, summary, RAW_RESPONSE_COLOR)
    else:
        plot_split_composition(response_ax, summary)
    plot_profile_context_panel(context_ax, canonical, summary, RAW_RESPONSE_COLOR)

    fig.suptitle(f"{summary.display_name} dataset profile", fontsize=16, color="#111827")

    output_path = output_dir / f"{summary.name}_profile.png"
    save_figure(fig, output_path, dpi=dpi)
    return output_path


def save_benchmark_overview(
    summaries: list[DatasetSummary],
    output_dir: Path,
    dpi: int,
) -> Path:
    """Create and save the cross-dataset benchmark overview figure.

    Args:
        summaries: Per-dataset scalar summaries.
        output_dir: Directory where the image should be written.
        dpi: Figure resolution.

    Returns:
        Output path of the saved overview figure.

    """
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.6), constrained_layout=True)
    flat_axes = axes.ravel()
    x_positions = np.arange(len(summaries))
    names = [summary.display_name for summary in summaries]
    dataset_colors = [DATASET_COLORS[summary.name] for summary in summaries]
    metric_specs = [
        (
            "Users",
            [summary.n_users for summary in summaries],
            "users",
            True,
        ),
        (
            "Items",
            [summary.n_items for summary in summaries],
            "items",
            True,
        ),
        (
            "Interactions",
            [summary.n_interactions for summary in summaries],
            "interactions",
            True,
        ),
        (
            "Observed density",
            [summary.density * 100 for summary in summaries],
            "density (%)",
            True,
        ),
    ]

    for ax, (title, values, ylabel, log_scale) in zip(
        flat_axes[:4],
        metric_specs,
        strict=True,
    ):
        ax.bar(x_positions, values, color=dataset_colors, edgecolor="white", linewidth=0.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(names, rotation=25, ha="right")
        if log_scale:
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)

    y_positions = np.arange(len(summaries))
    sign_negative = np.asarray([summary.negative_sign_rate for summary in summaries])
    sign_zero = np.asarray([summary.zero_sign_rate for summary in summaries])
    sign_positive = np.asarray([summary.positive_sign_rate for summary in summaries])
    flat_axes[4].barh(
        y_positions,
        sign_negative,
        color=SIGN_NEGATIVE_COLOR,
        label="sign < 0",
    )
    flat_axes[4].barh(
        y_positions,
        sign_zero,
        left=sign_negative,
        color=SIGN_ZERO_COLOR,
        label="sign = 0",
    )
    flat_axes[4].barh(
        y_positions,
        sign_positive,
        left=sign_negative + sign_zero,
        color=SIGN_POSITIVE_COLOR,
        label="sign > 0",
    )
    for y_position, summary in zip(y_positions, summaries, strict=True):
        flat_axes[4].plot(
            summary.positive_label_rate,
            y_position,
            marker="|",
            color=LABEL_NONPOSITIVE_COLOR,
            markersize=22,
            markeredgewidth=5.0,
        )
        flat_axes[4].plot(
            summary.positive_label_rate,
            y_position,
            marker="|",
            color=LABEL_POSITIVE_COLOR,
            markersize=18,
            markeredgewidth=2.4,
        )
    flat_axes[4].set_title("Feedback sign distribution")
    flat_axes[4].set_xlabel("share of rows; green marker = label > 0 rate")
    flat_axes[4].set_yticks(y_positions)
    flat_axes[4].set_yticklabels(names)
    flat_axes[4].xaxis.set_major_formatter(PercentFormatter(1.0))
    flat_axes[4].set_xlim(0.0, 1.0)
    flat_axes[4].grid(True, axis="x", alpha=0.25)
    flat_axes[4].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=SIGN_NEGATIVE_COLOR, label="sign < 0"),
            plt.Rectangle((0, 0), 1, 1, color=SIGN_ZERO_COLOR, label="sign = 0"),
            plt.Rectangle((0, 0), 1, 1, color=SIGN_POSITIVE_COLOR, label="sign > 0"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=3,
        fontsize=8,
        frameon=False,
    )

    width = 0.38
    flat_axes[5].bar(
        x_positions - width / 2,
        [summary.user_feature_dim for summary in summaries],
        width=width,
        label="user features",
        color=FEATURE_COLORS[0],
    )
    flat_axes[5].bar(
        x_positions + width / 2,
        [summary.item_feature_dim for summary in summaries],
        width=width,
        label="item features",
        color=FEATURE_COLORS[1],
    )
    flat_axes[5].set_title("Optional feature dimensions")
    flat_axes[5].set_ylabel("columns")
    flat_axes[5].set_xticks(x_positions)
    flat_axes[5].set_xticklabels(names, rotation=25, ha="right")
    flat_axes[5].legend()
    flat_axes[5].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Benchmark dataset overview", fontsize=17)

    output_path = output_dir / "benchmark_overview.png"
    save_figure(fig, output_path, dpi=dpi)
    return output_path


def print_comparative_summary(summaries: list[DatasetSummary]) -> None:
    """Print a compact comparative summary table.

    Args:
        summaries: Per-dataset scalar summaries.

    Returns:
        None. The summary is printed to stdout.

    """
    print("\n" + "=" * 108)
    print("DATASET SUMMARY")
    print("=" * 108)
    print(
        ""
        f"{'Dataset':<16} {'Users':>10} {'Items':>10} {'Interact.':>12} {'PairReuse%':>10} "
        f"{'Pos%':>8} {'UFeat':>7} {'IFeat':>7} {'Split':>24}",
    )
    print("-" * 108)
    for summary in summaries:
        print(
            f"{summary.display_name:<16} {summary.n_users:>10,} {summary.n_items:>10,} "
            f"{summary.n_interactions:>12,} {summary.repeated_pair_share:>9.2%} "
            f"{summary.pos_rate:>7.2%} {summary.user_feature_dim:>7} "
            f"{summary.item_feature_dim:>7} {summary.split_source:>24}",
        )


def main() -> int:
    """Load datasets, generate figures, and print a summary.

    Args:
        None.

    Returns:
        Shell-style exit code.

    """
    args = build_explore_all_datasets_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EDGRec DATASET VISUALIZATION")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")

    summaries: list[DatasetSummary] = []
    dataset_payloads: list[dict[str, Any]] = []
    for name in args.datasets:
        print(f"\n--- {display_name(name)} ---")
        canonical = load_dataset(name, args.data_dir)
        summary = summarize_dataset(name, canonical)
        dataset_payloads.append(build_dataset_summary_payload(canonical, summary))
        profile_path = save_dataset_profile(
            canonical=canonical,
            summary=summary,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
        print(
            ""
            f"users={summary.n_users:,} items={summary.n_items:,} "
            f"interactions={summary.n_interactions:,} density={summary.density * 100:.5f}% "
            f"pos_rate={summary.pos_rate:.2%}",
        )
        print(
            ""
            f"saved profile: {profile_path} | split={summary.split_label} | "
            f"context={summary.modalities_label}",
        )
        summaries.append(summary)

    overview_path = save_benchmark_overview(
        summaries=summaries,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
    summary_json_path, summary_markdown_path = save_summary_exports(
        dataset_payloads,
        args.output_dir,
    )
    print_comparative_summary(summaries)
    print(f"\nSaved benchmark overview: {overview_path}")
    print(f"Saved summary JSON: {summary_json_path}")
    print(f"Saved summary markdown: {summary_markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
