#!/usr/bin/env python
"""Export thesis-defense figures from SQLite-backed EDGRec evidence."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import textwrap
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from src.utils.crru import compute_validation_crru_components_for_k
from src.utils.project_paths import RESULTS_DIR, THESIS_DB_PATH

from scripts.query_results import (
    _compute_dataset_crru_scores_with_audit,
    _crru_epoch_time_s,
    _display_profile,
    _evidence_label,
    _is_default_report_row,
    _largest_training_item_interaction_count_for_row,
    _leaderboard_label,
    _query_report_rows,
    _row_value,
    _select_paper_runtime_probe_rows,
    _select_top_test_rows,
    connect,
)

DEFENSE_FIGURES_DIR = RESULTS_DIR / "defense_figures"
DATASET_SUMMARY_PATH = RESULTS_DIR / "dataset_visualizations" / "benchmark_summary.json"
DATASET_ORDER = ("amazonbook", "kuairec_v2", "movielens1m", "kuairand1k")
A4_TEXT_WIDTH_IN = 160.0 / 25.4
THESIS_FIGURE_DPI = 300
RETIRED_DEFENSE_ARTIFACTS = (
    "architecture_pipeline.mmd",
    "accuracy_efficiency_pareto.png",
    "evidence_map_vs_lightgcn.png",
    "edgrec_reference_vs_lightgcn.png",
    "score_mix_diagnostics.png",
    "branch_rank_diagnostics.png",
    "architecture_pipeline.png",
)
DATASET_LABELS = {
    "amazonbook": "Amazon Book",
    "kuairec_v2": "KuaiRec v2",
    "movielens1m": "MovieLens-1M",
    "kuairand1k": "KuaiRand-1K",
}
KUAIREC_DEFAULT_BIG_MATRIX_PRESETS = frozenset(
    {
        "kuairec_watchratio",
        "kuairec_big_matrix_watch_ratio_threshold_0_5",
    },
)
KUAIREC_SMALL_MATRIX_PRESETS = frozenset(
    {
        "kuairec_fullobs",
        "kuairec_small_matrix_full_observation",
    },
)
METHOD_STYLES = {
    "EDGRec family": {"color": "#1f77b4", "marker": "o"},
    "LightGCN paper-faithful": {"color": "#d17a22", "marker": "s"},
    "LightGCN sampled ablation": {"color": "#f0a642", "marker": "^"},
    "DICE-style sampled ablation": {"color": "#8f63b8", "marker": "P"},
    "DICE paper-faithful probe": {"color": "#b2182b", "marker": "s"},
    "Other": {"color": "#6d7882", "marker": "X"},
}
COMPONENT_LABELS = {
    "accuracy": "Ranking\naccuracy",
    "popularity_aware_personalization": "Pop-aware\npersonalization",
    "efficiency": "Resource\nefficiency",
    "crru": "CRRU",
}
PAPER_BLUE = "#2166ac"
PAPER_ORANGE = "#d17a22"
PAPER_GREEN = "#1b9e77"
PAPER_RED = "#b2182b"
PAPER_GRAY = "#6b7280"
CLAIM_MATRIX_LOG2_CLIP = 4.0
PAPER_LIGHT_GRAY = "#d1d5db"
KUAIREC_ABLATION_PROFILES = ("mainline", "no_popularity_head", "no_independence")
ARCHITECTURE_MERMAID = """
%% EDGRec thesis-defense architecture. Vertical layout for papers and slides.
flowchart TB
    subgraph D["1. Split-safe data"]
        direction TB
        D1["Dataset loaders<br/>canonical interactions"]
        D2["Train / validation / test masks"]
        D3["Observed train graph<br/>positive train edges only"]
        D4["Train-only context tensors<br/>popularity, recency, safe features"]
        D5["Bounded subgraph sampler<br/>k-hop fanout per batch"]
        D1 --> D2
        D2 --> D3
        D2 --> D4
        D3 --> D5
    end

    subgraph M["2. EDGRec scoring model"]
        direction TB
        M1["Embedding module<br/>users, items, optional item features"]
        M2["Dual LightGCN-style propagation"]
        M3["Interest branch<br/>preference signal"]
        M4["Conformity branch<br/>popularity / exposure signal"]
        M5["Item-only context head<br/>split-safe metadata"]
        M6["Bounded score mixer<br/>final ranking score"]
        M1 --> M2
        M2 --> M3
        M2 --> M4
        M1 --> M5
        M3 --> M6
        M4 --> M6
        M5 --> M6
    end

    subgraph O["3. Training objective"]
        direction TB
        O1["Recommendation BPR<br/>on final score"]
        O2["DICE-style branch BPR<br/>popularity-conditioned negatives"]
        O3["Bounded auxiliaries<br/>independence, L_pop, optional IPW calibration"]
        O4["LossSuite weighted sum<br/>scheduled and capped"]
        O1 --> O4
        O2 --> O4
        O3 --> O4
    end

    subgraph E["4. Evidence and reports"]
        direction TB
        E1["Evaluator<br/>NDCG, Recall, Hit, Pers, raw AvgPop"]
        E2["SQLite experiment store<br/>source of truth"]
        E3["Reports and figures<br/>query-results, Optuna, CRRU"]
        E1 --> E2
        E2 --> E3
    end

    D5 --> M1
    D4 --> M5
    M6 --> O1
    M3 --> O2
    M4 --> O2
    D4 --> O3
    O4 --> E2
    M6 --> E1

    classDef data fill:#e8f2ff,stroke:#2d3436,color:#111827;
    classDef model fill:#eef7e8,stroke:#2d3436,color:#111827;
    classDef loss fill:#fff0e6,stroke:#2d3436,color:#111827;
    classDef evidence fill:#f2f4f7,stroke:#2d3436,color:#111827;
    class D1,D2,D3,D4,D5 data;
    class M1,M2,M3,M4,M5,M6 model;
    class O1,O2,O3,O4 loss;
    class E1,E2,E3 evidence;
"""


@dataclass(frozen=True)
class DefenseRecord:
    """One completed report row prepared for plotting."""

    row: sqlite3.Row
    exp_id: int
    dataset: str
    label: str
    evidence: str
    profile: str
    method_group: str
    crru20: float | None
    crru40: float | None
    ndcg20: float | None
    recall20: float | None
    hit20: float | None
    personalization20: float | None
    avgpop20: float | None
    time_per_epoch_s: float | None
    peak_vram_mb: float | None
    score_mix_interest: float | None
    score_mix_conformity: float | None
    score_mix_context: float | None
    final_popularity_spearman: float | None
    branch_cosine: float | None
    interest_branch_ndcg20: float | None
    conformity_branch_ndcg20: float | None
    interest_branch_avgpop20: float | None
    conformity_branch_avgpop20: float | None

    @property
    def crru_mean(self) -> float | None:
        """Return the arithmetic mean of CRRU@20 and CRRU@40 when available."""
        if self.crru20 is None or self.crru40 is None:
            return None
        return (self.crru20 + self.crru40) / 2.0


@dataclass(frozen=True)
class DefenseData:
    """Prepared SQLite rows for the defense figure set."""

    records: list[DefenseRecord]
    top_records: list[DefenseRecord]
    probe_rows: list[sqlite3.Row]


@dataclass(frozen=True)
class DatasetProfile:
    """Dataset-level descriptive metadata for paper companion figures."""

    name: str
    display_name: str
    n_users: int
    n_items: int
    n_interactions: int
    density: float
    pos_rate: float | None
    item_feature_dim: int
    user_feature_dim: int
    feedback_type: str
    randomized_exposure_share: float | None


def _style_plots() -> None:
    """Apply a compact thesis-presentation Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2d3436",
            "axes.labelcolor": "#1f2933",
            "axes.titlecolor": "#111827",
            "axes.grid": True,
            "grid.color": "#d8dee4",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "savefig.dpi": 300,
        },
    )


def _remove_retired_defense_artifacts(output_dir: Path) -> None:
    """Remove stale defense artifacts superseded by current exports."""
    for filename in RETIRED_DEFENSE_ARTIFACTS:
        (output_dir / filename).unlink(missing_ok=True)


def _strip_axes(ax: plt.Axes, *, keep_left: bool = True, keep_bottom: bool = True) -> None:
    """Remove visual noise while keeping requested axes spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_visible(keep_bottom)


def _dataset_label(dataset: str) -> str:
    """Return a readable dataset label."""
    return DATASET_LABELS.get(dataset, dataset)


def _dataset_sort_key(dataset: str) -> tuple[int, str]:
    """Return a stable order key for datasets."""
    try:
        return DATASET_ORDER.index(dataset), dataset
    except ValueError:
        return len(DATASET_ORDER), dataset


def _finite(value: object | None) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_dataset_profiles(path: Path = DATASET_SUMMARY_PATH) -> list[DatasetProfile]:
    """Load dataset-profile metadata generated by dataset visualization tooling."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles: list[DatasetProfile] = []
    for item in payload.get("datasets", []):
        profiles.append(
            DatasetProfile(
                name=str(item["name"]),
                display_name=str(item.get("display_name") or item["name"]),
                n_users=int(item["n_users"]),
                n_items=int(item["n_items"]),
                n_interactions=int(item["n_interactions"]),
                density=float(item["density"]),
                pos_rate=_finite(item.get("pos_rate")),
                item_feature_dim=int(item.get("item_feature_dim") or 0),
                user_feature_dim=int(item.get("user_feature_dim") or 0),
                feedback_type=str(item.get("feedback_type") or "unknown"),
                randomized_exposure_share=_finite(item.get("randomized_exposure_share")),
            ),
        )
    if not profiles:
        raise RuntimeError(f"No dataset profiles found in {path}.")
    return profiles


def _format_compact_number(value: float) -> str:
    """Return a compact human-readable magnitude label."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def _format_metric_value(value: float | None, *, suffix: str = "") -> str:
    """Return a compact metric-value label."""
    if value is None:
        return "-"
    if suffix:
        return f"{value:.1f}{suffix}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _method_group(row: sqlite3.Row) -> str:
    """Return a compact plot family for one result row."""
    label = _leaderboard_label(row)
    evidence = _evidence_label(row)
    if evidence == "ablation":
        return "EDGRec family"
    if label == "EDGRec":
        return "EDGRec family"
    if label == "lightgcn_paper":
        return "LightGCN paper-faithful"
    if label == "lightgcn":
        return "LightGCN sampled ablation"
    if label == "dice_like":
        return "DICE-style sampled ablation"
    if label == "dice_paper":
        return "DICE paper-faithful probe"
    return "Other"


def _record_from_row(
    row: sqlite3.Row,
    *,
    crru_scores: Mapping[int, Mapping[int, float]],
) -> DefenseRecord:
    """Convert one query-results row into a plotting record."""
    exp_id = int(row["id"])
    row_scores = crru_scores.get(exp_id, {})
    return DefenseRecord(
        row=row,
        exp_id=exp_id,
        dataset=str(row["dataset"] or "-"),
        label=_leaderboard_label(row),
        evidence=_evidence_label(row),
        profile=_display_profile(row["profile_name"]),
        method_group=_method_group(row),
        crru20=_finite(row_scores.get(20)),
        crru40=_finite(row_scores.get(40)),
        ndcg20=_row_value(row, "test_ndcg_20"),
        recall20=_row_value(row, "test_recall_20"),
        hit20=_row_value(row, "test_hit_ratio_20"),
        personalization20=_row_value(row, "test_personalization_20"),
        avgpop20=_row_value(row, "test_average_popularity_20"),
        time_per_epoch_s=_crru_epoch_time_s(row),
        peak_vram_mb=_row_value(row, "peak_vram_mb"),
        score_mix_interest=_row_value(row, "test_score_mix_interest_mean"),
        score_mix_conformity=_row_value(row, "test_score_mix_conformity_mean"),
        score_mix_context=_row_value(row, "test_score_mix_context_mean"),
        final_popularity_spearman=_row_value(row, "test_final_popularity_spearman_20"),
        branch_cosine=_row_value(row, "test_interest_conformity_cosine_mean"),
        interest_branch_ndcg20=_row_value(row, "test_interest_branch_ndcg_20"),
        conformity_branch_ndcg20=_row_value(row, "test_conformity_branch_ndcg_20"),
        interest_branch_avgpop20=_row_value(row, "test_interest_branch_average_popularity_20"),
        conformity_branch_avgpop20=_row_value(row, "test_conformity_branch_average_popularity_20"),
    )


def _load_defense_data(top_n: int) -> DefenseData:
    """Load and filter completed full-data rows through query-results semantics."""
    with connect() as conn:
        report_rows = _query_report_rows(conn)
    default_rows = [row for row in report_rows if _is_default_report_row(row)]
    crru_scores, _audit = _compute_dataset_crru_scores_with_audit(default_rows)
    top_rows = _select_top_test_rows(default_rows, top_n=top_n, crru_scores=crru_scores)
    records = [_record_from_row(row, crru_scores=crru_scores) for row in default_rows]
    top_records = [_record_from_row(row, crru_scores=crru_scores) for row in top_rows]
    probe_rows = _select_paper_runtime_probe_rows(report_rows)
    if not records:
        raise RuntimeError("No completed full-data test rows found for defense figures.")
    return DefenseData(records=records, top_records=top_records, probe_rows=probe_rows)


def _records_by_dataset(records: Sequence[DefenseRecord]) -> dict[str, list[DefenseRecord]]:
    """Group records by dataset in stable dataset order."""
    grouped: dict[str, list[DefenseRecord]] = defaultdict(list)
    for record in records:
        grouped[record.dataset].append(record)
    return dict(sorted(grouped.items(), key=lambda item: _dataset_sort_key(item[0])))


def _selection_value(record: DefenseRecord) -> float:
    """Return the preferred ranking value for selecting representative records."""
    if record.crru_mean is not None:
        return record.crru_mean
    if record.ndcg20 is not None:
        return record.ndcg20
    return float("-inf")


def _best_record(
    records: Sequence[DefenseRecord],
    predicate: Callable[[DefenseRecord], bool],
) -> DefenseRecord | None:
    """Return the best record satisfying a predicate."""
    candidates = [record for record in records if predicate(record)]
    if not candidates:
        return None
    return max(candidates, key=_selection_value)


def _is_edgrec_family_record(record: DefenseRecord) -> bool:
    """Return whether a record belongs to the EDGRec model family."""
    return record.method_group == "EDGRec family"


def _selected_edgrec_reference_rows(
    records: Sequence[DefenseRecord],
    *,
    require_score_mix_telemetry: bool = False,
    require_branch_rank_telemetry: bool = False,
) -> dict[str, DefenseRecord]:
    """Return one EDGRec-family reference test row per dataset.

    The reference row is the highest-CRRU completed test row available through
    the same report semantics as `results/query_results.md`. Diagnostic figures
    optionally require score-mix or branch-rank telemetry because older
    supporting rows do not always contain those fields.
    """

    def has_required_telemetry(record: DefenseRecord) -> bool:
        if require_score_mix_telemetry and record.score_mix_interest is None:
            return False
        return not (
            require_branch_rank_telemetry
            and (
                record.interest_branch_ndcg20 is None
                or record.conformity_branch_ndcg20 is None
                or record.interest_branch_avgpop20 is None
                or record.conformity_branch_avgpop20 is None
            )
        )

    selected: dict[str, DefenseRecord] = {}
    for dataset, dataset_records in _records_by_dataset(records).items():
        current = _best_record(
            dataset_records,
            lambda record: _is_edgrec_family_record(record) and has_required_telemetry(record),
        )
        if current is not None:
            selected[dataset] = current
    return selected


def _selected_lightgcn_paper(records: Sequence[DefenseRecord]) -> dict[str, DefenseRecord]:
    """Return the best full-data LightGCN paper row per dataset."""
    selected: dict[str, DefenseRecord] = {}
    for dataset, dataset_records in _records_by_dataset(records).items():
        current = _best_record(dataset_records, lambda record: record.label == "lightgcn_paper")
        if current is not None:
            selected[dataset] = current
    return selected


def _record_config(record: DefenseRecord) -> dict[str, object]:
    """Return the parsed config JSON for a plotting record."""
    try:
        config_json = record.row["config_json"]
    except (IndexError, KeyError):
        return {}
    if not config_json:
        return {}
    try:
        loaded = json.loads(str(config_json))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _config_string(record: DefenseRecord, key: str) -> str | None:
    """Return a non-empty string config value when available."""
    value = _record_config(record).get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _record_string(record: DefenseRecord, key: str) -> str | None:
    """Return a non-empty source-row string value when available."""
    try:
        value = record.row[key]
    except (IndexError, KeyError):
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_kuairec_small_matrix_record(record: DefenseRecord) -> bool:
    """Return whether a KuaiRec row belongs to the small-matrix sensitivity view."""
    if record.dataset != "kuairec_v2":
        return False
    preset = _config_string(record, "preprocessing_preset")
    matrix_variant = _config_string(record, "matrix_variant")
    return preset in KUAIREC_SMALL_MATRIX_PRESETS or matrix_variant == "small_matrix"


def _is_kuairec_default_big_matrix_record(record: DefenseRecord) -> bool:
    """Return whether a KuaiRec row is in the thesis-default big-matrix view."""
    if record.dataset != "kuairec_v2":
        return False
    if _is_kuairec_small_matrix_record(record):
        return False
    preset = _config_string(record, "preprocessing_preset")
    matrix_variant = _config_string(record, "matrix_variant")
    if matrix_variant == "small_matrix":
        return False
    if matrix_variant == "big_matrix":
        return True
    if preset in KUAIREC_DEFAULT_BIG_MATRIX_PRESETS:
        return True
    evidence_text = " ".join(
        text
        for text in (
            _record_string(record, "batch_id"),
            _record_string(record, "profile_name"),
        )
        if text
    ).lower()
    return "kuairec_watchratio" in evidence_text or "big_matrix_watch_ratio" in evidence_text


def _is_kuairec_big_matrix_comparison_record(record: DefenseRecord) -> bool:
    """Return whether a record belongs in the KuaiRec big-matrix comparison view."""
    if record.dataset == "kuairec_v2":
        return _is_kuairec_default_big_matrix_record(record)
    return True


def _selected_kuairec_big_matrix_edgrec_reference_rows(
    records: Sequence[DefenseRecord],
    *,
    require_score_mix_telemetry: bool = False,
    require_branch_rank_telemetry: bool = False,
) -> dict[str, DefenseRecord]:
    """Return EDGRec references with KuaiRec restricted to explicit big-matrix rows."""
    protocol_records = [
        record for record in records if _is_kuairec_big_matrix_comparison_record(record)
    ]
    return _selected_edgrec_reference_rows(
        protocol_records,
        require_score_mix_telemetry=require_score_mix_telemetry,
        require_branch_rank_telemetry=require_branch_rank_telemetry,
    )


def _selected_reference_rows(
    records: Sequence[DefenseRecord],
    *,
    reference_mode: str,
    require_score_mix_telemetry: bool = False,
    require_branch_rank_telemetry: bool = False,
) -> dict[str, DefenseRecord]:
    """Return EDGRec reference rows for a named figure policy."""
    if reference_mode == "highest_crru":
        return _selected_edgrec_reference_rows(
            records,
            require_score_mix_telemetry=require_score_mix_telemetry,
            require_branch_rank_telemetry=require_branch_rank_telemetry,
        )
    if reference_mode == "kuairec_big_matrix":
        return _selected_kuairec_big_matrix_edgrec_reference_rows(
            records,
            require_score_mix_telemetry=require_score_mix_telemetry,
            require_branch_rank_telemetry=require_branch_rank_telemetry,
        )
    raise ValueError(f"Unknown reference mode: {reference_mode!r}")


def _reference_label(reference_mode: str) -> str:
    """Return a legend label for the selected EDGRec reference policy."""
    if reference_mode == "kuairec_big_matrix":
        return "EDGRec reference"
    return "EDGRec-family reference"


def _reference_note(reference_mode: str) -> str:
    """Return a footnote clause explaining the selected reference policy."""
    if reference_mode == "kuairec_big_matrix":
        return (
            "Stars mark EDGRec references; the KuaiRec reference is restricted to explicit "
            "big-matrix watch-ratio rows."
        )
    return "Stars mark the highest-CRRU completed EDGRec-family test row per dataset."


def _selected_best(records: Sequence[DefenseRecord]) -> dict[str, DefenseRecord]:
    """Return the best displayed row per dataset by CRRU."""
    return {
        dataset: max(dataset_records, key=_selection_value)
        for dataset, dataset_records in _records_by_dataset(records).items()
    }


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
    *,
    tight: bool = True,
) -> Path:
    """Save one figure with presentation-friendly resolution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    if tight:
        fig.savefig(path, dpi=THESIS_FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
    else:
        fig.savefig(path, dpi=THESIS_FIGURE_DPI)
    plt.close(fig)
    return path


def _centered_figure_note(fig: plt.Figure, text: str, *, y: float = 0.015) -> None:
    """Add one centered explanatory note below a figure."""
    fig.text(
        0.5,
        y,
        textwrap.fill(text, width=140),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )


def _format_seconds(value: float | None) -> str:
    """Return a compact seconds label for figure annotations."""
    if value is None:
        return "-"
    if value < 1.0:
        return f"{value:.3f}s"
    if value < 1000.0:
        return f"{value:.1f}s"
    return f"{value:,.0f}s"


def write_architecture_mermaid(output_dir: Path) -> Path:
    """Write a Markdown Mermaid architecture preview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mermaid = textwrap.dedent(ARCHITECTURE_MERMAID).strip()
    markdown_path = output_dir / "architecture_pipeline.md"
    (output_dir / "architecture_pipeline.mmd").unlink(missing_ok=True)
    markdown_path.write_text(
        "\n".join(
            [
                "# EDGRec Architecture Pipeline",
                "",
                "```mermaid",
                mermaid,
                "```",
                "",
            ],
        ),
        encoding="utf-8",
    )
    return markdown_path


def _add_diagram_box(
    ax: plt.Axes,
    *,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#1f2933",
    fontsize: float = 8.5,
    weight: str = "normal",
) -> None:
    """Add one labeled rounded box to a diagram axis."""
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.05,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        weight=weight,
        wrap=True,
    )


def _balanced_wrap(text: str, *, width: int) -> str:
    """Wrap figure text into compact, visually balanced lines."""
    wrapped = textwrap.wrap(text, width=width)
    return "\n".join(wrapped)


def plot_candidate_taxonomy(output_dir: Path) -> Path:
    """Plot method-label meanings so figures are readable without code context."""
    rows = [
        (
            "EDGRec family",
            "All completed full-data EDGRec variants tested on the test split.",
            "Comparable candidates; no visual split between thesis/supporting rows.",
            METHOD_STYLES["EDGRec family"],
        ),
        (
            "LightGCN paper-faithful",
            "PaperLightGCN adapter with full-graph training and locked paper defaults.",
            "Main accuracy/resource baseline when full test row exists.",
            METHOD_STYLES["LightGCN paper-faithful"],
        ),
        (
            "LightGCN sampled ablation",
            "EDGRec runtime single-branch LightGCN approximation with sampled training.",
            "Engineering ablation; not a paper-faithful baseline.",
            METHOD_STYLES["LightGCN sampled ablation"],
        ),
        (
            "DICE-style sampled ablation",
            "Legacy dual-branch EDGRec-runtime ablation with DICE-style losses.",
            "Mechanism/runtime comparison; not GCN-DICE paper.",
            METHOD_STYLES["DICE-style sampled ablation"],
        ),
        (
            "DICE paper-faithful probe",
            "PaperGCNDICE adapter; current evidence is runtime probe only.",
            "Feasibility/resource evidence until full test rows exist.",
            METHOD_STYLES["DICE paper-faithful probe"],
        ),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.45))
    ax.set_position((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Candidate taxonomy for thesis-defense figures", y=0.955)

    headers = ("Figure label", "What it means", "How to use it")
    x_positions = (0.045, 0.335, 0.665)
    widths = (0.255, 0.285, 0.285)
    for header, x_left, width in zip(headers, x_positions, widths, strict=True):
        _add_diagram_box(
            ax,
            xy=(x_left, 0.82),
            width=width,
            height=0.075,
            text=header,
            facecolor="#e5e7eb",
            fontsize=10,
            weight="bold",
        )

    row_height = 0.13
    y_start = 0.655
    for index, (label, meaning, use, style) in enumerate(rows):
        y_bottom = y_start - index * row_height
        for x_left, width in zip(x_positions, widths, strict=True):
            box = patches.FancyBboxPatch(
                (x_left, y_bottom),
                width,
                0.085,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.05,
                edgecolor="#1f2933",
                facecolor="#ffffff" if x_left == x_positions[0] else "#f8fafc",
            )
            ax.add_patch(box)
        label_text = {
            "LightGCN paper-faithful": "LightGCN\npaper-faithful",
            "LightGCN sampled ablation": "LightGCN sampled\n(non-paper)",
            "DICE-style sampled ablation": "DICE-style sampled\n(non-paper)",
            "DICE paper-faithful probe": "DICE paper-faithful\nruntime probe",
        }.get(label, label)
        marker_x = x_positions[0] + 0.030
        ax.scatter(
            marker_x,
            y_bottom + 0.043,
            s=85,
            marker=str(style["marker"]),
            color=str(style["color"]),
            edgecolor="#111827",
            zorder=4,
        )
        ax.text(
            x_positions[0] + 0.065,
            y_bottom + 0.043,
            label_text,
            ha="left",
            va="center",
            fontsize=8.35,
            color="#111827",
            weight="bold",
        )
        ax.text(
            x_positions[1] + 0.014,
            y_bottom + 0.043,
            _balanced_wrap(meaning, width=35),
            ha="left",
            va="center",
            fontsize=8.9,
            color="#111827",
        )
        ax.text(
            x_positions[2] + 0.014,
            y_bottom + 0.043,
            _balanced_wrap(use, width=35),
            ha="left",
            va="center",
            fontsize=8.9,
            color="#111827",
        )

    ax.text(
        0.5,
        0.055,
        "Rule used in plots: if a row is a completed EDGRec-family test run, it is a candidate "
        "for comparison regardless of whether the report calls it thesis profile, supporting, "
        "or ablation evidence.",
        ha="center",
        va="center",
        fontsize=9,
        color="#4b5563",
        wrap=True,
    )
    return _save_figure(fig, output_dir, "candidate_taxonomy.png")


def plot_dataset_regime_map(output_dir: Path) -> Path:
    """Plot dataset regimes as a compact companion to dataset-profile figures."""
    profiles = _load_dataset_profiles()
    active_names = set(DATASET_ORDER)
    item_sizes = np.array([profile.n_items for profile in profiles], dtype=float)
    size_min = float(np.sqrt(item_sizes.min()))
    size_max = float(np.sqrt(item_sizes.max()))

    def marker_size(profile: DatasetProfile) -> float:
        root_items = math.sqrt(profile.n_items)
        if size_max == size_min:
            return 260.0
        scaled = (root_items - size_min) / (size_max - size_min)
        return 95.0 + 520.0 * scaled

    color_map = {
        "amazonbook": "#4c78a8",
        "kuairec_v2": "#59a14f",
        "movielens1m": "#f28e2b",
        "kuairand1k": "#8f63b8",
    }
    fig, ax = plt.subplots(figsize=(A4_TEXT_WIDTH_IN, 4.15))
    for profile in profiles:
        active = profile.name in active_names
        color = color_map.get(profile.name, PAPER_LIGHT_GRAY)
        alpha = 0.95 if active else 0.35
        edgecolor = "#111827" if active else "#9ca3af"
        linewidth = 1.0 if active else 0.6
        ax.scatter(
            profile.density * 100.0,
            profile.n_interactions,
            s=marker_size(profile),
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=3 if active else 2,
        )
        offset_y = 10 if profile.name != "taobao" else -18
        ax.annotate(
            _dataset_label(profile.name) if active else profile.display_name,
            (profile.density * 100.0, profile.n_interactions),
            xytext=(8, offset_y),
            textcoords="offset points",
            fontsize=9 if active else 8,
            color="#111827" if active else PAPER_GRAY,
            weight="bold" if active else "normal",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed density (%)")
    ax.set_ylabel("Observed interactions")
    ax.set_title("Dataset regimes used to test EDGRec", loc="center", fontsize=10)
    ax.grid(True, which="major", linewidth=0.8, alpha=0.45)
    ax.grid(True, which="minor", linewidth=0.35, alpha=0.18)
    ax.text(
        0.98,
        0.04,
        "Marker area encodes item catalog; gray points are context datasets.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=PAPER_GRAY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 2.0},
    )
    _strip_axes(ax)

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="#111827",
            label=_dataset_label(dataset),
            markersize=8,
        )
        for dataset, color in color_map.items()
    ]
    legend_handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor=PAPER_LIGHT_GRAY,
            markeredgecolor="#9ca3af",
            label="context dataset",
            markersize=8,
            alpha=0.7,
        ),
    )
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, framealpha=0.95)
    fig.tight_layout(pad=0.4)
    return _save_figure(fig, output_dir, "paper_dataset_regime_map.png")


def _relative_benefit(
    edgrec_value: float,
    baseline_value: float,
    *,
    lower_is_better: bool,
) -> float:
    """Return positive relative benefit for EDGRec compared with a baseline."""
    if baseline_value == 0:
        return 0.0
    relative = (edgrec_value - baseline_value) / abs(baseline_value)
    return -relative if lower_is_better else relative


def _log_benefit_score(
    edgrec_value: float,
    baseline_value: float,
    *,
    lower_is_better: bool,
) -> float:
    """Return signed log2 fold-change for claim-matrix color, clipped to [-1, 1]."""
    if edgrec_value <= 0 or baseline_value <= 0:
        benefit = _relative_benefit(
            edgrec_value,
            baseline_value,
            lower_is_better=lower_is_better,
        )
        return max(-1.0, min(1.0, benefit))
    ratio = baseline_value / edgrec_value if lower_is_better else edgrec_value / baseline_value
    return max(-1.0, min(1.0, math.log2(ratio) / CLAIM_MATRIX_LOG2_CLIP))


def _paper_comparison_label(
    edgrec_value: float,
    baseline_value: float,
    *,
    metric: str,
    lower_is_better: bool,
) -> str:
    """Return a compact paper-figure label for a pairwise comparison."""
    if metric == "time_per_epoch_s":
        ratio = baseline_value / edgrec_value
        direction = "faster" if ratio >= 1.0 else "slower"
        ratio_text = ratio if ratio >= 1.0 else 1.0 / ratio
        return (
            f"{_format_seconds(edgrec_value)} vs {_format_seconds(baseline_value)}\n"
            f"{ratio_text:.1f}x {direction}"
        )
    benefit = _relative_benefit(
        edgrec_value,
        baseline_value,
        lower_is_better=lower_is_better,
    )
    if lower_is_better:
        suffix = "lower" if benefit >= 0 else "higher"
    else:
        suffix = "higher" if benefit >= 0 else "lower"
    benefit_text = f"{abs(benefit):.0%}"
    if 0 < abs(benefit) < 0.005:
        benefit_text = "<1%"
    if metric == "peak_vram_mb":
        values = (
            f"{_format_metric_value(edgrec_value)} vs {_format_metric_value(baseline_value)} MB"
        )
    else:
        values = f"{_format_metric_value(edgrec_value)} vs {_format_metric_value(baseline_value)}"
    return f"{values}\n{benefit_text} {suffix}"


def plot_paper_claim_matrix(
    data: DefenseData,
    output_dir: Path,
    *,
    reference_mode: str = "highest_crru",
    filename: str = "paper_claim_matrix.png",
) -> Path:
    """Plot a signed full-data claim matrix against LightGCN paper rows."""
    comparison_records = [
        record
        for record in data.records
        if reference_mode != "kuairec_big_matrix"
        or _is_kuairec_big_matrix_comparison_record(record)
    ]
    edgrec_reference = _selected_reference_rows(data.records, reference_mode=reference_mode)
    lightgcn = _selected_lightgcn_paper(comparison_records)
    probe_datasets = {str(row["dataset"] or "-") for row in data.probe_rows}
    datasets = sorted(edgrec_reference, key=_dataset_sort_key)
    rows = (
        ("NDCG@20", "ndcg20", False),
        ("Recall@20", "recall20", False),
        ("Time/epoch", "time_per_epoch_s", True),
        ("AvgPop@20", "avgpop20", True),
        ("Peak VRAM", "peak_vram_mb", True),
    )
    matrix = np.full((len(rows), len(datasets)), np.nan)
    labels: list[list[str]] = []
    for row_index, (_title, metric, lower_is_better) in enumerate(rows):
        label_row: list[str] = []
        for col_index, dataset in enumerate(datasets):
            edgrec = edgrec_reference.get(dataset)
            baseline = lightgcn.get(dataset)
            if edgrec is None:
                label_row.append("no EDGRec")
                continue
            if baseline is None:
                label_row.append("probe only" if dataset in probe_datasets else "no full row")
                continue
            edgrec_value = getattr(edgrec, metric)
            baseline_value = getattr(baseline, metric)
            if edgrec_value is None or baseline_value is None:
                label_row.append("missing")
                continue
            matrix[row_index, col_index] = _log_benefit_score(
                edgrec_value,
                baseline_value,
                lower_is_better=lower_is_better,
            )
            label_row.append(
                _paper_comparison_label(
                    edgrec_value,
                    baseline_value,
                    metric=metric,
                    lower_is_better=lower_is_better,
                ),
            )
        labels.append(label_row)

    cmap = LinearSegmentedColormap.from_list(
        "edgrec_lightgcn_benefit",
        ["#c44e52", "#f8fafc", "#3c8d4c"],
    )
    cmap.set_bad("#eef1f4")
    fig, ax = plt.subplots(figsize=(A4_TEXT_WIDTH_IN, 3.85))
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(datasets)), [_dataset_label(dataset) for dataset in datasets])
    ax.set_yticks(np.arange(len(rows)), [row[0] for row in rows])
    title = "Claim matrix: EDGRec-family reference vs full-data LightGCN"
    if reference_mode == "kuairec_big_matrix":
        title = "Claim matrix: EDGRec vs full-data LightGCN"
    ax.set_title(title, loc="center")
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.grid(which="major", visible=False)
    for row_index in range(len(rows)):
        for col_index in range(len(datasets)):
            color_value = matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                labels[row_index][col_index],
                ha="center",
                va="center",
                fontsize=6.4,
                color="white"
                if math.isfinite(color_value) and abs(color_value) > 0.65
                else "#111827",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025)
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.set_ticklabels(
        [
            "LightGCN\n>=16x",
            "LightGCN\n4x",
            "parity",
            "EDGRec\n4x",
            "EDGRec\n>=16x",
        ],
    )
    colorbar.set_label("Signed log fold-change", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.4)
    fig.subplots_adjust(left=0.15, right=0.87, bottom=0.14, top=0.88)
    return _save_figure(fig, output_dir, filename, tight=False)


def _scatter_size(record: DefenseRecord) -> float:
    """Return point size scaled by peak VRAM."""
    if record.peak_vram_mb is None:
        return 70.0
    return 45.0 + min(max(record.peak_vram_mb / 120.0, 0.0), 130.0)


def _scatter_records(
    ax: plt.Axes,
    records: Sequence[DefenseRecord],
    *,
    x_field: str,
    y_field: str,
) -> None:
    """Scatter records using method-family marker and color."""
    for record in records:
        x_value = getattr(record, x_field)
        y_value = getattr(record, y_field)
        if x_value is None or y_value is None or x_value <= 0:
            continue
        style = METHOD_STYLES.get(record.method_group, METHOD_STYLES["Other"])
        ax.scatter(
            x_value,
            y_value,
            s=_scatter_size(record),
            marker=str(style["marker"]),
            color=str(style["color"]),
            edgecolor="#1f2933",
            linewidth=0.55,
            alpha=0.80,
        )


def _highlight_record(ax: plt.Axes, record: DefenseRecord | None, *, x_field: str) -> None:
    """Highlight one selected record with a reader-facing reference star."""
    if record is None:
        return
    x_value = getattr(record, x_field)
    y_value = record.ndcg20
    if x_value is None or y_value is None or x_value <= 0:
        return
    ax.scatter(
        x_value,
        y_value,
        s=230,
        marker="*",
        color="#ffd23f",
        edgecolor="#111827",
        linewidth=1.0,
        zorder=5,
    )


def _legend_handles(method_groups: Sequence[str]) -> list[Line2D]:
    """Return method legend handles for scatter plots."""
    handles: list[Line2D] = []
    for group in method_groups:
        style = METHOD_STYLES.get(group, METHOD_STYLES["Other"])
        handles.append(
            Line2D(
                [],
                [],
                marker=str(style["marker"]),
                linestyle="None",
                markerfacecolor=str(style["color"]),
                markeredgecolor="#1f2933",
                label=group,
                markersize=8,
            ),
        )
    handles.append(
        Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            markerfacecolor="#ffd23f",
            markeredgecolor="#111827",
            label="EDGRec-family reference",
            markersize=12,
        ),
    )
    return handles


def _set_zero_origin_limits(
    ax: plt.Axes,
    *,
    x_values: Sequence[float | None],
    y_values: Sequence[float | None],
    x_upper: float | None = None,
    y_upper: float | None = None,
) -> None:
    """Anchor a scatter panel at zero while keeping modest headroom."""
    finite_x = [float(value) for value in x_values if value is not None and value >= 0]
    finite_y = [float(value) for value in y_values if value is not None and value >= 0]
    if finite_x or x_upper is not None:
        upper = x_upper if x_upper is not None else max(finite_x)
        ax.set_xlim(left=0.0, right=upper * 1.08 if upper > 0 else 1.0)
    if finite_y or y_upper is not None:
        upper = y_upper if y_upper is not None else max(finite_y)
        ax.set_ylim(bottom=0.0, top=upper * 1.08 if upper > 0 else 1.0)


def _padded_zoom_bounds(values: Sequence[float | None]) -> tuple[float, float] | None:
    """Return non-zero-origin bounds around finite non-negative values."""
    finite = [float(value) for value in values if value is not None and value >= 0]
    if not finite:
        return None
    lower = min(finite)
    upper = max(finite)
    span = upper - lower
    pad = max(abs(upper) * 0.1, 0.01) if span <= 0 else span * 0.08
    padded_lower = lower - pad
    padded_upper = upper + pad
    padded_lower = max(padded_lower, lower * 0.85) if lower > 0 else 0.0
    if padded_upper <= padded_lower:
        padded_upper = padded_lower + max(abs(padded_lower) * 0.10, 0.01)
    return padded_lower, padded_upper


def _set_zoomed_limits(
    ax: plt.Axes,
    *,
    x_values: Sequence[float | None],
    y_values: Sequence[float | None],
) -> None:
    """Set local data-range limits for zoomed companion panels."""
    x_bounds = _padded_zoom_bounds(x_values)
    y_bounds = _padded_zoom_bounds(y_values)
    if x_bounds is not None:
        ax.set_xlim(*x_bounds)
    if y_bounds is not None:
        ax.set_ylim(*y_bounds)


def _focused_x_upper(
    x_values: Sequence[float | None],
    *,
    required_values: Sequence[float | None] = (),
    quantile: float = 0.90,
) -> float | None:
    """Return a per-panel x maximum that hides extreme right-tail outliers."""
    finite = sorted(float(value) for value in x_values if value is not None and value >= 0)
    if not finite:
        return None
    required = [
        float(value)
        for value in required_values
        if value is not None and math.isfinite(float(value)) and float(value) >= 0
    ]
    if len(finite) < 4:
        return max([*finite, *required])
    upper = float(np.quantile(finite, quantile))
    if required:
        upper = max(upper, max(required))
    return min(max(upper, 0.0), max(finite))


def _records_within_x_upper(
    records: Sequence[DefenseRecord],
    *,
    field: str,
    x_upper: float | None,
) -> list[DefenseRecord]:
    """Return records visible inside a trimmed x-axis limit."""
    if x_upper is None:
        return list(records)
    visible: list[DefenseRecord] = []
    for record in records:
        value = getattr(record, field)
        if value is None or value <= x_upper:
            visible.append(record)
    return visible


def _paper_family(record: DefenseRecord) -> str:
    """Return a reduced method family for paper-level plots."""
    if record.method_group == "EDGRec family":
        return "EDGRec family"
    if record.method_group == "LightGCN paper-faithful":
        return "LightGCN paper-faithful"
    if record.method_group == "LightGCN sampled ablation":
        return "LightGCN sampled ablation"
    if record.method_group == "DICE-style sampled ablation":
        return "DICE-style sampled ablation"
    return "Other"


def _pareto_frontier(
    records: Sequence[DefenseRecord],
) -> list[tuple[float, float, DefenseRecord]]:
    """Return records on the accuracy-efficiency frontier."""
    valid = [
        (record.time_per_epoch_s, record.ndcg20, record)
        for record in records
        if record.time_per_epoch_s is not None
        and record.time_per_epoch_s > 0
        and record.ndcg20 is not None
    ]
    valid.sort(key=lambda item: item[0])
    frontier: list[tuple[float, float, DefenseRecord]] = []
    best_ndcg = float("-inf")
    for time_s, ndcg, record in valid:
        if ndcg > best_ndcg + 1e-12:
            frontier.append((float(time_s), float(ndcg), record))
            best_ndcg = float(ndcg)
    return frontier


def plot_paper_accuracy_efficiency_frontier(
    data: DefenseData,
    output_dir: Path,
    *,
    focused: bool = False,
    reference_mode: str = "highest_crru",
    filename: str | None = None,
) -> Path:
    """Plot test-set accuracy-efficiency frontiers by dataset."""
    comparison_records = [
        record
        for record in data.top_records
        if reference_mode != "kuairec_big_matrix"
        or _is_kuairec_big_matrix_comparison_record(record)
    ]
    full_comparison_records = [
        record
        for record in data.records
        if reference_mode != "kuairec_big_matrix"
        or _is_kuairec_big_matrix_comparison_record(record)
    ]
    edgrec_reference = _selected_reference_rows(data.records, reference_mode=reference_mode)
    lightgcn = _selected_lightgcn_paper(full_comparison_records)
    grouped = _records_by_dataset(comparison_records)
    family_styles = {
        "EDGRec family": {"color": PAPER_BLUE, "marker": "o", "alpha": 0.72},
        "LightGCN paper-faithful": {"color": PAPER_ORANGE, "marker": "s", "alpha": 0.85},
        "LightGCN sampled ablation": {"color": "#f2b447", "marker": "^", "alpha": 0.68},
        "DICE-style sampled ablation": {"color": "#7b5aa6", "marker": "P", "alpha": 0.66},
        "Other": {"color": PAPER_GRAY, "marker": "x", "alpha": 0.55},
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), sharey=False)
    for ax, dataset in zip(axes.ravel(), sorted(grouped, key=_dataset_sort_key), strict=False):
        records = grouped[dataset]
        selected = edgrec_reference.get(dataset)
        baseline = lightgcn.get(dataset)
        x_upper = None
        if focused:
            x_upper = _focused_x_upper(
                [record.time_per_epoch_s for record in records],
                required_values=[
                    selected.time_per_epoch_s if selected is not None else None,
                ],
            )
        visible_records = _records_within_x_upper(
            records,
            field="time_per_epoch_s",
            x_upper=x_upper,
        )
        for record in visible_records:
            if record.time_per_epoch_s is None or record.ndcg20 is None:
                continue
            if record.time_per_epoch_s <= 0:
                continue
            family = _paper_family(record)
            style = family_styles[family]
            ax.scatter(
                record.time_per_epoch_s,
                record.ndcg20,
                s=70 if family != "EDGRec family" else 82,
                color=str(style["color"]),
                marker=str(style["marker"]),
                alpha=float(style["alpha"]),
                edgecolor="#111827" if family != "Other" else PAPER_GRAY,
                linewidth=0.55,
                zorder=3,
            )
        frontier = _pareto_frontier(visible_records)
        if len(frontier) >= 2:
            ax.plot(
                [item[0] for item in frontier],
                [item[1] for item in frontier],
                color="#111827",
                linewidth=1.35,
                alpha=0.85,
                zorder=2,
            )
        if (
            selected is not None
            and selected.time_per_epoch_s is not None
            and selected.ndcg20 is not None
            and (x_upper is None or selected.time_per_epoch_s <= x_upper)
        ):
            ax.scatter(
                selected.time_per_epoch_s,
                selected.ndcg20,
                s=230,
                marker="*",
                color="#f6c141",
                edgecolor="#111827",
                linewidth=0.9,
                zorder=5,
            )
            ax.annotate(
                "EDGRec ref." if reference_mode == "highest_crru" else "EDGRec",
                (selected.time_per_epoch_s, selected.ndcg20),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=7.8,
                color="#111827",
            )
        if (
            baseline is not None
            and baseline.time_per_epoch_s is not None
            and baseline.ndcg20 is not None
            and (x_upper is None or baseline.time_per_epoch_s <= x_upper)
        ):
            ax.annotate(
                "LightGCN",
                (baseline.time_per_epoch_s, baseline.ndcg20),
                xytext=(6, -12),
                textcoords="offset points",
                fontsize=8,
                color=PAPER_ORANGE,
            )
        x_values = [record.time_per_epoch_s for record in visible_records]
        y_values = [record.ndcg20 for record in visible_records]
        if focused:
            _set_zoomed_limits(ax, x_values=x_values, y_values=y_values)
        else:
            _set_zero_origin_limits(
                ax,
                x_values=x_values,
                y_values=y_values,
                x_upper=x_upper,
            )
        ax.set_xlabel("Seconds per epoch (lower is better)")
        ax.set_ylabel("Test NDCG@20")
        ax.set_title(_dataset_label(dataset), loc="center")
        _strip_axes(ax)
    handles = [
        Line2D(
            [],
            [],
            marker=str(style["marker"]),
            linestyle="None",
            markerfacecolor=str(style["color"]),
            markeredgecolor="#111827",
            label=family,
            markersize=8,
            alpha=float(style["alpha"]),
        )
        for family, style in family_styles.items()
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            markerfacecolor="#f6c141",
            markeredgecolor="#111827",
            label=_reference_label(reference_mode),
            markersize=12,
        ),
    )
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.01))
    title = "Test-set accuracy-efficiency frontier"
    if focused:
        title = "Test-set accuracy-efficiency frontier, zoomed axes"
    fig.suptitle(title, y=0.965, ha="center")
    note = (
        "Black lines mark the observed Pareto frontier among the displayed completed "
        "full-data test rows: higher NDCG and lower seconds per epoch are preferred. "
        f"{_reference_note(reference_mode)}"
    )
    if focused:
        note = (
            "Zoomed companion view: axes use local data ranges and may not start at zero. "
            "Each panel also trims the extreme right tail of seconds/epoch while keeping the "
            "EDGRec reference row visible. Use the full-range companion for complete context. "
            f"{_reference_note(reference_mode)}"
        )
    _centered_figure_note(
        fig,
        note,
        y=0.06,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    output_filename = filename
    if output_filename is None:
        output_filename = (
            "paper_accuracy_efficiency_frontier_zoomed.png"
            if focused
            else "paper_accuracy_efficiency_frontier.png"
        )
    return _save_figure(fig, output_dir, output_filename)


def plot_paper_mechanism_diagnostics(data: DefenseData, output_dir: Path) -> Path:
    """Plot one consolidated mechanism diagnostic for EDGRec-family reference rows."""
    selected = [
        record
        for _dataset, record in sorted(
            _selected_edgrec_reference_rows(
                data.records,
                require_score_mix_telemetry=True,
                require_branch_rank_telemetry=True,
            ).items(),
            key=lambda item: _dataset_sort_key(item[0]),
        )
        if record.score_mix_interest is not None
    ]
    if not selected:
        raise RuntimeError("No EDGRec-family mechanism diagnostics available.")
    labels = [_dataset_label(record.dataset) for record in selected]
    y = np.arange(len(selected))
    interest = np.array([record.score_mix_interest or 0.0 for record in selected])
    conformity = np.array([record.score_mix_conformity or 0.0 for record in selected])
    context = np.array([record.score_mix_context or 0.0 for record in selected])

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.4))
    ax_mix, ax_diag, ax_ndcg, ax_pop = axes.ravel()

    ax_mix.barh(y, interest, color=PAPER_BLUE, label="Interest")
    ax_mix.barh(y, conformity, left=interest, color="#7b5aa6", label="Conformity")
    ax_mix.barh(y, context, left=interest + conformity, color=PAPER_GREEN, label="Context")
    for index, values in enumerate(zip(interest, conformity, context, strict=True)):
        left = 0.0
        for value in values:
            if value >= 0.08:
                ax_mix.text(
                    left + value / 2.0,
                    index,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.18 else "#111827",
                )
            left += value
    ax_mix.set_yticks(y, labels)
    ax_mix.invert_yaxis()
    ax_mix.set_xlim(0, 1)
    ax_mix.set_xlabel("")
    ax_mix.set_title("Score composition", loc="center")
    ax_mix.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=True,
        framealpha=0.94,
    )
    _strip_axes(ax_mix)

    ax_diag.axvline(0, color=PAPER_GRAY, linewidth=1.0)
    final_pop = [record.final_popularity_spearman for record in selected]
    branch_cos = [record.branch_cosine for record in selected]
    ax_diag.scatter(
        final_pop,
        y - 0.08,
        marker="o",
        s=82,
        color=PAPER_ORANGE,
        edgecolor="#111827",
        label="Final-popularity Spearman@20",
        zorder=3,
    )
    ax_diag.scatter(
        branch_cos,
        y + 0.08,
        marker="s",
        s=74,
        color="#7b5aa6",
        edgecolor="#111827",
        label="Interest-conformity cosine",
        zorder=3,
    )
    ax_diag.set_yticks(y, labels)
    ax_diag.invert_yaxis()
    ax_diag.set_xlim(-1.05, 1.05)
    ax_diag.set_xlabel("Diagnostic value")
    ax_diag.set_title("Alignment checks", loc="center")
    ax_diag.legend(loc="upper left", frameon=True, framealpha=0.94)
    _strip_axes(ax_diag)

    branch_specs = (
        (
            ax_ndcg,
            "Standalone branch NDCG@20",
            "NDCG@20",
            [record.interest_branch_ndcg20 for record in selected],
            [record.conformity_branch_ndcg20 for record in selected],
        ),
        (
            ax_pop,
            "Standalone branch AvgPop@20",
            "Raw AveragePopularity@20",
            [record.interest_branch_avgpop20 for record in selected],
            [record.conformity_branch_avgpop20 for record in selected],
        ),
    )
    for ax, title, xlabel, interest_values, conformity_values in branch_specs:
        has_interest_label = False
        has_conformity_label = False
        for index, (interest_value, conformity_value) in enumerate(
            zip(interest_values, conformity_values, strict=True),
        ):
            if interest_value is None or conformity_value is None:
                continue
            ax.plot(
                [interest_value, conformity_value],
                [index, index],
                color="#aeb7c2",
                linewidth=2.0,
                zorder=1,
            )
            ax.scatter(
                interest_value,
                index,
                s=82,
                color=PAPER_BLUE,
                edgecolor="#111827",
                label="Interest" if not has_interest_label else None,
                zorder=3,
            )
            has_interest_label = True
            ax.scatter(
                conformity_value,
                index,
                s=78,
                color="#7b5aa6",
                marker="s",
                edgecolor="#111827",
                label="Conformity" if not has_conformity_label else None,
                zorder=3,
            )
            has_conformity_label = True
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="center")
        legend_location = "upper right" if ax is ax_pop else "lower right"
        ax.legend(loc=legend_location, frameon=True, framealpha=0.94)
        _strip_axes(ax)
    fig.suptitle("EDGRec mechanism diagnostics for reference rows", y=0.985, ha="center")
    _centered_figure_note(
        fig,
        "Reference rows here are the highest-CRRU completed EDGRec-family test rows with "
        "both score-mix and standalone branch telemetry. These diagnostics are not causal "
        "effect identification.",
        y=0.02,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return _save_figure(fig, output_dir, "paper_mechanism_diagnostics.png")


def _kuairec_public_ablation_records(data: DefenseData) -> dict[str, DefenseRecord]:
    """Return current public KuaiRec ablation records keyed by variant."""
    records: dict[str, DefenseRecord] = {}
    for record in data.records:
        if record.dataset != "kuairec_v2" or record.evidence != "ablation":
            continue
        raw_keys = {
            str(record.profile or ""),
            _record_string(record, "profile_name"),
            _record_string(record, "intervention"),
        }
        for key in KUAIREC_ABLATION_PROFILES:
            if key in raw_keys:
                records[key] = record
    return records


def _plot_delta_label(ax: plt.Axes, value: float, *, y: float, formatter: str) -> None:
    """Place one delta label just outside the bar end."""
    x_min, x_max = ax.get_xlim()
    offset = (x_max - x_min) * 0.025
    if value >= 0:
        x = value + offset
        ha = "left"
    else:
        x = value - offset
        ha = "right"
    ax.text(
        x,
        y,
        formatter.format(value),
        ha=ha,
        va="center",
        fontsize=8,
        color="#111827",
    )


def plot_kuairec_ablation_deltas(data: DefenseData, output_dir: Path) -> Path:
    """Plot protocol-local KuaiRec public ablation deltas relative to mainline."""
    records = _kuairec_public_ablation_records(data)
    mainline = records.get("mainline")
    variants = [
        ("no_popularity_head", "No popularity\nhead"),
        ("no_independence", "No independence\nregularizer"),
    ]
    selected = [(key, label, records.get(key)) for key, label in variants if records.get(key)]
    if mainline is None or not selected:
        raise RuntimeError("KuaiRec public ablation rows are incomplete.")

    metric_specs: tuple[
        tuple[str, str, Callable[[DefenseRecord], float | None], bool, str],
        ...,
    ] = (
        ("NDCG@20", r"$\Delta$ NDCG@20", lambda record: record.ndcg20, True, "{:+.4f}"),
        (
            "AvgPop@20",
            r"$\Delta$ AvgPop@20",
            lambda record: record.avgpop20,
            False,
            "{:+.4f}",
        ),
        (
            "Pers.@20",
            r"$\Delta$ Personalization@20",
            lambda record: record.personalization20,
            True,
            "{:+.4f}",
        ),
        ("CRRU@20", r"$\Delta$ CRRU@20", lambda record: record.crru20, True, "{:+.4f}"),
        (
            "Time/epoch",
            r"$\Delta$ seconds/epoch",
            lambda record: record.time_per_epoch_s,
            False,
            "{:+.1f}",
        ),
        (
            "Peak VRAM",
            r"$\Delta$ peak VRAM (MB)",
            lambda record: record.peak_vram_mb,
            False,
            "{:+.0f}",
        ),
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(A4_TEXT_WIDTH_IN * 1.95, A4_TEXT_WIDTH_IN * 1.17),
        sharey=False,
    )
    y = np.arange(len(selected))
    y_labels = [label for _key, label, _record in selected]
    metric_axes = zip(axes.ravel(), metric_specs, strict=True)
    for panel_index, (
        ax,
        (_short_title, title, getter, higher_is_better, formatter),
    ) in enumerate(metric_axes):
        base_value = getter(mainline)
        deltas: list[float] = []
        for _key, _label, record in selected:
            value = getter(record) if record is not None else None
            if value is None or base_value is None:
                deltas.append(0.0)
            else:
                deltas.append(value - base_value)
        max_abs = max([abs(delta) for delta in deltas] + [1e-6])
        ax.set_xlim(-max_abs * 1.58, max_abs * 1.58)
        colors = [
            PAPER_GREEN if (delta >= 0) == higher_is_better or abs(delta) < 1e-12 else PAPER_RED
            for delta in deltas
        ]
        bars = ax.barh(
            y,
            deltas,
            height=0.52,
            color=colors,
            edgecolor="#111827",
            linewidth=0.55,
        )
        ax.axvline(0.0, color=PAPER_GRAY, linewidth=1.0)
        ax.set_title(title, loc="center")
        ax.set_yticks(y, y_labels if panel_index % 3 == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.55)
        ax.grid(axis="y", visible=False)
        _strip_axes(ax, keep_left=False)
        for index, bar in enumerate(bars):
            _plot_delta_label(
                ax,
                deltas[index],
                y=bar.get_y() + bar.get_height() / 2.0,
                formatter=formatter,
            )
    axes[0, 0].set_ylabel("Component removal")
    fig.suptitle(
        "KuaiRec v2 matched public ablations relative to EDGRec mainline",
        y=0.982,
        ha="center",
    )
    _centered_figure_note(
        fig,
        "Deltas use only the current public matched full-data ablation rows visible in "
        "query-results. Green indicates movement in the preferred direction for that metric; "
        "for AveragePopularity, time, and VRAM, lower is preferred. These rows support "
        "protocol-local design interpretation only.",
        y=0.02,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    return _save_figure(fig, output_dir, "kuairec_ablation_deltas.png")


def plot_accuracy_popularity_tradeoff(
    data: DefenseData,
    output_dir: Path,
    *,
    focused: bool = False,
) -> Path:
    """Plot NDCG versus raw train-popularity concentration."""
    edgrec_reference = _selected_edgrec_reference_rows(data.records)
    grouped = _records_by_dataset(data.top_records)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), sharey=False)
    for ax, dataset in zip(axes.ravel(), sorted(grouped, key=_dataset_sort_key), strict=False):
        records = grouped[dataset]
        selected = edgrec_reference.get(dataset)
        x_upper = None
        if focused:
            x_upper = _focused_x_upper(
                [record.avgpop20 for record in records],
                required_values=[selected.avgpop20 if selected is not None else None],
            )
        visible_records = _records_within_x_upper(
            records,
            field="avgpop20",
            x_upper=x_upper,
        )
        _scatter_records(ax, visible_records, x_field="avgpop20", y_field="ndcg20")
        _highlight_record(ax, selected, x_field="avgpop20")
        x_values = [record.avgpop20 for record in visible_records]
        y_values = [record.ndcg20 for record in visible_records]
        if focused:
            _set_zoomed_limits(ax, x_values=x_values, y_values=y_values)
        else:
            _set_zero_origin_limits(
                ax,
                x_values=x_values,
                y_values=y_values,
                x_upper=x_upper,
            )
        ax.set_title(_dataset_label(dataset), loc="center")
        ax.set_xlabel("Raw PyG AveragePopularity@20 (lower is less concentration)")
        ax.set_ylabel("Test NDCG@20")
    method_groups = sorted({record.method_group for record in data.top_records})
    handles = _legend_handles(method_groups)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.05,
        handletextpad=0.45,
        borderaxespad=0.2,
    )
    title = "Accuracy-popularity trade-off across completed full-data rows"
    if focused:
        title = "Accuracy-popularity trade-off, zoomed axes"
    fig.suptitle(
        title,
        y=0.965,
        ha="center",
    )
    if focused:
        _centered_figure_note(
            fig,
            "Zoomed axes; use the full-range companion for scale context.",
            y=0.055,
        )
        fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    filename = (
        "accuracy_popularity_tradeoff_zoomed.png" if focused else "accuracy_popularity_tradeoff.png"
    )
    return _save_figure(fig, output_dir, filename)


def _metrics_for_record(record: DefenseRecord) -> dict[str, float]:
    """Return CRRU input metrics for one result record."""
    values = {
        "NDCG@20": record.ndcg20,
        "Recall@20": record.recall20,
        "HitRatio@20": record.hit20,
        "Personalization@20": record.personalization20,
        "AveragePopularity@20": record.avgpop20,
        "NDCG@40": _row_value(record.row, "test_ndcg_40"),
        "Recall@40": _row_value(record.row, "test_recall_40"),
        "HitRatio@40": _row_value(record.row, "test_hit_ratio_40"),
        "Personalization@40": _row_value(record.row, "test_personalization_40"),
        "AveragePopularity@40": _row_value(record.row, "test_average_popularity_40"),
    }
    return {key: value for key, value in values.items() if value is not None}


def _component_summary(record: DefenseRecord) -> dict[str, float] | None:
    """Return averaged CRRU components over K=20 and K=40."""
    try:
        largest_count = _largest_training_item_interaction_count_for_row(record.row)
        components = [
            compute_validation_crru_components_for_k(
                _metrics_for_record(record),
                k=cutoff,
                peak_vram_mb=record.peak_vram_mb,
                epoch_time_s=record.time_per_epoch_s,
                largest_training_item_interaction_count=largest_count,
            )
            for cutoff in (20, 40)
        ]
    except ValueError:
        return None
    return {
        key: float(sum(component[key] for component in components) / len(components))
        for key in ("accuracy", "popularity_aware_personalization", "efficiency", "crru")
    }


def plot_crru_decomposition(data: DefenseData, output_dir: Path) -> Path:
    """Plot independent CRRU components for selected EDGRec and LightGCN rows."""
    edgrec_reference = _selected_edgrec_reference_rows(data.records)
    lightgcn = _selected_lightgcn_paper(data.records)
    datasets = sorted(edgrec_reference, key=_dataset_sort_key)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8), sharey=True)
    component_keys = tuple(COMPONENT_LABELS)
    x = np.arange(len(component_keys))
    width = 0.34
    for ax, dataset in zip(axes.ravel(), datasets, strict=False):
        edgrec_summary = None
        baseline_summary = None
        edgrec = edgrec_reference.get(dataset)
        baseline = lightgcn.get(dataset)
        if edgrec is not None:
            edgrec_summary = _component_summary(edgrec)
        if baseline is not None:
            baseline_summary = _component_summary(baseline)
        if edgrec_summary is not None:
            values = [edgrec_summary[key] for key in component_keys]
            bars = ax.bar(
                x - width / 2,
                values,
                width,
                label="EDGRec-family reference",
                color=METHOD_STYLES["EDGRec family"]["color"],
                edgecolor="#111827",
                linewidth=0.55,
            )
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        if baseline_summary is not None:
            values = [baseline_summary[key] for key in component_keys]
            bars = ax.bar(
                x + width / 2,
                values,
                width,
                label="LightGCN paper-faithful",
                color=METHOD_STYLES["LightGCN paper-faithful"]["color"],
                edgecolor="#111827",
                linewidth=0.55,
            )
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        elif dataset in edgrec_reference:
            ax.text(
                0.98,
                0.92,
                "No full-data\nLightGCN-paper row",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=PAPER_GRAY,
            )
        ax.set_title(_dataset_label(dataset), loc="center")
        ax.set_xticks(x, [COMPONENT_LABELS[key] for key in component_keys])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Component score")
        _strip_axes(ax)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor=METHOD_STYLES["EDGRec family"]["color"],
                markeredgecolor="#111827",
                label="EDGRec-family reference",
                markersize=8,
            ),
            Line2D(
                [],
                [],
                marker="s",
                linestyle="None",
                markerfacecolor=METHOD_STYLES["LightGCN paper-faithful"]["color"],
                markeredgecolor="#111827",
                label="LightGCN paper-faithful",
                markersize=8,
            ),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "CRRU component decomposition: independent utility terms",
        y=0.99,
        ha="center",
    )
    _centered_figure_note(
        fig,
        "Bars are independent CRRU terms, not a trend over the x-axis. EDGRec reference rows "
        "are highest-CRRU completed EDGRec-family leaderboard rows. Missing LightGCN bars mean "
        "no full-data LightGCN-paper row is available.",
        y=0.075,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    return _save_figure(fig, output_dir, "crru_component_decomposition.png")


def plot_kuairec_matrix_regime_sensitivity(data: DefenseData, output_dir: Path) -> Path:
    """Plot KuaiRec sparse-vs-dense regime sensitivity without code-facing labels."""
    kuairec_records = [record for record in data.records if record.dataset == "kuairec_v2"]
    small_edgrec = _best_record(
        kuairec_records,
        lambda record: _is_edgrec_family_record(record) and _is_kuairec_small_matrix_record(record),
    )
    big_mainline = _best_record(
        kuairec_records,
        lambda record: (
            _is_edgrec_family_record(record)
            and _is_kuairec_default_big_matrix_record(record)
            and record.label == "mainline"
        ),
    )
    if big_mainline is None:
        big_mainline = _best_record(
            kuairec_records,
            lambda record: (
                _is_edgrec_family_record(record) and _is_kuairec_default_big_matrix_record(record)
            ),
        )
    big_lightgcn = _best_record(
        kuairec_records,
        lambda record: (
            record.label == "lightgcn_paper" and _is_kuairec_default_big_matrix_record(record)
        ),
    )
    rows = [
        ("Dense small-matrix\nEDGRec sensitivity", small_edgrec, "#9ca3af", "//"),
        ("Big-matrix\nEDGRec", big_mainline, PAPER_BLUE, ""),
        ("Sparse big-matrix\nLightGCN baseline", big_lightgcn, PAPER_ORANGE, ""),
    ]
    if any(record is None for _label, record, _color, _hatch in rows):
        raise RuntimeError("Missing KuaiRec regime rows for sensitivity figure.")

    metrics = (
        ("NDCG@20", "ndcg20", "higher is better", ""),
        ("Recall@20", "recall20", "higher is better", ""),
        ("AveragePopularity@20", "avgpop20", "lower means less train-pop concentration", ""),
        ("Seconds per epoch", "time_per_epoch_s", "lower is better", "s"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.4))
    y = np.arange(len(rows))
    for ax, (title, field, direction, suffix) in zip(axes.ravel(), metrics, strict=True):
        values = [getattr(record, field) for _label, record, _color, _hatch in rows]
        finite_values = [float(value) for value in values if value is not None]
        for index, ((_label, _record, color, hatch), value) in enumerate(
            zip(rows, values, strict=True),
        ):
            if value is None:
                continue
            bars = ax.barh(
                index,
                value,
                color=color,
                edgecolor="#111827",
                linewidth=0.65,
                hatch=hatch,
                height=0.58,
            )
            label_text = (
                _format_seconds(value)
                if field == "time_per_epoch_s"
                else (_format_metric_value(value, suffix=suffix))
            )
            ax.bar_label(bars, labels=[label_text], padding=3, fontsize=8)
        ax.set_yticks(y, [label for label, _record, _color, _hatch in rows])
        ax.invert_yaxis()
        ax.set_xlim(left=0.0, right=max(finite_values) * 1.15 if finite_values else 1.0)
        ax.set_title(f"{title}\n{direction}", loc="center", fontsize=10)
        _strip_axes(ax)
    fig.suptitle(
        "KuaiRec v2: dense sensitivity vs big-matrix comparison",
        y=0.985,
        ha="center",
    )
    _centered_figure_note(
        fig,
        "The dense small-matrix row is shown to explain why near-ceiling KuaiRec metrics "
        "exist. The comparison row uses explicit big-matrix watch-ratio evidence and should "
        "be compared with the big-matrix LightGCN row.",
        y=0.02,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    return _save_figure(fig, output_dir, "kuairec_matrix_regime_sensitivity.png")


def _probe_time_per_epoch(row: sqlite3.Row) -> float | None:
    """Return seconds per epoch for a runtime probe row."""
    probe_time = _row_value(row, "runtime_probe_seconds_per_epoch")
    if probe_time is not None and probe_time > 0:
        return probe_time
    return _crru_epoch_time_s(row)


def _probe_label(preset: str) -> str:
    """Return a reader-facing paper-probe label."""
    labels = {
        "lightgcn_paper": "LightGCN paper-faithful probe",
        "dice_paper": "DICE paper-faithful probe",
    }
    return labels.get(preset, preset.replace("_", " "))


def plot_paper_baseline_feasibility(data: DefenseData, output_dir: Path) -> Path:
    """Plot runtime-probe seconds and slowdown against EDGRec reference rows."""
    edgrec_reference = _selected_reference_rows(
        data.records,
        reference_mode="kuairec_big_matrix",
    )
    pairs: list[tuple[str, float, float, float]] = []
    for row in data.probe_rows:
        dataset = str(row["dataset"] or "-")
        baseline_time = _probe_time_per_epoch(row)
        edgrec = edgrec_reference.get(dataset)
        if baseline_time is None or edgrec is None or edgrec.time_per_epoch_s is None:
            continue
        preset = str(row["preset"] or "-")
        label = f"{_dataset_label(dataset)}\n{_probe_label(preset)}"
        speedup = baseline_time / edgrec.time_per_epoch_s
        pairs.append((label, edgrec.time_per_epoch_s, baseline_time, speedup))
    if not pairs:
        raise RuntimeError("No paper baseline runtime probes available.")

    pairs.sort(key=lambda item: item[3], reverse=True)
    y = np.arange(len(pairs))
    labels = [item[0] for item in pairs]
    edgrec_times = [item[1] for item in pairs]
    probe_times = [item[2] for item in pairs]
    ratios = [item[3] for item in pairs]

    fig, (ax_time, ax_ratio) = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.8),
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.08},
        sharey=True,
    )
    bar_height = 0.34
    ax_time.barh(
        y - bar_height / 2,
        edgrec_times,
        height=bar_height,
        color=METHOD_STYLES["EDGRec family"]["color"],
        edgecolor="#111827",
        linewidth=0.55,
        label="EDGRec-family reference",
    )
    ax_time.barh(
        y + bar_height / 2,
        probe_times,
        height=bar_height,
        color=PAPER_RED,
        edgecolor="#111827",
        linewidth=0.55,
        label="Paper-faithful probe",
    )
    for index, (edgrec_time, probe_time) in enumerate(zip(edgrec_times, probe_times, strict=True)):
        ax_time.text(
            edgrec_time * 1.15,
            index - bar_height / 2,
            _format_seconds(edgrec_time),
            va="center",
            ha="left",
            fontsize=8,
            color="#111827",
        )
        ax_time.text(
            probe_time * 1.08,
            index + bar_height / 2,
            _format_seconds(probe_time),
            va="center",
            ha="left",
            fontsize=8,
            color="#111827",
        )
    ax_time.set_xscale("log")
    ax_time.set_xlabel("Seconds per epoch (log)")
    ax_time.set_yticks(y, labels)
    ax_time.invert_yaxis()
    ax_time.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax_time.set_title("Raw epoch cost", loc="center")
    _strip_axes(ax_time)

    ax_ratio.axvline(1.0, color="#4b5563", linewidth=1.0)
    ax_ratio.barh(
        y,
        ratios,
        height=0.42,
        color="#c44e52",
        edgecolor="#111827",
        linewidth=0.55,
    )
    for index, ratio in enumerate(ratios):
        ax_ratio.text(
            ratio * 1.10,
            index,
            f"{ratio:,.0f}x",
            va="center",
            ha="left",
            fontsize=9,
            color="#111827",
            weight="bold",
        )
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlim(0.8, max(ratios) * 2.6)
    ax_ratio.set_xlabel("Probe slowdown vs EDGRec (log)")
    ax_ratio.tick_params(axis="y", left=False, labelleft=False)
    ax_ratio.set_title("Slowdown ratio", loc="center")
    _strip_axes(ax_ratio, keep_left=False)

    fig.suptitle(
        "Runtime probes for paper-faithful baselines: resource evidence only",
        y=0.98,
        ha="center",
    )
    _centered_figure_note(
        fig,
        "EDGRec reference rows use the same KuaiRec big-matrix policy as the claim matrix. "
        "These probe rows justify feasibility limits; they are not final test-accuracy "
        "comparisons.",
        y=0.018,
    )
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.18, top=0.88, wspace=0.10)
    return _save_figure(fig, output_dir, "paper_baseline_feasibility.png", tight=False)


def _figure_metadata() -> tuple[tuple[str, str, str, str], ...]:
    """Return defense-use descriptions keyed by generated filename."""
    return (
        (
            "architecture_pipeline.md",
            "Markdown Mermaid source",
            "What is built?",
            "Use as a previewable Mermaid architecture source or reconstruct it with TikZ.",
        ),
        (
            "candidate_taxonomy.png",
            "method taxonomy",
            "What do the labels mean?",
            "Use as a review or slide legend when readers need paper-baseline vs "
            "sampled-ablation semantics; in the thesis body, prefer the same content as a "
            "LaTeX table if the PNG feels text-heavy.",
        ),
        (
            "paper_dataset_regime_map.png",
            "log-scale regime map",
            "Why these datasets?",
            "Use as the paper-quality setup figure; existing dataset-profile figures remain "
            "backup detail.",
        ),
        (
            "paper_claim_matrix.png",
            "annotated claim matrix",
            "What can be claimed?",
            "Use as the main full-data LightGCN comparison; it shows direction, magnitude, "
            "and raw values in one panel.",
        ),
        (
            "paper_accuracy_efficiency_frontier.png",
            "Pareto-frontier scatter",
            "Is EDGRec on a useful frontier?",
            "Use as the full-range trade-off view, including outlier context.",
        ),
        (
            "paper_accuracy_efficiency_frontier_zoomed.png",
            "zoomed Pareto-frontier scatter",
            "What happens near the visible cluster?",
            "Use as the readable companion after the full-range frontier; axes use local "
            "data ranges and right-tail outliers are intentionally trimmed per panel.",
        ),
        (
            "paper_mechanism_diagnostics.png",
            "consolidated diagnostic panels",
            "Does the mechanism behave plausibly?",
            "Use as the main mechanism figure; score-mix and branch-rank diagnostics are "
            "merged here.",
        ),
        (
            "kuairec_ablation_deltas.png",
            "matched ablation delta panels",
            "Which component choices are currently supported?",
            "Use as the compact RQ4 figure for the current public KuaiRec matched "
            "ablations; keep the wording protocol-local and do not generalize across "
            "datasets.",
        ),
        (
            "accuracy_popularity_tradeoff.png",
            "trade-off scatter",
            "Does lower popularity concentration cost accuracy?",
            "Use as the full-range popularity-concentration view, including outlier context.",
        ),
        (
            "accuracy_popularity_tradeoff_zoomed.png",
            "zoomed trade-off scatter",
            "Does the local popularity trade-off remain visible?",
            "Use as the readable companion after the full-range popularity plot; axes use "
            "local data ranges and right-tail outliers are intentionally trimmed per panel.",
        ),
        (
            "crru_component_decomposition.png",
            "grouped component bars",
            "What does CRRU reward?",
            "Use to defend the composite utility without implying a trend between "
            "independent terms.",
        ),
        (
            "paper_baseline_feasibility.png",
            "paired log-time points plus slowdown",
            "Why are some baselines probes?",
            "Use to defend DICE/large full-graph feasibility limits and evidence roles.",
        ),
        (
            "kuairec_matrix_regime_sensitivity.png",
            "protocol-sensitivity bars",
            "Why is the near-ceiling KuaiRec row excluded?",
            "Use only as a sensitivity companion showing that small-matrix/full-observation "
            "rows are a different protocol from the sparse KuaiRec comparison.",
        ),
    )


def _write_figure_index(
    output_dir: Path,
    *,
    generated_paths: Sequence[Path],
    data: DefenseData,
) -> Path:
    """Write a Markdown index explaining how each figure should be defended."""
    generated_names = {path.name for path in generated_paths}
    lines = [
        "# Thesis Defense Figures",
        "",
        "Generated by `uv run scripts/export_defense_figures.py`.",
        "Architecture is exported as previewable Markdown Mermaid; empirical charts are "
        "exported as PNG.",
        "",
        f"- SQLite source: `{THESIS_DB_PATH}`",
        f"- Completed report rows loaded: `{len(data.records)}`",
        f"- Leaderboard rows plotted in trade-off figures: `{len(data.top_records)}`",
        f"- Runtime probe rows loaded: `{len(data.probe_rows)}`",
        "",
        "## Text Policy",
        "",
        "- Keep plot canvases visually light. Prefer axis labels, legends, and compact numeric "
        "annotations inside figures; put interpretation and caveats in LaTeX captions, "
        "surrounding prose, or this index.",
        "- Text-heavy explanatory figures should become LaTeX tables where possible; keep PNG "
        "versions mainly for review or slides.",
        "",
        "## Selection Rules",
        "",
        "- EDGRec reference row: highest-CRRU completed EDGRec-family test row per dataset "
        "from the same report semantics as `results/query_results.md`.",
        "- Mechanism-reference row: highest-CRRU EDGRec-family test row with the telemetry "
        "needed by that diagnostic panel.",
        "- Runtime-probe rows: resource evidence only; one-epoch accuracy is diagnostic.",
        "",
        "## Figure Guide",
        "",
        "| Artifact | Figure form | Committee question | Defense use |",
        "| --- | --- | --- | --- |",
    ]
    for filename, form, question, use in _figure_metadata():
        if filename not in generated_names:
            continue
        lines.append(f"| `{filename}` | {form} | {question} | {use} |")
    lines.extend(
        [
            "",
            "## Recommended Slide Flow",
            "",
            "| Stage | Artifact | Why this order |",
            "| --- | --- | --- |",
            "| Dataset setup | `paper_dataset_regime_map.png` | Establish the tested regimes "
            "without repeating every descriptive dataset bar chart. |",
            "| Architecture | `architecture_pipeline.md` | Preview the Mermaid source directly "
            "or rebuild it as deck-native TikZ. |",
            "| Candidate semantics | `candidate_taxonomy.png` | Make clear that EDGRec "
            "candidates are "
            "not split by report provenance, and that sampled LightGCN/DICE-style rows are "
            "not paper-faithful baselines. |",
            "| Search selection | "
            "`results/optuna_figures/optuna_crru_selection_frontier_by_dataset.png` "
            "plus one Optuna diagnostic if needed | Show validation-time candidate selection "
            "without duplicating test-set claims. |",
            "| Test evidence | `paper_claim_matrix.png` | State the full-data LightGCN "
            "paper-faithful comparison with direction, magnitude, raw values, and "
            "missing-baseline status. |",
            "| Trade-offs | `paper_accuracy_efficiency_frontier.png` plus "
            "`paper_accuracy_efficiency_frontier_zoomed.png` | Defend the thesis as a "
            "Pareto/trade-off result, then use the zoomed companion for readable clusters. |",
            "| Mechanism diagnostics | `paper_mechanism_diagnostics.png` | Explain branch usage "
            "while stating that these are diagnostics, not causal identification. |",
            "| Ablation evidence | `kuairec_ablation_deltas.png` | Defend the currently "
            "available component-removal evidence as KuaiRec protocol-local, not as a "
            "dataset-general component theorem. |",
            "| Feasibility limits | `paper_baseline_feasibility.png` | Justify why DICE-paper and "
            "large full-graph baselines are sometimes resource probes, with seconds/epoch "
            "and slowdown shown together. |",
            "",
            "## Claim Boundaries",
            "",
            "- Ranking metrics and CRRU are thesis utility evidence, not causal-effect estimates.",
            "- Branch-rank, score-mix, Spearman, and cosine plots are diagnostics only.",
            "- Runtime-probe rows support feasibility and resource claims, "
            "not final accuracy claims.",
            "- The public ablation plot uses matched KuaiRec rows only; it supports design "
            "choices under that protocol and identifies missing ablations for future work.",
            "- KuaiRand compact randomized-exposure rows need separate wording from full "
            "standard-view runs.",
            "- `figure_review.md` contains the per-image audit, recommended main flow, "
            "backup-figure decisions, and caveats for defense use.",
            "",
            "## Existing Companion Figures",
            "",
            "- `results/dataset_visualizations/` owns dataset scale, density, split, feedback, "
            "and feature-availability context. Use them for backup detail; the regime map is "
            "the compact main-flow version.",
            "- `results/optuna_figures/` owns validation-search behavior, hyperparameter "
            "response, and selected-trial diagnostics. Use defense figures for full-data "
            "test evidence and claim boundaries.",
            "- `accuracy_popularity_tradeoff.png`, `accuracy_popularity_tradeoff_zoomed.png`, "
            "and `crru_component_decomposition.png` are backup figures after the paper-grade "
            "core set above. The duplicate NDCG-time "
            "scatter, evidence map, paired-dot raw comparison, standalone score-mix figure, "
            "standalone branch figure, raw `.mmd` source, and architecture PNG are "
            "intentionally not exported.",
            "",
        ],
    )
    path = output_dir / "figure_index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_thesis_review_index(output_dir: Path, generated_paths: Sequence[Path]) -> Path:
    """Write a compact review index for newly proposed companion figures."""
    lines = [
        "# Proposed Thesis Companion Figures",
        "",
        "These figures are review candidates. They do not replace the existing exported "
        "figures until inspected and accepted.",
        "",
        "## New Artifacts",
        "",
        "| Artifact | Role | Include only if... |",
        "| --- | --- | --- |",
        (
            "| `paper_claim_matrix.png` | Result matrix with the KuaiRec column "
            "restricted to explicit big-matrix watch-ratio rows. | You want one matrix "
            "that excludes KuaiRec small-matrix/full-observation sensitivity rows. |"
        ),
        (
            "| `paper_accuracy_efficiency_frontier.png` | "
            "Full-range NDCG-vs-time trade-off under the accepted comparison filter. | It "
            "adds a useful visual frontier beyond the component table. |"
        ),
        (
            "| `paper_accuracy_efficiency_frontier_zoomed.png` | "
            "Local zoom companion for the frontier. | The full-range figure is too "
            "compressed for slide or thesis-column readability. |"
        ),
        (
            "| `kuairec_matrix_regime_sensitivity.png` | Explains why dense KuaiRec "
            "sensitivity rows are not the big-matrix comparison. | Reviewers may confuse "
            "small-matrix/full-observation numbers with explicit big-matrix rows. |"
        ),
        "",
        "## Reader-Facing Rules",
        "",
        "- Captions should say `small-matrix sensitivity`, not `better KuaiRec run`.",
        "- The KuaiRec comparison should be described as explicit big-matrix watch-ratio "
        "evidence, not as a generic default row.",
        "- Do not show experiment IDs, database paths, or internal profile names in the "
        "final thesis figure text.",
        "- Runtime probes remain resource evidence only.",
        "",
        "Generated files:",
        "",
    ]
    for path in generated_paths:
        lines.append(f"- `{path.name}`")
    path = output_dir / "thesis_review_index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_thesis_review_figures(output_dir: Path, *, top_n: int) -> list[Path]:
    """Generate only new companion figures for review."""
    _style_plots()
    data = _load_defense_data(top_n)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_paper_claim_matrix(
            data,
            output_dir,
            reference_mode="kuairec_big_matrix",
            filename="paper_claim_matrix.png",
        ),
        plot_paper_accuracy_efficiency_frontier(
            data,
            output_dir,
            reference_mode="kuairec_big_matrix",
            filename="paper_accuracy_efficiency_frontier.png",
        ),
        plot_paper_accuracy_efficiency_frontier(
            data,
            output_dir,
            focused=True,
            reference_mode="kuairec_big_matrix",
            filename="paper_accuracy_efficiency_frontier_zoomed.png",
        ),
        plot_kuairec_matrix_regime_sensitivity(data, output_dir),
    ]
    paths.append(_write_thesis_review_index(output_dir, paths))
    return paths


def export_defense_figures(output_dir: Path, *, top_n: int) -> list[Path]:
    """Generate every defense figure and return output paths."""
    _style_plots()
    data = _load_defense_data(top_n)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_retired_defense_artifacts(output_dir)
    architecture_path = write_architecture_mermaid(output_dir)
    paths = [
        architecture_path,
        plot_candidate_taxonomy(output_dir),
        plot_dataset_regime_map(output_dir),
        plot_paper_claim_matrix(data, output_dir, reference_mode="kuairec_big_matrix"),
        plot_paper_accuracy_efficiency_frontier(
            data,
            output_dir,
            reference_mode="kuairec_big_matrix",
        ),
        plot_paper_accuracy_efficiency_frontier(
            data,
            output_dir,
            focused=True,
            reference_mode="kuairec_big_matrix",
        ),
        plot_paper_mechanism_diagnostics(data, output_dir),
        plot_kuairec_ablation_deltas(data, output_dir),
        plot_accuracy_popularity_tradeoff(data, output_dir),
        plot_accuracy_popularity_tradeoff(data, output_dir, focused=True),
        plot_crru_decomposition(data, output_dir),
        plot_paper_baseline_feasibility(data, output_dir),
    ]
    paths.append(_write_figure_index(output_dir, generated_paths=paths, data=data))
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Build the defense-figure CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFENSE_FIGURES_DIR,
        help="Directory for generated defense figures.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top rows per dataset to include in trade-off scatter plots.",
    )
    parser.add_argument(
        "--review-companions-only",
        action="store_true",
        help="Generate only new review companion figures without touching existing exports.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the defense-figure exporter."""
    args = build_parser().parse_args(argv)
    if args.review_companions_only:
        paths = export_thesis_review_figures(args.output_dir, top_n=args.top_n)
    else:
        paths = export_defense_figures(args.output_dir, top_n=args.top_n)
    print("Generated defense figures:")
    for path in paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
