#!/usr/bin/env python
"""Prune checkpoint history while keeping the best runs per dataset/model family."""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.utils.project_paths import CHECKPOINT_DIR, THESIS_DB_PATH

HASH_PATTERN = re.compile(r"_train-([0-9a-f]{16})\.pt$")
DEFAULT_FAMILIES = ("edgrec", "dice", "lightgcn")
FAMILY_ALIASES = {
    "edgrec": "edgrec",
    "dice_like": "dice",
    "dice_paper": "dice",
    "lgndice_paper": "dice",
    "lightgcn": "lightgcn",
    "lightgcn_paper": "lightgcn",
}
DELETE_FAILED_STATUSES = frozenset({"failed", "oom"})


@dataclass(frozen=True)
class ExperimentCheckpointRow:
    """One experiment row that can explain a checkpoint file."""

    exp_id: int
    dataset: str | None
    preset: str | None
    training_hash: str
    status: str | None
    batch_id: str | None
    profile_name: str | None
    score: float | None
    score_split: str | None
    updated_at: str | None


@dataclass(frozen=True)
class CheckpointCandidate:
    """One checkpoint file plus its SQLite-backed retention metadata."""

    path: Path
    size_bytes: int
    training_hash: str
    dataset: str
    family: str
    preset: str | None
    score: float | None
    score_split: str | None
    statuses: tuple[str, ...]
    search_only: bool
    updated_at: str | None


@dataclass(frozen=True)
class PruneDecision:
    """A keep/delete decision for one checkpoint file."""

    action: str
    path: Path
    size_bytes: int
    reason: str
    dataset: str | None = None
    family: str | None = None
    preset: str | None = None
    score: float | None = None
    score_split: str | None = None
    training_hash: str | None = None


@dataclass(frozen=True)
class PrunePlan:
    """Complete checkpoint pruning plan."""

    keep: tuple[PruneDecision, ...]
    delete: tuple[PruneDecision, ...]


def _parse_families(raw: str) -> tuple[str, ...]:
    families = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not families:
        raise ValueError("--families must resolve at least one family.")
    unsupported = sorted(set(families) - set(DEFAULT_FAMILIES))
    if unsupported:
        raise ValueError(f"Unsupported retention families: {', '.join(unsupported)}.")
    return families


def _normalize_family(preset: str | None) -> str | None:
    """Map historical and paper presets to the user's requested model families."""
    if preset is None:
        return None
    normalized = str(preset).lower()
    if normalized in FAMILY_ALIASES:
        return FAMILY_ALIASES[normalized]
    if normalized.startswith("lightgcn"):
        return "lightgcn"
    if "dice" in normalized:
        return "dice"
    return None


def _is_search_row(row: ExperimentCheckpointRow) -> bool:
    batch_id = row.batch_id or ""
    profile_name = row.profile_name or ""
    return batch_id.startswith("optuna-") or profile_name.startswith("edgrec-")


def _file_training_hash(path: Path) -> str | None:
    match = HASH_PATTERN.search(path.name)
    return match.group(1) if match is not None else None


def _finite_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score


def _load_rows_by_hash(
    db_path: Path,
    *,
    metric: str,
    split: str,
    fallback_split: str,
) -> dict[str, list[ExperimentCheckpointRow]]:
    """Load experiment rows keyed by training hash."""
    if not db_path.exists():
        return {}
    rows_by_hash: dict[str, list[ExperimentCheckpointRow]] = defaultdict(list)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT e.id, e.dataset, e.preset, e.training_hash, e.status,
                   e.batch_id, e.profile_name, e.updated_at,
                   MAX(CASE
                       WHEN m.metric_name = ? AND m.split = ?
                       THEN m.metric_value
                   END) AS primary_score,
                   MAX(CASE
                       WHEN m.metric_name = ? AND m.split = ?
                       THEN m.metric_value
                   END) AS fallback_score
            FROM experiments e
            LEFT JOIN metrics m ON m.experiment_id = e.id
            WHERE e.training_hash IS NOT NULL
              AND e.training_hash <> ''
            GROUP BY e.id
            """,
            (metric, split, metric, fallback_split),
        ).fetchall()

    for row in rows:
        primary_score = _finite_score(row["primary_score"])
        fallback_score = _finite_score(row["fallback_score"])
        score = primary_score if primary_score is not None else fallback_score
        score_split = split if primary_score is not None else fallback_split
        if score is None:
            score_split = None
        training_hash = str(row["training_hash"])
        rows_by_hash[training_hash].append(
            ExperimentCheckpointRow(
                exp_id=int(row["id"]),
                dataset=row["dataset"],
                preset=row["preset"],
                training_hash=training_hash,
                status=row["status"],
                batch_id=row["batch_id"],
                profile_name=row["profile_name"],
                score=score,
                score_split=score_split,
                updated_at=row["updated_at"],
            ),
        )
    return dict(rows_by_hash)


def _best_row(rows: list[ExperimentCheckpointRow]) -> ExperimentCheckpointRow | None:
    """Pick the row that should describe a checkpoint for reporting/ranking."""
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            row.status == "completed",
            row.score is not None,
            row.score if row.score is not None else float("-inf"),
            row.updated_at or "",
            row.exp_id,
        ),
        reverse=True,
    )[0]


def _decision_from_candidate(
    candidate: CheckpointCandidate,
    *,
    action: str,
    reason: str,
) -> PruneDecision:
    return PruneDecision(
        action=action,
        path=candidate.path,
        size_bytes=candidate.size_bytes,
        reason=reason,
        dataset=candidate.dataset,
        family=candidate.family,
        preset=candidate.preset,
        score=candidate.score,
        score_split=candidate.score_split,
        training_hash=candidate.training_hash,
    )


def _candidate_sort_key(candidate: CheckpointCandidate) -> tuple[object, ...]:
    return (
        candidate.score is not None,
        candidate.score if candidate.score is not None else float("-inf"),
        candidate.updated_at or "",
        candidate.training_hash,
    )


def build_prune_plan(
    *,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    db_path: Path = THESIS_DB_PATH,
    keep: int = 3,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    metric: str = "NDCG@40",
    split: str = "val",
    fallback_split: str = "test",
    delete_search: bool = True,
    delete_failed: bool = True,
    delete_running: bool = False,
    delete_unmatched: bool = False,
) -> PrunePlan:
    """Build a dry-run friendly checkpoint pruning plan."""
    if keep < 1:
        raise ValueError("--keep must be >= 1.")
    checkpoint_dir = checkpoint_dir.resolve()
    rows_by_hash = _load_rows_by_hash(
        db_path,
        metric=metric,
        split=split,
        fallback_split=fallback_split,
    )

    keep_decisions: list[PruneDecision] = []
    delete_decisions: list[PruneDecision] = []
    candidates_by_group: dict[tuple[str, str], list[CheckpointCandidate]] = defaultdict(list)

    for path in sorted(checkpoint_dir.glob("*.pt")):
        resolved_path = path.resolve()
        size_bytes = resolved_path.stat().st_size
        training_hash = _file_training_hash(resolved_path)
        if training_hash is None or training_hash not in rows_by_hash:
            decision = PruneDecision(
                action="delete" if delete_unmatched else "keep",
                path=resolved_path,
                size_bytes=size_bytes,
                reason="unmatched checkpoint without SQLite training_hash metadata",
                training_hash=training_hash,
            )
            if delete_unmatched:
                delete_decisions.append(decision)
            else:
                keep_decisions.append(decision)
            continue

        rows = rows_by_hash[training_hash]
        best_row = _best_row(rows)
        if best_row is None or best_row.dataset is None:
            decision = PruneDecision(
                action="delete" if delete_unmatched else "keep",
                path=resolved_path,
                size_bytes=size_bytes,
                reason="checkpoint has incomplete SQLite metadata",
                training_hash=training_hash,
            )
            if delete_unmatched:
                delete_decisions.append(decision)
            else:
                keep_decisions.append(decision)
            continue

        family = _normalize_family(best_row.preset)
        if family is None or family not in families:
            keep_decisions.append(
                PruneDecision(
                    action="keep",
                    path=resolved_path,
                    size_bytes=size_bytes,
                    reason="outside selected retention families",
                    dataset=best_row.dataset,
                    family=family,
                    preset=best_row.preset,
                    score=best_row.score,
                    score_split=best_row.score_split,
                    training_hash=training_hash,
                ),
            )
            continue

        statuses = tuple(sorted({str(row.status or "unknown") for row in rows}))
        candidate = CheckpointCandidate(
            path=resolved_path,
            size_bytes=size_bytes,
            training_hash=training_hash,
            dataset=str(best_row.dataset),
            family=family,
            preset=best_row.preset,
            score=best_row.score,
            score_split=best_row.score_split,
            statuses=statuses,
            search_only=all(_is_search_row(row) for row in rows),
            updated_at=best_row.updated_at,
        )
        if delete_search and candidate.search_only:
            delete_decisions.append(
                _decision_from_candidate(
                    candidate,
                    action="delete",
                    reason="search-only checkpoint; Optuna results are kept in SQLite",
                ),
            )
            continue
        if "running" in candidate.statuses and not delete_running:
            keep_decisions.append(
                _decision_from_candidate(
                    candidate,
                    action="keep",
                    reason="running experiment checkpoint kept by default",
                ),
            )
            continue
        if delete_failed and set(candidate.statuses) <= DELETE_FAILED_STATUSES:
            delete_decisions.append(
                _decision_from_candidate(
                    candidate,
                    action="delete",
                    reason="failed/OOM checkpoint outside completed-run retention",
                ),
            )
            continue
        candidates_by_group[(candidate.dataset, candidate.family)].append(candidate)

    for (dataset, family), group_candidates in sorted(candidates_by_group.items()):
        sorted_candidates = sorted(group_candidates, key=_candidate_sort_key, reverse=True)
        for index, candidate in enumerate(sorted_candidates):
            if index < keep:
                keep_decisions.append(
                    _decision_from_candidate(
                        candidate,
                        action="keep",
                        reason=f"top {keep} {family} checkpoint for {dataset}",
                    ),
                )
            else:
                delete_decisions.append(
                    _decision_from_candidate(
                        candidate,
                        action="delete",
                        reason=f"outside top {keep} {family} checkpoints for {dataset}",
                    ),
                )

    return PrunePlan(
        keep=tuple(sorted(keep_decisions, key=lambda item: str(item.path))),
        delete=tuple(sorted(delete_decisions, key=lambda item: str(item.path))),
    )


def _format_size(size_bytes: int) -> str:
    gib = size_bytes / 1024**3
    if gib >= 1:
        return f"{gib:.2f} GiB"
    mib = size_bytes / 1024**2
    return f"{mib:.1f} MiB"


def _decision_line(decision: PruneDecision) -> str:
    score = ""
    if decision.score is not None and decision.score_split is not None:
        score = f" | {decision.score_split}={decision.score:.6f}"
    group = ""
    if decision.dataset is not None or decision.family is not None:
        group = f" | {decision.dataset or '?'} / {decision.family or '?'}"
    return (
        f"- {_format_size(decision.size_bytes):>10} | {decision.path.name}"
        f"{group}{score} | {decision.reason}"
    )


def print_plan(plan: PrunePlan, *, execute: bool, limit: int) -> None:
    """Print a compact pruning plan."""
    delete_size = sum(decision.size_bytes for decision in plan.delete)
    keep_size = sum(decision.size_bytes for decision in plan.keep)
    print("=" * 72)
    print("CHECKPOINT RETENTION PLAN")
    print("=" * 72)
    print(f"Mode: {'execute' if execute else 'dry-run'}")
    print(f"Keep:   {len(plan.keep)} files / {_format_size(keep_size)}")
    print(f"Delete: {len(plan.delete)} files / {_format_size(delete_size)}")
    if plan.delete:
        print("\nFiles selected for deletion:")
        for decision in plan.delete[:limit]:
            print(_decision_line(decision))
        remaining = len(plan.delete) - limit
        if remaining > 0:
            print(f"... {remaining} more deletion candidates omitted")
    print("=" * 72)


def execute_plan(plan: PrunePlan) -> int:
    """Delete files selected by a plan and return reclaimed bytes."""
    deleted_bytes = 0
    for decision in plan.delete:
        if decision.path.exists():
            deleted_bytes += decision.path.stat().st_size
            decision.path.unlink()
    return deleted_bytes


def build_parser() -> argparse.ArgumentParser:
    """Build the checkpoint pruning CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Prune results/checkpoints while keeping the top N checkpoints per "
            "dataset and model family."
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--db-path", type=Path, default=THESIS_DB_PATH)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated retention families: edgrec,dice,lightgcn.",
    )
    parser.add_argument("--metric", default="NDCG@40")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument(
        "--fallback-split",
        default=None,
        choices=("val", "test"),
        help="Fallback metric split. Defaults to the opposite of --split.",
    )
    parser.add_argument(
        "--keep-search",
        dest="delete_search",
        action="store_false",
        help="Keep existing Optuna-search checkpoints instead of deleting them.",
    )
    parser.add_argument(
        "--keep-failed",
        dest="delete_failed",
        action="store_false",
        help="Keep failed/OOM checkpoints instead of deleting them.",
    )
    parser.add_argument(
        "--delete-running",
        action="store_true",
        help="Allow pruning checkpoints associated with running experiment rows.",
    )
    parser.add_argument(
        "--delete-unmatched",
        action="store_true",
        help="Delete checkpoint files that cannot be mapped to SQLite metadata.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete the selected files. Omit for a dry-run plan.",
    )
    parser.add_argument("--limit", type=int, default=80, help="Max deletion lines to print.")
    parser.set_defaults(delete_search=True, delete_failed=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    fallback_split = args.fallback_split or ("test" if args.split == "val" else "val")
    plan = build_prune_plan(
        checkpoint_dir=args.checkpoint_dir,
        db_path=args.db_path,
        keep=args.keep,
        families=_parse_families(args.families),
        metric=args.metric,
        split=args.split,
        fallback_split=fallback_split,
        delete_search=bool(args.delete_search),
        delete_failed=bool(args.delete_failed),
        delete_running=bool(args.delete_running),
        delete_unmatched=bool(args.delete_unmatched),
    )
    print_plan(plan, execute=bool(args.execute), limit=max(0, int(args.limit)))
    if args.execute:
        deleted_bytes = execute_plan(plan)
        print(f"Deleted {len(plan.delete)} files; reclaimed {_format_size(deleted_bytes)}.")
    else:
        print("Dry-run only. Re-run with --execute to delete selected checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
