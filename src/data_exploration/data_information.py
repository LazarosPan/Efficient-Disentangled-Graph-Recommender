"""Generate a robust dataset inventory report for thesis datasets.

This script scans ``data/<dataset>/`` recursively and writes a comprehensive
report to ``data/datasets_information.md``. It is designed to work even when
datasets are inconsistent, partially downloaded, or stored in mixed formats.

The purpose of the report is purely informational for the thesis pipeline - it
has no influence on the training code itself.

Report includes:
- Dataset-level size totals (bytes, MiB, GiB)
- Per-dataset file inventory across the selected datasets
- Per-file column names when a tabular schema can be read
- Per-file column dtypes when a tabular schema can be read
- Per-file size in MiB
- Optional machine-readable JSON export for feature-policy automation
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def load_exploration_api() -> tuple[Any, Any, Any]:
    """Load exploration helpers lazily so this script still runs directly."""
    module = importlib.import_module("src.data_exploration.data_exploration")
    return module.inspect_dataset_file, module.schema_overview, module.summarize_dataset


def load_cli_parser_api() -> Any:
    """Load the shared CLI parser after the repo-root path bootstrap."""
    module = importlib.import_module("src.utils.cli_parsers")
    return module.build_data_information_parser


def load_feature_policy_api() -> tuple[Any, Any, Any, Any]:
    """Load feature-policy helpers lazily after the repo-root path bootstrap."""
    module = importlib.import_module("src.data.feature_policy")
    return (
        module.enabled_feature_sources,
        module.feature_role_for_column,
        module.normalize_dataset_name,
        module.thesis_default_columns,
    )


INSPECT_DATASET_FILE, SCHEMA_OVERVIEW, SUMMARIZE_DATASET = load_exploration_api()
BUILD_DATA_INFORMATION_PARSER = load_cli_parser_api()
(
    ENABLED_FEATURE_SOURCES,
    FEATURE_ROLE_FOR_COLUMN,
    NORMALIZE_DATASET_NAME,
    THESIS_DEFAULT_COLUMNS,
) = load_feature_policy_api()


TEXT_EXTENSIONS = {
    ".txt",
    ".csv",
    ".tsv",
    ".dat",
    ".data",
    ".json",
    ".md",
    ".info",
    ".base",
    ".test",
    ".user",
    ".item",
    ".genre",
    ".occupation",
    ".pl",
    ".sh",
    "",
}

USER_COLUMN_MARKERS = {"user", "user_id", "userid", "uid"}
ITEM_COLUMN_MARKERS = {
    "item",
    "item_id",
    "itemid",
    "iid",
    "movieid",
    "movie_id",
    "video_id",
    "videoid",
}
INTERACTION_SIGNAL_MARKERS = {
    "rating",
    "watch_ratio",
    "is_click",
    "click",
    "behavior_type",
    "label",
    "play_duration",
    "duration_ms",
}
COMMON_INTERACTION_COLUMNS = {
    "timestamp",
    "time",
    "time_ms",
    "date",
    "ts",
    "hourmin",
    "rating",
    "watch_ratio",
    "is_click",
    "click",
    "label",
    "behavior_type",
    "play_duration",
    "playing_time",
    "duration_ms",
}
USER_FEATURE_HINTS = {
    "gender",
    "age",
    "occupation",
    "zip_code",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
    "onehot_feat",
    "search_active_level",
    "rec_active_level",
}
ITEM_FEATURE_HINTS = {
    "genre",
    "genres",
    "category",
    "category_id",
    "category_name",
    "caption",
    "topic_tag",
    "author_id",
    "music_id",
    "music_type",
    "video_type",
    "video_duration",
    "tag",
    "relevance",
    "show_cnt",
    "play_cnt",
    "play_user_num",
    "like_cnt",
    "comment_cnt",
    "follow_cnt",
    "share_cnt",
    "download_cnt",
    "collect_cnt",
    "reduce_similar_cnt",
}
POST_TREATMENT_MARKERS = {
    "show_cnt",
    "show_user_num",
    "play_cnt",
    "play_user_num",
    "play_duration",
    "complete_play_cnt",
    "valid_play_cnt",
    "long_time_play_cnt",
    "short_time_play_cnt",
    "play_progress",
    "like_cnt",
    "comment_cnt",
    "follow_cnt",
    "share_cnt",
    "download_cnt",
    "collect_cnt",
    "search",
    "click_cnt",
}
RANDOMIZATION_MARKERS = {"is_rand"}
SOCIAL_MARKERS = {"friend_list", "user_follow_id", "social_network"}
SEARCH_MARKERS = {
    "search",
    "search_session_id",
    "search_session_source",
    "keyword",
    "search_photo_related",
}
TEXT_MARKERS = {"title", "caption", "manual_cover_text", "tag", "topic_tag", "keyword"}
NON_CAUSAL_MARKERS = {
    "imdbid",
    "tmdbid",
    "role",
    "class_map",
    "format",
    "shape",
    "indices",
    "indptr",
    "data",
}
PRETREATMENT_CONTEXT_MARKERS = {
    "timestamp",
    "time",
    "time_ms",
    "date",
    "hourmin",
    "tab",
    "upload_dt",
    "upload_time",
    "upload_type",
    "visible_status",
}

LOADER_FACTS: dict[str, dict[str, Any]] = {
    "movielens1m": {
        "loader_name": "movielens1m",
        "metadata": [],
    },
    "movielens20m": {
        "loader_name": "movielens20m",
        "metadata": [],
    },
    "taobao": {
        "loader_name": "taobao",
        "metadata": [],
    },
    "kuairecv2": {
        "loader_name": "kuairec_v2",
        "metadata": [],
    },
    "kuairand1k": {
        "loader_name": "kuairand1k",
        "metadata": ["is_rand"],
    },
    "amazonbook": {
        "loader_name": "amazonbook",
        "metadata": [],
    },
}
ALLOWED_DATASETS = (
    "AmazonBook",
    "KuaiRand-1K",
    "KuaiRec_v2",
    "MovieLens20M",
    "Taobao",
    "MovieLens1M",
)
# Limits used to avoid expensive scanning or exports.  Set to `None`
# to disable the check entirely (we keep full datasets by default).
# Originally I added these when the CSV reader was slow and some files
# were multi-GB, but the user reminded us that they want *all* rows kept.
MAX_TEXT_SCAN_BYTES: int | None = None
MAX_JSON_PARSE_BYTES: int | None = None
MAX_FEATURE_EXPORT_BYTES: int | None = None
# MAX_TAOBAO_ROWS_FOR_GRAPH = None  # not enforced; we no longer truncate


@dataclass
class FileRecord:
    """Store per-file metrics.

    Args:
        relative_path: Path relative to dataset root.
        extension: Lowercase file extension.
        size_bytes: File size in bytes.
        line_count: Optional physical line count.
        parsed_count: Optional parsed semantic count.
        details: Additional parser metadata.

    """

    relative_path: str
    extension: str
    size_bytes: int
    line_count: int | None = None
    parsed_count: int | None = None
    details: dict[str, Any] | None = None
    schema: dict[str, Any] | None = None


@dataclass
class DatasetRecord:
    """Store aggregate metrics for a dataset folder.

    Args:
        name: Dataset folder name.
        root_path: Absolute dataset path.
        file_records: List of scanned files.
        has_raw_content: Whether any files exist under a raw folder.
        has_processed_data: Whether any files exist under processed/.
        processed_paths: Located processed/data.pt files.
        status: Derived status string.
        notes: Warnings/errors encountered while scanning.
        graph_info: Optional graph metadata extracted from processed files.

    """

    name: str
    root_path: Path
    file_records: list[FileRecord]
    has_raw_content: bool
    has_processed_data: bool
    processed_paths: list[str]
    status: str
    notes: list[str]
    graph_info: dict[str, Any] | None = None
    exploration_summary: dict[str, Any] | None = None

    @property
    def total_bytes(self) -> int:
        """Return total dataset size across all files."""
        return sum(record.size_bytes for record in self.file_records)


TextCountParser = Callable[[Path, int | None], tuple[int | None, dict[str, Any]]]


def safe_json(value: Any) -> Any:
    """Convert values into JSON-serializable primitives.

    Args:
        value: Any Python object.

    Returns:
        JSON-safe representation.

    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(item) for item in value]
    return str(value)


def format_bytes(size_bytes: int) -> str:
    """Format bytes in binary units.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string.

    """
    mib = size_bytes / (1024**2)
    gib = size_bytes / (1024**3)
    return f"{size_bytes:,} B ({mib:,.2f} MiB | {gib:,.2f} GiB)"


def safe_count_lines(file_path: Path) -> int | None:
    """Count lines for text-like files using a fast backend.

    This no longer iterates in Python; instead we try one of the following in
    order:
      1.  ``polars`` reader (reads csv-like files on the GPU/threads)
      2.  ``wc -l`` system call (portable and very efficient)

    If both fail, we fall back to the old Python loop as a last resort.  The
    previous max-size limit has been removed; the caller is responsible for
    skipping huge files if desired.
    """
    # use the lightweight `wc -l` command to count lines; no Python loop,
    # no external library needed, so memory usage stays minimal.  this works for
    # *any* text file and is the fastest option on Linux.
    try:
        import subprocess

        output = subprocess.check_output(["wc", "-l", str(file_path)])
        return int(output.strip().split()[0])
    except Exception:
        pass

    # final safety net: Python iteration (very slow, should not be hit)
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def parse_csv_counts(file_path: Path) -> tuple[int | None, dict[str, Any]]:
    """Compute parsed row counts for CSV files.

    Args:
        file_path: CSV file path.

    Returns:
        Tuple of parsed count and parser details.

    """
    details: dict[str, Any] = {}

    # first pass: sniff header and column count using csv module (cheap sampling)
    has_header = False
    first_row: list[str] | None = None
    try:
        with file_path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
            newline="",
        ) as handle:
            sample = handle.read(4096)
            handle.seek(0)
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                first_row = row
                break
    except Exception:
        pass

    details["has_header"] = has_header
    details["columns_detected"] = len(first_row) if first_row else 0

    # row count via wc; avoids loading entire CSV into memory
    try:
        import subprocess

        total_rows = int(
            subprocess.check_output(["wc", "-l", str(file_path)]).strip().split()[0],
        )
        parsed = total_rows - 1 if has_header and total_rows > 0 else total_rows
        return parsed, details
    except Exception:
        return None, details


def parse_dat_counts(
    file_path: Path,
    line_count: int | None,
) -> tuple[int | None, dict[str, Any]]:
    """Compute parsed record counts for ``.dat`` files.

    Args:
        file_path: DAT file path.
        line_count: Existing line count if already computed.

    Returns:
        Tuple of parsed count and parser details.

    """
    details: dict[str, Any] = {}
    if line_count is None:
        line_count = safe_count_lines(file_path)

    if line_count is None:
        return None, details

    details["delimiter_guess"] = "::"
    return line_count, details


def parse_amazonbook_interactions(file_path: Path) -> tuple[int | None, dict[str, Any]]:
    """Parse AmazonBook train/test interaction counts from txt files.

    This used to iterate row-by-row, which is painfully slow on large files.
    We now delegate the heavy lifting to shell utilities (`wc` + `awk`) which
    run in C and complete in seconds even for gigabyte tables.

    Assumes whitespace-separated tokens per line.
    """
    details: dict[str, Any] = {}

    try:
        import subprocess

        # count rows
        users = int(
            subprocess.check_output(["wc", "-l", str(file_path)]).strip().split()[0],
        )
        # sum(i=1..N) (NF - 1)
        awkcmd = [
            "awk",
            "{sum+=NF-1} END{print sum}",
            str(file_path),
        ]
        interactions = int(subprocess.check_output(awkcmd).strip())
        details["rows_with_user"] = users
        return interactions, details
    except Exception:
        return None, details


def parse_csv_text_count(
    file_path: Path,
    _line_count: int | None,
) -> tuple[int | None, dict[str, Any]]:
    """Adapt CSV count parsing to the shared text parser signature.

    Args:
        file_path: CSV file path.
        _line_count: Precomputed physical line count, unused for CSV parsing.

    Returns:
        Tuple of parsed count and parser details.

    """
    return parse_csv_counts(file_path)


def parse_amazonbook_text_count(
    file_path: Path,
    _line_count: int | None,
) -> tuple[int | None, dict[str, Any]]:
    """Adapt AmazonBook interaction parsing to the shared text parser signature.

    Args:
        file_path: AmazonBook train/test file path.
        _line_count: Precomputed physical line count, unused for interaction parsing.

    Returns:
        Tuple of parsed count and parser details.

    """
    return parse_amazonbook_interactions(file_path)


TEXT_COUNT_PARSERS: dict[str, TextCountParser] = {
    ".csv": parse_csv_text_count,
    ".dat": parse_dat_counts,
}


def text_count_parser_for(
    dataset_root: Path,
    file_path: Path,
    extension: str,
) -> TextCountParser | None:
    """Return the semantic count parser for one text-like dataset file.

    Args:
        dataset_root: Dataset root directory.
        file_path: Absolute file path.
        extension: Lowercase file extension.

    Returns:
        Parser callable when the file needs semantic parsing, otherwise ``None``.

    """
    parser = TEXT_COUNT_PARSERS.get(extension)
    if parser is not None:
        return parser

    if file_path.name in {"train.txt", "test.txt"} and "AmazonBook" in str(dataset_root):
        return parse_amazonbook_text_count

    return None


def try_numpy_metadata(file_path: Path) -> dict[str, Any]:
    """Extract lightweight metadata from NumPy files.

    Args:
        file_path: ``.npy`` or ``.npz`` path.

    Returns:
        Metadata dictionary (possibly empty).

    """
    try:
        import numpy as np
    except Exception:
        return {}

    try:
        if file_path.suffix.lower() == ".npy":
            array = np.load(file_path, mmap_mode="r")
            return {
                "numpy_shape": tuple(array.shape),
                "numpy_dtype": str(array.dtype),
            }
        if file_path.suffix.lower() == ".npz":
            with np.load(file_path, allow_pickle=False) as arrays:
                keys = list(arrays.keys())
                return {
                    "npz_arrays": len(keys),
                    "npz_keys_preview": keys[:10],
                }
    except Exception:
        return {}

    return {}


def try_json_metadata(file_path: Path) -> dict[str, Any]:
    """Extract lightweight metadata from small JSON files.

    Args:
        file_path: JSON path.

    Returns:
        Metadata dictionary (possibly empty).

    """
    if MAX_JSON_PARSE_BYTES is not None and file_path.stat().st_size > MAX_JSON_PARSE_BYTES:
        return {"json_parse": "skipped_large_file"}

    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            obj = json.load(handle)
        if isinstance(obj, dict):
            return {
                "json_type": "dict",
                "json_keys": len(obj.keys()),
                "json_keys_preview": list(obj.keys())[:10],
            }
        if isinstance(obj, list):
            return {
                "json_type": "list",
                "json_length": len(obj),
            }
        return {"json_type": type(obj).__name__}
    except Exception:
        return {"json_parse": "failed"}


def build_file_record(dataset_root: Path, file_path: Path) -> FileRecord:
    """Build metrics for one file.

    Args:
        dataset_root: Dataset root directory.
        file_path: Absolute file path.

    Returns:
        Populated ``FileRecord``.

    """
    extension = file_path.suffix.lower()
    relative_path = str(file_path.relative_to(dataset_root))
    size_bytes = file_path.stat().st_size

    line_count: int | None = None
    parsed_count: int | None = None
    details: dict[str, Any] = {}
    schema: dict[str, Any] | None = None

    if extension in TEXT_EXTENSIONS:
        line_count = safe_count_lines(file_path)
        parser = text_count_parser_for(dataset_root, file_path, extension)

        if parser is not None:
            parsed_count, parsed_details = parser(file_path, line_count)
            details.update(parsed_details)
        elif line_count is not None:
            parsed_count = line_count

    if extension in {".npy", ".npz"}:
        details.update(try_numpy_metadata(file_path))
    if extension == ".json":
        details.update(try_json_metadata(file_path))

    try:
        inspection = INSPECT_DATASET_FILE(dataset_root.name, file_path)
        schema = SCHEMA_OVERVIEW(inspection)
        if schema["delimiter"] is not None and "delimiter_guess" not in details:
            details["delimiter_guess"] = schema["delimiter"]
    except Exception as exc:
        schema = {"schema_error": str(exc)}

    return FileRecord(
        relative_path=relative_path,
        extension=extension,
        size_bytes=size_bytes,
        line_count=line_count,
        parsed_count=parsed_count,
        details=details or None,
        schema=schema,
    )


def attach_exploration_summary(record: DatasetRecord) -> DatasetRecord:
    """Attach dataset-level exploration metadata when a registry entry exists."""
    try:
        record.exploration_summary = SUMMARIZE_DATASET(record.name)
    except KeyError:
        record.exploration_summary = None
    except Exception as exc:
        record.notes.append(f"exploration_summary_failed:{exc}")
        record.exploration_summary = None
    return record


def normalize_dataset_key(name: str) -> str:
    """Normalize dataset names so report folders map to loader registry entries."""
    return NORMALIZE_DATASET_NAME(name)


def summarize_registered_paths(paths: tuple[str, ...], limit: int = 3) -> str:
    """Render a short summary for registered feature-source paths."""
    if not paths:
        return "-"
    unique_paths = tuple(sorted(set(paths)))
    if len(unique_paths) <= limit:
        return ", ".join(unique_paths)
    return ", ".join(unique_paths[:limit]) + f" (+{len(unique_paths) - limit} more)"


def describe_feature_support(dataset_name: str, aspect: str) -> str | None:
    """Describe which registered feature sources are active by policy.

    Args:
        dataset_name: Loader registry dataset name.
        aspect: Feature aspect such as ``user_features`` or ``item_features``.

    Returns:
        Human-readable coverage text for the aspect or ``None`` when the dataset
        has no registered sources for it.

    """
    thesis_default_sources = ENABLED_FEATURE_SOURCES(
        "thesis_default",
        dataset_name,
        aspect,
    )
    all_optional_sources = ENABLED_FEATURE_SOURCES(
        "all_optional",
        dataset_name,
        aspect,
    )
    optional_only_sources = tuple(
        source for source in all_optional_sources if source not in thesis_default_sources
    )
    aspect_label = f"canonical {aspect}"
    if thesis_default_sources:
        support = (
            "thesis-default "
            f"{aspect_label} from {summarize_registered_paths(thesis_default_sources)}"
        )
        if optional_only_sources:
            support += "; additional all_optional-only sources: " + summarize_registered_paths(
                optional_only_sources
            )
        return support
    if all_optional_sources:
        return (
            f"available only under all_optional in {aspect_label} from "
            f"{summarize_registered_paths(all_optional_sources)}"
        )
    return None


def get_loader_coverage(record: DatasetRecord) -> dict[str, Any]:
    """Return current loader-coverage facts for a dataset."""
    facts = LOADER_FACTS.get(normalize_dataset_key(record.name))
    if facts is not None:
        return {
            "loader_name": facts["loader_name"],
            "interactions": "native loader",
            "user_features": describe_feature_support(record.name, "user_features"),
            "item_features": describe_feature_support(record.name, "item_features"),
            "metadata": facts.get("metadata", []),
        }
    return {
        "loader_name": None,
        "interactions": None,
        "user_features": None,
        "item_features": None,
        "metadata": [],
    }


def schema_columns(file_record: FileRecord) -> list[str]:
    """Return normalized schema columns for a scanned file."""
    schema = file_record.schema or {}
    columns = schema.get("columns") or []
    return [str(column).strip().lower() for column in columns]


def has_exact_marker(values: list[str] | set[str], markers: set[str]) -> bool:
    """Check whether any normalized value matches one of the markers exactly."""
    return any(value in markers for value in values)


def has_substring_marker(values: list[str] | set[str], markers: set[str]) -> bool:
    """Check whether any marker appears as a substring in the provided values."""
    return any(any(marker in value for marker in markers) for value in values)


def relevant_feature_columns(columns: list[str]) -> list[str]:
    """Filter out core interaction identifiers to expose side-feature candidates."""
    feature_columns: list[str] = []
    for column in columns:
        if column in COMMON_INTERACTION_COLUMNS:
            continue
        if column in USER_COLUMN_MARKERS or column in ITEM_COLUMN_MARKERS:
            continue
        if column.endswith("id") and column not in {
            "music_id",
            "author_id",
            "category_id",
            "tagid",
            "tag_id",
        }:
            continue
        feature_columns.append(column)
    return feature_columns


def candidate_feature_columns(columns: list[str]) -> list[str]:
    """Return auditable candidate columns, excluding only pure entity identifiers."""
    candidates: list[str] = []
    for column in columns:
        if column in USER_COLUMN_MARKERS or column in ITEM_COLUMN_MARKERS:
            continue
        candidates.append(column)
    return candidates


def is_interaction_source(file_record: FileRecord) -> bool:
    """Identify files that can serve as interaction sources."""
    relative_path = file_record.relative_path.lower()
    file_name = Path(relative_path).name
    columns = schema_columns(file_record)

    if file_name in {
        "train.txt",
        "valid.txt",
        "test.txt",
        "ratings.dat",
        "ratings.csv",
        "userbehavior.csv",
        "big_matrix.csv",
        "small_matrix.csv",
        "record.csv",
        "train_record.csv",
        "val_record.csv",
        "test_record.csv",
    }:
        return True

    if "user_features" in relative_path or "social_network" in relative_path:
        return False

    has_user = has_exact_marker(columns, USER_COLUMN_MARKERS)
    has_item = has_exact_marker(columns, ITEM_COLUMN_MARKERS)
    has_signal = has_exact_marker(columns, INTERACTION_SIGNAL_MARKERS)
    return has_user and has_item and (has_signal or file_record.parsed_count is not None)


def is_user_feature_source(file_record: FileRecord) -> bool:
    """Identify user-side covariate files."""
    relative_path = file_record.relative_path.lower()
    if any(token in relative_path for token in ["user_features", "users.dat", "user.txt"]):
        return True

    columns = schema_columns(file_record)
    feature_columns = relevant_feature_columns(columns)
    return has_exact_marker(columns, USER_COLUMN_MARKERS) and has_substring_marker(
        feature_columns,
        USER_FEATURE_HINTS,
    )


def is_item_feature_source(file_record: FileRecord) -> bool:
    """Identify item-side covariate files.

    This also treats item descriptors embedded in interaction tables as
    item-feature sources for the audit.

    """
    relative_path = file_record.relative_path.lower()
    columns = schema_columns(file_record)
    feature_columns = relevant_feature_columns(columns)

    path_hint = any(
        token in relative_path
        for token in [
            "movies.dat",
            "movies.csv",
            "video_features",
            "genome",
            "caption",
            "category",
            "categories",
            "item_daily_features",
            "item_categories",
        ]
    )
    column_hint = has_substring_marker(columns + feature_columns, ITEM_FEATURE_HINTS)
    item_id_hint = has_exact_marker(columns, ITEM_COLUMN_MARKERS)
    return (item_id_hint and bool(feature_columns) and column_hint) or path_hint or column_hint


def is_metadata_source(file_record: FileRecord) -> bool:
    """Identify causal metadata or context files that are not direct model features today."""
    relative_path = file_record.relative_path.lower()
    columns = schema_columns(file_record)

    if has_exact_marker(
        columns,
        RANDOMIZATION_MARKERS | SOCIAL_MARKERS,
    ) or has_substring_marker(columns, SEARCH_MARKERS):
        return True
    return bool(
        any(
            token in relative_path
            for token in [
                "social_network",
                "links.csv",
                "tags.csv",
                "popularity",
                "role.json",
                "class_map.json",
            ]
        )
    )


def collect_aspect_sources(record: DatasetRecord) -> dict[str, list[FileRecord]]:
    """Collect files relevant to each causal-audit aspect."""
    aspects = {
        "interactions": [],
        "user_features": [],
        "item_features": [],
        "metadata": [],
    }
    for file_record in record.file_records:
        if is_interaction_source(file_record):
            aspects["interactions"].append(file_record)
        if is_user_feature_source(file_record):
            aspects["user_features"].append(file_record)
        if is_item_feature_source(file_record):
            aspects["item_features"].append(file_record)
        if is_metadata_source(file_record):
            aspects["metadata"].append(file_record)
    return aspects


def summarize_source_paths(files: list[FileRecord], limit: int = 3) -> str:
    """Render a short source-file summary for markdown tables."""
    if not files:
        return "-"
    paths = sorted({file_record.relative_path for file_record in files})
    if len(paths) <= limit:
        return ", ".join(paths)
    return ", ".join(paths[:limit]) + f" (+{len(paths) - limit} more)"


def summarize_columns(files: list[FileRecord]) -> set[str]:
    """Union column names across a set of scanned files."""
    columns: set[str] = set()
    for file_record in files:
        columns.update(schema_columns(file_record))
    return columns


def loader_support_text(
    record: DatasetRecord,
    aspect: str,
    files: list[FileRecord],
) -> str:
    """Describe whether the current loader path uses a given aspect."""
    coverage = get_loader_coverage(record)
    if aspect == "metadata":
        metadata_keys = coverage.get("metadata", [])
        if metadata_keys:
            return "preserved in canonical metadata: " + ", ".join(metadata_keys)
        return "available in files only" if files else "not present"

    support = coverage.get(aspect)
    if support:
        return str(support)
    if files:
        return "available in files only"
    return "not present"


def model_use_text(record: DatasetRecord, aspect: str) -> str:
    """Describe current end-model consumption for a given aspect."""
    coverage = get_loader_coverage(record)
    if coverage.get("loader_name") is None:
        return "not reachable by current experiment path"
    if aspect == "interactions":
        return "consumed by training and evaluation"
    if aspect == "item_features":
        return "used by the embedding layer when canonical item_features exist"
    if aspect == "user_features":
        return "retained in canonical/graph objects, not used by model"
    return "retained only for analysis; not used by model"


def strongest_causal_asset(record: DatasetRecord, opportunities: list[str]) -> str:
    """Select the strongest single causal asset for summary tables."""
    if not opportunities:
        return "interaction-only baseline"
    priority_order = [
        "randomized exposure metadata",
        "graded sign or explicit negative signal",
        "item-side covariates available",
        "user-side covariates available",
        "temporal ordering available",
        "predefined split available",
        "interaction-only baseline",
    ]
    for candidate in priority_order:
        if candidate in opportunities:
            return candidate
    return opportunities[0]


def choose_primary_risk(risks: list[str]) -> str:
    """Return the main caution to surface in summaries."""
    if not risks:
        return "no major risk flagged"
    return risks[0]


def build_causal_audit(record: DatasetRecord) -> dict[str, Any]:
    """Build a causal-audit summary that separates file availability from actual usage."""
    aspects = collect_aspect_sources(record)
    requirements = (record.exploration_summary or {}).get("edgrec_requirements", {})
    coverage = get_loader_coverage(record)
    thesis_default_user_sources = ENABLED_FEATURE_SOURCES(
        "thesis_default",
        record.name,
        "user_features",
    )

    interaction_columns = summarize_columns(aspects["interactions"])
    user_feature_columns = summarize_columns(aspects["user_features"])
    item_feature_columns = summarize_columns(aspects["item_features"])
    metadata_columns = summarize_columns(aspects["metadata"])
    feature_columns = user_feature_columns | item_feature_columns

    opportunities: list[str] = []
    if requirements.get("supports_timestamp_split"):
        opportunities.append("temporal ordering available")
    elif requirements.get("supports_predefined_split"):
        opportunities.append("predefined split available")
    if requirements.get("supports_sign_split"):
        opportunities.append("graded sign or explicit negative signal")
    if RANDOMIZATION_MARKERS & metadata_columns:
        opportunities.append("randomized exposure metadata")
    if aspects["user_features"]:
        opportunities.append("user-side covariates available")
    if aspects["item_features"]:
        opportunities.append("item-side covariates available")
    if not opportunities:
        opportunities.append("interaction-only baseline")

    risks: list[str] = []
    if POST_TREATMENT_MARKERS & feature_columns:
        risks.append("behavioral aggregate features may leak post-exposure outcomes")
    if has_substring_marker(
        interaction_columns | metadata_columns | feature_columns,
        SEARCH_MARKERS,
    ):
        risks.append(
            "search and recommendation signals are mixed and need causal separation",
        )
    if SOCIAL_MARKERS & metadata_columns:
        risks.append("social links may act as confounders and need explicit treatment")
    if has_substring_marker(feature_columns, TEXT_MARKERS):
        risks.append(
            "textual descriptors need encoding and leakage checks before promotion",
        )
    if thesis_default_user_sources and coverage.get("loader_name"):
        risks.append("user features are loaded today but remain unused by the model")
    if requirements.get("preprocessing_cost") == "high":
        risks.append(
            "preprocessing cost is high before the dataset can join the formal matrix",
        )

    if RANDOMIZATION_MARKERS & metadata_columns:
        priority = "highest"
        next_step = (
            "audit randomized vs non-random exposure slices before adding new causal objectives"
        )
    elif aspects["user_features"] or aspects["item_features"]:
        priority = "high"
        next_step = (
            "separate pre-treatment descriptors from post-treatment aggregates "
            "and promote only safe features"
        )
    elif coverage.get("loader_name"):
        priority = "medium"
        next_step = "use as an interaction-only baseline unless new covariates are engineered"
    else:
        priority = "low"
        next_step = "implement a canonical loader before considering formal experiments"

    return {
        "aspects": aspects,
        "opportunities": opportunities,
        "risks": risks,
        "priority": priority,
        "next_step": next_step,
        "loader_name": coverage.get("loader_name") or "not in current loader registry",
        "strongest_asset": strongest_causal_asset(record, opportunities),
        "primary_risk": choose_primary_risk(risks),
    }


def format_causal_audit_summary(record: DatasetRecord) -> list[str]:
    """Render dataset-level causal audit bullets."""
    audit = build_causal_audit(record)
    opportunities = ", ".join(audit["opportunities"]) if audit["opportunities"] else "-"
    risks = ", ".join(audit["risks"][:3]) if audit["risks"] else "no major risk flagged"
    return [
        f"- Current experiment path: {audit['loader_name']}",
        f"- Strongest causal asset: {audit['strongest_asset']}",
        f"- Highest-priority audit level: {audit['priority']}",
        f"- Main opportunities: {opportunities}",
        f"- Main risks: {risks}",
        f"- Recommended next step: {audit['next_step']}",
    ]


def aspect_note(aspect: str, files: list[FileRecord]) -> str:
    """Provide a concise causal interpretation for an audit aspect."""
    columns = summarize_columns(files)
    if aspect == "interactions":
        notes: list[str] = []
        if RANDOMIZATION_MARKERS & columns:
            notes.append("contains randomized exposure indicator")
        if has_exact_marker(
            list(columns),
            {"rating", "watch_ratio", "behavior_type", "is_hate"},
        ):
            notes.append("supports label/sign construction")
        if has_exact_marker(list(columns), {"timestamp", "time_ms", "ts", "date"}):
            notes.append("supports temporal splitting")
        return "; ".join(notes) if notes else "primary interaction source"
    if aspect == "user_features":
        if POST_TREATMENT_MARKERS & columns:
            return "candidate user covariates mixed with post-treatment engagement counters"
        return "candidate pre-treatment user covariates"
    if aspect == "item_features":
        if POST_TREATMENT_MARKERS & columns:
            return "mix of item descriptors and post-treatment exposure aggregates"
        if has_substring_marker(columns, TEXT_MARKERS):
            return "descriptor-rich item covariates that need encoding"
        return "candidate item-side descriptors for feature fusion"
    if RANDOMIZATION_MARKERS & columns:
        return "supports exposure-aware causal evaluation"
    if has_substring_marker(columns, SEARCH_MARKERS):
        return "context mixes search and recommendation behavior"
    if SOCIAL_MARKERS & columns:
        return "social context may act as confounding metadata"
    return "context or auxiliary metadata"


def render_causal_audit_table(record: DatasetRecord) -> list[str]:
    """Render a markdown table for causal-audit aspects."""
    audit = build_causal_audit(record)
    aspects = audit["aspects"]
    rows: list[list[str]] = []
    for aspect, label in [
        ("interactions", "Interactions"),
        ("user_features", "User Features"),
        ("item_features", "Item Features"),
        ("metadata", "Metadata / Context"),
    ]:
        files = aspects[aspect]
        loader_text = loader_support_text(record, aspect, files)
        if not files and loader_text == "not present":
            continue
        rows.append(
            [
                label,
                summarize_source_paths(files),
                loader_text,
                model_use_text(record, aspect),
                aspect_note(aspect, files),
            ],
        )

    if not rows:
        return ["- No causal feature sources detected."]

    lines = [
        render_table_row(
            [
                "Aspect",
                "Source Files",
                "Current Loader Coverage",
                "Current Model Use",
                "Causal Notes",
            ],
        ),
        render_table_row(["---", "---", "---", "---", "---"]),
    ]
    lines.extend(render_table_row(row) for row in rows)
    return lines


def infer_entity_level(file_record: FileRecord, column: str) -> str:
    """Infer whether a candidate column belongs to interaction, user, item, or context level."""
    if is_user_feature_source(file_record):
        return "user"
    if is_item_feature_source(file_record):
        return "item"
    if is_metadata_source(file_record):
        return "context"
    if is_interaction_source(file_record):
        if column in PRETREATMENT_CONTEXT_MARKERS or column in RANDOMIZATION_MARKERS:
            return "context"
        return "interaction"
    return "context"


def infer_column_aspect(file_record: FileRecord) -> str:
    """Map a file to the audit aspect used by current loader/model summaries."""
    if file_record.relative_path.lower().endswith("item_categories.csv"):
        return "item_features"
    if is_user_feature_source(file_record):
        return "user_features"
    if is_item_feature_source(file_record):
        return "item_features"
    if is_metadata_source(file_record):
        return "metadata"
    return "interactions"


def infer_pipeline_stage(
    record: DatasetRecord,
    file_record: FileRecord,
    column: str,
) -> str:
    """Describe how far fields from a file currently travel in the pipeline."""
    aspect = infer_column_aspect(file_record)
    coverage = get_loader_coverage(record)

    if coverage.get("loader_name") is None:
        return "raw_only"
    if aspect == "interactions":
        return "model_consumed"
    if aspect == "item_features":
        default_columns = THESIS_DEFAULT_COLUMNS(
            record.name,
            aspect,
            file_record.relative_path,
        )
        if default_columns and column.lower() in {value.lower() for value in default_columns}:
            return "model_consumed"
        return "raw_only"
    if aspect == "user_features":
        default_columns = THESIS_DEFAULT_COLUMNS(
            record.name,
            aspect,
            file_record.relative_path,
        )
        if default_columns and column.lower() in {value.lower() for value in default_columns}:
            return "graph_retained"
        return "raw_only"
    metadata_keys = coverage.get("metadata", [])
    if column.lower() in {str(key).lower() for key in metadata_keys}:
        return "analysis_retained"
    return "raw_only"


def infer_causal_role(
    record: DatasetRecord,
    file_record: FileRecord,
    column: str,
) -> str:
    """Heuristically classify a candidate column by causal role."""
    relative_path = file_record.relative_path.lower()
    normalized = column.lower()
    aspect = infer_column_aspect(file_record)

    if aspect in {"user_features", "item_features"}:
        registered_role = FEATURE_ROLE_FOR_COLUMN(
            record.name,
            aspect,
            file_record.relative_path,
            column,
        )
        if registered_role is not None:
            return registered_role

    if normalized in NON_CAUSAL_MARKERS:
        return "non_causal"
    if normalized in POST_TREATMENT_MARKERS:
        return "post_treatment"
    if normalized in RANDOMIZATION_MARKERS:
        return "pre_treatment"
    if normalized in SOCIAL_MARKERS or normalized in SEARCH_MARKERS:
        return "proxy"
    if normalized in PRETREATMENT_CONTEXT_MARKERS:
        return "pre_treatment"
    if normalized == "feat" and relative_path.endswith("item_categories.csv"):
        return "pre_treatment"
    if normalized in {
        "rating",
        "watch_ratio",
        "behavior_type",
        "is_click",
        "click",
        "label",
        "play_duration",
        "duration_ms",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
    }:
        return "post_treatment"
    if (
        normalized.endswith("_cnt")
        or normalized.endswith("_user_num")
        or normalized.endswith("_duration")
        or normalized.endswith("_progress")
    ):
        return "post_treatment"
    if normalized.startswith("onehot_feat") or normalized.endswith("_range"):
        return "pre_treatment"
    if normalized in {
        "gender",
        "age",
        "occupation",
        "zip_code",
        "genres",
        "genre",
        "category_id",
        "category_name",
        "author_id",
        "music_id",
        "music_type",
        "video_type",
        "video_duration",
        "server_width",
        "server_height",
        "tag",
        "topic_tag",
        "caption",
        "manual_cover_text",
        "first_level_category_id",
        "second_level_category_id",
        "third_level_category_id",
    }:
        if "tags.csv" in relative_path:
            return "proxy"
        return "pre_treatment"
    if normalized.endswith("id") and normalized not in {
        "author_id",
        "music_id",
        "category_id",
        "tagid",
        "tag_id",
        "first_level_category_id",
        "second_level_category_id",
        "third_level_category_id",
    }:
        return "non_causal"
    if normalized in TEXT_MARKERS:
        if "tags.csv" in relative_path:
            return "proxy"
        return "pre_treatment"
    return "unknown"


def quick_safe_use_check(
    file_record: FileRecord,
    column: str,
    causal_role: str,
    pipeline_stage: str,
) -> tuple[str, str]:
    """Return a minimal policy verdict for whether a field is safe to use now."""
    normalized = column.lower()
    relative_path = file_record.relative_path.lower()

    if causal_role == "non_causal":
        return "exclude", "identifier or bookkeeping field"
    if causal_role in {"post_treatment", "post_treatment_excluded"}:
        return (
            "defer",
            "likely downstream of exposure or outcome; keep out of default causal features",
        )
    if causal_role in {"proxy", "proxy_only"}:
        return (
            "ablation_only",
            "potentially useful but entangled with exposure, search, or social context",
        )
    if causal_role == "unknown":
        return "review", "needs manual causal review before promotion"
    if normalized in TEXT_MARKERS or "caption" in relative_path or normalized == "title":
        return (
            "encode_then_test",
            "pre-treatment descriptor but needs encoding and leakage review",
        )
    if pipeline_stage == "raw_only":
        return (
            "load_then_test",
            "looks safe enough to prototype after adding loader support",
        )
    if pipeline_stage == "graph_retained":
        return (
            "model_extension_needed",
            "already loaded but not consumed by the current model",
        )
    return (
        "safe_candidate",
        "eligible for quick utility probes in the current feature-aware path",
    )


def collect_candidate_feature_rows(record: DatasetRecord) -> list[dict[str, str]]:
    """Enumerate auditable candidate columns for one dataset."""
    rows: list[dict[str, str]] = []
    for file_record in sorted(record.file_records, key=lambda item: item.relative_path):
        columns = candidate_feature_columns(schema_columns(file_record))
        if not columns:
            continue
        aspect = infer_column_aspect(file_record)
        for column in columns:
            pipeline_stage = infer_pipeline_stage(record, file_record, column)
            causal_role = infer_causal_role(record, file_record, column)
            safe_use, rationale = quick_safe_use_check(
                file_record,
                column,
                causal_role,
                pipeline_stage,
            )
            rows.append(
                {
                    "file": file_record.relative_path,
                    "aspect": aspect,
                    "entity_level": infer_entity_level(file_record, column),
                    "column": column,
                    "causal_role": causal_role,
                    "pipeline_stage": pipeline_stage,
                    "quick_check": safe_use,
                    "rationale": rationale,
                },
            )
    return rows


def render_candidate_feature_table(record: DatasetRecord) -> list[str]:
    """Render a per-column causal-feature audit table."""
    rows = collect_candidate_feature_rows(record)
    if not rows:
        return ["- No candidate columns detected."]

    lines = [
        render_table_row(
            [
                "File",
                "Aspect",
                "Entity",
                "Column",
                "Causal Role",
                "Pipeline Stage",
                "Quick Check",
                "Rationale",
            ],
        ),
        render_table_row(["---", "---", "---", "---", "---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(
            render_table_row(
                [
                    row["file"],
                    row["aspect"],
                    row["entity_level"],
                    row["column"],
                    row["causal_role"],
                    row["pipeline_stage"],
                    row["quick_check"],
                    row["rationale"],
                ],
            ),
        )
    return lines


def build_feature_audit_payload(
    records: list[DatasetRecord],
    data_root: Path,
) -> dict[str, Any]:
    """Build a machine-readable feature-audit export."""
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    datasets: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.name.lower()):
        audit = build_causal_audit(record)
        datasets.append(
            {
                "dataset": record.name,
                "root": str(record.root_path),
                "loader_name": audit["loader_name"],
                "strongest_asset": audit["strongest_asset"],
                "primary_risk": audit["primary_risk"],
                "priority": audit["priority"],
                "next_step": audit["next_step"],
                "file_schemas": [
                    {
                        "path": file_record.relative_path,
                        "schema_type": (file_record.schema or {}).get("schema_type"),
                        "column_count": (file_record.schema or {}).get("column_count"),
                        "columns": list((file_record.schema or {}).get("columns", [])),
                        "dtypes": dict((file_record.schema or {}).get("dtypes", {})),
                        "has_header": (file_record.schema or {}).get("has_header"),
                        "delimiter": (file_record.schema or {}).get("delimiter"),
                        "parser": (file_record.schema or {}).get("parser"),
                        "schema_source": (file_record.schema or {}).get("schema_source"),
                        "encoding": (file_record.schema or {}).get("encoding"),
                    }
                    for file_record in sorted(
                        schema_file_records(record),
                        key=lambda item: item.relative_path,
                    )
                ],
                "candidate_columns": collect_candidate_feature_rows(record),
            },
        )

    return {
        "generated_at": generated_at,
        "scan_root": str(data_root),
        "datasets": datasets,
    }


def detect_status(
    has_raw_content: bool,
    has_processed_data: bool,
    file_count: int,
) -> str:
    """Compute dataset availability status.

    Args:
        has_raw_content: Whether raw files are present.
        has_processed_data: Whether processed files are present.
        file_count: Total discovered files.

    Returns:
        Status label.

    """
    if file_count == 0:
        return "EMPTY"
    if has_processed_data and has_raw_content:
        return "READY (raw + processed)"
    if has_processed_data:
        return "READY (processed only)"
    if has_raw_content:
        return "PARTIAL (raw only)"
    return "PARTIAL"


def load_pyg_data_from_processed(processed_path: Path) -> Any:
    """Load a PyG Data/HeteroData object from ``processed/data.pt``.

    Args:
        processed_path: Processed data file path.

    Returns:
        Loaded PyG data object.

    Raises:
        RuntimeError: If object cannot be reconstructed.

    """
    import torch

    try:
        payload = torch.load(processed_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(processed_path, map_location="cpu")

    if isinstance(payload, tuple) and len(payload) >= 3 and isinstance(payload[0], dict):
        data_dict = payload[0]
        data_cls = payload[2]
        if hasattr(data_cls, "from_dict"):
            return data_cls.from_dict(data_dict)

    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]

    if hasattr(payload, "edge_index") or hasattr(payload, "edge_types"):
        return payload

    raise RuntimeError("unsupported processed payload format")


def infer_graph_info(data_obj: Any, source_path: str) -> dict[str, Any]:
    """Infer graph metadata from a loaded PyG data object.

    Args:
        data_obj: PyG Data or HeteroData object.
        source_path: Relative source path for report.

    Returns:
        Graph metadata dictionary.

    """
    node_types = list(getattr(data_obj, "node_types", []))
    edge_types = list(getattr(data_obj, "edge_types", []))

    def _collect_splits(obj: Any) -> dict[str, Any]:
        """Gather train/val/test masks or index tensors from a data object."""
        splits: dict[str, Any] = {}
        # homogeneous Data objects store them as attributes
        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if isinstance(v, torch.Tensor) and any(
                    prefix in k for prefix in ["train", "val", "test"]
                ):
                    splits[k] = list(v.size())
        # hetero stores have dict-like semantics
        if hasattr(obj, "node_types") and hasattr(obj, "edge_types"):
            for store in list(getattr(obj, "node_stores", [])) + list(
                getattr(obj, "edge_stores", []),
            ):
                for k, v in store.items():
                    if isinstance(v, torch.Tensor) and any(
                        prefix in k for prefix in ["train", "val", "test"]
                    ):
                        keyname = f"{store._key}.{k}"
                        splits[keyname] = list(v.size())
        return splits

    if node_types and edge_types:
        node_counts: dict[str, int] = {}
        node_features: dict[str, int] = {}
        edge_counts: dict[str, int] = {}

        for node_type in node_types:
            node_store = data_obj[node_type]
            node_counts[node_type] = int(getattr(node_store, "num_nodes", 0) or 0)
            x_value = getattr(node_store, "x", None)
            if x_value is not None and hasattr(x_value, "shape") and len(x_value.shape) == 2:
                node_features[node_type] = int(x_value.shape[1])
            else:
                node_features[node_type] = 0

        for edge_type in edge_types:
            edge_key = str(edge_type)
            edge_store = data_obj[edge_type]
            edge_counts[edge_key] = int(getattr(edge_store, "num_edges", 0) or 0)

        splits_info = _collect_splits(data_obj)

        result = {
            "source": source_path,
            "graph_type": "heterogeneous",
            "node_types": node_types,
            "edge_types": [str(item) for item in edge_types],
            "total_nodes": int(sum(node_counts.values())),
            "total_edges": int(sum(edge_counts.values())),
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "node_feature_dims": node_features,
            "edge_layouts": "sparse"
            if any(hasattr(data_obj[et], "adj_t") for et in edge_types)
            else "dense",
        }
        if splits_info:
            result["splits"] = splits_info
        return result

    # Homogeneous case
    edge_layout = "dense"
    if hasattr(data_obj, "adj_t") and data_obj.adj_t is not None:
        edge_layout = "sparse"

    return {
        "source": source_path,
        "graph_type": "homogeneous",
        "num_nodes": int(getattr(data_obj, "num_nodes", 0) or 0),
        "num_edges": int(getattr(data_obj, "num_edges", 0) or 0),
        "num_node_features": int(getattr(data_obj, "num_node_features", 0) or 0),
        "edge_layout": edge_layout,
    }


def extract_graph_info_from_processed(
    dataset_root: Path,
    processed_paths: list[Path],
) -> tuple[dict[str, Any] | None, str | None]:
    """Try extracting graph metadata from ``processed/data.pt``.

    Args:
        dataset_root: Dataset root directory.
        processed_paths: Candidate processed files.

    Returns:
        Tuple (graph_info, error_message).

    """
    if not processed_paths:
        return None, None

    for processed_path in processed_paths:
        try:
            data_obj = load_pyg_data_from_processed(processed_path)
            graph_info = infer_graph_info(
                data_obj,
                str(processed_path.relative_to(dataset_root)),
            )
            return graph_info, None
        except Exception:
            continue

    return None, "failed to parse processed/data*.pt"


def scan_dataset(dataset_root: Path) -> DatasetRecord:
    """Scan one dataset directory recursively.

    Args:
        dataset_root: Dataset root folder under data/<dataset>/.

    Returns:
        Populated ``DatasetRecord``.

    """
    notes: list[str] = []
    file_records: list[FileRecord] = []

    all_files = [path for path in dataset_root.rglob("*") if path.is_file()]
    for file_path in sorted(all_files):
        try:
            file_records.append(build_file_record(dataset_root, file_path))
        except Exception as exc:
            notes.append(f"file_scan_failed:{file_path.name}:{exc}")

    raw_files = [path for path in dataset_root.rglob("raw/**/*") if path.is_file()]
    processed_files = [path for path in dataset_root.rglob("processed/**/*") if path.is_file()]
    processed_paths = [path for path in dataset_root.rglob("processed/data*.pt") if path.is_file()]

    has_raw_content = len(raw_files) > 0
    has_processed_data = len(processed_files) > 0
    status = detect_status(has_raw_content, has_processed_data, len(file_records))

    graph_info, graph_error = extract_graph_info_from_processed(
        dataset_root,
        processed_paths,
    )
    if graph_error:
        notes.append(graph_error)

    record = DatasetRecord(
        name=dataset_root.name,
        root_path=dataset_root,
        file_records=file_records,
        has_raw_content=has_raw_content,
        has_processed_data=has_processed_data,
        processed_paths=[str(path.relative_to(dataset_root)) for path in processed_paths],
        status=status,
        notes=notes,
        graph_info=graph_info,
    )
    return attach_exploration_summary(record)


def schema_file_records(record: DatasetRecord) -> list[FileRecord]:
    """Return files that expose a structured schema."""
    return [
        file_record
        for file_record in record.file_records
        if (file_record.schema or {}).get("column_count") is not None
    ]


def format_delimiter(delimiter: Any) -> str:
    """Render a delimiter value for markdown."""
    if delimiter is None:
        return "-"
    if delimiter == "\t":
        return "\\t"
    if delimiter == " ":
        return "space"
    return str(delimiter)


def format_header_flag(schema: dict[str, Any] | None) -> str:
    """Render whether the parsed file used a header row."""
    if not schema or schema.get("has_header") is None:
        return "-"
    return "yes" if schema.get("has_header") else "no"


def format_schema_pairs(
    schema: dict[str, Any] | None,
    limit: int | None = None,
) -> str:
    """Render a file schema as ``name:type`` pairs."""
    if not schema:
        return "-"
    columns = [str(column) for column in schema.get("columns", [])]
    if not columns:
        return "-"
    dtypes = {str(column): str(dtype) for column, dtype in (schema.get("dtypes") or {}).items()}
    pairs = [f"{column}:{dtypes.get(column, '-')}" for column in columns]
    if limit is not None and len(pairs) > limit:
        return ", ".join(pairs[:limit]) + f" (+{len(pairs) - limit} more)"
    return ", ".join(pairs)


def count_schema_sources(record: DatasetRecord, *sources: str) -> int:
    """Count structured files that use the provided schema sources."""
    return sum(
        1
        for file_record in schema_file_records(record)
        if (file_record.schema or {}).get("schema_source") in sources
    )


def max_schema_columns(record: DatasetRecord) -> int | None:
    """Return the widest structured schema found in a dataset."""
    column_counts = [
        int((file_record.schema or {}).get("column_count") or 0)
        for file_record in schema_file_records(record)
    ]
    return max(column_counts) if column_counts else None


def render_schema_inventory(record: DatasetRecord) -> list[str]:
    """Render a schema-first inventory for structured files."""
    structured_files = sorted(
        schema_file_records(record),
        key=lambda item: item.relative_path,
    )
    if not structured_files:
        return ["- No structured file schemas detected."]

    lines: list[str] = []
    for file_record in structured_files:
        schema = file_record.schema or {}
        lines.append(f"#### {file_record.relative_path}")
        lines.append("")
        lines.append(f"- Header row: {format_header_flag(schema)}")
        lines.append(f"- Delimiter: {format_delimiter(schema.get('delimiter'))}")
        lines.append(f"- Parser: {schema.get('parser') or '-'}")
        lines.append(f"- Schema source: {schema.get('schema_source') or '-'}")
        if schema.get("encoding"):
            lines.append(f"- Encoding: {schema['encoding']}")
        lines.append(
            f"- Columns ({schema.get('column_count') or 0}): {format_schema_pairs(schema)}",
        )
        lines.append("")
    return lines


def render_table_row(columns: list[str]) -> str:
    """Render one markdown table row.

    Args:
        columns: Ordered cell values.

    Returns:
        Markdown row string.

    """
    safe_columns = [value.replace("\n", " ").replace("|", "\\|") for value in columns]
    return "| " + " | ".join(safe_columns) + " |"


def build_markdown_report(records: list[DatasetRecord], data_root: Path) -> str:
    """Build markdown report content.

    Args:
        records: Dataset records.
        data_root: Root folder that was scanned.

    Returns:
        Markdown document.

    """
    now = datetime.now(UTC).isoformat(timespec="seconds")
    total_bytes = sum(record.total_bytes for record in records)
    total_files = sum(len(record.file_records) for record in records)

    lines: list[str] = []
    lines.append("# Dataset Information Report")
    lines.append("")
    lines.append(f"- Generated at: {now}")
    lines.append(f"- Scan root: `{data_root}`")
    lines.append(f"- Selected datasets: {', '.join(ALLOWED_DATASETS)}")
    lines.append(f"- Datasets scanned: {len(records)}")
    lines.append(f"- Files scanned: {total_files}")
    lines.append(f"- Total size: {format_bytes(total_bytes)}")
    lines.append("")

    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(
        render_table_row(
            [
                "Dataset",
                "Files",
                "Total Size",
            ],
        ),
    )
    lines.append(
        render_table_row(
            ["---", "---:", "---:"],
        ),
    )
    ordered_records = sorted(
        records,
        key=lambda item: (-item.total_bytes, item.name.lower()),
    )
    for record in ordered_records:
        lines.append(
            render_table_row(
                [
                    record.name,
                    str(len(record.file_records)),
                    f"{record.total_bytes / (1024**2):,.2f} MiB",
                ],
            ),
        )
    lines.append("")

    for record in ordered_records:
        lines.append(f"## {record.name}")
        lines.append("")
        lines.append(f"- Files: {len(record.file_records)}")
        lines.append(f"- Total size: {format_bytes(record.total_bytes)}")
        lines.append("")
        lines.append(
            render_table_row(
                [
                    "Path",
                    "Extension",
                    "Size (MiB)",
                    "Column Names & DTypes",
                ],
            ),
        )
        lines.append(
            render_table_row(
                ["---", "---", "---:", "---"],
            ),
        )

        for file_record in sorted(
            record.file_records,
            key=lambda item: item.relative_path.lower(),
        ):
            schema = file_record.schema or {}
            schema_error = schema.get("schema_error")
            column_types = format_schema_pairs(schema) if not schema_error else "-"
            lines.append(
                render_table_row(
                    [
                        file_record.relative_path,
                        file_record.extension or "<no_ext>",
                        f"{file_record.size_bytes / (1024**2):,.2f} MiB",
                        column_types or "-",
                    ],
                ),
            )

        lines.append("")

    return "\n".join(lines)


def resolve_paths(output_path: str | None) -> tuple[Path, Path]:
    """Resolve top-level data root and report output path.

    The repository now uses `data/<dataset>/` (each with its own raw/processed
    subfolders) instead of `data/raw/<dataset>`.  This helper returns the
    root under which dataset directories live along with the requested report
    path.

    Args:
        output_path: Optional output path from CLI.

    Returns:
        Tuple ``(data_root, report_path)``.

    """
    # the script now lives under src/data_exploration/, so ``parents[1]``
    # yields the ``src/`` folder; we need the repository root one level above.
    repo_root = REPO_ROOT
    data_root = repo_root / "data"

    if output_path:
        report_path = Path(output_path)
        if not report_path.is_absolute():
            report_path = repo_root / report_path
    else:
        report_path = repo_root / "data" / "datasets_information.md"

    return data_root, report_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed args namespace.

    """
    return BUILD_DATA_INFORMATION_PARSER().parse_args()


def main() -> None:
    """Entry point for dataset report generation."""
    args = parse_args()
    data_root, report_path = resolve_paths(args.output)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # only scan the selected dataset folders under `data/`
    dataset_dirs = sorted(
        [path for path in data_root.iterdir() if path.is_dir() and path.name in ALLOWED_DATASETS],
        key=lambda item: ALLOWED_DATASETS.index(item.name),
    )
    records = [scan_dataset(dataset_dir) for dataset_dir in dataset_dirs]

    report = build_markdown_report(records, data_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    if args.audit_json:
        audit_path = Path(args.audit_json)
        if not audit_path.is_absolute():
            audit_path = REPO_ROOT / audit_path
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_payload = build_feature_audit_payload(records, data_root)
        audit_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

    print("=" * 72)
    print("DATASET INFORMATION REPORT GENERATED")
    print("=" * 72)
    print(f"Scan root: {data_root}")
    print(f"Datasets scanned: {len(records)}")
    print(f"Output report: {report_path}")
    if args.audit_json:
        print(f"Feature audit JSON: {audit_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
