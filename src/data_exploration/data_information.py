"""Generate a robust dataset inventory report for thesis datasets.

This script scans ``data/<dataset>/`` recursively and writes a comprehensive
report to ``data/datasets_information.md``. It is designed to work even when
datasets are inconsistent, partially downloaded, or stored in mixed formats.

The purpose of the report is purely informational for the thesis pipeline – it
has no influence on the training code itself.

Report includes:
- Dataset-level size totals (bytes, MiB, GiB)
- Per-file inventory with extension and size
- Text/tabular counts (line count + parsed count where possible)
- Optional array metadata (NumPy / NPZ)
- Optional processed graph metadata from ``processed/data*.pt``
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
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


INSPECT_DATASET_FILE, SCHEMA_OVERVIEW, SUMMARIZE_DATASET = load_exploration_api()


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
# Limits used to avoid expensive scanning or exports.  Set to `None`
# to disable the check entirely (we keep full datasets by default).
# Originally I added these when the CSV reader was slow and some files
# were multi‑GB, but the user reminded us that they want *all* rows kept.
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
    previous max‑size limit has been removed; the caller is responsible for
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
        with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
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
            subprocess.check_output(["wc", "-l", str(file_path)])
            .strip()
            .split()[0]
        )
        parsed = total_rows - 1 if has_header and total_rows > 0 else total_rows
        return parsed, details
    except Exception:
        return None, details


def parse_dat_counts(file_path: Path, line_count: int | None) -> tuple[int | None, dict[str, Any]]:
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
            subprocess.check_output(["wc", "-l", str(file_path)])
            .strip()
            .split()[0]
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

        if extension == ".csv":
            parsed_count, parsed_details = parse_csv_counts(file_path)
            details.update(parsed_details)
        elif extension == ".dat":
            parsed_count, parsed_details = parse_dat_counts(file_path, line_count)
            details.update(parsed_details)
        elif file_path.name in {"train.txt", "test.txt"} and "AmazonBook" in str(dataset_root):
            parsed_count, parsed_details = parse_amazonbook_interactions(file_path)
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


def detect_status(has_raw_content: bool, has_processed_data: bool, file_count: int) -> str:
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
                if isinstance(v, torch.Tensor) and any(prefix in k for prefix in ["train", "val", "test"]):
                    splits[k] = list(v.size())
        # hetero stores have dict-like semantics
        if hasattr(obj, "node_types") and hasattr(obj, "edge_types"):
            for store in list(getattr(obj, "node_stores", [])) + list(getattr(obj, "edge_stores", [])):
                for k, v in store.items():
                    if isinstance(v, torch.Tensor) and any(prefix in k for prefix in ["train", "val", "test"]):
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
            
            # Detect Graph Transformation / Tensor formats
            layout = "unknown"
            if hasattr(edge_store, "edge_index") and edge_store.edge_index is not None:
                layout = "dense_edge_index"
            elif hasattr(edge_store, "adj_t") and edge_store.adj_t is not None:
                layout = "sparse_adj_t"
            if layout not in node_features:
                pass  # We can capture formatting broadly if wanted, here we focus on counts

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
            "edge_layouts": "sparse" if any(hasattr(data_obj[et], "adj_t") for et in edge_types) else "dense",
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


def extract_graph_info_from_processed(dataset_root: Path, processed_paths: list[Path]) -> tuple[dict[str, Any] | None, str | None]:
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
            graph_info = infer_graph_info(data_obj, str(processed_path.relative_to(dataset_root)))
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
    processed_paths = [
        path
        for path in dataset_root.rglob("processed/data*.pt")
        if path.is_file()
    ]

    has_raw_content = len(raw_files) > 0
    has_processed_data = len(processed_files) > 0
    status = detect_status(has_raw_content, has_processed_data, len(file_records))

    graph_info, graph_error = extract_graph_info_from_processed(dataset_root, processed_paths)
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


def format_columns(columns: list[str]) -> str:
    """Render the full column-name list for markdown tables."""
    if not columns:
        return "-"
    return ", ".join(columns)


def format_ucagnn_summary(summary: dict[str, Any] | None) -> list[str]:
    """Render dataset-level U-CaGNN suitability bullets."""
    if not summary:
        return ["- U-CaGNN suitability: no registered semantic summary available"]

    requirements = summary["ucagnn_requirements"]
    fields = requirements["fields"]
    return [
        f"- Registry dataset: {summary['name']}",
        f"- Pairwise triplets ready: {'yes' if requirements['supports_pairwise_triplets'] else 'no'}",
        f"- Timestamp split ready: {'yes' if requirements['supports_timestamp_split'] else 'no'}",
        f"- Sign-aware split ready: {'yes' if requirements['supports_sign_split'] else 'no'}",
        f"- Popularity signal ready: {'yes' if requirements['supports_popularity_signal'] else 'no'}",
        f"- Side or multimodal features: {'yes' if requirements['supports_multimodal_or_side_features'] else 'no'}",
        f"- Preprocessing cost: {requirements['preprocessing_cost']}",
        (
            "- Inferred fields: "
            f"user={fields['user_field'] or '-'}, "
            f"item={fields['item_field'] or '-'}, "
            f"label={fields['rating_field'] or '-'}, "
            f"timestamp={fields['timestamp_field'] or '-'}"
        ),
        (
            f"- File-level semantic inspection errors: {len(summary.get('file_errors', []))}"
            if summary.get('file_errors')
            else "- File-level semantic inspection errors: 0"
        ),
        f"- Assessment: {summary['note']}",
    ]


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
    now = dt.datetime.now().isoformat(timespec="seconds")
    total_bytes = sum(record.total_bytes for record in records)
    total_files = sum(len(record.file_records) for record in records)

    lines: list[str] = []
    lines.append("# Dataset Information Report")
    lines.append("")
    lines.append(f"- Generated at: {now}")
    lines.append(f"- Scan root: `{data_root}`")
    lines.append(f"- Datasets scanned: {len(records)}")
    lines.append(f"- Files scanned: {total_files}")
    lines.append(f"- Total size: {format_bytes(total_bytes)}")
    lines.append("")

    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(render_table_row([
        "Dataset",
        "Status",
        "Files",
        "Total Size",
        "Pairwise",
        "Sign",
        "Preprocessing",
        "Raw",
        "Processed",
    ]))
    lines.append(render_table_row(["---", "---", "---:", "---:", "---", "---", "---", "---", "---"]))
    ordered_records = sorted(records, key=lambda item: (-item.total_bytes, item.name.lower()))
    for record in ordered_records:
        requirements = (record.exploration_summary or {}).get("ucagnn_requirements", {})
        lines.append(
            render_table_row(
                [
                    record.name,
                    record.status,
                    str(len(record.file_records)),
                    f"{record.total_bytes / (1024**2):,.2f} MiB",
                    "yes" if requirements.get("supports_pairwise_triplets") else "no",
                    "yes" if requirements.get("supports_sign_split") else "no",
                    requirements.get("preprocessing_cost", "-"),
                    "yes" if record.has_raw_content else "no",
                    "yes" if record.has_processed_data else "no",
                ]
            )
        )
    lines.append("")

    for record in ordered_records:
        lines.append(f"## {record.name}")
        lines.append("")
        lines.append(f"- Status: {record.status}")
        lines.append(f"- Root: `{record.root_path}`")
        lines.append(f"- Total files: {len(record.file_records)}")
        lines.append(f"- Dataset size: {format_bytes(record.total_bytes)}")
        lines.append(f"- Raw files present: {'yes' if record.has_raw_content else 'no'}")
        lines.append(f"- Processed files present: {'yes' if record.has_processed_data else 'no'}")
        if record.processed_paths:
            lines.append(f"- Processed file(s): {', '.join(record.processed_paths)}")

        extension_counter = Counter(file.extension or "<no_ext>" for file in record.file_records)
        if extension_counter:
            ext_items = ", ".join(
                f"{ext}: {count}" for ext, count in sorted(extension_counter.items(), key=lambda item: item[0])
            )
            lines.append(f"- Extension breakdown: {ext_items}")

        if record.notes:
            lines.append("- Notes:")
            for note in record.notes:
                lines.append(f"  - {note}")

        lines.append("")
        lines.append("### U-CaGNN Suitability")
        lines.append("")
        lines.extend(format_ucagnn_summary(record.exploration_summary))

        lines.append("")
        lines.append("### Files")
        lines.append("")
        lines.append(render_table_row([
            "Path",
            "Extension",
            "Size",
            "Line Count",
            "Parsed Count",
            "Column Count",
            "Column Names",
            "Details",
        ]))
        lines.append(render_table_row(["---", "---", "---:", "---:", "---:", "---:", "---", "---"]))

        for file_record in sorted(record.file_records, key=lambda item: (-item.size_bytes, item.relative_path)):
            details_text = ""
            if file_record.details:
                details_text = "; ".join(f"{key}={value}" for key, value in sorted(file_record.details.items()))
            schema = file_record.schema or {}
            schema_error = schema.get("schema_error")
            column_count = schema.get("column_count")
            column_names = format_columns(schema.get("columns", [])) if not schema_error else "-"
            if schema_error:
                details_text = f"{details_text}; schema_error={schema_error}" if details_text else f"schema_error={schema_error}"
            lines.append(
                render_table_row(
                    [
                        file_record.relative_path,
                        file_record.extension or "<no_ext>",
                        f"{file_record.size_bytes / (1024**2):,.2f} MiB",
                        str(file_record.line_count) if file_record.line_count is not None else "-",
                        str(file_record.parsed_count) if file_record.parsed_count is not None else "-",
                        str(column_count) if column_count is not None else "-",
                        column_names,
                        details_text or "-",
                    ]
                )
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
    parser = argparse.ArgumentParser(description="Generate dataset information report.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for report (default: data/datasets_information.md)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for dataset report generation."""
    args = parse_args()
    data_root, report_path = resolve_paths(args.output)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # each immediate child of `data/` is treated as a dataset folder
    dataset_dirs = sorted([path for path in data_root.iterdir() if path.is_dir()])
    records = [scan_dataset(dataset_dir) for dataset_dir in dataset_dirs]

    report = build_markdown_report(records, data_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("=" * 72)
    print("DATASET INFORMATION REPORT GENERATED")
    print("=" * 72)
    print(f"Scan root: {data_root}")
    print(f"Datasets scanned: {len(records)}")
    print(f"Output report: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()