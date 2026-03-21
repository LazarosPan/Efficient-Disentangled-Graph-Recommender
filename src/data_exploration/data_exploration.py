"""Dataset exploration helpers for U-CaGNN planning.

This module inspects candidate datasets with the future U-CaGNN ingestion
contract in mind: user-item interaction schema, timestamp support, sign-aware
edge construction, side features, and preprocessing cost.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
	from scipy.io import whosmat
except Exception:  # pragma: no cover - optional dependency for .mat inspection
	whosmat = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

DEFAULT_PREVIEW_ROWS = 5
MAX_PREVIEW_CHARS = 12_000
STRUCTURED_TEXT_SUFFIXES = {".csv", ".tsv", ".dat", ".txt"}


CANDIDATE_DATASETS: dict[str, dict[str, Any]] = {
	"MovieLens1M": {
		"root": DATA_ROOT / "MovieLens1M" / "raw",
		"kind": "ratings_dat",
		"files": ["ratings.dat", "movies.dat", "users.dat"],
	},
	"MovieLensSmall": {
		"root": DATA_ROOT / "MovieLens" / "raw" / "ml-latest-small",
		"kind": "csv_interactions",
		"files": ["ratings.csv", "movies.csv", "tags.csv"],
	},
	"MovieLens20M": {
		"root": DATA_ROOT / "MovieLens20M" / "raw",
		"kind": "csv_interactions",
		"files": ["ratings.csv", "movies.csv", "tags.csv"],
	},
	"Taobao": {
		"root": DATA_ROOT / "Taobao" / "raw",
		"kind": "csv_no_header",
		"files": ["UserBehavior.csv", "description.txt"],
	},
	"KuaiRec_v2": {
		"root": DATA_ROOT / "KuaiRec_v2" / "data",
		"kind": "csv_interactions",
		"files": ["small_matrix.csv", "big_matrix.csv", "user_features.csv", "item_daily_features.csv"],
	},
	"KuaiRand-1K": {
		"root": DATA_ROOT / "KuaiRand-1K" / "data",
		"kind": "csv_interactions",
		"files": [
			"log_random_4_22_to_5_08_1k.csv",
			"log_standard_4_08_to_4_21_1k.csv",
			"log_standard_4_22_to_5_08_1k.csv",
			"user_features_1k.csv",
			"video_features_basic_1k.csv",
		],
	},
	"KuaiSAR_v2": {
		"root": DATA_ROOT / "KuaiSAR_v2",
		"kind": "csv_interactions",
		"files": ["rec_inter.csv", "src_inter.csv", "user_features.csv", "item_features.csv"],
	},
	"AmazonBook": {
		"root": DATA_ROOT / "AmazonBook" / "raw",
		"kind": "interaction_lists",
		"files": ["train.txt", "test.txt", "item_list.txt", "user_list.txt"],
	},
	"AmazonCDs": {
		"root": DATA_ROOT / "AmazonCDs" / "raw",
		"kind": "split_ratings_txt",
		"files": ["train.txt", "valid.txt", "test.txt", "info.txt"],
	},
	"AmazonMusic": {
		"root": DATA_ROOT / "AmazonMusic" / "raw",
		"kind": "split_ratings_txt",
		"files": ["train.txt", "valid.txt", "test.txt", "info.txt"],
	},
	"KuaiRec_SIGformer": {
		"root": DATA_ROOT / "KuaiRec_SIGformer" / "raw",
		"kind": "signed_splits",
		"files": ["train.txt", "valid.txt", "test.txt", "info.txt"],
	},
	"KuaiRand_SIGformer": {
		"root": DATA_ROOT / "KuaiRand_SIGformer" / "raw",
		"kind": "signed_splits",
		"files": ["train.txt", "test.txt", "valid.txt", "info.txt"],
	},
	"Netflix": {
		"root": DATA_ROOT / "netflix" / "raw" / "output",
		"kind": "preprocessed_sparse",
		"files": ["train_record.csv", "val_record.csv", "test_record.csv", "train_coo_record.npz", "popularity.npy"],
	},
	"Douban_Book": {
		"root": DATA_ROOT / "Douban_Book",
		"kind": "heterogeneous_txt_npz",
		"files": ["train.txt", "test.txt", "user.txt", "author.txt", "s_adj_mat.npz"],
	},
	"Douban": {
		"root": DATA_ROOT / "Douban" / "raw",
		"kind": "mat_only",
		"files": ["training_test_dataset.mat"],
	},
	"Yelp": {
		"root": DATA_ROOT / "Yelp" / "raw",
		"kind": "graph_binary",
		"files": ["adj_full.npz", "feats.npy", "class_map.json", "role.json"],
	},
	"AmazonProducts": {
		"root": DATA_ROOT / "AmazonProducts" / "raw",
		"kind": "graph_binary",
		"files": ["adj_full.npz", "feats.npy", "class_map.json", "role.json"],
	},
}


DATASET_NAME_ALIASES: dict[str, str] = {
	"MovieLens": "MovieLensSmall",
	"netflix": "Netflix",
}


FILE_COLUMN_HINTS: dict[str, list[str]] = {
	"ratings.dat": ["user_id", "movie_id", "rating", "timestamp"],
	"movies.dat": ["movie_id", "title", "genres"],
	"users.dat": ["user_id", "gender", "age", "occupation", "zip_code"],
	"UserBehavior.csv": ["user_id", "item_id", "category_id", "behavior_type", "timestamp"],
}


def dataset_registry() -> pd.DataFrame:
	"""Return a tabular view of the built-in dataset registry."""
	rows = []
	for name, config in CANDIDATE_DATASETS.items():
		rows.append(
			{
				"dataset": name,
				"root": str(config["root"].relative_to(REPO_ROOT)),
				"kind": config["kind"],
				"files": ", ".join(config["files"]),
			}
		)
	return pd.DataFrame(rows)


def get_dataset_config(name: str) -> dict[str, Any]:
	"""Return the configured metadata for a dataset."""
	name = DATASET_NAME_ALIASES.get(name, name)
	if name not in CANDIDATE_DATASETS:
		raise KeyError(f"Unknown dataset: {name}")
	return CANDIDATE_DATASETS[name]


def _safe_json(value: Any) -> Any:
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	if isinstance(value, dict):
		return {str(key): _safe_json(item) for key, item in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [_safe_json(item) for item in value]
	return str(value)


def _read_text_head(path: Path, max_lines: int = 10) -> list[str]:
	lines: list[str] = []
	with path.open("r", encoding="utf-8", errors="ignore") as handle:
		for _, line in zip(range(max_lines), handle):
			lines.append(line.rstrip("\n"))
	return lines


def _guess_delimiter(sample: str, default: str | None = None) -> str:
	if default is not None:
		return default
	if "::" in sample:
		return "::"
	try:
		dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
		return dialect.delimiter
	except csv.Error:
		if "\t" in sample:
			return "\t"
		if "," in sample:
			return ","
		return " "


def inspect_text_file(
	path: str | Path,
	delimiter: str | None = None,
	n_rows: int = DEFAULT_PREVIEW_ROWS,
	header: str | int | None = "infer",
	column_names: list[str] | None = None,
) -> dict[str, Any]:
	"""Inspect a delimited text file without loading it fully into memory."""
	file_path = Path(path)
	head_lines = _read_text_head(file_path, max_lines=max(n_rows + 2, 8))
	sample = "\n".join(head_lines)
	detected_delimiter = _guess_delimiter(sample, default=delimiter)

	if detected_delimiter == "::":
		normalized_sample = sample.replace("::", "\t")
		frame = pd.read_csv(
			io.StringIO(normalized_sample),
			sep="\t",
			header=header,
			names=column_names,
			engine="python",
			nrows=n_rows,
		)
	elif detected_delimiter == " ":
		frame = pd.read_csv(
			io.StringIO(sample),
			sep=r"\s+",
			header=header,
			names=column_names,
			engine="python",
			nrows=n_rows,
		)
	else:
		frame = pd.read_csv(
			io.StringIO(sample),
			sep=detected_delimiter,
			header=header,
			names=column_names,
			engine="python",
			nrows=n_rows,
		)

	return {
		"path": str(file_path.relative_to(REPO_ROOT)),
		"type": "text_table",
		"delimiter": detected_delimiter,
		"columns": [str(column) for column in frame.columns],
		"dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
		"sample_rows": frame.head(n_rows).to_dict(orient="records"),
	}


def inspect_interaction_list_file(path: str | Path, n_rows: int = DEFAULT_PREVIEW_ROWS) -> dict[str, Any]:
	"""Inspect files where each line is `user_id item_1 item_2 ...`."""
	file_path = Path(path)
	rows: list[dict[str, Any]] = []
	interaction_counts: list[int] = []
	for line in _read_text_head(file_path, max_lines=n_rows):
		tokens = line.split()
		if not tokens:
			continue
		user_id = tokens[0]
		item_ids = tokens[1:]
		interaction_counts.append(len(item_ids))
		rows.append(
			{
				"user_id": int(user_id) if user_id.isdigit() else user_id,
				"num_items": len(item_ids),
				"first_items": item_ids[:5],
			}
		)
	return {
		"path": str(file_path.relative_to(REPO_ROOT)),
		"type": "interaction_list",
		"columns": ["user_id", "item_ids..."],
		"avg_items_per_preview_row": float(np.mean(interaction_counts)) if interaction_counts else 0.0,
		"sample_rows": rows,
	}


def inspect_dat_file(
	path: str | Path,
	separator: str = "::",
	n_rows: int = DEFAULT_PREVIEW_ROWS,
	column_names: list[str] | None = None,
) -> dict[str, Any]:
	"""Inspect MovieLens-style DAT files with multi-character separators."""
	return inspect_text_file(
		path,
		delimiter=separator,
		n_rows=n_rows,
		header=None,
		column_names=column_names,
	)


def inspect_npy_file(path: str | Path) -> dict[str, Any]:
	"""Inspect NumPy array metadata without materializing the full array."""
	file_path = Path(path)
	array = np.load(file_path, mmap_mode="r")
	return {
		"path": str(file_path.relative_to(REPO_ROOT)),
		"type": "npy",
		"dtype": str(array.dtype),
		"shape": list(array.shape),
	}


def inspect_npz_file(path: str | Path) -> dict[str, Any]:
	"""Inspect NPZ container keys and sparse hints."""
	file_path = Path(path)
	container = np.load(file_path, allow_pickle=False)
	keys = list(container.files)
	meta: dict[str, Any] = {}
	for key in keys[:10]:
		value = container[key]
		meta[key] = {
			"dtype": str(value.dtype),
			"shape": list(value.shape),
		}
	sparse_hint = {"looks_like_sparse": {"indices", "indptr", "data", "shape"}.issubset(set(keys))}
	return {
		"path": str(file_path.relative_to(REPO_ROOT)),
		"type": "npz",
		"keys": keys,
		"arrays": meta,
		**sparse_hint,
	}


def inspect_mat_file(path: str | Path) -> dict[str, Any]:
	"""Inspect MATLAB file metadata if SciPy support is available."""
	file_path = Path(path)
	if whosmat is None:
		return {
			"path": str(file_path.relative_to(REPO_ROOT)),
			"type": "mat",
			"warning": "scipy.io.whosmat unavailable",
		}
	try:
		return {
			"path": str(file_path.relative_to(REPO_ROOT)),
			"type": "mat",
			"variables": [
				{"name": name, "shape": list(shape), "dtype": dtype}
				for name, shape, dtype in whosmat(file_path)
			],
		}
	except Exception as exc:
		return {
			"path": str(file_path.relative_to(REPO_ROOT)),
			"type": "mat",
			"warning": str(exc),
		}


def inspect_json_file(path: str | Path, max_keys: int = 10) -> dict[str, Any]:
	"""Inspect JSON top-level structure and a small key preview."""
	file_path = Path(path)
	with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
		data = json.load(handle)
	if isinstance(data, dict):
		preview_keys = list(data.keys())[:max_keys]
		json_type = "dict"
		size_hint = len(data)
	elif isinstance(data, list):
		preview_keys = data[:max_keys]
		json_type = "list"
		size_hint = len(data)
	else:
		preview_keys = [str(data)]
		json_type = type(data).__name__
		size_hint = None
	return {
		"path": str(file_path.relative_to(REPO_ROOT)),
		"type": "json",
		"json_type": json_type,
		"size_hint": size_hint,
		"preview": _safe_json(preview_keys),
	}


def inspect_file(path: str | Path, **kwargs: Any) -> dict[str, Any]:
	"""Dispatch inspection based on file suffix and known patterns."""
	file_path = Path(path)
	suffix = file_path.suffix.lower()
	if suffix == ".npy":
		return inspect_npy_file(file_path)
	if suffix == ".npz":
		return inspect_npz_file(file_path)
	if suffix == ".mat":
		return inspect_mat_file(file_path)
	if suffix == ".json":
		return inspect_json_file(file_path)
	if suffix == ".dat":
		dat_kwargs = {key: value for key, value in kwargs.items() if key in {"separator", "n_rows", "column_names"}}
		return inspect_dat_file(file_path, **dat_kwargs)
	return inspect_text_file(file_path, **kwargs)


def inspect_dataset_file(dataset_name: str, path: str | Path, n_rows: int = DEFAULT_PREVIEW_ROWS) -> dict[str, Any]:
	"""Inspect one file using dataset-aware parsing rules when available."""
	file_path = Path(path)
	config = get_dataset_config(dataset_name)
	file_name = file_path.name
	suffix = file_path.suffix.lower()

	if config["kind"] == "interaction_lists" and file_name in {"train.txt", "test.txt"}:
		return inspect_interaction_list_file(file_path, n_rows=n_rows)

	if config["kind"] == "csv_no_header" and suffix == ".csv":
		return inspect_text_file(
			file_path,
			delimiter=",",
			n_rows=n_rows,
			header=None,
			column_names=FILE_COLUMN_HINTS.get(file_name),
		)

	if file_name in FILE_COLUMN_HINTS:
		return inspect_file(
			file_path,
			n_rows=n_rows,
			header=None,
			column_names=FILE_COLUMN_HINTS[file_name],
		)

	if suffix in STRUCTURED_TEXT_SUFFIXES:
		return inspect_file(file_path, n_rows=n_rows)

	if suffix in {".npy", ".npz", ".mat", ".json"}:
		return inspect_file(file_path)

	return {
		"path": str(file_path.relative_to(REPO_ROOT)),
		"type": "unstructured",
	}


def schema_overview(inspection: dict[str, Any]) -> dict[str, Any]:
	"""Build a compact schema summary from a file inspection payload."""
	columns = [str(column).replace("\n", " ").strip() for column in inspection.get("columns", [])]
	return {
		"schema_type": inspection.get("type", "unknown"),
		"column_count": len(columns) if columns else None,
		"columns": columns,
		"delimiter": inspection.get("delimiter"),
	}


def _infer_ucagnn_fields(columns: list[str]) -> dict[str, Any]:
	lowered = {column.lower(): column for column in columns}

	def find(*keywords: str) -> str | None:
		for keyword in keywords:
			for lowered_name, original_name in lowered.items():
				if keyword in lowered_name:
					return original_name
		return None

	return {
		"user_field": find("user"),
		"item_field": find("item", "movie", "video", "iid"),
		"rating_field": find("rating", "watch_ratio", "is_click", "label"),
		"timestamp_field": find("timestamp", "time_ms", "time", "date", "ts"),
		"popularity_candidate": find("pop", "show_cnt", "play_cnt"),
		"explicit_negative_candidate": find("hate", "dislike", "is_hate"),
	}


def _derive_ucagnn_requirements(summary: dict[str, Any]) -> dict[str, Any]:
	file_inspections = summary["files"]
	text_columns: list[str] = []
	for inspection in file_inspections:
		text_columns.extend(inspection.get("columns", []))

	fields = _infer_ucagnn_fields(text_columns)
	lower_columns = [column.lower() for column in text_columns]

	sign_support = any(
		candidate in lower_columns
		for candidate in ["is_hate", "behavior", "behavior_type", "watch_ratio", "rating"]
	) or summary["kind"] in {"signed_splits", "preprocessed_sparse"}

	multimodal_support = any(
		marker in lower_columns
		for marker in ["feat", "caption", "genre", "category", "author", "music_id"]
	) or summary["kind"] in {"graph_binary", "heterogeneous_txt_npz"}

	preprocessing_cost = "low"
	if summary["kind"] in {"heterogeneous_txt_npz", "graph_binary", "mat_only"}:
		preprocessing_cost = "high"
	elif summary["kind"] in {"interaction_lists", "preprocessed_sparse", "signed_splits"}:
		preprocessing_cost = "medium"
	elif summary["name"] in {"Taobao", "KuaiRand-1K", "KuaiSAR_v2", "KuaiRec_v2"}:
		preprocessing_cost = "medium"

	return {
		"fields": fields,
		"supports_pairwise_triplets": fields["user_field"] is not None and fields["item_field"] is not None,
		"supports_timestamp_split": fields["timestamp_field"] is not None or any(
			file_name.startswith(("train", "valid", "test")) for file_name in summary["present_files"]
		),
		"supports_sign_split": sign_support,
		"supports_popularity_signal": fields["popularity_candidate"] is not None,
		"supports_multimodal_or_side_features": multimodal_support,
		"preprocessing_cost": preprocessing_cost,
	}


def _dataset_note(name: str, requirements: dict[str, Any]) -> str:
	if name == "MovieLens1M":
		return "Strong baseline for fairness, timestamps, and rating-derived positive-negative splits."
	if name == "Taobao":
		return "Large-scale multi-behavior dataset; good for implicit sign derivation and scaling experiments."
	if name == "KuaiRec_v2":
		return "Strong candidate for richer feedback and side features; likely easiest Kuai dataset to adapt to U-CaGNN."
	if name == "KuaiRand-1K":
		return "Most valuable for randomized exposure, but heavier than KuaiRec because of sequential logs and large feature tables."
	if name == "KuaiSAR_v2":
		return "Rich dataset, but search and recommendation are mixed, which makes it less aligned with the first U-CaGNN benchmark phase."
	if requirements["preprocessing_cost"] == "high":
		return "Likely off-scope for phase 1 because the structure is not a simple user-item recommendation table."
	return "Potentially usable if transformed into the unified U-CaGNN ingestion contract."


def summarize_dataset(name: str) -> dict[str, Any]:
	"""Inspect the configured files for one candidate dataset."""
	resolved_name = DATASET_NAME_ALIASES.get(name, name)
	config = get_dataset_config(name)
	root = Path(config["root"])
	files: list[dict[str, Any]] = []
	present_files: list[str] = []
	missing_files: list[str] = []
	file_errors: list[str] = []

	for file_name in config["files"]:
		file_path = root / file_name
		if not file_path.exists():
			missing_files.append(file_name)
			continue
		present_files.append(file_name)
		try:
			files.append(inspect_dataset_file(resolved_name, file_path))
		except Exception as exc:
			file_errors.append(f"{file_name}: {exc}")
			files.append(
				{
					"path": str(file_path.relative_to(REPO_ROOT)),
					"type": "inspection_error",
					"error": str(exc),
				}
			)

	summary = {
		"name": resolved_name,
		"requested_name": name,
		"root": str(root.relative_to(REPO_ROOT)),
		"kind": config["kind"],
		"present_files": present_files,
		"missing_files": missing_files,
		"file_errors": file_errors,
		"files": files,
	}
	requirements = _derive_ucagnn_requirements(summary)
	summary["ucagnn_requirements"] = requirements
	summary["note"] = _dataset_note(resolved_name, requirements)
	return summary


def summarize_candidates(dataset_names: list[str] | None = None) -> list[dict[str, Any]]:
	"""Summarize multiple datasets."""
	names = dataset_names or list(CANDIDATE_DATASETS)
	return [summarize_dataset(name) for name in names]


def candidate_verdict_table(dataset_names: list[str] | None = None) -> pd.DataFrame:
	"""Return a comparative table tuned for dataset selection decisions."""
	summaries = summarize_candidates(dataset_names)
	rows = []
	for summary in summaries:
		requirements = summary["ucagnn_requirements"]
		fields = requirements["fields"]
		rows.append(
			{
				"dataset": summary["name"],
				"kind": summary["kind"],
				"pairwise_ready": requirements["supports_pairwise_triplets"],
				"timestamp_ready": requirements["supports_timestamp_split"],
				"sign_ready": requirements["supports_sign_split"],
				"popularity_ready": requirements["supports_popularity_signal"],
				"side_features": requirements["supports_multimodal_or_side_features"],
				"preprocessing_cost": requirements["preprocessing_cost"],
				"user_field": fields["user_field"],
				"item_field": fields["item_field"],
				"label_field": fields["rating_field"],
				"timestamp_field": fields["timestamp_field"],
				"note": summary["note"],
			}
		)
	return pd.DataFrame(rows)


def to_json_ready(obj: Any) -> Any:
	"""Convert explorer outputs into JSON-serializable structures."""
	return _safe_json(obj)


def main() -> None:
	"""Print a compact comparative verdict for the configured datasets."""
	table = candidate_verdict_table(
		[
			"MovieLens1M",
			"Taobao",
			"KuaiRec_v2",
			"KuaiRand-1K",
			"KuaiSAR_v2",
			"AmazonBook",
			"Netflix",
		]
	)
	print(table.to_string(index=False))


if __name__ == "__main__":
	main()
