# U-CaGNN Data Pipeline Skill

Use this skill when working on data loading, graph construction, negative sampling, or dataset handling.

## Key Files
- `docs/ucagnn_implementation/data-pipeline.md` - Data flow with paper cross-references
- `src/data/loaders.py` - Dataset loading (load_dataset)
- `src/utils/dataset_loader_utils.py` - Shared dataset-loader helpers for local-path resolution and safe primitive field parsing
- `src/data/feature_policy.py` - Thesis-safe optional feature policy and dataset allowlists
- `src/data/canonical.py` - CanonicalInteractions format
- `src/data/graph_builder.py` - Graph construction (build_graph)
- `src/data/negative_sampler.py` - NegativeSampler
- `src/utils/interaction_indexing.py` - Contiguous user/item ID remapping and max-normalized popularity

## Graph Construction Methods
| Method | Function | When to Use |
|--------|----------|-------------|
| `"dense"` | Bipartite edges from training | Default, no embeddings needed |
| `"knn"` | Bipartite + kNN edges | After first epoch (needs embeddings) |
| `"cagra"` | Bipartite + CAGRA ANN | Large-scale, needs RAPIDS cuVS |

## Paper Sources
| Decision | Source |
|----------|--------|
| No self-loops (loop=False) | LightGCN section 3.1 |
| 80/10/10 temporal split | FMMRec, DICE |
| CAGRA ANN acceleration | NVIDIA CAGRA 2024 |

## Quick Reference
```python
from src.data.loaders import load_dataset
from src.data.graph_builder import build_graph

canonical = load_dataset(config.dataset, config.data_dir)
data = build_graph(canonical, config, embeddings=None)  # dense
data = build_graph(canonical, config, embeddings=model.get_stacked_embeddings())  # knn/cagra
```

For capped smoke or tiny-validation runs, pass `max_rows=...` through the experiment path. The shared loader now reuses capped loads in-process, so the validator can stay aligned with feature-enabled formal runs without rescanning the same dataset files for every recipe.

## Current Data Notes
- `build_graph()` now carries canonical `user_features`, `item_features`, and `metadata` onto the PyG `Data` object when they exist.
- The current feature-aware model path is item-feature-first: `data.item_features` and `data.popularity` are passed into `UCaGNN`, while user features remain available for later extensions.
- `load_dataset(..., max_rows=...)` now caches capped loads in-process, so tiny validation can reuse the same feature-enriched canonical dataset across many recipe cases instead of rescanning the same files repeatedly.
- The shared loader path now also accepts `feature_policy=...`. `thesis_default` is the default and enforces the thesis-safe allowlist; `all_optional` restores the broader optional side-feature scans for explicit ablations. The shared policy definitions live in `src/data/feature_policy.py` so data-side rules stay with the loaders and audit tooling.
- Numeric CSV side-feature parsing shared by `kuairec_v2` and `kuairand1k` now lives in `src/utils/csv_features.py`; keep only dataset-specific feature-policy assembly inside the individual loaders.
- Keep only strictly mechanical loader helpers in `src/utils/dataset_loader_utils.py`, such as local raw-directory resolution and safe primitive field parsing. Helpers should return `None` on parse failure; loaders own the explicit fallback choice plus any malformed-value accounting or warning policy.
- Contiguous user/item ID remapping and max-normalized popularity now live in `src/utils/interaction_indexing.py`; keep dataset-specific parsing, labels, signs, and feature assembly inside the individual loaders.
- Under `thesis_default`, `kuairec_v2` keeps only safe item descriptor columns from `item_daily_features.csv` plus `item_categories.csv` and caption-category IDs, and `kuairand1k` keeps only safe descriptor columns from `video_features_basic_1k.csv`; both datasets stop loading deferred user-feature blocks by default.
- `src/data_exploration/data_information.py` now emits a causal feature audit in `data/datasets_information.md` that separates file availability, loader coverage, and current model consumption for interactions, user features, item features, and metadata.
- The dataset audit now also emits a per-column candidate-column table and can export a machine-readable JSON payload with `--audit-json ...` for feature-policy automation.
- Use the terms precisely: `optional features` are canonical side-feature matrices, `optional feature scans` are the extra raw-file reads needed to construct them, and `feature consumption` refers to whether the current model or evaluator actually uses those fields.
- Formal and tiny runs should use the same loader registry, canonical schema, feature toggles, and `feature_policy`; tiny runs may differ only through row caps, sampled interactions, and other runtime controls.
