# U-CaGNN Data Pipeline Skill

Use this skill when working on data loading, graph construction, negative sampling, or dataset handling.

## Key Files
- `docs/ucagnn_implementation/data-pipeline.md` - Data flow with paper cross-references
- `src/data/loaders.py` - Dataset loading (load_dataset)
- `src/utils/dataset_loader_utils.py` - Shared dataset-loader helpers for local-path resolution, safe primitive field parsing, and numeric downcasting
- `src/data/feature_policy.py` - Structured feature-safety registry and policy helpers
- `src/data/canonical.py` - CanonicalInteractions format
- `src/data/graph_builder.py` - Graph construction (build_graph)
- `src/data/negative_sampler.py` - NegativeSampler
- `src/utils/interaction_indexing.py` - Contiguous user/item ID remapping plus max-normalized/time-windowed popularity

## Graph Construction Methods
| Method | Function | When to Use |
|--------|----------|-------------|
| `"knn"` | Train bipartite edges + optional kNN edges | Portable fallback when `cuvs` is unavailable |
| `"cagra"` | Train bipartite edges + optional CAGRA ANN | Default method; falls back to kNN when `cuvs` is unavailable |

## Paper Sources
| Decision | Source |
|----------|--------|
| No self-loops (loop=False) | LightGCN section 3.1 |
| Per-user 80/10/10 temporal split default; global temporal remains available | FMMRec, DICE |
| CAGRA ANN acceleration | NVIDIA CAGRA 2024 |

## Quick Reference
```python
from src.data.loaders import load_dataset
from src.data.graph_builder import build_graph

canonical = load_dataset(config.dataset, config.data_dir)
data = build_graph(canonical, config, embeddings=None)  # train-interaction graph
data = build_graph(canonical, config, embeddings=model.get_stacked_embeddings())  # knn/cagra
```

Named preprocessing presets stay inside the same loader registry via `load_dataset(..., preprocessing_preset=...)`; use them instead of inventing new dataset aliases.

Without embeddings, both supported graph methods currently reduce to the train-interaction bipartite graph.

For capped smoke or tiny-validation runs, pass `max_rows=...` through the experiment path. The shared loader now reuses capped loads in-process, so the validator can stay aligned with feature-enabled formal runs without rescanning the same dataset files for every recipe.

## Current Data Notes
- `build_graph()` now computes `data.edge_norm` — precomputed full-graph symmetric degree normalization `1/sqrt(deg_u * deg_v)` for every edge.  Both `SubgraphSampler` (training subgraph path) and `Evaluator` (full-graph eval path) consume `edge_norm` to apply identical normalization, eliminating the train/eval degree-normalization inconsistency that arises when `LGConv` computes degrees from the local subgraph edge index.
- `SubgraphBatch` now carries `sub_edge_norm: torch.Tensor | None` — the per-edge precomputed norms for the sampled subgraph edges, aligned to `sub_edge_index`.  `SubgraphSampler` accepts `edge_norm` at construction time and subsets it during every `sample()` call.
- `build_graph()` now carries canonical `user_features`, `item_features`, `metadata`, and the new causal descriptors (`raw_target`, `behavior_type`, `exposure_flag`, `source_domain`, `feedback_type`, `preprocessing_preset`) onto the PyG `Data` object when they exist.
- `build_graph()` now recomputes `data.popularity` from the final training split rather than reusing all-interaction counts, so held-out validation/test rows never leak into popularity-driven losses, sampling, or evaluation metrics. `config.popularity_window_seconds` can further restrict that summary to the recent train window.
- The current feature-aware model path is item-feature-first: `data.item_features` and split-safe `data.popularity` are passed into `UCaGNN`, while user features remain available for later extensions.
- `load_dataset(..., max_rows=...)` now caches capped loads in-process, so tiny validation can reuse the same feature-enriched canonical dataset across many recipe cases instead of rescanning the same files repeatedly.
- Derived split resolution now defaults to `per_user_temporal` whenever a loader does not provide masks. Keep `global_temporal` only as an explicit opt-in for compatibility or analysis baselines.
- The shared loader path now also accepts `feature_policy=...` and `preprocessing_preset=...`. `thesis_default` is the default and promotes only `safe_pre_treatment` columns from the structured registry; `all_optional` restores the broader optional side-feature scans for explicit ablations.
- Numeric CSV side-feature parsing shared by `kuairec_v2` and `kuairand1k` now lives in `src/utils/csv_features.py`; that helper may narrow feature matrices to compact storage dtypes after loading, so keep only dataset-specific feature-policy assembly inside the individual loaders.
- Keep only strictly mechanical loader helpers in `src/utils/dataset_loader_utils.py`, such as local raw-directory resolution and safe primitive field parsing. Helpers should return `None` on parse failure; loaders own the explicit fallback choice plus any malformed-value accounting or warning policy.
- Contiguous user/item ID remapping and popularity helpers now live in `src/utils/interaction_indexing.py`; remapped IDs may be stored in narrower integer dtypes before `build_graph()` promotes them to `torch.long`, so keep dataset-specific parsing, labels, signs, preset semantics, and feature assembly inside the individual loaders.
- Under `thesis_default`, `kuairec_v2` keeps only safe item descriptor columns from `item_daily_features.csv` plus `item_categories.csv` and caption-category IDs, and `kuairand1k` keeps only safe descriptor columns from `video_features_basic_1k.csv`; both datasets stop loading deferred user-feature blocks by default.
- Loader semantics are now normalized at the canonical boundary: MovieLens keeps raw ratings, Taobao keeps behavior labels plus an ordinal raw target, KuaiRec keeps raw watch ratio plus matrix provenance, KuaiRand keeps randomized exposure plus behavior/domain labels, and Amazon-Book records the graph-only preset.
- `src/data_exploration/data_information.py` now emits a causal feature audit in `data/datasets_information.md` that separates file availability, loader coverage, and current model consumption for interactions, user features, item features, and metadata.
- The dataset audit now also emits a per-column candidate-column table and can export a machine-readable JSON payload with `--audit-json ...` for feature-policy automation.
- Use the terms precisely: `optional features` are canonical side-feature matrices, `optional feature scans` are the extra raw-file reads needed to construct them, and `feature consumption` refers to whether the current model or evaluator actually uses those fields.
- Formal and tiny runs should use the same loader registry, canonical schema, feature toggles, and `feature_policy`; tiny runs may differ only through row caps, sampled interactions, and other runtime controls.
