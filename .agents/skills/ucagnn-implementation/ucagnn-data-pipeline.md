# U-CaGNN Data Pipeline

Use this file for the live data contract: loader registry, canonical interactions, feature policy, graph construction, and sampling.

## Key files

- `.agents/skills/ucagnn-implementation/ucagnn-data-pipeline.md`
- `src/data/loaders/_registry.py`
- `src/data/canonical.py`
- `src/data/feature_policy.py`
- `src/data/graph_builder.py`
- `src/data/subgraph_sampler.py`
- `src/data/negative_sampler.py`
- `src/utils/csv_features.py`
- `src/utils/interaction_indexing.py`

## Runtime path

```mermaid
flowchart LR
    A[load_dataset] --> B[CanonicalInteractions]
    B --> C[build_graph embeddings_none]
    C --> D{graph_policy}
    D -->|observed| E[Runtime graph]
    D -->|cagra_augmented| F[Bootstrap embeddings]
    F --> G[build_graph embeddings_present]
    G --> E
```

Boundary: loaders emit `CanonicalInteractions`; graph build starts from observed train positives; CAGRA augments only after observed-graph bootstrap embeddings exist.

## Loader boundary

- `load_dataset(...)` is the public loader surface.
- Default preprocessing presets are resolved in `src/data/loaders/_registry.py`.
- Full loads are uncached. Capped loads (`max_rows` set) are cached in-process so tiny validation can reuse the same canonical dataset.
- `feature_policy` and `preprocessing_preset` cross the same loader boundary as the dataset name. A preprocessing preset may own a stricter or broader feature policy; the registry resolves that before calling the concrete loader.

### Repository preprocessing defaults

| Dataset | Default preset | Important alternatives |
| --- | --- | --- |
| `movielens1m` | `movielens_explicit` | none |
| `movielens20m` | `movielens_explicit` | `movielens_explicit_dense_genome` |
| `taobao` | `taobao_multibehavior` | `taobao_multibehavior_raw` |
| `kuairec_v2` | `kuairec_watchratio` | `kuairec_watchratio_raw`, `kuairec_fullobs` |
| `amazonbook` | `amazonbook_graph_only` | none |
| `kuairand1k` | `kuairand_causal` | `kuairand_random_only` |

Dataset semantic notes:

| Dataset/view | Live contract |
| --- | --- |
| `kuairec_watchratio` | default KuaiRec; `big_matrix`; watch-ratio ranking signal |
| `kuairec_fullobs` | explicit `small_matrix` comparison; near-full observation; not default ranking story |
| `kuairand_causal` | default KuaiRand; standard + randomized exposure rows |
| `kuairand_random_only` | explicit randomized-exposure diagnostic view |
| `amazonbook_graph_only` | graph-only default; no side-feature thesis story |

KuaiRec matrix owner: preprocessing preset. Direct `matrix_variant=...` exists only for loader compatibility and must not conflict with an explicit preset.

## `CanonicalInteractions`

| Group | Fields |
| --- | --- |
| Core arrays | `user_id`, `item_id`, `label`, `timestamp`, `sign`, `popularity` |
| Maps and sizes | `n_users`, `n_items`, `user_map`, `item_map` |
| Optional side info | `user_features`, `item_features` |
| Causal descriptors | `raw_target`, `behavior_type`, `exposure_flag`, `source_domain`, `feedback_type`, `preprocessing_preset` |
| Repeat-collapse summaries | `repeat_count`, `repeat_*`, optional `repeat_behavior_counts`, `repeat_behavior_labels` |
| Split metadata | optional `train_mask`, `val_mask`, `test_mask`, plus `metadata` |
| Propensity supervision | optional `item_propensity_targets` with shape `(n_items,)` |

Split/repeat/sampling rules:

| Owner | Contract |
| --- | --- |
| `get_splits()` | prefer loader masks; else derive val from train/test; else configured derived split |
| `compute_item_recency()` | training split only |
| train-derived summaries | reuse same final train mask; no category-specific splits |
| repeat collapse | one raw user-item pair cannot span train/val/test |
| retained repeat row | max priority, timestamp tie-break |
| repeat summaries | preserve count, mean, max, latest, first/last timestamps, behavior counts/labels |
| `sample_canonical_interactions()` | tiny-run sampling owner |
| tiny samples | preserve split coverage; remap IDs; recompute popularity; slice all aligned fields |

## Feature policy

- `thesis_default` loads only safe pre-treatment features from the structured registry.
- `all_optional` keeps exploratory sources such as proxy-only feature files.
- Post-treatment aggregates stay out of thesis-default model features.
- Under `thesis_default`, KuaiRand's `video_features_statistic_1k.csv` stays excluded from model features, but its `show_cnt` column is reused separately as a propensity calibration target.
- Free-text or comment-style columns are not part of the live thesis-default path; the runtime uses only structured numeric, temporal, and categorical-safe features.

`src/utils/csv_features.py` encoding policy:

| Field type | Encoding |
| --- | --- |
| repeated entity rows | first source row wins before encoding |
| numeric/temporal | min-max to `[0,1]` within source |
| categorical-like | deterministic codes in `[0,1]`; `0` reserved for missing |
| embedding-time buffers | normalized again before context head |

Purpose: daily files cannot overwrite thesis-default static descriptors or reintroduce raw timestamp/ID scale.

## Graph construction

| `graph_policy` | Path | Current behavior |
| --- | --- | --- |
| `observed` | `load_runtime_data()` -> `build_graph(..., embeddings=None)` | Uses only train-split interaction edges. |
| `cagra_augmented` | `load_runtime_data()` -> bootstrap embeddings -> `build_graph(..., embeddings=...)` | Keeps observed train edges and adds neutral ANN edges from CAGRA. |

Current graph rules:

- `build_graph()` attaches original observed split masks plus label-aware `*_positive_mask` fields.
- Interaction graph edges, BPR training positives, and train-time popularity use only positive training labels.
- Original observed masks remain available for seen-item exclusion and split bookkeeping.
- `build_graph()` always recomputes `data.popularity` from positive rows in the final training split.
- `build_graph()` precomputes `data.edge_norm` once, so training and evaluation share the same degree normalization.
- Optional canonical payloads are copied onto the PyG `Data` object through one shared boundary helper.
- `cagra_augmented` is strict: it requires item features and raises on CAGRA failures instead of silently degrading.

## Train-derived user history and exposure context

- `build_recent_train_history()` creates `recent_train_items` and `recent_train_mask` from the final training split only.
- Those buffers are per-user histories: the latest training interactions for each user, never global "recent" or popularity-only items.
- Subgraph training reuses the same train-derived user history and does not create separate splits for interest, recency, or context.

Propensity target contract:

| Dataset | Field | Source | Default scorer use |
| --- | --- | --- | --- |
| `kuairand1k` | `item_propensity_targets` | log1p-normalized `show_cnt` | zero-filled unless calibrated IPW active |
| all others | `None` | none | inactive |

`show_cnt` is post-treatment. It can calibrate explicit IPW, but it must not affect default scoring.

## Sampling

- `NegativeSampler` is vectorized and mixes uniform and popularity-weighted draws via `hard_negative_ratio`.
- It receives train-positive `(user, item)` pairs from `TrainerRuntime` and filters sampled negatives against every known positive training item for the same user, not only the current positive item.

DICE sampling:

| Item | Contract |
| --- | --- |
| Strategy | `negative_sampling_strategy="dice"` |
| Metadata | `sample_with_metadata()` returns aligned high-popularity mask |
| U-CaGNN large batch | DICE high/low routing, then vectorized known-positive filtering |
| `dice_paper` | exact per-user positive-count correction retained |
| Consumer | `LossSuite` consumes mask directly |
| Fallback | threshold reconstruction only for older/manual payloads |

- `SubgraphSampler` extracts sampled k-hop subgraphs with per-hop fan-out limits from `num_neighbors`.
- `SubgraphBatch` carries:
  - `sub_edge_index`, `sub_edge_sign`, `sub_edge_norm`,
  - global user and item ids for metadata lookup,
  - local user, positive-item, and negative-item ids for scoring and loss computation,
  - optional `dice_negative_mask` batch metadata for DICE-style branch losses.
