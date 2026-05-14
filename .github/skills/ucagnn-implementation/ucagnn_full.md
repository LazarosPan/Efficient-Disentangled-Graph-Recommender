# U-CaGNN Implementation Documentation

U-CaGNN (Unified Causal Graph Neural Network) is a recommendation system that disentangles user interest from popularity conformity using dual-branch graph convolution, fused scoring, train-split item metadata, and branch-local auxiliary losses. The codebase is built on PyTorch and PyTorch Geometric.

---

## Module Dependency Diagram

```
                    +------------------+
                    |  UCaGNNConfig    |  (controls everything)
                    +--------+---------+
                             |
          +------------------+------------------+
          |                  |                  |
  +-------v--------+  +------v------+  +--------v--------+
  | Data Pipeline  |  |   Models    |  |   Loss Suite    |
  | (loaders,      |  | (A,B,C,F)   |  | (fused + aux)   |
  | graph, sampler)|  +------+------+  +--------+--------+
  +-------+-------+         |                   |
             |           +------v------+           |
             +---------->| MiniBatch   |<----------+
               |  Trainer    |
               +------+------+
                             |
                      +------v------+
                      | GPU Profiler|
                      +-------------+
```

---

## Navigation Guide

| I want to understand... | Read |
|------------------------|------|
| System overview, data flow, tensor shapes | [architecture.md](architecture.md) |
| Embedding, GCN, scoring, propensity, orchestrator | [models.md](models.md) |
| Fused BPR, branch BPR, independence, within-branch contrastive, optional DirectAU, popularity losses | [losses.md](losses.md) |
| Dataset loaders, canonical format, graph builder, negative sampler | [data-pipeline.md](data-pipeline.md) |
| Training loop, evaluation metrics, GPU profiling | [training.md](training.md) |
| Validation and formal run entry points | [training.md#quick-validation-workflow](training.md#quick-validation-workflow) |
| Mini-batch training runtime | [training.md#minibatchtrainer](training.md#minibatchtrainer) |
| Every config parameter with defaults | [config-reference.md](config-reference.md) |
| How to switch between LightGCN / DICE / full U-CaGNN | [config-reference.md#presets](config-reference.md#presets) |
| Which config toggle controls which module | [architecture.md#ablation-map](architecture.md#ablation-map) |
| Design decision justifications | [theoretical_justifications.md](theoretical_justifications.md) |

---

## Recommended Reading Order

1. **architecture.md** -- System overview, data flow, tensor shapes, ablation map
2. **data-pipeline.md** -- Loaders, canonical format, graph builder, negative sampler, subgraph sampler
3. **models.md** -- Neural network modules (A-F) and UCaGNN orchestrator
4. **losses.md** -- Fused BPR, branch BPR, independence, within-branch contrastive, optional DirectAU, popularity losses
5. **training.md** -- Mini-batch training loop, evaluation, GPU profiling, and validation workflow
6. **config-reference.md** -- Every config parameter with defaults and paper evidence
7. **theoretical_justifications.md** -- Formal reasoning for design choices

Implementation preference: extend existing entry points and collapse duplication before adding parallel scripts or public surfaces.

Depth ownership is now explicit in `UCaGNNConfig`: `single_branch_gnn_layers` controls the LightGCN / non-dual-branch path, while `interest_gnn_layers` and `conformity_gnn_layers` control the dual-branch U-CaGNN path. `max_gnn_layers` remains the derived sampler-facing depth used to validate `num_neighbors`, and the base dual-branch default is now the matching two-hop fan-out `[10, 5]`.

Current data defaults are no longer purely incidental: derived splits default to `per_user_temporal`, loaders may record a `preprocessing_preset`, and graph-time popularity can optionally be restricted with `popularity_window_seconds` while remaining train-split-only. Mechanical loader downcasting now also has one shared bulk helper in `src/utils/dataset_loader_utils.py`, so repeated array extraction can reuse the same narrowing policy without duplicating one-call-per-column boilerplate. The graph-boundary tensor conversion in `src/data/graph_builder.py` also guards against read-only NumPy payloads by copying only when needed before `torch.from_numpy(...)`, which keeps Polars-backed canonical arrays from emitting non-writable-tensor warnings during validation runs.

The graph builder now keeps ANN augmentation explicitly aligned with the reproducibility contract: kNN/CAGRA edges are mirrored so the graph remains undirected, the CAGRA path converts embeddings through CuPy/CUDA-array inputs, threads `config.seed` into `SearchParams.rand_xor_mask`, and falls back to kNN when CAGRA import, build, or search fails at runtime.

---

## Theoretical Lineage

Each U-CaGNN component traces to a specific paper. This table maps every major design choice to its published origin:

| Component | Paper Origin | Key Insight Extracted |
|-----------|-------------|----------------------|
| GCN backbone | LightGCN (He et al., 2020) | No transforms, no activations, symmetric D^{-1/2}AD^{-1/2} norm, uniform alpha layer combination |
| Dual user embeddings | DICE (Zheng et al., 2020) | Collider-based interest/conformity disentanglement; +31% Recall@20 on ML-10M |
| Sign-aware edge weighting | SIGformer (2024) | Negative edges critical (-21.5% Recall@20 without); extracted as 2 learnable scalars |
| Adaptive fusion gate | MGCE / repository synthesis | User-conditioned mixing over interest, conformity, and popularity scores |
| Within-branch contrastive | Contrastive causal-recys synthesis | Batch-safe branch-local InfoNCE on aligned positive user-item pairs |
| DirectAU alignment/uniformity | DirectAU (2023) | Optional branch-local geometry regularization without replacing BPR |
| Counterfactual scoring | CausE (2018) / MCLN (2023) | Treatment effect = interest - conformity; kept as a diagnostic score |
| Popularity loss + recency metadata | DICE (2020) / DDCE (2023) / repository synthesis | Anchor popularity-aware scoring with train-split item popularity and recency summaries |
| Curriculum scheduling | CaDCR (2025) | Phase in losses progressively by difficulty; multi-task curriculum learning |
| IPW debiasing | Survey consensus (S1-S4, 2023-2025) | Propensity reweighting with clipping to address high-variance instability |
| CAGRA graph construction | CAGRA (2024) | GPU-accelerated ANN; 33-77x batch speedups over HNSW; out_degree 32-96 |

---

## Source File Map

```
src/
  utils/
    __init__.py              Exports UCaGNNConfig
    config.py                UCaGNNConfig dataclass (all hyperparameters + presets)
    cli_parsers.py           Shared argparse constants plus parser/helper builders for experiments, scripts, and data-exploration entry points
    csv_features.py          Shared mixed-type CSV side-feature loader plus feature-block stacking helpers
    interaction_indexing.py  Shared contiguous ID remapping plus max-normalized or time-windowed popularity helpers

  data/
    __init__.py              Exports CanonicalInteractions, build_graph, NegativeSampler, SubgraphSampler, SubgraphBatch
    canonical.py             CanonicalInteractions dataclass (universal dataset format, split helpers, causal descriptors)
    feature_policy.py        Structured feature-safety registry plus thesis-default/all-optional policy helpers
    graph_builder.py         build_graph(): canonical -> PyG Data (dense/knn/cagra)
    negative_sampler.py      NegativeSampler: uniform + popularity-weighted sampling
    subgraph_sampler.py      SubgraphSampler + SubgraphBatch: k-hop subgraph extraction for mini-batch training
    loaders/
      __init__.py            Re-exports from _registry.py (docstring + public API only)
      _registry.py           LOADERS dict + load_dataset() dispatcher with preprocessing preset support
      movielens1m.py         ML-1M from local raw files (ratings -> label/sign + stable side features)
      movielens20m.py        ML-20M from CSV (20M rows, numpy fast path + raw-inferred genres)
      amazonbook.py          Amazon-Book from local LightGCN raw files (implicit positive interactions)
      taobao.py              Taobao UserBehavior (behavior-typed: buy/cart/fav/pv)
      kuairec_v2.py          KuaiRec v2 (watch-ratio based, with side features)
      kuairand1k.py          KuaiRand-1K (click/like/follow/hate signals, with side features)

  models/
    __init__.py              Exports UCaGNN
    embeddings.py            Module A: EmbeddingModule (dual user + item + popularity)
    lightgcn.py              Module B: LightGCNBranch + DualBranchGCN (sign-aware)
    scoring.py               Module C: ScoringModule (interest + conformity + popularity + fused scoring)
    propensity.py            Module F: PropensityEstimator (2-layer MLP for IPW)
    ucagnn.py                UCaGNN orchestrator (forward + forward_subgraph + cached/full-catalog scoring)

  losses/
    __init__.py              Exports LossSuite
    loss_suite.py            LossSuite: fused BPR + branch-local auxiliary losses, within-branch contrastive, optional DirectAU, and popularity supervision

  training/
    __init__.py              Exports MiniBatchTrainer, Evaluator, thesis metric constants
    mini_batch_trainer.py    MiniBatchTrainer: CUDA-first sampled subgraphs + local GCN per batch
    evaluator.py             Evaluator: batched Recall@K + NDCG@K via PyG metrics with device-side graph caching

  profiling/
    __init__.py              Exports GPUProfiler, profile_stage
    gpu_profiler.py          Stage-level timing + VRAM tracking + PyG model/data summaries

  data_exploration/
    data_exploration.py      Dataset analysis utilities (not part of training pipeline)
    explore_all_datasets.py  Full-dataset canonical-loader visualizer with thesis-facing labels that rewrites benchmark overview + per-dataset profiles
    data_information.py      Dataset metadata and statistics

scripts/
  quick_validate.py         Single tiny-scale validation entry point with shared tiny recipe-config scaffolding
  query_results.py          SQLite-first result inspection CLI with named-column output
  cleanup_experiment_artifacts.py  Reset MLflow artifacts, checkpoints, and formal-run state
  reset_experiment_db.py    Reset the thesis SQLite database

experiments/
  run_experiment.py         Main single-run entry point; supports sampled preflight runs
  run_benchmark.py          Formal matrix runner and formal-run entry point across dataset x preset
  run_ablation.py           Ablation runner over named component variants
  experiment_catalog.json   Declarative formal recipe catalog
```

---

## Formal Experiment Workflow

The formal experiment matrix is defined along two semantic axes. Profile-owned `scoring_weight_modes` can add explicit score-mix comparisons when needed, but the default day-to-day `ucagnn` profile now keeps learned fusion only:

1. `dataset`
2. `preset`
Support parameters such as `batch_size`, `epochs`, and `num_neighbors` are not formal matrix axes. They remain config-level values that should be checked through `quick-validate` before long runs begin. Formal profiles may still sweep several support-parameter fan-out shapes through a JSON-safe `num_neighbors_options` field (for example `[[10, 5], [5, 3]]`); the benchmark runner expands those into separate resolved runs while each run still keeps one concrete `config.num_neighbors` vector.

Use the orchestration layer as follows:

```bash
uv run formal-run
uv run quick-validate
uv run experiment --list-recipes
```

`formal-run` is the default thesis entry point. `quick-validate` is the default tiny-scale validation entry point. It exercises the canonical recipe matrix, ablation variants, and representative observability/evaluation probes so implementation changes are checked against the same mini-batch runtime path used in formal runs. MLflow stays off by default there; pass `--mlflow` only when you explicitly want the MLflow probe. Repository pytest collection is scoped to `tests/` so vendored reference code under `external/` does not interfere with the main validation workflow.

The orchestration layer now treats `build_config()` as the single config-assembly contract for both CLI namespaces and script-built mapping inputs. Benchmark and quick-validation code should pass plain dictionaries into `build_config()` instead of manufacturing intermediate `argparse.Namespace` adapters, the formal-run resume path should keep `benchmark_args` in that normalized dict form instead of rebuilding internal Namespace transport objects, and `build_config()` itself should normalize namespace-like inputs once at the boundary instead of stacking generic field-access wrapper helpers on top of both dict and Namespace callers. The benchmark/formal-run path now shares that contract explicitly: config-bearing benchmark payload fields are normalized once through `normalize_benchmark_config_overrides(...)`, each concrete matrix item rebuilds its run-local config input dict through `build_benchmark_config_inputs(...)`, quick validation now reuses the same runtime config-input assembly via `build_runtime_config_inputs(...)` before handing the mapping back to `build_config()`, both config-input builders share one internal present-field helper so runtime and benchmark wiring stay aligned, and benchmark-only exclusions such as `num_neighbors_options` remain explicit in one place. Downstream helpers such as `build_benchmark_plan()` and `run_benchmark()` should consume an already-normalized formal-run payload directly instead of renormalizing it. Presets now apply before explicit CLI/profile overrides, so changing branch depths or fan-out really changes the resolved runtime config and checkpoint identity rather than silently snapping back to preset defaults. When a formal profile carries a deeper `num_neighbors` bundle than a shallower preset needs, the benchmark runner now trims that vector to the preset's active depth instead of failing config validation for baselines such as LightGCN. Saved formal-run state is intentionally strict: unexpected saved fields or removed graph methods should force a fresh formal run instead of reviving compatibility shims. Formal profile alias normalization and catalog lookup belong in `experiments/recipes.py`, with the current implementation resolving formal profiles in one cached normalization pass rather than layering separate helper wrappers for each normalization step. Canonical recipe filtering also lives there now, so `quick_validate.py` reuses `recipe_names(include_aliases=False)` instead of reading the catalog directly. Semantic plan matching in `experiments/run_benchmark.py` should be derived from the normalized `benchmark_args` payload with runtime-only overrides excluded.

Checkpoint resume now uses an explicit identity split. `training_identity` / `training_hash` encode every training-defining field that must match for safe resume, while `evaluation_identity` / `evaluation_hash` capture same-checkpoint evaluation settings such as `eval_scoring_mode` and `eval_ks`. The default checkpoint filename includes `training_hash`, so runs with different training semantics do not collide under one canonical experiment stem. Evaluation-only overrides should therefore remain compatible with the same checkpoint, but changing any training-defining field must force a new checkpoint path or raise when the user explicitly points at an incompatible existing path. Use `--overwrite-checkpoint` when you intentionally want to delete and replace an existing checkpoint.

Experiment-level CLI parsers live in `experiments/cli_parsers.py` (`build_run_experiment_parser`, `build_benchmark_parser`, `build_formal_run_parser`, `build_ablation_parser`). Utility-script parsers live in `src/utils/cli_parsers.py`. Command files import their parser directly from `experiments/cli_parsers.py` with no thin `build_parser()` facade wrappers. Shared parser constants and reusable argument-group helpers stay centralized as well: benchmark dataset/tier definitions plus dataset-selector normalization and expansion helpers live in `src/utils/cli_parsers.py`, benchmark and ablation now share one `add_execution_tracking_group(...)` helper for the standard device/data-dir plus batch-execution flags, LR-scheduler choices stay in `src/utils/config.py`, repository-local results/checkpoint path constants now live in `src/utils/project_paths.py`, and small orchestration helpers such as CLI logging setup, batch-id generation, summary counters, and metric fallback logic belong in `scripts/_workflow_helpers.py`. Prefer extending the appropriate centralized module over adding local `add_argument(...)` blocks, keep choice validation in those centralized parsers instead of re-validating the same values inside entry-point `main()` functions, and invoke the experiment commands through the packaged entry points (`uv run experiment`, `uv run formal-run`, `uv run ablation`) instead of relying on repo-root `sys.path` bootstrapping from direct file execution.

## Thesis Evaluation Focus

The runtime continues to log the full PyG link-prediction bundle, but thesis-facing reporting should stay narrow and interpretable:

- `NDCG@20` and `Recall@20` cover visible-list ranking quality.
- `AveragePopularity@20` tracks whether visible-list gains come from pushing more popular items; lower values are better.
- `NDCG@40` and `Recall@40` are the deeper-list robustness cutoffs.
- `AveragePopularity@40` tracks whether debiasing survives deeper into the ranked list.
- Formal score-mix sweeps should keep `lightgcn` and `dice_like` on the fixed path, while `ucagnn` may run both learned and fixed mixing as an explicit ablation of the proposed model.

Mechanism checks should compare the same checkpoint under `default`, `interest_only`, and `conformity_suppressed` scoring. This is the repository's main causal-behavior evaluation path. PropCare-style causal-uplift metrics remain optional future work unless a separate treatment/propensity/effect evaluation contract is added.


# Architecture

High-level design of the U-CaGNN system: module interaction, data flow through a training iteration, tensor shape ledger, and ablation map.

---

## System Diagram

```
                          +-----------+
                          | Raw Data  |  (MovieLens, Taobao, KuaiRec, ...)
                          +-----+-----+
                                |
                      CanonicalInteractions   <-- src/data/loaders/*.py
                                |
                    +-----------+-----------+
                    |                       |
              build_graph()          NegativeSampler
           src/data/graph_builder.py    src/data/negative_sampler.py
                    |                       |
              PyG Data object         neg_item_ids per batch
                    |                       |
        +-----------+-----------+-----------+
        |                                   |
      +-----v------+                     +------v------+
      |  Module A   |                     | MiniBatch   |
      | Embeddings  |                     |   Trainer   |
      +-----+------+                     +------+------+
        |                                   |
        | init_embs dict                    | batches (user, pos, neg)
        |                                   |
  +-----v------+                            |
  |  Module B   | <---- edge_index, edge_sign
  | DualBranch  |
  |   GCN       |
  +-----+------+
        |
        | propagated dict
        |
  +-----v------+
  |  Module C   | <---- user_ids, item_ids
  |  Scoring    |
  +-----+------+
        |
        | pos_scores, neg_scores dicts
        |
  +-----v------+        +----------+
  |  Module F   |------->| LossSuite|
  | Propensity  |  ipw   | (5 loss  |
  +-------------+  wts   |  terms)  |
                         +----+-----+
                              |
                         total loss
                              |
                         optimizer.step()
```

---

## Module Labels (Thesis Notation)

| Label | Module | Class | File |
|-------|--------|-------|------|
| A | Embeddings | `EmbeddingModule` | `src/models/embeddings.py` |
| B | GCN Propagation | `DualBranchGCN` / `LightGCNBranch` | `src/models/lightgcn.py` |
| C | Scoring | `ScoringModule` | `src/models/scoring.py` |
| F | Propensity (IPW) | `PropensityEstimator` | `src/models/propensity.py` |
| -- | Orchestrator | `UCaGNN` | `src/models/ucagnn.py` |
| D | Loss Suite | `LossSuite` | `src/losses/loss_suite.py` |

---

## Synthesis Rationale

U-CaGNN is not a single paper's method -- it deliberately synthesizes techniques from multiple published works, each contributing a specific capability:

- **LightGCN (He et al., 2020)** provides the GCN backbone. The paper proved that feature transforms "negatively increase the difficulty for model training" and nonlinear activations have "no positive effect on collaborative filtering." This justifies the parameter-free normalized adjacency propagation used in the repository.
- **DICE (Zheng et al., 2020)** provides dual interest/conformity disentanglement via a collider causal model (Interest -> Rating <- Popularity). GCN-DICE achieved +31% Recall@20 over single-embedding baselines.
- **SIGformer (2024)** provides the insight that negative edge information is critical (-21.5% Recall@20 without it). U-CaGNN extracts this as lightweight sign-aware edge weighting (2 learnable scalars) without SIGformer's expensive spectral eigendecomposition.
- **DCCL (2023)** provides the contrastive regularization (NT-Xent) for separating semantic intents, ensuring dual branches encode distinct but user-consistent signals.
- **CaDCR (2025)** provides the curriculum scheduling framework -- progressive multi-task learning that phases in losses by difficulty.
- **CausE (2018) / MCLN (2023)** provide counterfactual scoring: the treatment effect `Y_interest - Y_conformity` measures genuine causal preference.
- **CAGRA (2024)** provides GPU-accelerated approximate nearest neighbor graph construction (33-77x speedups over HNSW) for scalable embedding-space edge building.

The thesis contribution is the *unified architecture* that combines these techniques under a single config-driven framework, enabling controlled ablation of each component.

---

## Data Flow: One Training Iteration

1. **Batch construction** -- `MiniBatchTrainer` shuffles training interactions, slices a batch of `(user_ids, pos_item_ids)`, then calls `NegativeSampler.sample()` to get `neg_item_ids`.

2. **Forward pass** (`UCaGNN.forward`):
      - **A: Embeddings** -- `EmbeddingModule.get_all_embeddings()` returns a dict of raw embedding weight matrices (pre-GNN). When `use_features=True` and `item_features` exist, this stage also builds `item_interest` and `item_conformity` from projected canonical item features and popularity-conditioned modulation.
   - **B: GCN** -- `DualBranchGCN.forward()` concatenates user + item embeddings, builds one sparse normalized adjacency matrix from `edge_index` + optional sign-aware weights, propagates through repeated sparse adjacency matmuls, then splits the output back into user/item tensors. Returns `propagated` dict.
   - **C: Scoring** -- `ScoringModule.forward()` looks up user/item embeddings from `propagated`, computes dot-product scores: `interest`, `conformity`, `cf`, `final`.
   - **F: Propensity** -- `PropensityEstimator.forward()` maps item embeddings through a 2-layer MLP to produce propensity scores; `UCaGNN` computes `1 / propensity` inline as IPW weights.

3. **Loss computation** (`LossSuite.forward`):
   - Computes up to 5 loss terms (L_rec, L_ortho, L_contr, L_cf, L_pop), each gated by config lambdas and curriculum phase.
   - Returns weighted sum as `total`.

4. **Backward + optimizer step** -- standard PyTorch `loss.backward()` + `Adam.step()`.

5. **Evaluation** -- after each epoch, `Evaluator.evaluate()` propagates the full graph once, scores all items for validation users in batches, masks observed non-target items, and computes Recall@K and NDCG@K via PyG metrics.

### Mini-Batch Runtime

The repository now runs only the mini-batch path:

- Step 1 includes `SubgraphSampler.sample()` to extract a k-hop subgraph around each batch's nodes.
- Step 2 calls `UCaGNN.forward_subgraph()` which runs GCN on the local subgraph edges using the precomputed full-graph normalization factors carried into the sampled subgraph.
- A background CPU prefetch pipeline overlaps negative sampling and subgraph extraction with the current GPU forward/backward pass.

---

## Tensor Shape Ledger

Symbols: `U` = n_users, `I` = n_items, `N = U + I`, `B` = batch_size, `D` = embed_dim, `E` = n_edges.

| Stage | Tensor | Shape | Dtype | Source |
|-------|--------|-------|-------|--------|
| Input | `edge_index` | `(2, E)` | long | `build_graph` |
| Input | `edge_sign` | `(E,)` | float32 | `build_graph` (aligned to `edge_index`) |
| Input | `user_ids`, `pos_item_ids`, `neg_item_ids` | `(B,)` | long | Trainer batch |
| Embedding (A) | `user_interest.weight` | `(U, D)` | float32 | `nn.Embedding` |
| Embedding (A) | `user_conformity.weight` | `(U, D)` | float32 | `nn.Embedding` (dual only) |
| Embedding (A) | `item_embed.weight` | `(I, D)` | float32 | `nn.Embedding` |
| Embedding (A) | `item_pop.weight` | `(I, D_pop)` | float32 | optional popularity emb |
| Embedding (A) | `item_features` | `(I, F_i)` | float32 | optional canonical item features |
| Embedding (A) | `item_interest` | `(I, D)` | float32 | fused ID + projected feature view |
| Embedding (A) | `item_conformity` | `(I, D)` | float32 | fused ID + popularity-modulated feature view |
| GCN input (B) | `x_int = cat(user_interest, item)` | `(N, D)` | float32 | concat for propagation |
| GCN output (B) | `propagated["user_interest"]` | `(U, D)` | float32 | post sparse propagation |
| GCN output (B) | `propagated["item_interest"]` | `(I, D)` | float32 | post sparse propagation |
| GCN output (B) | `propagated["user_conformity"]` | `(U, D)` | float32 | post sparse propagation (dual only) |
| GCN output (B) | `propagated["item_conformity"]` | `(I, D)` | float32 | post sparse propagation (dual only) |
| Edge weights (B) | `edge_weight` | `(E,)` or `None` | float32 | sign-aware weighting |
| Scoring (C) | `interest_score` | `(B,)` | float32 | dot product |
| Scoring (C) | `conformity_score` | `(B,)` | float32 | dot product (dual only) |
| Scoring (C) | `counterfactual_score` | `(B,)` | float32 | interest - conformity |
| Scoring (C) | `final_score` | `(B,)` | float32 | weighted sum |
| Propensity (F) | `ipw_weights` | `(B,)` | float32 | 1/P(exposure) |
| Eval matrix | `score_users_from_propagated` output | `(B_eval, I)` | float32 | cached full-catalog scoring |

---

## Ablation Map

Each config toggle enables/disables specific modules and losses. Setting a toggle to `False` (or lambda to `0.0`) cleanly removes that component.

| Config Toggle | Modules Affected | Losses Affected | Default |
|--------------|------------------|-----------------|---------|
| `use_dual_branch` | A (dual user embs), B (two GCN branches), C (dual scoring) | L_ortho, L_contr, L_cf, L_pop | `True` |
| `use_sign_aware` | B (alpha_pos/alpha_neg edge weights) | -- | `True` |
| `use_counterfactual` | C (counterfactual_score = interest - conformity) | L_cf | `True` |
| `use_ipw` | F (PropensityEstimator) | L_rec (IPW-weighted BPR) | `True` |
| `use_popularity_emb` | A (item_pop embedding) | -- | `True` |
| `lambda_ortho > 0` | -- | L_ortho | `0.02` |
| `lambda_contr > 0` | -- | L_contr | `0.1` |
| `lambda_cf > 0` | -- | L_cf | `0.08` |
| `lambda_pop > 0` | D (PopularityPredictor MLP) | L_pop | `0.15` |

### Presets

| Preset | Equivalent to | Paper Baseline | Graph | Losses |
|--------|---------------|----------------|-------|--------|
| `preset_lightgcn()` | Non-causal LightGCN baseline | He et al. 2020 vanilla LightGCN | dense | L_rec only |
| `preset_dice_like()` | DICE-inspired dual-branch | Zheng et al. 2020 GCN-DICE | knn | L_rec + L_ortho + L_pop |
| `preset_full()` | Full U-CaGNN | U-CaGNN thesis contribution | knn | All 5 losses |

---

## PyG Integration Points

| Feature | PyG Built-in | Custom Code | Why |
|---------|-------------|-------------|-----|
| GCN layer | Sparse adjacency matmul over PyTorch COO tensors | `LightGCNBranch` wraps repeated propagation + alpha combination | The branch uses the same LightGCN normalized adjacency update as `LGConv`, but avoids gather-scatter message materialization by reusing one sparse adjacency matrix across all propagation layers. Self-connections remain omitted because alpha layer combination captures their effect. |
| BPR loss | `BPRLoss` from `torch_geometric.nn.models.lightgcn` | `bpr_loss()` wraps it + adds IPW extension | IPW weighting is U-CaGNN-specific |
| kNN graph | `knn_graph` from `torch_geometric.nn` | `_build_knn()` combines bipartite + kNN | Need bipartite interaction edges always present |
| Metrics | `LinkPredRecall`, `LinkPredNDCG` | `Evaluator` orchestrates batched eval | PyG metric API does the heavy math |
| Profiling | `count_parameters`, `get_model_size`, `get_data_size` | `GPUProfiler` adds stage timing + VRAM tracking | PyG gives static summaries; we need per-stage dynamics |
| Data object | `torch_geometric.data.Data` | `build_graph()` constructs it with custom attributes | Standard PyG Data, extended with sign/mask/popularity attrs |
| CAGRA ANN | -- (uses `cuvs.neighbors.cagra`) | `_build_cagra()` with fallback to kNN | GPU-accelerated ANN; not part of PyG |

# Models

All neural network modules that compose the U-CaGNN model. Each section covers one file in `src/models/`.

---

## Module Map

```
UCaGNN (orchestrator)
  |-- EmbeddingModule      (Module A)
  |-- DualBranchGCN        (Module B)
  |     |-- LightGCNBranch (x1 or x2)
  |-- ScoringModule        (Module C)
  |-- PropensityEstimator  (Module F, optional)
```

---

## EmbeddingModule (Module A)

**File:** `src/models/embeddings.py`

**Purpose:** Learnable lookup tables for users and items. When `use_dual_branch=True`, users get two separate embeddings (interest vs. conformity) so the GCN branches operate on independent representations. When `use_features=True` and canonical item features are available, Module A also projects those features into the embedding space and fuses them into branch-specific item representations.

### Constructor

```python
EmbeddingModule(
    n_users: int,
    n_items: int,
    config: UCaGNNConfig,
    item_features: Tensor | None = None,
    item_popularity: Tensor | None = None,
)
```

### Embedding Tables Created

| Config | Table | Shape | Init |
|--------|-------|-------|------|
| `use_dual_branch=True` | `user_interest` | `(U, D)` | Uniform(-1, 1) |
| `use_dual_branch=True` | `user_conformity` | `(U, D)` | Uniform(-1, 1) |
| `use_dual_branch=False` | `user_embed` | `(U, D)` | Uniform(-1, 1) |
| always | `item_embed` | `(I, D)` | Uniform(-1, 1) |
| `use_popularity_emb=True` | `item_pop` | `(I, D_pop)` | Uniform(-1, 1) |
| `use_features=True` and features present | `item_feature_proj` | `(F_i, D)` | Linear projection of canonical item features |
| `use_features=True` and features present | `popularity_modulator` | `(1, D)` | Popularity-conditioned gate for conformity-side feature fusion |

### Key Methods

| Method | Returns | Used by |
|--------|---------|---------|
| `get_all_embeddings()` | `dict[str, Tensor]` of raw weight matrices | `UCaGNN.forward()` -- fed into GCN |
| `get_subgraph_embeddings()` | `dict[str, Tensor]` restricted to sampled node IDs | `UCaGNN.forward_subgraph()` |
| `get_stacked_embeddings()` | `(U+I, D)` concatenated node embeddings | `build_graph()` for kNN/CAGRA construction |

Internally, branch-aware user selection and optional popularity-embedding selection now flow through shared private helpers, so `get_all_embeddings()`, `get_subgraph_embeddings()`, and `get_stacked_embeddings()` stay aligned when the branch contract changes.

### Feature-Aware Item Fusion

When item side features are available, Module A keeps the original item ID embedding and adds two fused views:

```python
projected_features = LayerNorm(Linear(item_features))
item_interest = item_id_embedding + sigmoid(item_interest_gate) * projected_features
item_conformity = item_id_embedding + sigmoid(item_conformity_gate) * (
    projected_features * popularity_modulator(item_popularity)
)
```

This keeps the feature path optional and branch-specific:
- `item_interest` biases toward semantic/content signals.
- `item_conformity` uses the same projected features, but scales them with a popularity-conditioned gate before propagation.
- If features are absent or `use_features=False`, the module falls back to the original ID-only item embedding path.

### Design Decision: Why Dual User Embeddings?

Users still need separate representations because their interest signal (genuine preference) and conformity signal (popularity following) are causally distinct. Items now start from a shared ID embedding but may expose branch-specific fused inputs (`item_interest`, `item_conformity`) so the two branches can react differently to the same canonical side information.

**Causal mechanism (DICE collider model):** The core insight from Zheng et al. (2020) is that Interest -> Rating <- Popularity forms a *collider* structure. When a user rates a popular item highly, the observation is confounded: the rating could stem from genuine interest *or* from popularity-driven conformity. A single embedding cannot distinguish these two causes. Dual embeddings break this confounding by giving each causal pathway its own representation. DICE demonstrated this yields +31% Recall@20 improvement over single-embedding baselines on MovieLens-10M (GCN-DICE: 0.1812 vs GCN-None: 0.1378).

**Embedding initialization note:** The LightGCN paper uses Xavier initialization, but U-CaGNN follows DDCE's recommendation of Uniform(-1, 1), which the DDCE paper specifically validates for dual-embedding architectures where normal initialization causes slow convergence. The consensus embedding dimension d=64 is shared across LightGCN, DICE, MGCE, FMMRec, and DDCE.

**Alpha monitoring note:** When `use_sign_aware=True`, the learned `alpha_pos` and `alpha_neg` scalars are intentionally unconstrained (following thesis_plan.md). Their values are logged per epoch via `ExperimentLogger` to monitor for drift. If instability is observed, consider adding a constraint (e.g., softmax normalization).

---

## LightGCNBranch / DualBranchGCN (Module B)

**File:** `src/models/lightgcn.py`

**Purpose:** Message passing over the user-item bipartite graph using LightGCN (He et al., 2020). Optionally runs two independent branches for interest and conformity, with sign-aware edge weighting.

### LightGCNBranch

```python
LightGCNBranch(n_layers: int)
```

A stack of `n_layers` `A_norm @ x` propagation steps with uniform alpha-weighted layer combination:

```
out = alpha[0]*x + alpha[1]*conv1(x) + alpha[2]*conv2(conv1(x)) + ...
```

where `alpha[i] = 1 / (n_layers + 1)` for all `i` (stored as a buffer, not learned). The LightGCN paper explicitly tested learning alpha weights and found they "did not yield improvements" over uniform 1/(K+1).

**Input/Output:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `x` (input) | `(N, D)` | Concatenated user + item embeddings |
| `adj` | `(N, N)` sparse | Sparse normalized adjacency matrix |
| return | `(N, D)` | Layer-averaged propagated embeddings |

### DualBranchGCN

```python
DualBranchGCN(config: UCaGNNConfig)
```

The orchestrator that creates either one or two `LightGCNBranch` instances based on `use_dual_branch`.

When `use_dual_branch=True`, U-CaGNN uses explicit asymmetric branch depth through `interest_gnn_layers` and `conformity_gnn_layers`. `DualBranchGCN.forward()` also consumes branch-specific item inputs when `EmbeddingModule` provides them.

**Forward signature:**

```python
def forward(
    embeddings: dict[str, Tensor],   # from EmbeddingModule
    edge_index: Tensor,              # (2, E)
    edge_sign: Tensor | None,       # (E,) aligned to edge_index
    n_users: int,
    n_items: int,
) -> dict[str, Tensor]              # propagated embeddings
```

**Output keys (dual branch):** `user_interest`, `item_interest`, `user_conformity`, `item_conformity`
**Output keys (single branch):** `user`, `item`

### Sign-Aware Edge Weighting

When `use_sign_aware=True`, two learnable scalars modulate edge contributions:

```
edge_weight[e] = 1.0                              if sign[e] > 0 or sign[e] == 0
edge_weight[e] = clamp(alpha_neg / alpha_pos)    if sign[e] < 0
```

Initialized from `alpha_pos=0.7`, `alpha_neg=0.3`, but normalized around a unit LightGCN baseline so observed positive interactions are never weaker than neutral ANN augmentation edges. The learned ratio still lets the model attenuate negative interactions relative to that baseline.

The sign-aware weighting path only changes the effective weights when the current graph contains both positive and negative signed edges. One-sided or neutral-only graphs keep the plain LightGCN unit baseline.

**Paper connection (SIGformer, 2024):** This sign-aware mechanism is inspired by SIGformer, which demonstrated that removing negative edge information causes a -21.5% Recall@20 drop on signed graph benchmarks. SIGformer uses expensive spectral eigendecomposition and Transformer attention to handle signed edges; U-CaGNN extracts the core insight as a lightweight enhancement (just 2 learnable scalars) without that computational overhead.

### Design Decision: Why LightGCN as GCN Backbone?

The LightGCN paper (He et al., 2020) provides rigorous ablation evidence for each simplification:

1. **No weight matrices (W):** Feature transformation matrices "negatively increase the difficulty for model training" -- they add parameters that don't help collaborative filtering where IDs are the only features.
2. **No nonlinear activations:** Activations have "no positive effect on the effectiveness of collaborative filtering" -- unlike vision/NLP, CF embeddings don't benefit from nonlinear transformations.
3. **Symmetric normalization D^{-1/2}AD^{-1/2}:** Mandatory -- "removing either side will drop the performance largely."
4. **Uniform alpha=1/(K+1) not learned:** Learning alpha weights "did not yield improvements" over the uniform setting.
5. **Optimal 2-4 layers:** Performance peaks at 2-3 layers across all benchmarks (Amazon-Book: Recall@20 peaks at L=3 with 0.0410). Over-smoothing is mitigated by layer combination, which captures the effect of self-connections.

Multiple papers confirm the layer finding: MCLN peaks at 2 layers, CaDSI peaks at 2 layers, MGCE degrades after 4-5 layers. Pushing either branch depth above 4 is likely detrimental.

### Why Sparse Adjacency Matmul?

LightGCN's update is exactly repeated multiplication by a normalized adjacency matrix, so the repository now materializes that sparse adjacency once per forward pass and reuses it across all propagation layers. This keeps the same LightGCN semantics as `LGConv(normalize=False)` with explicit `edge_weight`, but avoids the gather-scatter edge-message materialization that was inflating memory on large graphs. The LightGCN paper further proves that self-connections are unnecessary because the "layer combination operation... essentially captures the same effect."

### Config Toggles

| Toggle | Effect |
|--------|--------|
| `use_dual_branch` | 1 branch vs. 2 branches |
| `use_sign_aware` | Enables alpha_pos/alpha_neg learnable edge weights |
| `single_branch_gnn_layers` | LightGCN / non-dual-branch propagation depth |
| `interest_gnn_layers` | Interest-branch propagation depth |
| `conformity_gnn_layers` | Conformity-branch propagation depth |

---

## ScoringModule (Module C)

**File:** `src/models/scoring.py`

**Purpose:** Compute recommendation scores from propagated embeddings via dot products. Combines interest and conformity signals with a diagnostic counterfactual term.

### Constructor

```python
ScoringModule(config: UCaGNNConfig)
```

When `scoring_weight_mode="fixed"`, the module uses config weights directly. When `scoring_weight_mode="learned"`, it learns simplex-constrained mixture weights over the active score components.

### Forward

```python
def forward(
    propagated: dict[str, Tensor],  # from DualBranchGCN
    user_ids: Tensor,               # (B,)
    item_ids: Tensor,               # (B,)
) -> dict[str, Tensor]
```

### Score Computation (Dual Branch)

```
interest_score        = dot(user_interest[u], item_interest[i])
conformity_score      = dot(user_conformity[u], item_conformity[i])
counterfactual_score  = interest_score - conformity_score      (if use_counterfactual)
final_score           = alpha * interest + beta * conformity + gamma * popularity
```

### Score Computation (Single Branch)

```
final_score = dot(user[u], item[i])
```

All other scores are set to zero tensors for API compatibility.

### Output Dict

| Key | Shape | Description |
|-----|-------|-------------|
| `interest_score` | `(B,)` | Interest branch dot product |
| `conformity_score` | `(B,)` | Conformity branch dot product |
| `counterfactual_score` | `(B,)` | Counterfactual difference |
| `final_score` | `(B,)` | Weighted combination for ranking |

### Config Toggles

| Parameter | Default | Role |
|-----------|---------|------|
| `scoring_weight_mode` | `fixed` | `fixed` uses config weights directly; `learned` learns simplex-constrained score weights |
| `alpha_interest` | 0.5 | Weight on interest score |
| `beta_conformity` | 0.3 | Weight on conformity score |
| `gamma_popularity` | 0.2 | Weight on popularity score |
| `use_counterfactual` | True | Enables `counterfactual_score = interest - conformity` diagnostics |

### Evaluation-Time Intervention Modes

`ScoringModule` now exposes evaluation-time scoring modes without changing the training objective:

| Mode | Formula |
|------|---------|
| `default` | `alpha * interest + beta * conformity + gamma * popularity` |
| `interest_only` | `interest` |
| `conformity_only` | `conformity` |
| `conformity_suppressed` | `alpha * interest + gamma * popularity` |

These modes are used by the cached evaluation path (`UCaGNN.score_users_from_propagated()` -> `ScoringModule.score_final_all_items()`) during validation and test evaluation through `config.eval_scoring_mode`. The model also accepts `config.train_scoring_mode`: `preset_lightgcn()` keeps the default score, `preset_dice_like()` trains and evaluates on the fixed `default` interest+conformity score, and `preset_full()` trains and evaluates on the fused `default` score unless a same-checkpoint evaluation script overrides `eval_scoring_mode`.

For semantic diagnostics, `UCaGNN.get_all_score_components()` exposes the full-catalog interest, conformity, counterfactual, and final score matrices, along with the selected user-side propagated embeddings needed for branch-separation analysis.

`ScoringModule` now owns both score paths: pairwise batch scoring for training via `forward()` and full-catalog component assembly for evaluation via `score_all_items()`. Both routes reuse the same internal branch-selection and component-composition helpers so counterfactual and score-mixing semantics stay identical across training and evaluation.

---

## PropensityEstimator (Module F)

**File:** `src/models/propensity.py`

**Purpose:** Estimate item exposure probability P(exposure | item) for inverse propensity weighting (IPW). Debiases the BPR loss by upweighting rare items and downweighting popular ones.

### Constructor

```python
PropensityEstimator(config: UCaGNNConfig)
```

### Architecture

```
item_embedding (D) -> Linear(D, hidden) -> ReLU -> Linear(hidden, 1) -> Sigmoid -> clamp
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `propensity_hidden` | 128 | MLP hidden size |
| `propensity_clip_min` | 0.01 | Lower clamp for numerical stability |
| `propensity_clip_max` | 0.99 | Upper clamp |

### Key Methods

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `forward(item_emb)` | `(B, D)` | `(B,)` | Propensity scores in `[clip_min, clip_max]`; caller computes `1 / forward(...)` for IPW weights |

### Design Decision: Why a Learned Estimator?

Simple heuristics (e.g., `1/count`) are noisy and don't account for how popularity interacts with item content. A learned MLP conditioned on the item embedding captures richer propensity patterns. Clamping prevents extreme weights that destabilize training.

**Survey evidence:** Inverse Propensity Weighting (IPW) is identified by all 4 causal inference surveys (S1-S4, 2023-2025) as a foundational debiasing technique. The consensus finding is that IPW suffers from "high variance when propensities near 0 or 1" -- the clamping to `[clip_min, clip_max]` directly addresses this. Recommended mitigations across surveys include clipping (implemented here), trimming, and doubly robust learning.

---

## UCaGNN (Orchestrator)

**File:** `src/models/ucagnn.py`

**Purpose:** Top-level model class that wires together Modules A, B, C, and F. Provides `forward()` for full-graph training, `forward_subgraph()` for mini-batch training, `score_users_from_propagated()` for cached evaluation, and `get_all_score_components()` for evaluation diagnostics. Different configurations produce different model variants without code changes.

### Constructor

```python
UCaGNN(n_users: int, n_items: int, config: UCaGNNConfig)
```

Creates: `EmbeddingModule`, `DualBranchGCN`, `ScoringModule`, and optionally `PropensityEstimator`.

`UCaGNN` also now routes inverse-propensity weighting through a single internal item-embedding selector, exposes a shared `propagate_embeddings()` helper for precomputed embedding bundles, and reuses one training-payload path across `forward()`, `forward_subgraph()`, and cached-propagation scoring. That keeps propagation, pairwise scoring, and the loss-suite payload contract aligned across full-graph, subgraph, and cached execution paths.

### forward()

```python
def forward(
    edge_index: Tensor,       # (2, E)
    user_ids: Tensor,         # (B,)
    pos_item_ids: Tensor,     # (B,)
    neg_item_ids: Tensor,     # (B,)
    edge_sign: Tensor | None,
) -> dict[str, Tensor | dict]
```

**Returns:**

| Key | Type | Description |
|-----|------|-------------|
| `pos_scores` | `dict[str, Tensor]` | Scores for positive items |
| `neg_scores` | `dict[str, Tensor]` | Scores for negative items |
| `embeddings` | `dict[str, Tensor]` | Initial (pre-GNN) embeddings |
| `propagated` | `dict[str, Tensor]` | Post-GNN embeddings |
| `ipw_weights` | `Tensor (B,)` | IPW weights (ones if `use_ipw=False`) |

### score_users_from_propagated()

Used at evaluation time once `get_propagated_for_eval()` has already built the cached full-graph embeddings.

```python
def score_users_from_propagated(
    propagated: dict[str, Tensor],
    user_ids: Tensor,              # (B,)
    scoring_mode: str | None = None,
) -> Tensor                        # (B, I)
```

Delegates the cached scoring matrix build to `ScoringModule.score_final_all_items()`.

### get_all_score_components()

Used for evaluation diagnostics when the full component matrices are needed instead of only the fused score.

```python
def get_all_score_components(
    edge_index: Tensor,       # (2, E)
    user_ids: Tensor,         # (B,)
    edge_sign: Tensor | None,
    scoring_mode: str | None = None,
    edge_norm: Tensor | None = None,
) -> dict[str, Tensor]
```

### forward_subgraph()

Used by `MiniBatchTrainer` for mini-batch GCN training on subgraphs.

```python
def forward_subgraph(
    batch: SubgraphBatch,
) -> dict[str, Tensor | dict]    # same format as forward()
```

**How it works:** Reads the sampled node ids and local edge tensors from `SubgraphBatch`, indexes into the full embedding tables for the sampled user/item subset, runs GCN propagation on the local `sub_edge_index` (with local normalization), then scores using the batch's local user/item indices. Returns the same output format as `forward()` so downstream loss computation is unchanged.

**Import note:** `forward_subgraph()` is defined in `src/models/ucagnn.py` alongside `forward()`, `score_users_from_propagated()`, and `get_all_score_components()`.

### Model Variants via Config

| Variant | Config | Modules Active |
|---------|--------|----------------|
| LightGCN baseline | `preset_lightgcn()` | A (single user emb), B (single branch), C (dot product only) |
| DICE-like | `preset_dice_like()` | A (dual), B (dual), C (interest + conformity) |
| Full U-CaGNN | `preset_full()` | A (dual + pop), B (dual + sign-aware), C (all scores), F (IPW) |


# Losses

All loss functions and the multi-task orchestrator. Each section covers one file in `src/losses/`.

---

## Loss Overview

| Loss | Symbol | File | Role | Paper Origin | Config Lambda |
|------|--------|------|------|-------------|---------------|
| BPR | L_rec | `loss_suite.py` | Pairwise ranking (positive > negative) | LightGCN (He et al., 2020) | `lambda_rec` (1.0) |
| Orthogonality | L_ortho | `loss_suite.py` | Decorrelate interest/conformity | DICE (Zheng et al., 2020); similar to FMMRec orthogonality constraint | `lambda_ortho` (0.01) |
| Contrastive | L_contr | `loss_suite.py` | Pull same-user views together, push others apart | DCCL (2023) NT-Xent for semantic intent separation | `lambda_contr` (0.05) |
| Counterfactual | L_cf | `loss_suite.py` | Score-scale regularizer: keeps branch magnitudes comparable | CausalEmbed (Bonner & Vasile, 2018) Ω(θ_treat − θ_control) | `lambda_cf` (0.05) |
| Popularity | L_pop | `loss_suite.py` | Conformity branch predicts item popularity | DICE (2020) / DDCE (2023) conformity-predicts-popularity | `lambda_pop` (0.1) |
| **Total** | L_total | `loss_suite.py` | Weighted sum with curriculum scheduling | CaDCR (2025) multi-task curriculum | -- |

---

All loss implementations now live as private helpers inside `src/losses/loss_suite.py`. `LossSuite` remains the only public loss-layer API.

## L_rec: BPR Ranking Loss

**File:** `src/losses/loss_suite.py`

### Formula

$$L_{rec} = -\frac{1}{B} \sum_{b=1}^{B} w_b \cdot \log \sigma(s_{pos}^{(b)} - s_{neg}^{(b)})$$

where $w_b$ = IPW weight (1.0 if IPW disabled).

### Function Signature

```python
def bpr_loss(
    pos_scores: Tensor,          # (B,)
    neg_scores: Tensor,          # (B,)
    weights: Tensor | None,      # (B,) IPW weights
) -> Tensor                      # scalar
```

### Implementation Notes

- **Unweighted case**: delegates to PyG's `BPRLoss` (from `torch_geometric.nn.models.lightgcn`) for maximum efficiency.
- **Weighted case**: custom implementation using `F.logsigmoid` + element-wise multiplication with IPW weights.
- Always active (not gated by curriculum).

### Role in Training

Encourages the model to rank positive items higher than negative items. The IPW extension debiases by giving rare items higher loss weight, counteracting popularity bias in the training data.

`L_rec` now uses whatever score view `UCaGNN.build_training_output()` exposes through `config.train_scoring_mode`. `preset_lightgcn()` keeps the default single-branch score, `preset_dice_like()` uses the fixed default interest+conformity score, and `preset_full()` keeps the fused default score unless evaluation later overrides only `config.eval_scoring_mode`.

---

## L_ortho: Orthogonality Loss

**File:** `src/losses/loss_suite.py`

### Formula

$$L_{ortho} = \frac{1}{B} \sum_{b=1}^{B} \cos(z_{int}^{(b)}, z_{conf}^{(b)})^2$$

### Function Signature

```python
def orthogonality_loss(
    interest: Tensor,      # (B, D)
    conformity: Tensor,    # (B, D)
) -> Tensor                # scalar
```

### Implementation Notes

- Uses `F.cosine_similarity` for numerically stable computation.
- Complexity: O(B * D) -- much cheaper than distance correlation (dCor) which is O(B^2 * D).
- Squaring the cosine similarity penalizes both positive and negative correlation equally.
- Distance correlation (dCor) is identified by Survey S4 as a computational bottleneck with "high training overhead." Cosine-squared trades statistical power for O(B*D) tractability, following the FMMRec precedent which also uses an orthogonality constraint rather than dCor for its fair/biased embedding separation.

### Role in Training

Decorrelates interest and conformity user embeddings so they capture distinct signals. Without this, both branches may converge to the same representation, defeating the purpose of dual-branch disentanglement.

---

## L_contr: Contrastive Loss (NT-Xent)

**File:** `src/losses/loss_suite.py`

### Formula

$$L_{contr} = -\frac{1}{2B} \sum_{k=1}^{B} \left[ \log \frac{e^{sim(z_i^{(k)}, z_j^{(k)})/\tau}}{\sum_{m \neq k} e^{sim(z_i^{(k)}, z_m)/\tau}} + \log \frac{e^{sim(z_j^{(k)}, z_i^{(k)})/\tau}}{\sum_{m \neq k} e^{sim(z_j^{(k)}, z_m)/\tau}} \right]$$

where $sim$ = cosine similarity, $\tau$ = temperature.

### Function Signature

```python
def contrastive_loss(
    z_i: Tensor,        # (B, D) - first view (interest)
    z_j: Tensor,        # (B, D) - second view (conformity)
    tau: float = 0.1,   # temperature
) -> Tensor             # scalar
```

### Implementation Notes

- Both views are L2-normalized before similarity computation.
- Concatenates views to form a `(2B, 2B)` similarity matrix.
- Self-similarity is masked out via identity matrix exclusion.
- Log-sum-exp for numerical stability.
- Returns zero (with grad) for batch size <= 1 (edge case).
- The O(B^2) similarity matrix is the same bottleneck identified in DCCL and Survey S2 (which quantifies contrastive learning at O(B^2*d) per batch). At batch dimension 2048, this remains manageable. CaDCR (2025) suggests curriculum phasing to introduce contrastive loss only after initial embedding stabilization, which is supported by the curriculum scheduling in LossSuite.

### Role in Training

Pulls the two views of the same user together while pushing views of different users apart. This ensures that while interest and conformity are *orthogonal* (L_ortho), they still encode information about the *same* user (not random noise).

### Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contrastive_tau` | 0.1 | Temperature -- lower = sharper distinctions |

---

## L_cf: Counterfactual Divergence Loss

**File:** `src/losses/loss_suite.py`

### Formula

$$L_{cf} = \frac{1}{B} \sum_{b=1}^{B} (Y_{int}^{(b)} - Y_{conf}^{(b)})^2$$

### Function Signature

```python
def counterfactual_loss(
    interest_scores: Tensor,     # (B,)
    conformity_scores: Tensor,   # (B,)
) -> Tensor                      # scalar
```

### Role in Training

Regularizes interest and conformity *score magnitudes* to remain comparable, so the counterfactual contrast `interest − conformity` stays interpretable as embeddings diverge.  Embedding divergence itself is driven by L_ortho (orthogonality) and L_contr (contrastive separation); L_cf acts as a score-scale calibrator that prevents the two branches from drifting to incompatible magnitude ranges.

**Causal theory connection:** This implements the CausalEmbed (Bonner & Vasile, 2018) regularization Ω(θ_treat − θ_control), which penalizes parameter-space distance between treatment and control branches to keep the causal contrast well-conditioned.  CausE (2018) established product-level treatment effect estimation with domain-adapted parameters; MCLN (2023) extended counterfactual reasoning to multimodal content.

---

## L_pop: Popularity Loss

**File:** `src/losses/loss_suite.py`

### Formula

$$L_{pop} = \frac{1}{B} \sum_{b=1}^{B} (\hat{p}^{(b)} - p^{(b)})^2$$

where $\hat{p}$ = predicted popularity from conformity embeddings, $p$ = ground-truth normalized popularity.

### Classes and Functions

**PopularityPredictor** (nn.Module):

```python
PopularityPredictor(embed_dim: int)
```

Architecture: `Linear(D, 1) -> squeeze(-1)` -- projects conformity embeddings to a scalar popularity prediction.

**popularity_loss** (function):

```python
def popularity_loss(pop_pred: Tensor, pop_target: Tensor) -> Tensor
```

Standard MSE loss.

### Role in Training

Anchors the conformity branch to the popularity signal. By training the conformity embeddings to predict item popularity, the interest branch is freed to capture genuine user preference -- the causal signal we actually want for recommendations.

**Paper lineage:** This follows DICE's core principle that conformity embeddings should predict popularity, since conformity *is* the popularity-following behavior. DDCE (2023) validates this with ablation results: removing the popularity disentanglement component (DDCE w/o i) drops HR from 0.0507 to 0.0485 on KuaiRec. MGCE (2024) further refines this with asymmetric GCN depth -- 1 layer for interest, 2 layers for conformity -- arguing that conformity requires deeper propagation to capture group homogeneity effects.

---

## LossSuite (Orchestrator)

**File:** `src/losses/loss_suite.py`

### Constructor

```python
LossSuite(config: UCaGNNConfig)
```

Creates `PopularityPredictor` if `use_dual_branch=True` and `lambda_pop > 0`.

### Forward

```python
def forward(
    model_output: dict,            # from UCaGNN.forward()
    item_popularity: Tensor,       # (I,) normalized counts
    pos_item_ids: Tensor,          # (B,)
    epoch: int = 0,                # for curriculum
) -> dict[str, Tensor]            # individual losses + "total"
```

### Total Loss Formula

```
L_total = lambda_rec   * L_rec
        + lambda_ortho * L_ortho
        + lambda_contr * L_contr
        + lambda_cf    * L_cf
        + lambda_pop   * L_pop
```

### Curriculum Scheduling

Losses are phased in based on epoch thresholds to stabilize early training:

| Phase | Epochs | Active Losses | Rationale |
|-------|--------|--------------|-----------|
| 1 | `[0, phase1_end)` | L_rec only | Learn basic ranking first |
| 2 | `[phase1_end, phase2_end)` | + L_ortho, L_contr | Begin disentangling branches |
| 3 | `[phase2_end, ...)` | + L_cf, L_pop | Add counterfactual + popularity once branches are separated |

**Default:** `auxiliary_losses_start_epoch = 15`, `popularity_supervision_start_epoch = 30` (CaDCR-inspired staged curriculum -- Phase 1: ranking only, Phase 2: + branch separation, Phase 3: + full causal disentanglement).

### Gating Logic

Each loss requires:
1. `use_dual_branch = True` (ortho, contr, cf, pop all need two branches)
2. Its lambda > 0
3. Current epoch >= phase threshold

If any condition fails, the loss is set to a zero tensor (no gradient contribution).

### Output Dict

| Key | Description |
|-----|-------------|
| `"rec"` | BPR loss value |
| `"ortho"` | Orthogonality loss value |
| `"contr"` | Contrastive loss value |
| `"cf"` | Counterfactual loss value |
| `"pop"` | Popularity loss value |
| `"total"` | Weighted sum |


# Data Pipeline

How raw datasets become PyG graph objects ready for training. Covers the canonical format, all 6 dataset loaders, the graph builder, and the negative sampler.

---

## Pipeline Overview

```
Raw files (CSV, PyG auto-download)
        |
   Dataset Loader          src/data/loaders/<dataset>.py
        |
  CanonicalInteractions    src/data/canonical.py
        |
     build_graph()         src/data/graph_builder.py
        |
    PyG Data object         (edge_index, masks, sign, popularity, optional features)
        |
        +-----------------------------+
        |                             |
   NegativeSampler              SubgraphSampler
   src/data/negative_sampler.py src/data/subgraph_sampler.py
   (used per batch in Trainer)  (used per batch in MiniBatchTrainer)
```

---

## CanonicalInteractions

**File:** `src/data/canonical.py`

The universal intermediate format. Every loader produces one; downstream code never touches raw file formats.

### Fields

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `user_id` | `np.ndarray` int64 | `(N,)` | Re-indexed from 0 |
| `item_id` | `np.ndarray` int64 | `(N,)` | Re-indexed from 0 |
| `label` | `np.ndarray` float32 | `(N,)` | 1.0 = positive, 0.0 = negative |
| `timestamp` | `np.ndarray` int64 | `(N,)` | Unix seconds (0 if unavailable) |
| `sign` | `np.ndarray` float32 | `(N,)` | Continuous sentiment in [-1, 1] |
| `popularity` | `np.ndarray` float32 | `(I,)` | Per-item interaction count, normalized to [0, 1] |
| `n_users` | `int` | -- | Total unique users |
| `n_items` | `int` | -- | Total unique items |
| `user_map` | `dict[int, int]` | -- | original_id -> reindexed_id |
| `item_map` | `dict[int, int]` | -- | original_id -> reindexed_id |
| `user_features` | `np.ndarray` or `None` | `(U, F_u)` | Optional side features |
| `item_features` | `np.ndarray` or `None` | `(I, F_i)` | Optional side features |
| `repeat_count` | `np.ndarray` or `None` | `(N,)` | Number of raw rows aggregated into each kept user-item pair |
| `repeat_mean_target` | `np.ndarray` or `None` | `(N,)` | Mean target value across the raw repeated rows |
| `repeat_max_target` | `np.ndarray` or `None` | `(N,)` | Maximum target value retained by the representative row |
| `repeat_latest_target` | `np.ndarray` or `None` | `(N,)` | Latest target value by timestamp among the raw repeated rows when timestamps exist |
| `repeat_first_timestamp` | `np.ndarray` or `None` | `(N,)` | Earliest timestamp among the raw repeated rows when timestamps exist |
| `repeat_last_timestamp` | `np.ndarray` or `None` | `(N,)` | Latest timestamp among the raw repeated rows when timestamps exist |
| `repeat_behavior_counts` | `np.ndarray` or `None` | `(N, B)` | Optional per-pair behavior-count matrix for datasets that expose repeated behavior events |
| `repeat_behavior_labels` | `np.ndarray` or `None` | `(B,)` | Optional labels describing the `repeat_behavior_counts` columns |
| `metadata` | `dict` or `None` | -- | Optional dataset-specific arrays or descriptors aligned to interactions, users, or items |

### Key Methods

| Method | Description |
|--------|-------------|
| `__len__()` | Number of interactions |
| `__repr__()` | Summary with user/item counts, interaction count, positive rate |
| `temporal_split(train_ratio, val_ratio)` | Returns `(train_mask, val_mask, test_mask)` boolean arrays based on timestamp ordering |

### Label and Sign: Causal Semantics

The `label` and `sign` fields carry causal meaning beyond simple data encoding:

- **`label`** maps to DICE's collider model: positive interactions (label=1.0) are potentially confounded by popularity -- a user may have clicked because of genuine interest *or* because the item was popular. The dual-branch architecture exists to disentangle these causes.
- **`sign`** maps to SIGformer's finding that negative edge information is critical (-21.5% Recall@20 without it). Sign values enable differential propagation intensities in the GCN. Graded sign values (e.g., Taobao's behavior types: buy=1.0, cart=0.5, fav=0.25, pv=-0.25) allow the model to learn nuanced edge weights rather than binary positive/negative.
- Datasets with `sign=0.0` for all interactions (Amazon-Book) effectively disable sign-aware weighting, making U-CaGNN equivalent to unsigned LightGCN propagation on those datasets.

### Design Decision: Why a Dataclass?

A simple `@dataclass` (not a PyG `InMemoryDataset`) keeps the intermediate format framework-agnostic. Loaders produce NumPy arrays; conversion to PyG tensors happens once in `build_graph()`. This makes loaders easy to test without PyG.

---

## Dataset Loaders

### Dataset Selection Rationale

The 6 datasets span the major benchmarks used across the causal recommendation literature: MovieLens (used by DICE, CausE, FMMRec, CaDSI for explicit rating evaluation), Amazon-Book (used by CaDCR, LightGCN for implicit large-scale evaluation), Taobao (used by MGCE, MCLN for behavior-typed multimodal evaluation), and KuaiRec/KuaiRand (identified by all 4 surveys as gold-standard unbiased interaction logs for rigorous causal method validation, replacing small RCTs like Coat and Yahoo! R3).

**File:** `src/data/loaders/__init__.py`

All loaders are registered in a `LOADERS` dict and accessed via:

```python
from src.data.loaders import load_dataset
canonical = load_dataset("movielens1m", data_dir="data")
```

The dataset-specific allowlists for optional side-feature scans now live in `src/data/feature_policy.py`, alongside the rest of the data-layer policy code used by the loaders and dataset audit. That module also exposes which registered feature files are enabled under `thesis_default` versus `all_optional`, so file-selection facts do not drift between the loader path and the audit/reporting path.

Local raw-directory resolution and numeric downcasting that are shared across multiple loaders now live in `src/utils/dataset_loader_utils.py`. `downcast_numeric_array(...)` now narrows arrays by explicit `np.iinfo` / `np.finfo` range checks across the standard NumPy integer and floating widths, so positive integer IDs can move into unsigned storage when safe and large floats stay in `float64` only when narrower widths are not valid. Loader-local parse fallbacks, malformed-value counting, and warning summaries should stay in the concrete loaders themselves, so dataset-specific coercion policy remains close to the code that owns each CSV schema. Keep the shared helper limited to genuinely mechanical concerns; dataset-specific labels, signs, split logic, metadata, and feature-policy behavior should stay in the concrete loader modules.

The mixed-type CSV side-feature parser shared by KuaiRec and KuaiRand now lives in `src/utils/csv_features.py`. That module also owns the shared `stack_feature_blocks(...)` combiner, the policy-gated CSV loader wrapper, and the small `PolicyCsvFeatureSpec` + `load_policy_csv_feature_blocks(...)` helper used when a loader needs to stack several row-aligned feature tables without repeating the same resolve-and-load control flow. The shared path keeps numeric columns numeric, ordinal-encodes categorical columns (including ID-like feature columns), and normalizes retained date/datetime strings to integer Unix seconds. Keep the shared helper narrowly focused on row alignment, compact numeric coercion, policy-gated CSV loading, and mechanical stacking; dataset-specific caption/category parsing and feature composition should remain in the individual loaders.

User/item contiguous ID remapping and normalized popularity scores now come from the shared helper in `src/utils/interaction_indexing.py`. Keep that utility limited to canonical indexing mechanics so each loader still owns its raw parsing, labels, signs, and feature-policy behavior. The pairwise-collapse helpers there now share one internal validation/slicing contract, so the retained-index path, plain collapse path, repeat-aware summary path, loader-facing canonical repeat-field packaging, and shared `repeat_collapse` metadata payload stay aligned without reimplementing the same row-length checks, aligned-array slicing, or summary-to-canonical wiring. Final `CanonicalInteractions` packing for shared indexed payloads now goes through `src/data/canonical.py::build_indexed_canonical_interactions(...)`, which centralizes the user/item maps, counts, default popularity derivation, and optional canonical side fields used by Amazon-Book, Taobao, KuaiRec, KuaiRand, and the shared explicit-rating path.

For tiny validation and smoke-style runs, the shared loader path also accepts `max_rows=...`. Capped loads are now reused in-process, so the validator can keep the same feature-enabled dataset semantics as formal runs without rescanning the same raw files for every recipe case.

The repository-level dataset report generated by `src/data_exploration/data_information.py` now includes a causal feature audit. That audit makes three layers explicit for each dataset: what exists in raw files, what the current canonical loaders already preserve, and what the current model actually consumes. The loader-coverage text for optional feature files now comes from the same feature-policy registry used by the loaders, so the audit remains synchronized with the real thesis-default and all-optional file choices. The suitability summary now also distinguishes **true timestamp support** from merely having predefined train/test split files, which keeps graph-only baselines such as Amazon-Book from being mistaken for temporal datasets.

The same audit now enumerates candidate columns file-by-file and labels each candidate with:
- a causal role (`pre_treatment`, `post_treatment`, `proxy`, `non_causal`, or `unknown`),
- a current pipeline stage (`raw_only`, `analysis_retained`, `graph_retained`, or `model_consumed`), and
- a quick policy verdict (`safe_candidate`, `encode_then_test`, `load_then_test`, `model_extension_needed`, `ablation_only`, `defer`, `review`, or `exclude`).

`item_categories.csv` in KuaiRec should now be interpreted as an item-feature source rather than an interaction-like heuristic artifact: its `feat` column is the multi-hot category block that the current thesis-default loader path promotes into canonical item features.

Use the terminology consistently:
- `optional features` means canonical side-feature matrices such as `user_features` or `item_features`.
- `optional feature scans` means extra raw-file reads used to build those matrices.
- `feature consumption` means the current model or evaluator actually uses those fields.
- `feature_policy` means which subset of optional scans is allowed into the current canonical path. `thesis_default` is the repository default; `all_optional` is reserved for explicit ablations.

`src/data_exploration/explore_all_datasets.py` also writes `benchmark_summary.json` plus `benchmark_summary.md` into `results/dataset_visualizations/` alongside the PNG figures. Those text exports preserve the same canonical statistics used by the plots, including repeated-pair share, distinct user-item pair counts, randomized-exposure share when available, response-signal summaries, and the small context summaries that would otherwise need to be read visually from the figures. They are now also the quickest way to confirm that thesis-default preprocessing is actually changing the data surface as intended—for example, the official `MovieLens 20M` raw files still expose a 20-column default genre feature width after the loader's raw-file vocabulary inference, and thesis-default `Taobao` / `KuaiRec v2` now report `0.00%` repeated-pair share after pairwise collapse.

### Loader Summary

| Name | Function | Source Format | Size | Label Logic | Sign Logic | Side Features |
|------|----------|--------------|------|-------------|------------|---------------|
| `movielens1m` | `load_movielens1m` | local `ratings.dat` raw files | ~1M ratings | rating >= 4 -> positive | (rating - 3) / 2 | user features, movie genres, raw rating retained |
| `movielens20m` | `load_movielens20m` | `ratings.csv` (manual) | ~20M ratings | rating >= 4 -> positive | (rating - 3) / 2 | movie genres by default; dense genome relevance only under `all_optional` / dense preset |
| `amazonbook` | `load_amazonbook` | local LightGCN split files | ~2.9M implicit | all positive (implicit) | 0.0 (neutral) | none; preset recorded as graph-only |
| `taobao` | `load_taobao` | `UserBehavior.csv` (manual) | ~100M behaviors | buy/cart/fav -> pos, pv -> neg | buy=1, cart=0.5, fav=0.25, pv=-0.25 | item categories |
| `kuairec_v2` | `load_kuairec_v2` | `small_matrix.csv` by default; `big_matrix.csv` for explicit watch-ratio runs | dual-matrix watch logs | watch_ratio >= 0.5 -> pos | clip(wr, 0, 2) - 1 | thesis-default: 6 daily descriptors + multi-hot categories = **40 item features**; `all_optional` restores user features + full item scans |
| `kuairand1k` | `load_kuairand1k` | `log_standard*.csv` (manual) | ~11M interactions | click/like/follow -> pos | like=+1, follow=+0.7, long_view=+0.3, comment=0, hate=-1 | thesis-default: 9 video descriptors = **9 item features**; repeated pairs collapsed with repeat-count / priority summaries preserved; `all_optional` restores user features + statistic tables |

### Per-Loader Details

#### MovieLens 1M

Uses the repository-local `ratings.dat`, `users.dat`, and `movies.dat` files rather than PyG auto-download paths. Re-indexes user and movie IDs to contiguous 0-based ranges, keeps the raw explicit rating in `raw_target`, loads stable user descriptors from `users.dat`, and infers the movie-genre vocabulary directly from `movies.dat` before building the multi-hot item feature matrix.

#### MovieLens 20M

Manual CSV loading with `np.loadtxt` for speed on 20M rows. Same label/sign logic as ML-1M. The loader can also enrich items with genres from `movies.csv` and genome relevance scores from `genome-scores.csv`. The genre matrix is now inferred from the raw `movies.csv` vocabulary instead of a duplicated hardcoded genre list, so the official dataset still yields 20 genre columns while smaller fixtures keep only the columns they actually contain. The `np.loadtxt(...)` path now forces a 2D array shape even for single-row fixtures, which keeps tiny validation files on the same canonical path as full dataset loads. In capped validation runs, the resulting canonical object is cached and reused across recipe cases so those auxiliary scans are paid once per dataset configuration instead of once per recipe.

Both MovieLens loaders now share a private helper in `src/data/loaders/_explicit_ratings.py` for the explicit-feedback mechanics they truly have in common: raw-id reindexing, label/sign derivation, popularity computation, and final canonical assembly. Dataset-specific parsing and feature extraction remain local to each concrete loader.

This loader is the clearest example of `optional feature scans`: the interaction table always comes from `ratings.csv`, while extra scans over `movies.csv` and `genome-scores.csv` are only used to build optional item features.

#### Amazon-Book

Uses the repository-local LightGCN-format `train.txt`/`test.txt` files under `data/AmazonBook/raw`. All interactions are implicit positive: `label=1.0`, `sign=0.0`. The dataset has predefined train/test splits but **no true timestamp field**, so the audit should treat it as split-ready rather than temporally ordered. Malformed user or item tokens are now skipped with warnings during parsing instead of crashing the loader.

#### Taobao (UserBehavior)

Behavior-typed interactions with a graded sign scale:

| Behavior | Label | Sign |
|----------|-------|------|
| buy | 1.0 | +1.0 |
| cart | 1.0 | +0.5 |
| fav | 1.0 | +0.25 |
| pv (page view) | 0.0 | -0.25 |

Item features: category ID (re-indexed to contiguous integers, stored as single-column float). Unknown behavior labels and malformed core rows are now skipped with warnings so they do not silently enter the training graph as neutral noise.

#### KuaiRec v2

Watch-ratio based: `label = (watch_ratio >= 0.5)`, `sign = clip(watch_ratio, 0, 2) - 1`. The repository now defaults to the `kuairec_fullobs` view on `small_matrix`, so the nearly fully observed matrix is the default causal-ready path and its canonical `raw_target` keeps the unmodified `watch_ratio`. The explicit `kuairec_watchratio` path still targets `big_matrix`; that scale-oriented view remains available, but it is no longer the default and still clips `watch_ratio` to `[0, 5]` before the repeated-pair collapse path runs. The loader reuses the shared mixed-type CSV feature helper in `src/utils/csv_features.py` for row-aligned side-feature tables, the same module's policy-gated CSV loader wrapper for row-aligned feature sources, and `stack_feature_blocks(...)` when several optional feature blocks must be merged into one canonical matrix. Under the default `feature_policy="thesis_default"`, the loader keeps the audited descriptor subset from `item_daily_features.csv` (`author_id`, `music_id`, `video_type`, `upload_dt`, `upload_type`, `visible_status` — 6 columns), the multi-hot category block from `item_categories.csv`, and the three numeric caption-category IDs from `kuairec_caption_category.csv`, giving **item_feature_dim = 40**. Within the shared helper, categorical-like columns are ordinal-encoded and the retained `upload_dt` string is normalized to integer Unix seconds instead of collapsing to zero during numeric coercion. `item_categories.csv` and `kuairec_caption_category.csv` are parsed with CSV-aware readers so quoted category lists and caption text containing commas do not corrupt the feature matrix. The wider user-feature scans and post-treatment item aggregates remain reachable only through `feature_policy="all_optional"`. Malformed core rows and non-finite `watch_ratio` values are now dropped with warnings before they reach the canonical tensors. The `watch_ratio_policy` metadata field is `"raw"` for the default full-observation preset and `"clipped_to_5"` for the explicit `kuairec_watchratio` path.

Because `item_daily_features.csv` and related tables can mix stable descriptors with behavioral aggregates, the audit report and shared feature policy now act as the gatekeeper for what enters the default causal feature path versus what is held back for explicit ablations.

#### KuaiRand-1K

Multi-signal interactions (click, like, follow, hate). Scans all `log_standard*.csv` and `log_random*.csv` files in the data directory. Label is `click OR like OR follow`. Sign is `+1` for like, `+0.7` for follow, `+0.3` for click-plus-long-view, `-1` for hate, and `0` otherwise; comment-only rows now stay neutral until the repository has a justified sentiment signal rather than being hard-coded as positive. Under the default `feature_policy="thesis_default"`, the loader keeps 9 columns from `video_features_basic_1k.csv` (`author_id`, `video_type`, `upload_dt`, `upload_type`, `visible_status`, `server_width`, `server_height`, `music_id`, `music_type`), giving **item_feature_dim = 9**; it skips the user-feature block plus the post-treatment statistics table. The shared CSV helper ordinal-encodes categorical-like columns, compresses ID-like feature columns into stable positive codes, and normalizes `upload_dt` to integer Unix seconds so retained temporal context does not vanish during parsing. `feature_policy="all_optional"` restores the wider scans for controlled ablations. Plain feature sources now go through the shared policy-gated CSV helper so the loader does not repeat the same resolve-and-load branches for each file. Both `kuairand_causal` and `kuairand_random_only` presets now collapse repeated user-item pairs using the shared repeat-aware summary path (priority = watch-ratio computed from play_time/duration, tie-breaker = timestamp); the canonical output preserves `repeat_count`, first/last timestamps, and mean/max watch-ratio priority for each kept pair, while the number of dropped rows is still stored in `metadata["repeat_collapse"]["dropped_rows"]`. Capped tiny runs reuse cached capped loads so the validator still exercises the same policy path without repeated rescans. `is_rand` metadata remains preserved, malformed core rows are skipped, and non-finite optional float fields are coerced back to neutral defaults instead of propagating NaNs into the dataset.

---

## Graph Builder

**File:** `src/data/graph_builder.py`

### Function Signature

```python
def build_graph(
    canonical: CanonicalInteractions,
    config: UCaGNNConfig,
    embeddings: Tensor | None = None,
) -> Data
```

### What It Does

1. **Node ID offset** -- item IDs are shifted by `n_users` so the bipartite graph has unique node IDs: users are `[0, U)`, items are `[U, U+I)`.

2. **Temporal split** -- calls `get_splits()` (which prefers predefined masks when available, falling back to `canonical.temporal_split()`) to create `train_mask`, `val_mask`, `test_mask`.

3. **Edge construction** -- uses only training interactions for edges.

4. **Popularity construction** -- recomputes `data.popularity` from the final training split only, so validation/test interactions never leak into popularity-driven losses, sampling, or metrics.

5. **Output** -- a PyG `Data` object with custom attributes.

### Graph Construction Strategies

| Path | Config Surface | Description | Paper Rationale | Fallback |
|------|----------------|-------------|-----------------|----------|
| `cagra` | Implicit default with `cagra_*` settings | Train bipartite edges plus optional CAGRA GPU-accelerated ANN edges (via `cuvs`) | CAGRA paper (2024): 33-77x batch speedups over HNSW; out_degree 32-96 | Falls back to the train-interaction graph if CAGRA import, CuPy conversion, index build, or search fails, and also if embeddings are absent |

All strategies add edges in **both directions** (undirected graph).

### Output Data Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `edge_index` | `(2, E)` | Graph edges (bipartite + optional CAGRA ANN augmentation) |
| `edge_sign` | `(E,)` | Aligned to `edge_index`; train signs duplicated for undirected edges; kNN/CAGRA edges get 0.0 |
| `train_mask` | `(N_interactions,)` bool | Training split |
| `val_mask` | `(N_interactions,)` bool | Validation split |
| `test_mask` | `(N_interactions,)` bool | Test split |
| `popularity` | `(I,)` | Train-split normalized item popularity |
| `labels` | `(N_interactions,)` | Binary labels |
| `user_nodes` | `(N_interactions,)` | User IDs per interaction |
| `item_nodes` | `(N_interactions,)` | Offset item IDs per interaction |
| `n_users` | `int` | User count |
| `n_items` | `int` | Item count |
| `user_features` | `(U, F_u)` optional | Canonical user feature matrix when present |
| `item_features` | `(I, F_i)` optional | Canonical item feature matrix when present |
| `metadata` | `dict` optional | Canonical metadata passed through unchanged |

### Feature-Carry Semantics

`build_graph()` does not transform canonical side features beyond NumPy-to-tensor conversion. It simply preserves them on the PyG `Data` object so later model stages can decide how to use them. That canonical-to-graph payload transfer now goes through one shared helper in `src/data/graph_builder.py`, which keeps the optional tensor conversions, copied string arrays, repeat-summary fields, and metadata passthrough aligned in one place instead of a long hand-maintained `if` chain.

In the current implementation, `data.item_features` and split-safe `data.popularity` are the inputs consumed by `UCaGNN` for feature-aware item embedding fusion. `data.user_features` is retained for future user-side extensions but is not yet consumed by the model.

That distinction matters for audit interpretation: a field can be present in raw files, loaded into the canonical dataset, and even carried into the graph object without contributing to current ranking or semantic evaluation. The per-column audit is intended to make that explicit before new feature engineering work begins.

### CAGRA Details

Uses NVIDIA `cuvs.neighbors.cagra` for GPU-accelerated approximate nearest neighbor graph construction:
- Keeps CUDA embeddings on-device through the CuPy/DLPack path when available, materializes the returned neighbor table once as a CPU `torch.long` tensor, and builds the ANN edge index in Torch to avoid extra NumPy host-array churn.
- Configurable via `cagra_out_degree`, `cagra_initial_degree`, `cagra_team_size`, `cagra_metric`, `cagra_itopk_size`.
- Falls back gracefully to kNN on machines without CUDA/cuvs.

**Default configuration:** `out_degree=32`, `initial_degree=64` (2× out_degree, thesis-speed defaults; cuVS library defaults are 64/128), `team_size=0` (auto), `metric="inner_product"` (matches dot-product scoring used by LightGCN), `itopk_size=64` (search-time only, no build cost). Increase `out_degree`/`initial_degree` for higher-quality ANN recall at the cost of longer graph build time.

---

## Negative Sampler

**File:** `src/data/negative_sampler.py`

### Constructor

```python
NegativeSampler(
    n_items: int,
    popularity: Tensor,              # (I,) for weighted sampling
    n_negatives: int = 1,
    hard_negative_ratio: float = 0.0,
)
```

### Sampling Strategies

Two strategies mixed via `hard_negative_ratio`:

| Strategy | Method | When |
|----------|--------|------|
| Uniform random | `torch.randint` (fully vectorized) | `1 - hard_negative_ratio` fraction |
| Popularity-weighted | `torch.multinomial` from pre-computed weights | `hard_negative_ratio` fraction |

### Sample Method

```python
def sample(
    batch_size: int,
    positive_items: Tensor | None,  # (B,) for collision avoidance
    device: str | torch.device,
) -> Tensor                          # (B, n_negatives)
```

### Collision Avoidance

Best-effort: if any sampled negative equals its corresponding positive item, it is replaced with a fresh `randint` draw. This is not guaranteed collision-free (the replacement could also collide), but it is fast and sufficient in practice since `n_items` is typically much larger than `n_negatives`.

### Causal Rationale for Hard Negatives

Popularity-weighted hard negatives (controlled by `hard_negative_ratio`) stress-test disentanglement: popular items are most likely to be liked due to conformity rather than genuine interest. This is precisely the confounding mechanism in DICE's collider model (Interest -> Rating <- Popularity). By oversampling popular items as negatives, the model is forced to distinguish "this user doesn't genuinely like this popular item" from "this item is popular and the user might conform" -- directly exercising the dual-branch separation.

### Design Decision: Why Not Exact Exclusion?

Exact exclusion (sampling from `all_items - user_history`) requires per-user sets and is O(B * |history|). For large-scale datasets (20M+ interactions), the uniform+replace approach is orders of magnitude faster and empirically equivalent (negative collision rate < 0.01% for typical item catalog sizes).

---

## Subgraph Sampler

**File:** `src/data/subgraph_sampler.py`

Used by `MiniBatchTrainer` to extract k-hop subgraphs around batch nodes for mini-batch GCN training.

### SubgraphBatch

A `@dataclass` returned by `SubgraphSampler.sample()`:

| Field | Type | Description |
|-------|------|-------------|
| `sub_edge_index` | `Tensor (2, E_sub)` | Edges within the subgraph (reindexed to local node IDs) |
| `sub_edge_sign` | `Tensor (E_sub,)` | Edge signs aligned to `sub_edge_index` |
| `local_user_ids` | `Tensor (B,)` | Batch user IDs in local subgraph indexing |
| `local_pos_item_ids` | `Tensor (B,)` | Batch positive item IDs in local subgraph indexing |
| `local_neg_item_ids` | `Tensor (B,)` | Batch negative item IDs in local subgraph indexing |
| `global_node_ids` | `Tensor (N_sub,)` | Mapping from local to global node IDs |
| `n_sub_users` | `int` | Number of unique users in the subgraph |
| `n_sub_items` | `int` | Number of unique items in the subgraph |
| `sub_popularity` | `Tensor (I_sub,)` | Popularity values for subgraph items |

### Constructor

```python
SubgraphSampler(
    edge_index: Tensor,          # (2, E) full graph edges
    edge_sign: Tensor,           # (E,) full graph edge signs
    n_users: int,
    n_items: int,
    popularity: Tensor,          # (I,) global item popularity
    num_neighbors: list[int],    # per-layer fan-out (e.g., [10, 10])
)
```

### sample()

```python
def sample(
    user_ids: Tensor,            # (B,) batch user IDs
    pos_item_ids: Tensor,        # (B,) batch positive item IDs (offset by n_users)
    neg_item_ids: Tensor,        # (B,) batch negative item IDs (offset by n_users)
) -> SubgraphBatch
```

### Design Decisions

- **`k_hop_subgraph` rationale:** Uses PyG's `k_hop_subgraph` for exact k-hop neighborhood extraction rather than stochastic neighbor sampling (e.g., GraphSAINT). Exact extraction is deterministic and avoids variance from sampling, at the cost of potentially larger subgraphs for high-degree nodes. Fan-out is controlled via `num_neighbors` to bound subgraph size.
- **Users-first layout:** The subgraph reindexes nodes with users first (`[0, n_sub_users)`) and items second (`[n_sub_users, n_sub_users + n_sub_items)`), matching the full graph's node layout convention.
- **Local index reuse:** `SubgraphSampler.sample()` now sorts each local user/item ID partition once and reuses that sorted view for batch-user, positive-item, and negative-item remapping, avoiding repeated sort allocations on the same subgraph item IDs.
- **Fan-out control:** `num_neighbors` limits the number of neighbors sampled per layer, preventing subgraph explosion on high-degree nodes. For a 2-layer GCN with `num_neighbors=[10, 10]`, the worst-case subgraph is ~100 neighbors per seed node.


# Training

The training loop, evaluation pipeline, and GPU profiling system. Covers `src/training/` and `src/profiling/`.

Shared lifecycle code lives in `src/utils/trainer_runtime.py`. The shared runtime class is `TrainerRuntime`, and the concrete trainer `MiniBatchTrainer` in `src/training/mini_batch_trainer.py` keeps its mode-specific batch loop so subgraph sampling and local-index handling remain easy to audit.
On CUDA, that shared runtime also owns AMP/autocast setup, cached device-side popularity, and a reusable epoch-level batch progress bar. The runtime standardizes AMP on bfloat16 autocast and no longer exposes a separate fp16/GradScaler path.
Experiment entry now enforces one repository-wide reproducibility contract: seed Python, NumPy, and PyTorch once per run, enable deterministic torch algorithms, disable cuDNN benchmarking and TF32 shortcuts, and keep that backend policy centralized in `src/utils/reproducibility.py`. The precision policy itself remains bf16 AMP.
To avoid per-batch host/device sync pressure from tqdm, the runtime now accumulates epoch loss on-device and only syncs a Python loss scalar for the progress-bar postfix every `config.progress_bar_loss_cadence` batches (and on the final batch of the epoch).
Profiling is now opt-in through `config.enable_profiling`; default training runs prioritize throughput unless a script or experiment explicitly enables stage profiling. `GPUProfiler` instances also start disabled, so direct construction does not introduce synchronized stage timing until the runtime opts in.

`config.use_torch_compile` remains available, but it is now opt-in rather than default. In the current mini-batch-only runtime the sampled subgraph shapes and per-batch edge tensors are dynamic enough that `torch.compile(dynamic=True)` frequently recompiles `DualBranchGCN`, which hurts throughput on long formal runs instead of improving it.

**Sampler path:** `MiniBatchTrainer.train()` now prefers a CUDA-resident `SubgraphSampler` when training on GPU. The full graph is staged on the accelerator once, negative sampling plus sampled-BFS subgraph extraction run on-device, and the trainer falls back to the original four-worker CPU prefetch path only if staging the full graph would exhaust VRAM. In that CPU fallback, prepared `SubgraphBatch` objects stay pinned and transfer to CUDA with `non_blocking=True` immediately before the forward pass. Epoch shuffling now uses a seeded `torch.randperm(...)`, and each prepared batch keeps its own deterministic RNG seed. The isfinite check uses `.all()` (stays on GPU) instead of `.item()` (GPU-CPU sync) to avoid per-batch synchronization.

**EMA support:** When `config.use_ema=True`, `TrainerRuntime` creates a `torch.optim.swa_utils.AveragedModel` with `get_ema_multi_avg_fn(config.ema_decay)`. EMA weights are updated after each optimizer step, used for validation evaluation, and captured as `best_state` (with `module.` prefix stripped) for model restoration. EMA state is saved/restored with checkpoints for correct resume.

**Device transfer helpers:** All tensor device moves during training and evaluation route through shared helpers in `src/utils/trainer_runtime.py` to enforce consistent non-blocking policy, and CUDA AMP context selection now reuses the same module-level helper there as well:
- `move_tensor_to_device(tensor, device, dtype=None)` — Moves a tensor with `non_blocking=True` on CUDA, else `False`.
- `move_optional_tensor_to_device(tensor, device, dtype=None)` — Handles optional tensors; returns `None` if input is `None`.
- `stage_graph_tensors_for_device(data, device)` — Returns `(edge_index, edge_sign, edge_norm)` tuple; handles optional edge fields via `getattr()`.
- `model_device(module)` — Reads the canonical parameter device for evaluator/model hand-offs.
- `empty_cuda_cache(device)` — Centralizes CUDA allocator cache clears behind a device-aware no-op on CPU.
- `autocast_context(use_amp, amp_dtype=torch.bfloat16)` — Returns the shared CUDA autocast context used by the trainer runtime, mini-batch forward path, and evaluator.

This centralization eliminates repeated inline `tensor.to(device, non_blocking=...)`, raw `next(model.parameters()).device`, ad hoc `torch.cuda.empty_cache()` calls, and ad hoc autocast branches across `mini_batch_trainer.py`, `evaluator.py`, and optimizer state management. Device transfer calls now have one canonical policy: synchronous on CPU (default), asynchronous on CUDA.

---

## MiniBatchTrainer

**File:** `src/training/mini_batch_trainer.py`

### Constructor

```python
MiniBatchTrainer(
    model: UCaGNN,
    loss_suite: LossSuite,
    data: Data,                      # PyG Data from build_graph()
    config: UCaGNNConfig,
    profiler: GPUProfiler | None,
    experiment_logger: ExperimentLogger | None,
    exp_id: int | None,
)
```

Inherits from `TrainerRuntime` which sets up:
- Adam optimizer over both model and loss_suite parameters (the `PopularityPredictor` in `LossSuite` has learnable weights).
- `NegativeSampler` configured from `config.n_negatives` and `config.hard_negative_ratio`.
- `Evaluator` for validation metrics.
- Early stopping state: `best_ndcg`, `patience_counter`, `best_state`.
- `SubgraphSampler` for k-hop subgraph extraction around batch seed nodes.

**Hyperparameter evidence:** Adam with lr=1e-3 matches the LightGCN paper ("0.001 using Adam"). Patience=10 follows DDCE ("early stopping after 10 consecutive epochs"). The thesis runtime standardizes on K=20 and K=40, matching the main cutoffs used across CaDCR-style reporting while keeping one visible-list and one deeper-list view.

**Gradient clipping:** `torch.nn.utils.clip_grad_norm_` with `max_norm=1.0` (configurable via `config.grad_clip_norm`) is applied after `loss.backward()` and before `optimizer.step()`. This follows DICE's recommendation to prevent gradient explosions, which is especially important with 5 loss terms and IPW weights that can reach 100x.

**Curriculum scheduling (CaDCR connection):** The epoch-based curriculum follows CaDCR's multi-task approach: progressive tasks with increasing difficulty. Phase 1 keeps fused BPR and the branch BPR auxiliaries on from the start, Phase 2 adds independence, within-branch contrastive, and optional DirectAU geometry, and Phase 3 adds popularity supervision once the branch structure is stable. The runtime defaults are `auxiliary_losses_start_epoch=15` / `auxiliary_losses_start_epoch` and `popularity_supervision_start_epoch=30` / `popularity_supervision_start_epoch`. Those thresholds remain configurable for local sensitivity checks, but the public thesis ablation matrix is intentionally limited to `mainline`, `no_popularity_head`, `no_independence`, and `no_features`.

**Loss schedule:** supported runs keep `config.loss_schedule="baseline"`. Fused BPR stays active throughout the curriculum, and only the auxiliary terms phase in via `auxiliary_loss_schedule`, `auxiliary_losses_start_epoch`, and `popularity_supervision_start_epoch`.

### Per-batch flow

The training loop now has two execution modes:

- CUDA sampler mode: negative sampling and sampled-BFS subgraph extraction run on the accelerator, so the hot path stays on-device before the forward pass.
- CPU fallback mode: while the current batch runs steps 3-5 on GPU, the next batch's steps 1-2 execute concurrently in a background thread and arrive pinned for asynchronous transfer.

1. **Sample negatives** -- `NegativeSampler` provides hard negative items (via `_prepare_batch()`).
2. **Sample subgraph** -- `SubgraphSampler.sample()` extracts a k-hop subgraph around the batch's user and item nodes (via `_prepare_batch()`).
3. **Forward on subgraph** -- `UCaGNN.forward_subgraph()` indexes into full embedding tables using subgraph node IDs, runs GCN on the local `sub_edge_index`, and scores with local indices. Profiled as `"forward"`.
4. **Local popularity** -- popularity values are indexed from the global `data.popularity` tensor using the subgraph's item mapping.
5. **Loss + backward** -- standard loss computation and optimizer step on the subgraph outputs. The branch-alignment losses (`L_ortho`, `L_contr`) are evaluated on the current batch users, not every context user in the sampled subgraph, so mini-batch memory stays tied to the training batch rather than growing quadratically with subgraph size. Profiled as `"loss"` and `"backward"`.
6. **Validation** -- shared `_finalize_epoch()` handles evaluation (using EMA weights when available), early stopping, checkpointing.

### Batch Structure

Each training batch contains:

| Tensor | Shape | Source |
|--------|-------|--------|
| `batch_users` | `(B,)` | Shuffled training user IDs |
| `batch_pos_items` | `(B,)` | Corresponding positive item IDs |
| `batch_neg_items` | `(B,)` | Sampled negative item IDs |

Mini-batch training extracts a k-hop subgraph around batch nodes via `SubgraphSampler`, then runs GCN propagation only on the local subgraph. This keeps VRAM usage proportional to the batch neighbourhood size rather than the full graph, enabling training on large datasets.

`batch_size` controls both the number of interactions per loss/optimizer step and the size of the extracted subgraph. Fixed-batch runs still default to `4096`, but CUDA runs can now enable `auto_batch_size` so the runtime mirrors the real epoch-0 shuffle, probes several representative mini-batches, synchronizes CUDA before releasing probe allocations, and picks the largest surviving candidate before checkpoint identity, canonical naming, and logging are frozen. The experiment entry also defaults `PYTORCH_ALLOC_CONF=expandable_segments:True` so the allocator can reuse segments more safely across probe/train handoff. The built-in candidate ladders are dataset-aware support defaults rather than thesis axes and now extend down to `256`, which matters for dense sampled subgraphs such as Amazon-Book. `num_neighbors` limits per-hop fan-out to control subgraph density and VRAM; the current default is `[10, 5]`, and formal profiles can optionally expand several such shapes through `num_neighbors_options`.

When running on CUDA, the shared runtime wraps the forward/loss path in autocast so the batch loop benefits from Tensor Core acceleration without changing the trainer API. `TrainerRuntime` now also enables cuDNN autotuning plus TF32 matmul mode (`torch.set_float32_matmul_precision("high")`, `torch.backends.cuda.matmul.allow_tf32=True`) so the scoring and feature-projection MLPs can use the faster Tensor Core kernels on modern NVIDIA GPUs. The epoch progress bar reports batch progress for the active epoch while leaving the existing log summary at epoch boundaries.

### Early Stopping

- Primary metric: `NDCG@40`.
- If validation NDCG improves: save model state, reset patience counter.
- If `config.use_early_stopping=True` and there is no improvement for `config.patience` consecutive epochs after the curriculum warmup window: stop and restore best state.
- Default patience: 10 epochs, but patience counting is deferred until `max(auxiliary_losses_start_epoch, popularity_supervision_start_epoch)` so staged training is not terminated before later loss phases activate.
- Disabling early stopping does not disable best-model tracking. The trainer still records the best validation checkpoint and restores it after the full epoch budget completes.

Validation and test metrics are computed with `config.eval_scoring_mode`, so early stopping can now be driven by the default score mixture or by an intervention-style evaluation view such as `interest_only` or `conformity_suppressed`.
`config.train_scoring_mode` now follows the preset contract explicitly: `preset_lightgcn()` keeps the default score, `preset_dice_like()` uses the fixed default interest+conformity score, and `preset_full()` keeps the fused default score unless a same-checkpoint evaluation script overrides `config.eval_scoring_mode`.
Validation/test output is intentionally restricted to the thesis-facing PyG metrics: `NDCG@20`, `Recall@20`, `AveragePopularity@20`, `HitRatio@20`, `Personalization@20`, `NDCG@40`, `Recall@40`, `AveragePopularity@40`, `HitRatio@40`, and `Personalization@40`.
The evaluator builds those metrics as a PyG `LinkPredMetricCollection` keyed by metric name. MetricCollection removes per-metric runtime update loops, but it still needs one metric instance per metric and cutoff at construction time.
The evaluator now rematerializes `edge_index`, `edge_sign`, `edge_norm`, and `popularity` on the active evaluation device for each validation/test call instead of holding a persistent device-side full-graph cache across epochs. On CUDA it also runs the full-graph propagation and scoring path under bf16 autocast, explicitly casts the evaluation embedding bundle to bf16, and reclaims the CUDA-resident sampler before validation so full-graph propagation has the VRAM budget back on large datasets such as MovieLens20M.
`src/training/evaluator.py` is also the single source of truth for the thesis-primary metric tuple and lower-is-better metric polarity, so downstream benchmark, ablation, reporting, and scoring-mode scripts import those constants instead of re-declaring them.
Treat both AveragePopularity metrics as lower-is-better.
The main mechanism check should reuse the same checkpoint under `default`, `interest_only`, and `conformity_suppressed` scoring. This keeps the causal analysis aligned with the current runtime contract: branch-sensitive ranking behavior is tested directly, while PropCare-style `CPrec` and `CDCG` remain out of scope unless a separate treatment/propensity/effect evaluation table is introduced.
Use `uv run evaluate-scoring-modes --checkpoint-path ...` when you want the thesis mechanism table from one trained checkpoint without retraining separate runs under different evaluation modes. The command reuses `run_experiment.py` helpers for checkpoint validation and runtime dataset/graph/model reconstruction instead of carrying a parallel copy of that logic.
PyG 2.7 also exposes Diversity and Personalization. The runtime evaluator now keeps `Personalization@20/40` as part of the thesis-facing metric set and defines the degenerate tiny-split case with fewer than two evaluated users as `0.0` so smoke validation stays finite. Diversity remains audit-only until the runtime contract grows category metadata for it.
External implementation audits may discuss non-PyG causal-uplift evaluators such as PropCare's semi-simulated `CPrec` or `CDCG` pipeline, but those are documentation-only reference analyses unless the runtime data contract is extended with treatment, propensity, and causal-effect labels.

### Checkpointing

```python
trainer.save_checkpoint("checkpoints/ucagnn.pt")
trainer.load_checkpoint("checkpoints/ucagnn.pt")
```

Saves/loads: model state dict, loss_suite state dict, optimizer state dict, config, and best NDCG score.

### Return Value

`MiniBatchTrainer.train()` returns a history dict:

| Key | Type | Description |
|-----|------|-------------|
| `"train_loss"` | `list[float]` | Average total loss per epoch |
| `"val_metrics"` | `list[dict]` | Validation metrics per epoch |

Named recipes own the matrix-defining fields they declare. Conflicting CLI flags should be considered an invalid command; use a different recipe alias or drop `--recipe` and pass `--preset` with explicit matrix flags instead.

Formal matrix runners now support explicit batch orchestration metadata through `--batch-id` and `--resume-batch`. This is intentionally separate from checkpoint auto-resume inside a single run: SQLite tracks whether each matrix item finished as `completed`, failed with `oom`, or failed generically, and batch resume skips only those terminal rows.

`experiments/run_benchmark.py::formal_main()` now provides the repository's simplest formal entry point through `uv run formal-run`. It persists the last formal batch plan to `results/formal_run_state.json`, so re-running the same semantic profile or calling `uv run formal-run --resume-latest` continues the formal matrix after an interrupted workday or overnight run.

`results/formal_run_state.json` is only a generated resume pointer. It stores the current user-facing `profile_name` id, deterministic `profile_slug`, operational `batch_id`, runtime args, and simple run timestamps/status. The formal matrix definition itself still lives in `experiments/experiment_catalog.json`, so the state file should not be treated as a hand-maintained or authoritative profile snapshot. Keep the nested `benchmark_args` payload normalized once at the boundary so saved JSON, semantic plan matching, and resumed execution all reuse the same dict schema instead of drifting across separate helper lists.

`formal-run` now separates two identities that used to be conflated:
- `profile_name`: the short explicit JSON identifier passed to `--profile` (for example `dev` or `final`), meant to be easy to remember and visible directly in `experiments/experiment_catalog.json`.
- `profile_slug`: the deterministic semantic signature derived from that bundle, used to make the exact support-parameter contract visible in logs and saved state. Auto-batch profiles now use an `abauto` slug fragment instead of pretending they are pinned to a fixed `bs4096` schedule.
- `batch_id`: the operational execution label used to resume or restart a concrete long-running sweep.

Both are logged in SQLite and MLflow so result inspection can answer two different questions cleanly: which formal protocol produced the run, and which specific overnight batch execution it belonged to.

Formal profiles always run on full datasets. Sampled-interaction and loader-cap controls remain available for quick validation and other smoke-scale workflows, but they are intentionally excluded from the formal wrapper so the thesis entry point stays semantically clean.

The default formal profile currently focuses on the `ucagnn` preset only, disables early stopping, runs the full 60 planned epochs, keeps learned fused scoring, and spells out the mainline asymmetric propagation shape with `interest_gnn_layers=1`, `conformity_gnn_layers=2`, and `num_neighbors=[10, 5]`. This keeps all ongoing thesis compute on the proposed architecture while preserving the complete training trajectory before early-stop behavior is treated as part of the formal protocol.

A second formal profile is reserved for the end-stage matched comparison pass. It reintroduces `lightgcn` and `dice_like` under the same evaluator and logging pipeline, keeps that baseline sweep out of day-to-day tuning runs, and can now expand several `num_neighbors` fan-out shapes under one profile bundle.

Although the semantic matrix is now `dataset × preset`, benchmark and `formal-run` still execute that plan with datasets as the innermost loop. Profile-owned `scoring_weight_modes` can still add explicit fixed-vs-learned comparisons where needed, but the default day-to-day `ucagnn` profile keeps learned fusion only. In practice this means one preset bundle is tried across all datasets before the runner advances to the next bundle, so dataset-specific failures surface early instead of after a single dataset has already consumed every preset variant.

Benchmark and `formal-run` no longer expose seed as a matrix axis or public orchestration flag. They use the repository's single default seed internally, while the formal matrix remains `dataset × preset` and any score-mix comparison is an explicit profile-owned support choice. The `lightgcn` preset stays fixed-only because learned score weights are inapplicable without dual branches.

Benchmark and `ablation` now follow the same OOM policy as `formal-run`: log the failure and continue. There is no fallback retry flag anymore.

Keep the ablation runner's resume query direct at its only call site; a single `ExperimentLogger.find_latest_batch_experiment()` use does not need its own wrapper helper.

The SQLite tracking layer now exposes three convenience exploration views on top of `experiment_summary`: `experiment_completed_summary`, `experiment_attention_summary`, and `experiment_error_summary`. They are the intended read path for quick experiment review when you want only clean completions, anything still needing attention, or strict failures. `query-results --view ...` is the supported CLI on top of those views.

The supported query-view mapping now lives in `ExperimentLogger.VIEW_TABLES`, and `scripts/query_results.py` imports that registry instead of re-declaring the valid view names. Keep view ownership in the logger layer so schema definition and CLI filtering stay aligned.

Keep one-use logger write-path details local to the public methods that own them: config serialization belongs in `ExperimentLogger.log_experiment()`, and per-epoch profiler-stage aggregation belongs in `ExperimentLogger.log_epoch()` rather than living behind extra private wrapper helpers for those single call sites.

Because `scripts/query_results.py` already opts into `sqlite3.Row`, its detail and profiling renderers should read named columns rather than positional indexes so query column reordering does not silently break the CLI.

---

## Quick Validation Workflow

Formal training should be preceded by one small but representative smoke pass:

```bash
uv run quick-validate
```

`quick_validate.py` is now the repository's only supported tiny-scale validation entry point. The older preflight and feature-probe scripts were removed so the workflow stays concentrated on one file instead of several partially overlapping harnesses.

By default the validator runs across all six thesis datasets and covers:
- every canonical experiment recipe in the frozen preset matrix,
- every named ablation variant,
- observability probes for profiling, checkpoint save/load/auto-resume, and feature-aware execution paths, and
- the supported evaluation scoring modes: `default`, `interest_only`, and `conformity_suppressed`.

Checkpoint payload loading stays centralized in `experiments/run_experiment.py::load_checkpoint_payload()`, so quick validation uses the same schema guard as runtime auto-resume and same-checkpoint scoring-mode evaluation.

`quick_validate.py` still keeps its own tiny dataset caps and inline timing around `run_experiment()`, rather than routing those one-command details through `scripts/_workflow_helpers.py`. Its parser definition now comes from the shared `src/utils/cli_parsers.py` module, while the tiny-run workflow logic stays local to the script.

The script keeps runs small through aggressive loader caps, sampled interactions, and one-epoch defaults, so it behaves like a tiny end-to-end experiment suite rather than a benchmark runner. Use category filters only when debugging a specific surface.
Feature-aware runs follow the formal config default, so the validator exercises the same side-feature path as formal experiments. Capped dataset loads are reused in-process, which keeps repeated recipe coverage practical without dropping feature enrichment from the default tiny run.
MLflow is disabled by default in quick validation, so the standard command does not create `results/mlflow.db` or `mlruns/`. Pass `--mlflow` only when you explicitly want the MLflow observability probe.
Use `query-results` as the supported SQLite inspection path after runs. The repository currently does not expose a supported plotting command in the main workflow.
`query-results` now also supports `--view`, `--batch-id`, and `--status`, and its default table truncates long `profile_name` values so dynamically generated profile slugs remain aligned in the summary table.
Keep one-command runtime details local even though parser definitions are centralized. `scripts/_workflow_helpers.py` should stay limited to genuinely shared helpers such as CLI logging setup, batch-id generation, shared summary counters, and shared metric fallback logic, while one-command concerns like tiny dataset caps, inline timing, and same-checkpoint JSON writing remain local to their scripts.

The intended validation invariant is: formal and tiny runs use the same loader registry, canonical schema, and feature-engineering path, with only row scaling and runtime controls changed. In practice this means `loader_max_rows` and `sample_interactions` may shrink the data, but they should not silently redefine which fields exist or which feature transforms run.

### Sampled Runner Support

`experiments/run_experiment.py` accepts `--sample-interactions` for preflight-style runs. This samples the canonical interaction table before graph construction while preserving split coverage as much as possible. It is intended for smoke checks, not for formal metrics.

This split-preserving row scaling is the current answer to representativeness in tiny runs: the path remains semantically aligned with the formal experiment, while the sample size is reduced for cost. If a future feature block requires skipping an expensive optional scan during tiny validation, add that exception explicitly and keep at least one dedicated feature-path observability probe enabled.

### MLflow Tracking URI Resolution

`experiments/run_experiment.py` resolves MLflow tracking in this order:

1. `--mlflow-tracking-uri`
2. `MLFLOW_TRACKING_URI`
3. project default `results/mlflow.db`

This keeps formal runs pinned to the repository-local MLflow database unless an explicit override is provided. Plain `mlflow ui` or `mlflow server` commands without `--backend-store-uri` use MLflow's separate default `mlflow.db`, which is not the thesis tracking database for this repository.

Run metadata is logged with non-duplicated MLflow fields: matrix-defining values are stored as params, while operational state remains in tags. Each run also records `run_started_at_utc`, `project_version`, `git_commit`, `training_hash`, `evaluation_hash`, and an optional `change_note` so ordering, code provenance, and semantic identity remain visible in both MLflow and SQLite.

`results/mlflow.db` is the MLflow backend store and `mlruns/` holds artifacts. Benchmark runs default to MLflow experiment `ucagnn-benchmark`; ablations default to `ucagnn-ablation`. Checkpoints are logged to MLflow under the `checkpoints/` artifact subpath. Use `uv run reset-experiment-db` to delete the repository-local thesis SQLite database and sidecars. Use `uv run cleanup-experiment-artifacts` to delete repository-local MLflow state, the generated `results/formal_run_state.json` resume pointer, and checkpoints in one step.

### Formal Matrix Runner

`experiments/run_benchmark.py` now iterates the frozen orchestration axes:

```text
dataset × preset
```

Dual-branch presets (`ucagnn`, `dice_like`) also sweep `scoring_weight_mode ∈ {fixed, learned}` inside that matrix. `lightgcn` stays fixed-only.

Use `--dry-run` to inspect the matrix before executing it.

---

## Evaluator

**File:** `src/training/evaluator.py`

### Constructor

```python
Evaluator(config: UCaGNNConfig)
```

Creates PyG metric objects for each k in `config.eval_ks`:
- `LinkPredPrecision(k=k)`
- `LinkPredRecall(k=k)`
- `LinkPredF1(k=k)`
- `LinkPredMAP(k=k)`
- `LinkPredNDCG(k=k)`
- `LinkPredMRR(k=k)`
- `LinkPredHitRatio(k=k)`
- `LinkPredCoverage(k=k, num_dst_nodes=n_items)`
- `LinkPredAveragePopularity(k=k, popularity=data.popularity)`

### evaluate()

```python
@torch.no_grad()
def evaluate(
    model: UCaGNN,
    data: Data,
    mask: Tensor,           # boolean mask selecting interactions
    batch_size: int = 512,
) -> dict[str, float]       # e.g., {"Recall@10": 0.15, "NDCG@20": 0.12}
```

### Evaluation Flow

1. **Extract** unique users from the masked interactions.
2. **Build sparse ground truth** from the masked `(user, item)` pairs.
3. **Batch over users**: for each batch, call `model.score_users_from_propagated()` on the cached propagated embeddings to get the `(B, I)` score matrix.
4. **Mask all observed non-target interactions** before `topk`, so train/val positives outside the target split cannot occupy the ranked list.
5. **Top-k selection**: `torch.topk(scores, max_k)` to get predicted item indices.
6. **Update metrics**: feed predictions and ground-truth to the `LinkPredMetricCollection` once, which updates the full configured metric bundle for the batch.
7. **Compute and return** final metric values.

### Design Notes

- Users with no ground-truth items in the mask are skipped.
- All computation stays on GPU (metrics are moved to the model's device).
- Eval batch size (512) is separate from training batch size to manage VRAM during full-catalog scoring.
- `model.score_users_from_propagated()` honors `config.eval_scoring_mode`, allowing intervention-aware Recall/NDCG without retraining.

---

## GPU Profiler

**File:** `src/profiling/gpu_profiler.py`

### Classes

**StageMetrics** (dataclass):

| Field | Description |
|-------|-------------|
| `name` | Stage identifier (e.g., "forward", "backward") |
| `elapsed_ms` | Wall-clock time in milliseconds |
| `vram_before_mb` | VRAM allocated before stage |
| `vram_after_mb` | VRAM allocated after stage |
| `vram_peak_mb` | Peak VRAM during stage |
| `vram_delta_mb` | Property: `after - before` |

**GPUProfiler** (dataclass):

| Method | Description |
|--------|-------------|
| `reset()` | Clear stage list and reset CUDA peak memory stats |
| `stage(name)` | Context manager that records timing and VRAM for a named stage |
| `summary()` | Formatted multi-line string with per-stage breakdown and percentages |
| `model_summary(model)` | Static: parameter count and model size via PyG's `count_parameters` and `get_model_size` |
| `data_summary(data)` | Static: data object size via PyG's `get_data_size` |

### Usage in Trainer

The profiler is optional. When provided, each training stage is wrapped:

```python
with profile_stage("forward", self.profiler):
    output = self.model(...)
```

The `profile_stage` convenience function is a no-op if profiler is `None`.

### Profiled Stages

| Stage | What It Measures |
|-------|-----------------|
| `forward` | Full UCaGNN forward pass (embed + GCN + score + IPW) |
| `loss` | LossSuite computation (all 5 loss terms) |
| `backward` | Gradient computation + optimizer step |
| `eval` | Full validation evaluation pass |

Negative sampling and subgraph extraction now run in the CPU prefetch pipeline ahead of the profiled GPU forward/backward stages, so they are intentionally not reported as separate profiler stages.

### Example Output

```
=== GPU Profile ===
    forward                 45.2ms (42.5%) | VRAM peak 1024 MB
    loss                     8.1ms ( 7.6%) | VRAM peak 1024 MB
    backward                41.9ms (39.4%) | VRAM peak 1536 MB
    eval                    11.2ms (10.5%) | VRAM peak 1024 MB
    TOTAL                  106.4ms
```

---

## Module Re-exports

| Package | `__init__.py` Exports | Used By |
|---------|----------------------|---------|
| `src/training/` | `MiniBatchTrainer`, `Evaluator`, thesis metric constants | Main entry point |
| `src/profiling/` | `GPUProfiler`, `profile_stage` | Trainer |


# Config Reference

Complete parameter reference for `UCaGNNConfig`, the single dataclass that controls the entire system.

**File:** `src/utils/config.py`

---

## Usage

```python
from src.utils.config import UCaGNNConfig

# Default (full U-CaGNN)
config = UCaGNNConfig()

# LightGCN baseline
config = UCaGNNConfig().preset_lightgcn()

# DICE-like variant
config = UCaGNNConfig().preset_dice_like()

# Custom
config = UCaGNNConfig(
    embed_dim=128,
    single_branch_gnn_layers=2,
    interest_gnn_layers=1,
    conformity_gnn_layers=2,
    dataset="kuairec_v2",
)
```

Public experiment naming note: use `ucagnn` as the main CLI preset/recipe name. The internal config method name stays `preset_full()`.

---

## Architecture Toggles

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_dual_branch` | `bool` | `True` | Separate interest/conformity user embeddings and GCN branches |
| `use_sign_aware` | `bool` | `True` | Learnable alpha_pos/alpha_neg edge weights in GCN |
| `use_counterfactual` | `bool` | `True` | Enable `counterfactual_score = interest - conformity` diagnostics |
| `use_ipw` | `bool` | `True` | Enable PropensityEstimator for inverse propensity weighting |
| `use_popularity_emb` | `bool` | `True` | Additional item popularity embedding table |
| `use_torch_compile` | `bool` | `False` | Opt-in only. Dynamic mini-batch subgraphs currently hit Dynamo recompiles often enough that compile is not the default training path. |
| `use_ema` | `bool` | `False` | Optional EMA via `torch.optim.swa_utils.AveragedModel`; smooths generalization |
| `ema_decay` | `float` | `0.999` | EMA exponential decay rate (higher = slower update) |

---

## Graph Construction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cagra_k` | `int` | `20` | Number of ANN neighbors returned per query for CAGRA augmentation |
| `cagra_out_degree` | `int` | `32` | CAGRA graph degree (final). Thesis-speed default; cuVS library default is 64. |
| `cagra_initial_degree` | `int` | `64` | CAGRA intermediate graph degree. Thesis-speed default; cuVS library default is 128. |
| `cagra_team_size` | `int` | `0` | CAGRA search team size. 0 = auto-select (cuVS default). |
| `cagra_metric` | `str` | `"inner_product"` | Distance metric; inner_product matches LightGCN dot-product scoring. |
| `cagra_itopk_size` | `int` | `64` | Intermediate top-k per step during search. Higher = better recall. |

---

## Embedding / GNN Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_dim` | `int` | `64` | Embedding dimension (D) for users and items |
| `pop_embed_dim` | `int` | `16` | Popularity embedding dimension (D_pop) |
| `single_branch_gnn_layers` | `int` | `2` | Dedicated LightGCN / non-dual-branch depth |
| `interest_gnn_layers` | `int` | `1` | Interest-branch depth for dual-branch runs |
| `conformity_gnn_layers` | `int` | `2` | Conformity-branch depth for dual-branch runs |
| `dropout` | `float` | `0.1` | Dropout rate (reserved, surfaced in the catalog, not currently applied) |

**Paper evidence for defaults:**
- `embed_dim=64`: consensus across LightGCN, DICE, MGCE, FMMRec, and DDCE -- all use d=64 as their primary embedding dimension.
- `single_branch_gnn_layers=2`: dedicated LightGCN / non-dual-branch depth.
- `interest_gnn_layers=1`, `conformity_gnn_layers=2`: current dual-branch default and the mainline `ucagnn` shape.
- `dropout=0.1`: surfaced in the formal profiles as a reserved config knob; the current model path does not yet consume it.

---

## Scoring Weights

Control how the three score components combine into `final_score`:

```
final_score = alpha * interest + beta * conformity + gamma * cf
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scoring_weight_mode` | `"fixed" \| "learned"` | `"fixed"` | Whether the default score mixture uses the config weights directly or learns simplex-constrained mixture weights |
| `alpha_interest` | `float` | `0.5` | Weight on interest score |
| `beta_conformity` | `float` | `0.3` | Weight on conformity score |
| `gamma_popularity` | `float` | `0.2` | Weight on popularity score |
| `train_scoring_mode` | `"default" \| "interest_only" \| "conformity_only" \| "conformity_suppressed"` | `"default"` | Controls which score view feeds the ranking loss |

**Scoring weight rationale:** Defaults weight interest most heavily (50%), conformity secondary (30%), and popularity as a lightweight calibration/debiasing signal (20%). DICE uses interest only (equivalent to alpha=1, beta=0, gamma=0); nonzero beta and gamma differentiate U-CaGNN from DICE by incorporating conformity and popularity into the final ranking score. `scoring_weight_mode="learned"` keeps the same three components but lets the model learn a simplex-constrained mixture over the active components.

## Evaluation Scoring Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_scoring_mode` | `"default" \| "interest_only" \| "conformity_only" \| "conformity_suppressed"` | `"default"` | Controls how Recall/NDCG are computed at validation and test time |

`train_scoring_mode` and `eval_scoring_mode` are independent. `preset_lightgcn()` trains and evaluates with the default single-branch score, `preset_dice_like()` aligns both to the fixed default interest+conformity score, and `preset_full()` aligns both to the fused `default` view. Same-checkpoint mechanism evaluation changes only `eval_scoring_mode`, reusing the trained checkpoint and propagated embeddings without retraining. Preset helpers now also rewrite their preset-owned fields explicitly, so switching presets on one `UCaGNNConfig` instance does not preserve stale values from an earlier preset. For fixed score weights, the single-component intervention modes are renormalized to the pure component score.

---

## Loss Lambdas

Control the weight of each loss term in the total loss. Setting a lambda to `0.0` disables that loss entirely.

```
L_total = lambda_rec * L_rec + lambda_ortho * L_ortho + lambda_contr * L_contr
        + lambda_cf * L_cf + lambda_pop * L_pop
```

| Parameter | Type | Default | Loss Term |
|-----------|------|---------|-----------|
| `lambda_rec` | `float` | `1.0` | L_rec (BPR ranking) |
| `lambda_ortho` | `float` | `0.01` | L_ortho (orthogonality) |
| `lambda_contr` | `float` | `0.05` | L_contr (NT-Xent contrastive) |
| `lambda_cf` | `float` | `0.05` | L_cf (counterfactual divergence) |
| `lambda_pop` | `float` | `0.1` | L_pop (popularity prediction) |

---

## Contrastive Loss

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contrastive_tau` | `float` | `0.1` | NT-Xent temperature (lower = sharper) |

**Note:** Default tau=0.1 is standard (SimCLR uses the range 0.1-0.5). DCCL does not report a specific tau value. Should be tuned per dataset -- lower tau produces sharper intent distinctions but increases gradient magnitude.

---

## Propensity (IPW)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `propensity_hidden` | `int` | `128` | Hidden layer size for propensity MLP |
| `propensity_clip_min` | `float` | `0.01` | Lower bound for propensity scores |
| `propensity_clip_max` | `float` | `0.99` | Upper bound for propensity scores |

**Implied IPW weight range:** With the default clip bounds `[0.01, 0.99]`, the inverse propensity weights `1/P(exposure)` range from `1/0.99 = 1.01` to `1/0.01 = 100.0`. If training is unstable (loss spikes, gradient explosions), increase `propensity_clip_min` to reduce the maximum IPW weight (e.g., `propensity_clip_min=0.1` yields a max weight of 10.0).

---

## Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | `1e-3` | Adam learning rate |
| `weight_decay` | `float` | `1e-5` | L2 regularization |
| `batch_size` | `int` | `4096` | Fallback training batch size |
| `epochs` | `int` | `60` | Maximum training epochs |
| `grad_clip_norm` | `float` | `1.0` | Max gradient norm for clipping (DICE recommendation) |
| `patience` | `int` | `10` | Early stopping patience (epochs without improvement) |
| `use_early_stopping` | `bool` | `True` | Whether training may terminate before `epochs` based on validation `NDCG@40` |
| `lr_scheduler` | `"none" \| "plateau"` | `"none"` | Optional LR scheduler. `"plateau"` enables ReduceLROnPlateau monitoring NDCG. |
| `lr_scheduler_factor` | `float` | `0.5` | Factor to reduce LR by when plateau detected |
| `lr_scheduler_patience` | `int` | `5` | Epochs with no improvement before LR reduction |
| `eval_ks` | `list[int]` | `[20, 40]` | Thesis-standard K values for Recall@K and NDCG@K |
**Paper evidence for training defaults:**
- `lr=1e-3`: LightGCN paper: "0.001 using Adam."
- `batch_size=4096`: the repository now uses this as the fixed-batch default. CUDA runs can instead enable `auto_batch_size=True`, which probes representative shuffled batches with dataset-aware candidates (`[16384, 8192, 4096, 2048, 1024]` for MovieLens 1M, `[8192, 4096, 2048, 1024, 512, 256]` for Amazon-Book, `[4096, 2048, 1024, 512, 256]` for MovieLens 20M / KuaiRec, `[2048, 1024, 512, 256, 4096]` for KuaiRand after the fallback append) and freezes the largest surviving value into the run identity. This keeps batch size as an efficiency support parameter rather than a thesis axis.
- `weight_decay=1e-5`: LightGCN optimal is 1e-4; DDCE uses 1e-5. The slightly lower value may suit the additional loss terms in U-CaGNN.
- `epochs=60`: Aligns with thesis_plan.md (3 curriculum phases x 20 epochs each). The previous 100-epoch default was not grounded in any paper; 60 matches CaDCR/MGCE convergence behavior.
- `patience=10`: DDCE: "early stopping after 10 consecutive epochs" of validation decline.
- `use_early_stopping=True`: keeps non-formal runs aligned with the current best-checkpoint workflow; the default formal profile overrides this to `False` so the development-focused `ucagnn` sweep completes the full 60 epochs while still restoring the best validation checkpoint afterward.
- `use_torch_compile=False`: the current mini-batch-only runtime feeds highly dynamic subgraph structures into `DualBranchGCN`, and the observed `torch.compile(dynamic=True)` path hits enough recompiles to hurt throughput instead of helping it. Keep compile opt-in until a profiling-backed stable path exists.
- `eval_ks=[20, 40]`: matches the repository's thesis-standard visible-list and deeper-list cutoffs while staying aligned with the common K=20/K=40 reporting pattern in the causal recommendation literature.
- `grad_clip_norm=1.0`: DICE recommends `max_norm=1.0` to prevent gradient explosions, especially important with 5 loss terms and IPW weights up to 100x.

Formal profile note: the repository's `uv run formal-run --profile ...` entry point now freezes a predefined support-parameter bundle outside `UCaGNNConfig` orchestration, then passes the resolved values into the benchmark runner. This keeps semantic thesis protocols distinct from operational batch-resume metadata. The current formal profiles no longer pin an explicit batch-size override in the catalog; they enable `auto_batch_size` with a shared descending candidate ladder and let the runtime resolve the actual CUDA batch size per dataset before logging and checkpoint hashing. Formal profiles may also declare `num_neighbors_options`, which expands to several resolved benchmark items while preserving one concrete `num_neighbors` vector per run. The default formal bundle remains development-focused: it runs only the `ucagnn` preset and sets `use_early_stopping=False` so iterative U-CaGNN sweeps always complete the full epoch budget. A second final-comparison profile reintroduces `lightgcn` and `dice_like` only for the end-stage matched baseline pass.

When full-graph validation still hits a CUDA OOM on very large catalogs such as KuaiRand even after the sampler copy is released, the runtime now retries once after temporarily moving optimizer state tensors to CPU. That keeps the OOM handling explicit while letting the evaluation path use VRAM that would otherwise stay occupied by Adam state during the validation step.
The formal bundle now also forwards support parameters such as `hard_negative_ratio`, `auxiliary_losses_start_epoch`, and `popularity_supervision_start_epoch` through the formal-run orchestration path instead of leaving them stranded in the catalog.

---

## Training Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_neighbors` | `list[int]` | `[10, 5]` | Per-layer fan-out for subgraph sampling. Length must equal `max_gnn_layers`; once explicit branch-depth overrides are applied after preset defaults, mismatched fan-out lists now fail loudly instead of silently reverting to the preset shape. The default matches the current 1/2-hop dual-branch configuration and keeps the second hop smaller. |
| `sample_interactions` | `int \| None` | `None` | Optional interaction budget for sampled runs such as quick validation. When set, the runner samples interactions, preserves train/val/test coverage as much as possible, and reindexes the sampled user/item universe to a smaller temporary graph. |

**`__post_init__` validation:** `len(num_neighbors)` must equal the effective maximum branch depth. With asymmetric branches this is `max(interest_gnn_layers, conformity_gnn_layers)`.

**Fan-out tuning:** Higher `num_neighbors` values improve normalization accuracy but increase subgraph size (and VRAM). Default `[10, 5]` balances neighborhood coverage with memory for the current two-hop setup.

**Sampled-run guidance:** `sample_interactions` is intended for smoke tests and preflight only. It should not be used for formal thesis measurements because it changes the effective dataset size.

---

## Negative Sampling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_negatives` | `int` | `1` | Number of negative samples per positive |
| `hard_negative_ratio` | `float` | `0.0` | Fraction of negatives that are popularity-weighted (0.0 = all uniform) |

---

## Curriculum Schedule

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auxiliary_losses_start_epoch` | `int` | `15` | Epoch when independence, within-branch contrastive, and optional DirectAU auxiliaries activate (0 = from start) |
| `popularity_supervision_start_epoch` | `int` | `30` | Epoch when popularity supervision activates (0 = from start) |
| `loss_schedule` | `"baseline"` | `"baseline"` | Fused BPR is active throughout the curriculum; this field remains baseline-only for supported runs. |

**Default behavior:** CaDCR-inspired staged curriculum: Phase 1 (epochs 0-14) fused BPR + branch BPR, Phase 2 (epochs 15-29) adds independence, within-branch contrastive, and optional DirectAU geometry, Phase 3 (epochs 30+) adds popularity supervision. Set both thresholds to 0 for joint training from epoch 0.

**Ablation note:** Curriculum thresholds remain configurable for local sensitivity checks, but the public thesis ablation matrix is intentionally limited to `mainline`, `no_popularity_head`, `no_independence`, and `no_features`.

**Loss schedule note:** Supported runs keep `loss_schedule="baseline"`. Do not reintroduce delayed-BPR schedules; the intended contract is BPR from epoch 0 with only the auxiliary terms phased in.

---

## Side Features

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_features` | `bool` | `True` | Enable canonical side-feature usage when available; current model logic uses item features in Module A and falls back to ID-only item embeddings otherwise |
| `feature_policy` | `"thesis_default" \| "all_optional"` | `"thesis_default"` | Controls which optional side-feature scans are permitted into the canonical path. `thesis_default` enforces the thesis-safe allowlist; `all_optional` restores the broader optional scans for explicit ablations. |

---

## Data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `str` | `"movielens1m"` | Dataset name (must match a key in `LOADERS`) |
| `data_dir` | `str` | `"data"` | Root directory for dataset files |
| `train_ratio` | `float` | `0.8` | Fraction of interactions for training (temporal split) |
| `val_ratio` | `float` | `0.1` | Fraction for validation (remainder = test) |
| `seed` | `int` | `13` | Random seed |

---

## Device

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` | `"cuda"` | PyTorch device (falls back to CPU if CUDA unavailable) |

---

## Presets

Methods that mutate the config in-place and return `self` for chaining.

### `preset_lightgcn()`

Non-causal LightGCN baseline. Disables all causal components:

| Changed Parameter | Value |
|-------------------|-------|
| `use_dual_branch` | `False` |
| `use_sign_aware` | `False` |
| `use_counterfactual` | `False` |
| `use_ipw` | `False` |
| `use_popularity_emb` | `False` |
| `lambda_ortho/contr/cf/pop` | `0.0` |
| `beta_conformity` | `0.0` |
| `gamma_counterfactual` | `0.0` |

### `preset_dice_like()`

DICE-inspired dual-branch with orthogonality only:

| Changed Parameter | Value |
|-------------------|-------|
| `use_dual_branch` | `True` |
| `use_counterfactual` | `False` |
| `use_ipw` | `False` |
| `lambda_contr` | `0.0` |
| `lambda_cf` | `0.0` |
| `gamma_counterfactual` | `0.0` |

### `preset_full()`

Wave-1 U-CaGNN mainline with learned fused scoring, asymmetric branch depth, BPR active from epoch 0, and ramped auxiliary regularization.

Public experiment naming note: this internal preset method is exposed at the CLI and in the catalog as `ucagnn`.

### Expected Performance Ranges

Based on published results from the corresponding papers:

| Preset | Benchmark | Recall@20 | NDCG@20 | Source |
|--------|-----------|-----------|---------|--------|
| `preset_lightgcn()` | Amazon-Book | ~0.0730 | ~0.0546 | CaDCR Table 2 (LightGCN baseline) |
| `preset_dice_like()` | MovieLens-10M | ~0.1812 | ~0.1228 | DICE Table (GCN-DICE) |
| `preset_full()` | -- | -- | -- | Thesis contribution; expected to improve over both baselines |

# Theoretical Justifications for U-CaGNN Design Decisions

This document provides formal reasoning behind the key design choices in U-CaGNN, linking each decision to causal inference theory, prior work, and implementation specifics.

---

## 1. Sign Assignment Heuristics

### 1.1 Taobao: Why pv = -0.25?

The Taobao (UserBehavior) dataset records four behavior types, forming a behavioral hierarchy:

| Behavior | Sign | Rationale |
|----------|------|-----------|
| buy | +1.0 | Strongest positive signal: user completed a purchase |
| cart | +0.5 | Active interest: user added to cart but didn't buy |
| fav | +0.25 | Mild interest: user bookmarked for later |
| pv (page view) | -0.25 | Exposure without engagement |

**Why -0.25 and not -1.0?** A page view is an *exposure event*. The user saw the item but took no further action. This is fundamentally different from an explicit "dislike" (-1.0). The user may have been mildly interested (they did click through to the page), but the *absence of further engagement* signals weak negative preference.

The 0.5 gap between `fav` (+0.25) and `pv` (-0.25) represents the critical transition from "user acted on the item" to "user did not act beyond viewing." This gap is wider than the gaps within positive behaviors (buy-cart: 0.5, cart-fav: 0.25) because the act/no-act boundary is the most informative signal for the recommendation task.

**Reference**: Multi-behavior recommendation literature (e.g., MBGCN, NMTR) establishes behavior hierarchies. The specific values are tunable hyperparameters.

**Implementation**: `src/data/loaders/taobao.py`, lines 12-13.

### 1.2 MovieLens: Why rating >= 4?

The threshold `rating >= 4` for positive labels follows the standard established by He et al. (2017, "Neural Collaborative Filtering"):

- MovieLens 1M uses whole-star ratings (1-5), so >= 3.5 is impossible
- Rating 3 = "neutral" (user neither liked nor disliked)
- Rating 4 = "liked" (clear positive signal)
- This creates a meaningful positive class (~55-60% of ratings)

The sign mapping `(rating - 3) / 2` maps the 1-5 scale to [-1, +1]:
- Rating 1 -> -1.0, Rating 2 -> -0.5, Rating 3 -> 0.0, Rating 4 -> +0.5, Rating 5 -> +1.0

**Implementation**: `src/data/loaders/movielens1m.py`, lines 55-56.

### 1.3 KuaiRand-1K: Graded Engagement Sign

The graded sign hierarchy for KuaiRand-1K reflects engagement depth:

| Signal | Sign | Rationale |
|--------|------|-----------|
| like | +1.0 | Explicit positive feedback |
| follow | +0.7 | Strong positive: user wants future content |
| comment | 0.0 | Neutral until sentiment is observed rather than assumed |
| click + long_view | +0.3 | Passive but sustained engagement (>18s) |
| neutral (click only) | 0.0 | Clicked but didn't engage further |
| hate | -1.0 | Explicit negative feedback |

**Why is `is_rand` preserved as metadata?** The `is_rand` flag indicates whether the item was shown through randomized exposure (vs. algorithmic recommendation). This is critical for causal analysis because randomized exposure satisfies the "no confounding" assumption, enabling unbiased estimation of treatment effects. The flag enables:
1. Evaluating on random exposure only (debiased metrics)
2. Training the propensity estimator with known exposure probabilities
3. Validating IPW effectiveness by comparing random vs. algorithmic subsets

**Implementation**: `src/data/loaders/kuairand1k.py`.

### 1.4 Are Signs Learnable?

Sign assignment is a *preprocessing heuristic*, not a learned parameter. However, the model adapts to sign quality through two mechanisms:

1. **Learnable `alpha_pos` / `alpha_neg`**: The DualBranchGCN learns separate aggregation weights for positive and negative edges, effectively learning how much to trust the sign assignments.
2. **IPW handles exposure bias dynamically**: The PropensityEstimator learns P(exposure|item) and reweights the loss, compensating for systematic biases in sign assignment.

Making signs fully learnable would require a differentiable sign predictor, which adds complexity without clear benefit -- the current approach works because sign assignment errors are bounded (misclassifying neutral as mildly positive/negative has limited impact due to the graded scale).

---

## 2. Counterfactual Scoring: Why INT - CONF?

### 2.1 The Scoring Formula

The final prediction score is:

```
score = alpha * INT + beta * CONF + gamma * (INT - CONF)
```

Where:
- `INT` = interest branch score (user's intrinsic preference)
- `CONF` = conformity branch score (popularity-driven preference)
- `INT - CONF` = Individual Treatment Effect (ITE) from causal inference

### 2.2 Causal Interpretation

In the potential outcomes framework:
- `INT` approximates Y(1) -- the outcome if the user's interest drives the interaction
- `CONF` approximates Y(0) -- the outcome if conformity/popularity drives the interaction
- `INT - CONF` = Y(1) - Y(0) = ITE -- the causal effect of genuine interest

This decomposition comes from CausE (Bonner & Vasile, 2018), which established that separating interest from conformity requires:
1. Two separate representation branches
2. A counterfactual contrast term to measure the *difference* between branches

### 2.3 Why Not Just Set CONF = 0?

Setting `CONF = 0` (ignoring conformity) would:
- Kill the conformity branch's gradient signal (no loss flows through it)
- Prevent the model from learning to *disentangle* interest from conformity
- Reduce the model to a standard GCN with no causal debiasing

The conformity branch *must* receive gradient signal to learn what conformity looks like, so that the interest branch can learn what it is *not*.

**Implementation**: `src/models/scoring.py` (CounterfactualScoring class).

---

## 3. Inverse Propensity Weighting (IPW)

### 3.1 Usage in Prior Work

| Method | Uses IPW? | Notes |
|--------|-----------|-------|
| DICE (Zheng et al., 2021) | Yes | Popularity-based propensity |
| FMMRec | Yes | Feature-aware propensity |
| CaDCR | Yes | Cross-domain propensity |
| MGCE | No | Uses contrastive learning instead |
| SIGformer | No | Uses signed graph attention |

### 3.2 Design Choice

U-CaGNN implements IPW as a *toggleable* component (`use_ipw` flag in config). This allows:
1. **Ablation**: Measure IPW's contribution on each dataset
2. **Efficiency**: Skip propensity computation when unnecessary
3. **Comparison**: Match non-IPW baselines (MGCE, SIGformer) fairly

### 3.3 Architecture: MLP, Not GNN

The PropensityEstimator uses a 2-layer MLP (input -> 128 hidden -> 1 output) rather than a GNN because propensity P(exposure|item) is a *per-item* property:
- An item's exposure probability depends on its popularity, recency, and platform-level factors
- These are item-level statistics, not graph-relational properties
- A GNN would be over-parameterized for this task (and add unnecessary computation)

The 128 hidden size (2x the 64-dim input) follows standard MLP design practice.

**Implementation**: `src/models/propensity.py`.

---

## 4. Embedding Dimensions

### 4.1 embed_dim = 64

This is the standard dimension used by LightGCN (He et al., 2020) and DICE (Zheng et al., 2021). Key considerations:
- 64 provides sufficient capacity for millions of user-item interactions
- Larger dimensions (128, 256) show diminishing returns on recommendation quality
- Memory scales linearly: `(n_users + n_items) * embed_dim * 4 bytes`
- Training speed is heavily influenced by embedding dimension due to GNN message passing

### 4.2 propensity_hidden = 128

This is the *MLP hidden size*, not an embedding dimension. The factor of 2x (128 = 2 * 64) follows standard MLP design: the hidden layer expands the representation space to enable non-linear transformations before compressing to a scalar propensity score.

### 4.3 pop_embed_dim = 16

Popularity is a low-dimensional signal (single scalar per item, bucketed). A 16-dim embedding is sufficient to capture the relationship between popularity buckets and recommendation relevance. This is concatenated with the main embedding, so keeping it small minimizes parameter overhead.

---

## 5. Default Lambda Values

### 5.1 Source

The default loss weights come from ablation studies in DICE and MGCE:

| Lambda | Default | Source | Role |
|--------|---------|--------|------|
| `lambda_rec` | 1.0 | Standard | Primary BPR loss weight |
| `lambda_ortho` | 0.02 | DICE | Orthogonality between INT/CONF branches |
| `lambda_contr` | 0.1 | MGCE | Contrastive loss for embedding quality |
| `lambda_cf` | 0.08 | CausE | Counterfactual regularization |
| `lambda_pop` | 0.15 | U-CaGNN | Popularity prediction auxiliary loss |

### 5.2 Sensitivity

These values are *not* universal. Each dataset has different characteristics (density, sign distribution, popularity skew) that affect optimal lambda values. The experiment infrastructure supports systematic sensitivity sweeps:

```bash
# Example: sweep lambda_ortho on MovieLens1M
for val in 0.0 0.01 0.02 0.05 0.1; do
    uv run experiment \
    --dataset movielens1m --preset ucagnn \
        --override lambda_ortho=$val
done
```

**Implementation**: `src/utils/config.py`, lines 37-41.

---

## 6. Predefined Splits vs. Temporal Splits

### 6.1 Why Predefined Splits Matter

AmazonBook provides predefined train/test splits from the LightGCN benchmark. Using these splits instead of synthetic temporal ordering ensures:
1. **Reproducibility**: Results are directly comparable to published LightGCN/DICE metrics
2. **No information leakage**: The original split was designed to prevent temporal leakage
3. **Benchmark validity**: Using different splits invalidates comparisons with published numbers

### 6.2 Fallback Strategy

The `CanonicalInteractions.get_splits()` method implements a clean fallback:
- If predefined masks are set (all three: train, val, test) -> use them
- Otherwise -> fall back to `temporal_split()` based on timestamps

This allows each loader to provide the most appropriate splitting strategy while maintaining a uniform interface for the graph builder.

**Implementation**: `src/data/canonical.py` (`get_splits` method), `src/data/graph_builder.py` (line 42).

---

## 7. PyG Performance Optimization Opportunities

The following optimizations are documented for future experimentation but are **not** implemented by default, as they require profiling to confirm benefit on each dataset/hardware configuration.

### 7.1 `torch.compile()` for GNN Acceleration

Per PyG's compiled GNN documentation, wrapping the model with `torch.compile(model, dynamic=True)` can yield 2-3x speedup on static graph structures. This is particularly effective for LGConv since it performs only linear message passing (no learnable message function).

```python
# Future: wrap model after construction
model = torch.compile(model, dynamic=True)
```

**Current repository stance**: this remains opt-in only. The present training runtime is mini-batch-only and feeds dynamic sampled subgraphs plus per-batch edge-sign tensors into `DualBranchGCN`; in practice that has been enough to trigger repeated Dynamo recompiles, so the default config keeps `use_torch_compile=False` until a compile-stable path is profiled and verified.

### 7.2 SparseTensor for Memory-Efficient Aggregation

Per PyG's memory-efficient aggregation documentation, LightGCN is an ideal candidate for sparse-matrix propagation because it does not use central node features in messages — it only performs neighbor aggregation. The repository now implements this idea with PyTorch COO sparse adjacency matmul in `LightGCNBranch`; a future `SparseTensor` migration remains possible if profiling shows it beats the current path on the target GPUs.

```python
# Future: convert edge_index to SparseTensor
from torch_sparse import SparseTensor
adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                   sparse_sizes=(n_nodes, n_nodes))
```

**Caveat**: Requires `torch-sparse` and changes across model + graph_builder + training loop.

### 7.3 CPU Affinity and Multi-Worker DataLoaders

Not relevant for current full-batch training setup. Would become relevant if mini-batch training is adopted for larger datasets.

---

## 8. Mini-Batch Training Rationale

U-CaGNN now uses a single training mode: `mini_batch`.

### 8.1 Why the Runtime Standardizes on Mini-Batch

The thesis runtime keeps the semantic experiment matrix on `dataset × preset` and treats batch size plus per-hop fan-out as support parameters, not separate methodological axes. A single mini-batch trainer simplifies the orchestration and keeps the training contract consistent across small and large datasets.

### 8.2 Current Mini-Batch Design

For each batch, the trainer extracts a k-hop subgraph around the batch users plus positive and negative items. Key design choices:

- **CPU-side preparation**: negative sampling and subgraph extraction stay on CPU in a background thread.
- **Pinned transfer path**: the sampled `SubgraphBatch` is pinned and moved to CUDA with `non_blocking=True` immediately before the forward pass.
- **Users-first layout**: `SubgraphSampler` rearranges the local node set so `DualBranchGCN` can treat subgraphs exactly like the full graph layout.
- **Batch-scoped disentanglement losses**: `L_ortho` and `L_contr` operate on the current batch users, not every context user in the sampled subgraph.

### 8.3 Practical Tuning Guidance

- **Batch size**: increase batch size first when GPU memory allows; it improves both hardware utilization and the representativeness of the sampled subgraph.
- **Fan-out**: use `num_neighbors` as the main VRAM/throughput knob. The current formal default `[15, 10]` keeps a denser first hop without letting the second hop explode.
- **Contrastive cost**: `L_contr` remains an `O(B^2 d)` term, so batch size and fan-out should be tuned together rather than independently.

This is the only training path reflected in the current implementation and experiment orchestration.

---

## References

1. Bonner, S. & Vasile, F. (2018). Causal Embeddings for Recommendation. RecSys.
2. Zheng, Y. et al. (2021). Disentangling User Interest and Conformity for Recommendation with Causal Embedding. WWW.
3. He, X. et al. (2017). Neural Collaborative Filtering. WWW.
4. He, X. et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR.
5. Wang, W. et al. (2022). Causal Representation Learning for Out-of-Distribution Recommendation. WWW.
6. Wei, T. et al. (2021). Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System. KDD.
7. Mostafa, H. (2022). Full-Graph vs. Mini-Batch GNN Training: Comprehensive Comparison and Guidelines.
