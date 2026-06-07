# U-CaGNN Implementation Overview

This file is the integration map for the current implementation. It stays intentionally short: use the routed docs for slice-specific detail.

## Reading map

| Need | Open |
| --- | --- |
| model modules and refined scorer | `ucagnn-architecture.md` |
| loaders, canonical schema, graph build, samplers | `ucagnn-data-pipeline.md` |
| presets and config rules | `ucagnn-config.md` |
| objectives and schedule semantics | `ucagnn-losses.md` |
| runtime flow, evaluator, checkpoints, logging | `ucagnn-training.md` |

## End-to-end flow

```mermaid
flowchart LR
    A[build_config] --> B[load_dataset]
    B --> C[CanonicalInteractions]
    C --> D[build_graph embeddings_none]
    D --> E{graph_policy}
    E -->|observed| F[Runtime graph]
    E -->|cagra_augmented| G[Bootstrap embeddings]
    G --> H[build_graph embeddings_present]
    H --> F
    C --> I[item_propensity_targets optional]
    F --> J[build_runtime_model]
    J --> K[MiniBatchTrainer and LossSuite]
    I --> L[data.propensity_targets optional]
    L --> K
    K --> M[Evaluator]
    K --> N[ExperimentLogger and checkpoints]
```

The diagram shows the full runtime join points. Slice-specific rules stay in the owner docs: config precedence in `ucagnn-config.md`, graph policy and propensity-target loading in `ucagnn-data-pipeline.md`, loss activation in `ucagnn-losses.md`, and checkpointing or tracking in `ucagnn-training.md`. The evaluation path keeps thesis metrics on the training-validation loop and runs refined scorer diagnostics only on the final post-training test pass, including standalone raw interest/conformity branch ranking diagnostics for dual-branch models.
Auto-batch CUDA OOM decisions are logged with the original exception summary, PyTorch allocator allocated/reserved/peak memory, and probe-stage/subgraph dimensions when available so resource-monitor readings can be compared with actual allocator pressure. Training-window telemetry separately logs PyTorch allocated/reserved peaks, peak `nvidia-smi memory.used`, average training GPU utilization, and max sampled training GPU utilization before validation/test evaluation starts. The experiment runner sets `PYTORCH_ALLOC_CONF=expandable_segments:True` before importing `torch` unless the user already supplied a CUDA allocator policy, and it accepts legacy `PYTORCH_CUDA_ALLOC_CONF` as a source value.
DICE-style independence remains a distance-correlation auxiliary, but the user and item entity sets are hash-sampled up to `distance_correlation_max_pairs` before quadratic distance matrices are built. DirectAU uniformity similarly hash-samples rows up to `uniformity_max_pairs` before `torch.pdist`. Sampled-BFS fanout uses bounded CSR offset gathers, and U-CaGNN sampled-subgraph propagation uses uncoalesced CUDA sparse COO with CPU chunked edge-list fallback instead of per-batch CUDA sparse COO coalescing. U-CaGNN's DICE sampler uses fast high/low routing plus vectorized known-positive filtering; the paper DICE baseline keeps exact per-user pool-count correction.

## Source map

| Path | Responsibility |
| --- | --- |
| `src/utils/config.py` | `UCaGNNConfig` defaults, validation, preset overrides |
| `src/utils/experiment_naming.py` | shared canonical experiment names for checkpoints and result reports |
| `src/data/loaders/_registry.py` | dataset registry and default preprocessing presets |
| `src/data/canonical.py` | canonical interaction schema, split logic, item recency |
| `src/data/feature_policy.py` | safe-vs-optional feature registry |
| `src/data/graph_builder.py` | graph construction, optional field transfer, train-only popularity, CAGRA augmentation |
| `src/data/subgraph_sampler.py` | sampled k-hop subgraph extraction |
| `src/data/negative_sampler.py` | vectorized negative sampling |
| `src/models/embeddings.py` | embedding layer |
| `src/models/lightgcn.py` | propagation layer |
| `src/models/baselines/lightgcn.py` | canonical paper LightGCN adapter |
| `src/models/baselines/dice.py` | canonical paper GCN-DICE adapter |
| `src/models/scoring.py` | scoring layer |
| `src/models/propensity.py` | propensity layer |
| `src/models/ucagnn.py` | model orchestration and public train/eval surfaces |
| `src/losses/loss_suite.py` | total objective assembly |
| `src/utils/trainer_runtime.py` | shared runtime, optimizer, scheduler, checkpointing |
| `src/training/mini_batch_trainer.py` | sole trainer |
| `src/training/evaluator.py` | batched full-graph evaluation |
| `experiments/run_experiment.py` | single-run orchestration and runtime assembly |
| `experiments/run_benchmark.py` | formal-run orchestration and strict saved-state handling |
| `experiments/ablation_configs.py` | thesis-facing ablation variants |
| `src/utils/experiment_logger.py` | SQLite experiment store |
| `scripts/query_results.py` | SQLite-first result inspection |
