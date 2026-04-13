# U-CaGNN Training Skill

Use this skill when working on training loop, evaluation, checkpoints, profiling, or experiment logging.

## Key Files
- `docs/ucagnn_implementation/training.md` - Training pipeline with paper cross-references
- `src/utils/trainer_runtime.py` - Shared trainer setup, checkpointing, profiling helpers, and early-stopping state
- `src/training/mini_batch_trainer.py` - MiniBatchTrainer class (sole trainer)
- `src/training/evaluator.py` - Evaluator (PyG link-prediction metrics at K)
- `src/profiling/gpu_profiler.py` - GPUProfiler and profile_stage
- `src/utils/experiment_logger.py` - SQLite ExperimentLogger

## Paper Sources
| Feature | Source |
|---------|--------|
| Adam lr=1e-3 (fused on CUDA) | LightGCN |
| `set_to_none=True` in zero_grad | PyTorch performance guide |
| cuDNN benchmark + bf16 AMP | PyTorch CUDA best practices |
| ReduceLROnPlateau (optional) | PyTorch |
| Gradient clipping max_norm=1.0 | DICE |
| Early stopping patience=10 | DDCE |
| 60 epochs default | thesis_plan.md |

## Training Pipeline
```python
from src.training.mini_batch_trainer import MiniBatchTrainer
from src.utils.experiment_logger import ExperimentLogger

experiment_logger = ExperimentLogger()
exp_id = experiment_logger.log_experiment(config.dataset, config)

trainer = MiniBatchTrainer(
    model,
    loss_suite,
    data,
    config,
    profiler,
    experiment_logger=experiment_logger,
    exp_id=exp_id,
)
history = trainer.train()

# Test metrics
test_metrics = trainer.evaluator.evaluate(model, data, data.test_mask)
for metric, value in test_metrics.items():
    experiment_logger.log_metric(exp_id, metric, value, split="test")

experiment_logger.close()
trainer.save_checkpoint("results/checkpoints/ucagnn_best.pt")
```

## Ownership Notes
- `src/training/mini_batch_trainer.py` is the sole trainer; it keeps its mode-specific epoch loop for readability and behavior isolation.
- **Sampler path**: `MiniBatchTrainer.train()` now prefers a CUDA-resident `SubgraphSampler` when training on GPU. The full graph is staged on the accelerator once, negative sampling plus sampled-BFS subgraph extraction run on-device, and the trainer falls back to the original four-worker CPU prefetch path only if staging the full graph would exhaust VRAM. Right before full-graph validation, the trainer drops that CUDA sampler copy and rebuilds it afterward so the evaluator can use the same VRAM budget instead of competing with a second resident graph copy. The CPU fallback keeps prepared `SubgraphBatch` instances pinned and transfers them with `non_blocking=True` right before the forward pass. Each prepared batch keeps a deterministic per-batch RNG seed.
- `src/utils/trainer_runtime.py` owns only duplicated lifecycle machinery through `TrainerRuntime`: module/device setup, optimizer creation (fused Adam on CUDA, `set_to_none=True`), optional CUDA AMP/autocast (fixed to bfloat16), cached device-side popularity, reusable epoch progress bars, on-device epoch-loss accumulation, optional LR scheduler, checkpoint save/load (including scheduler and EMA state), profiling toggles, resume history, shared end-of-epoch evaluation/logging/checkpointing, small scalar/lifecycle helpers, and early-stopping helpers. `TrainerRuntime._get_train_interactions()` now returns CPU index-space tensors; the CUDA sampler path moves only the current batch users/items onto the sampler device so the runtime does not keep a second full interaction copy on GPU.
- **EMA support**: When `config.use_ema=True`, `TrainerRuntime` creates an `AveragedModel` with `get_ema_multi_avg_fn(config.ema_decay)`. EMA weights are updated after each optimizer step, used for validation evaluation, and captured as `best_state` for model restoration. EMA state is saved/loaded with checkpoints. The `module.` key prefix is stripped when capturing best_state so `load_state_dict()` into the base model works directly.
- CUDA performance: On GPU, `TrainerRuntime.__init__` enables `cudnn.benchmark = True`, turns on TF32 matmul (`torch.set_float32_matmul_precision("medium")`, `torch.backends.cuda.matmul.allow_tf32=True`), and relies on bf16 AMP for the mixed-precision policy. The experiment entry path also defaults `PYTORCH_ALLOC_CONF=expandable_segments:True` unless the user has already set an allocator policy explicitly.
- `config.use_amp`, fixed `config.amp_dtype="bfloat16"`, `config.show_progress_bar`, and `config.progress_bar_loss_cadence` control the CUDA mixed-precision path and the epoch-level tqdm bar. Keep AMP enabled by default for interactive training; `progress_bar_loss_cadence` exists specifically to avoid syncing a loss scalar every batch just to refresh tqdm.
- `config.use_torch_compile` is opt-in rather than default. The current mini-batch-only runtime feeds dynamic sampled subgraphs into `DualBranchGCN`, and the observed `torch.compile(dynamic=True)` path can hit enough recompiles to reduce throughput instead of improving it.
- The isfinite check uses `torch.isfinite(total_loss).all()` instead of `.item()` to avoid a GPU-CPU synchronization point every batch.
- Early stopping and plateau LR scheduling both defer their patience logic until `max(config.curriculum_phase1_end, config.curriculum_phase2_end)`, so staged CaDCR-style losses can activate before the run decays LR or terminates. When `config.use_early_stopping=False`, the trainer still tracks/restores the best validation state but never terminates before the configured epoch budget.
- `config.enable_profiling` now defaults to `False`; enable it explicitly for observability or profiling-focused runs so throughput-oriented training does not pay for synchronized stage timing by default. `GPUProfiler` instances also start disabled and are enabled only when the runtime opts in for a profiled epoch.
- LR scheduler: When `config.lr_scheduler == "plateau"`, the shared runtime delays `ReduceLROnPlateau` updates until the curriculum warmup has completed, so LR decay uses the same post-curriculum validation window as early stopping. Scheduler state is saved/restored with checkpoints.
- Treat `batch_size` as both an interaction-loss knob and a VRAM knob in mini-batch training: it controls how many interactions contribute per step and (via `SubgraphSampler`) the size of the extracted subgraph. The current throughput-oriented repo default is `4096`; if a full-dataset run hits OOM, reduce this before changing model semantics.
- In mini-batch mode, the branch-local auxiliary losses (`L_independence`, `L_align`, `L_uniform`) should operate on the current batch users/items rather than every propagated context node from the sampled subgraph.
- `num_neighbors` controls `SubgraphSampler` per-hop fan-out and is the primary tool for managing subgraph density and VRAM. The current base dual-branch default is `[10, 5]`, matching the default `interest_gnn_layers=1` / `conformity_gnn_layers=2` contract.
- Named recipes own matrix-defining fields such as `graph_method`; conflicting CLI flags should be rejected rather than silently treated as an auto-scaling mechanism.
- Formal batch runners expose `--batch-id` and `--resume-batch` so benchmark and ablation sweeps can skip terminal rows (`completed`, `oom`, `failed`) without hiding failed feasibility cases.
- `experiments/run_benchmark.py::formal_main()` is now the simple formal orchestration entry point behind `uv run formal-run`. It persists `results/formal_run_state.json` and is the preferred path when the user wants one command that resumes formal experiments after interruption.
- `results/formal_run_state.json` is only a generated resume pointer for `formal-run` (current `profile_name`, `batch_id`, runtime args, timestamps, exit status). The source of truth for the formal matrix remains `experiments/experiment_catalog.json`; do not treat the state file as a static profile definition.
- `formal-run` now separates semantic protocol identity from execution identity: `--profile` selects a catalog bundle from `experiments/experiment_catalog.json`, while the persisted `profile_name` is a deterministic slug derived from that bundle and `batch_id` is only the resumable execution label stored in SQLite and MLflow.
- Formal profiles should own the full thesis matrix and support-parameter bundle directly in the catalog. The formal wrapper should stay thin and should not reintroduce parallel definitions for epochs, learning rate, sampled interactions, or fallback behavior.
- The current default formal profile is development-focused: it runs only the `ucagnn` preset, disables early stopping, keeps the full 60-epoch budget, uses learned fused scoring, and makes the mainline branch asymmetry explicit with `interest_gnn_layers=1`, `conformity_gnn_layers=2`, `num_neighbors=[10, 5]`, and `loss_schedule="baseline"` so fused BPR stays active from epoch 0.
- A second formal profile is reserved for the end-stage matched comparison pass, where `lightgcn` and `dice_like` re-enter under the same evaluator and logging stack.
- The semantic formal matrix remains `dataset × preset × graph_method`; profile-owned `scoring_weight_modes` can add score-mix comparisons when explicitly requested, but the default day-to-day `ucagnn` profile now keeps learned fusion only. The benchmark/formal execution order should keep datasets as the innermost loop so a dataset-specific failure appears during the first method sweep instead of after one dataset has already exhausted every method combination.
- Benchmark and `formal-run` no longer expose seed as a public orchestration flag. Keep the thesis matrix on `dataset × preset × graph_method`, and treat any score-mix comparison as an explicit profile-owned support choice rather than the default mainline sweep.
- Benchmark, ablation, and `formal-run` now share the same OOM policy: keep the original OOM row as thesis evidence and continue without an automatic fallback retry.
- SQLite experiment review now has three convenience views: completed runs, attention-required runs, and strict error runs. Prefer `query-results --view completed|attention|errors` over manual SQL when triaging long batches.
- The supported query view registry now lives in `ExperimentLogger.VIEW_TABLES`; keep `scripts/query_results.py` as a thin CLI over that logger-owned mapping instead of re-declaring view names.
- Formal runs now persist `profile_name` alongside status, batch id, and hardware metadata, so query and MLflow inspection can distinguish the scientific profile from the operational batch.
- Checkpoint payload loading is shared between runtime auto-resume, quick validation observability probes, and same-checkpoint scoring-mode evaluation through `experiments/run_experiment.py::load_checkpoint_payload()`. Invalid checkpoint payloads should not be treated as resumable state.
- Runtime dataset/graph/model reconstruction helpers now live in `experiments/run_experiment.py`, so `evaluate-scoring-modes` / `scripts/evaluate_scoring_modes.py` should reuse them instead of carrying a parallel reconstruction path.
- `experiments/run_experiment.py` keeps the single-run CLI surface in `build_parser()` and constructs `MiniBatchTrainer` directly inside `run_experiment()`; keep thin wrappers out of that path unless they remove real duplication.

## What Gets Logged Automatically
| Data | SQLite Table | Split |
|------|--------------|-------|
| Training loss | metrics | train |
| Validation metrics | metrics | val |
| Profiling stages | profiling | -- |
| Alpha values (sign-aware) | metrics | train |
| PyG link-prediction metrics | metrics | val/test |

## Evaluation Notes
- Validation and test metrics now honor `config.eval_scoring_mode`, so the thesis metric suite can be computed under intervention-style scoring without changing the checkpointed model weights.
- Dual-branch presets now align training and evaluation with the intended score contract: `preset_full()` optimizes and reports the fused `default` score, while `preset_dice_like()` keeps the `interest_only` intervention view.
- Validation and test evaluation logs only the thesis-facing PyG metrics: `NDCG@20`, `Recall@20`, `AveragePopularity@20`, `NDCG@40`, `Recall@40`, and `AveragePopularity@40`.
- Validation and test ranking must mask all observed non-target interactions before `topk`; otherwise train/val positives can occupy the ranked list and collapse held-out Recall/NDCG.
- The evaluator builds those six metrics via `LinkPredMetricCollection`. MetricCollection removes per-metric runtime update loops; construction still needs one metric instance per metric and cutoff.
- The evaluator rematerializes `edge_index`, `edge_sign`, `edge_norm`, and `popularity` on the evaluation device for each validation/test call instead of holding a persistent full-graph cache across epochs. When AMP is enabled on CUDA, the full-graph propagation and scoring path also runs under bf16 autocast and casts the evaluation embedding bundle to bf16 to reduce validation memory pressure on large datasets.
- `src/training/evaluator.py` is also the source of truth for the thesis-primary metric subset and lower-is-better metric polarity, so downstream benchmark, ablation, reporting, and scoring-mode scripts should import those constants instead of re-declaring them.
- Treat `AveragePopularity@20` and `AveragePopularity@40` as debiasing readouts where lower values are better. Summary aggregation, delta interpretation, and heatmap colors should reflect that polarity.
- The preferred mechanism comparison is same-checkpoint evaluation under `default`, `interest_only`, and `conformity_suppressed`. Leave `conformity_only` and `counterfactual_only` available for debugging rather than thesis headline tables.
- Use `uv run evaluate-scoring-modes --checkpoint-path ...` to run the thesis mechanism table from one saved checkpoint without retraining separate runs.
- PyG 2.7 also exposes Diversity and Personalization. They are allowed by the repo's metric audit, but the current evaluator does not log them because they require extra category inputs or additional pairwise recommendation computation.
- External implementation audits may discuss non-PyG causal-uplift evaluators such as PropCare's semi-simulated `CPrec` or `CDCG` pipeline, but those remain reference analyses unless the runtime data contract is extended with treatment, propensity, and causal-effect labels.

## MLflow Routing
- `experiments/run_experiment.py` uses `--mlflow-tracking-uri` first, then `MLFLOW_TRACKING_URI`, then the project default `results/mlflow.db`.
- `results/mlflow.db` is the MLflow backend store and `mlruns/` holds artifacts.
- Benchmark runs default to MLflow experiment `ucagnn-benchmark`; ablations default to `ucagnn-ablation`.
- Checkpoints are logged to MLflow under the `checkpoints/` artifact subpath.
- Use `uv run reset-experiment-db` to delete the repository-local thesis SQLite database and its sidecars.
- Use `uv run cleanup-experiment-artifacts` to delete `results/mlflow.db`, the generated `results/formal_run_state.json` resume pointer, `mlruns/`, and `results/checkpoints/` in one step.
- Matrix fields should not be duplicated across MLflow params and tags; ordering/provenance should use explicit params such as `run_started_at_utc`, `project_version`, and `git_commit`.

## Fast Validation Workflow
- Use `quick-validate` as the default ultra-fast post-change validation path across all six datasets.
- The older preflight and feature-probe scripts have been removed; their retained smoke coverage now lives inside `quick_validate.py` so there is one supported tiny-scale validation entry point.
- `quick_validate.py` is now the unified tiny-scale pipeline validator. By default it exercises the canonical recipe matrix, all ablation variants, observability probes across the selected datasets (profiling, checkpoint/resume, feature path), and evaluation scoring modes with aggressive row caps and sampled interactions.
- Shared tiny-run dataset budgets and other low-level script helpers now live in `scripts/_workflow_helpers.py`; keep only mechanical helpers there and leave CLI/config wiring inside each command.
- Feature usage now follows the formal config default (`use_features=True`, `feature_policy="thesis_default"`), so tiny validation covers the same feature-aware model path as formal runs. The capped dataset loader path stays practical by reusing cached capped loads across repeated recipe cases.
- Treat tiny validation as row-scaled, not schema-changed: the intended invariant is that formal and tiny runs share the same canonical fields and feature-engineering path, while `loader_max_rows`, `sample_interactions`, `epochs`, `batch_size`, and semantic-eval caps control runtime.
- Use category filters such as `--categories recipes` or `--categories observability` only when debugging a specific surface; the default command is intended to be the single broad post-change validation entry point.
- Quick validation keeps MLflow disabled by default, so `uv run quick-validate` does not create MLflow tables or artifact files unless `--mlflow` is passed explicitly.
- Use `query-results` as the supported SQLite inspection path after runs. The repository currently does not expose a supported plotting command in the main workflow.
- `query-results` now supports `--view`, `--batch-id`, and `--status` filters for resumable benchmark and ablation inspection.
- Keep the tiny validation CLIs locally explicit. Shared helpers should stay limited to low-level utilities such as dataset-limit lookup, timed `run_experiment()` execution, and JSON report writing; config wiring belongs in each script and should call `build_config()` directly with plain mappings instead of manufacturing fake CLI namespaces.


