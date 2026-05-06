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
| Deterministic seeded runtime + bf16 AMP | PyTorch reproducibility guide + CUDA best practices |
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
- **Sampler path**: `MiniBatchTrainer.train()` now prefers a CUDA-resident `SubgraphSampler` when training on GPU. The full graph is staged on the accelerator once, negative sampling plus sampled-BFS subgraph extraction run on-device, and the trainer falls back to the original four-worker CPU prefetch path only if staging the full graph would exhaust VRAM. Right before full-graph validation, the trainer drops that CUDA sampler copy and rebuilds it afterward so the evaluator can use the same VRAM budget instead of competing with a second resident graph copy. The CPU fallback keeps prepared `SubgraphBatch` instances pinned and transfers them with `non_blocking=True` right before the forward pass. Epoch shuffling now uses a seeded `torch.randperm(...)`, and each prepared batch keeps its own deterministic RNG seed.
- `src/utils/trainer_runtime.py` owns only duplicated lifecycle machinery through `TrainerRuntime`: module/device setup, optimizer creation (fused Adam on CUDA, `set_to_none=True`), optional CUDA AMP/autocast (fixed to bfloat16), cached device-side popularity, reusable epoch progress bars, on-device epoch-loss accumulation, optional LR scheduler, checkpoint save/load (including scheduler and EMA state), profiling toggles, resume history, shared end-of-epoch evaluation/logging/checkpointing, small scalar/lifecycle helpers, and early-stopping helpers. `TrainerRuntime._get_train_interactions()` now returns CPU index-space tensors; the CUDA sampler path moves only the current batch users/items onto the sampler device so the runtime does not keep a second full interaction copy on GPU.
- **EMA support**: When `config.use_ema=True`, `TrainerRuntime` creates an `AveragedModel` with `get_ema_multi_avg_fn(config.ema_decay)`. EMA weights are updated after each optimizer step, used for validation evaluation, and captured as `best_state` for model restoration. EMA state is saved/loaded with checkpoints. The `module.` key prefix is stripped when capturing best_state so `load_state_dict()` into the base model works directly.
- Reproducibility contract: experiment entry now seeds Python, NumPy, and PyTorch once per run, sets deterministic torch algorithms, disables cuDNN benchmarking/TF32, and keeps that backend policy centralized in `src/utils/reproducibility.py`. The experiment entry path also defaults `PYTORCH_ALLOC_CONF=expandable_segments:True` unless the user has already set an allocator policy explicitly.
- `config.use_amp`, fixed `config.amp_dtype="bfloat16"`, `config.show_progress_bar`, and `config.progress_bar_loss_cadence` control the CUDA mixed-precision path and the epoch-level tqdm bar. Keep AMP enabled by default for interactive training; `progress_bar_loss_cadence` exists specifically to avoid syncing a loss scalar every batch just to refresh tqdm.
- `config.use_torch_compile` is opt-in rather than default. The current mini-batch-only runtime feeds dynamic sampled subgraphs into `DualBranchGCN`, and the observed `torch.compile(dynamic=True)` path can hit enough recompiles to reduce throughput instead of improving it.
- The isfinite check uses `torch.isfinite(total_loss).all()` instead of `.item()` to avoid a GPU-CPU synchronization point every batch.
- Early stopping and plateau LR scheduling both defer their patience logic until `max(config.curriculum_phase1_end, config.curriculum_phase2_end)`, so staged CaDCR-style losses can activate before the run decays LR or terminates. When `config.use_early_stopping=False`, the trainer still tracks/restores the best validation state but never terminates before the configured epoch budget.
- `config.enable_profiling` now defaults to `False`; enable it explicitly for observability or profiling-focused runs so throughput-oriented training does not pay for synchronized stage timing by default. `GPUProfiler` instances also start disabled and are enabled only when the runtime opts in for a profiled epoch. `GPUProfiler.epoch_timer()` is a lightweight wall-clock context manager (no CUDA sync) that records `epoch_elapsed_ms` regardless of profiling state — use it to track total epoch duration without stage-level overhead.
- LR scheduler: When `config.lr_scheduler == "plateau"`, the shared runtime delays `ReduceLROnPlateau` updates until the curriculum warmup has completed, so LR decay uses the same post-curriculum validation window as early stopping. Scheduler state is saved/restored with checkpoints.
- Treat `batch_size` as both an interaction-loss knob and a VRAM knob in mini-batch training: it controls how many interactions contribute per step and (via `SubgraphSampler`) the size of the extracted subgraph. The current throughput-oriented repo default is `4096`; if a full-dataset run hits OOM, reduce this before changing model semantics.
- In mini-batch mode, the branch-local auxiliary losses (`L_independence`, `L_align`, `L_uniform`) should operate on the current batch users/items rather than every propagated context node from the sampled subgraph.
- `num_neighbors` controls `SubgraphSampler` per-hop fan-out and is the primary tool for managing subgraph density and VRAM. The current base dual-branch default is `[10, 5]`, matching the default `interest_gnn_layers=1` / `conformity_gnn_layers=2` contract.
- Named recipes own the matrix-defining fields they declare; conflicting CLI flags should be rejected rather than silently treated as an auto-scaling mechanism.
- Formal batch runners expose `--batch-id` and `--resume-batch` so benchmark and ablation sweeps can skip terminal rows (`completed`, `oom`, `failed`) without hiding failed feasibility cases.
- `experiments/run_benchmark.py::formal_main()` is now the simple formal orchestration entry point behind `uv run formal-run`. It persists `results/formal_run_state.json` and is the preferred path when the user wants one command that resumes formal experiments after interruption.
- `results/formal_run_state.json` is only a generated resume pointer for `formal-run` (current `profile_name`, `batch_id`, runtime args, timestamps, exit status). The source of truth for the formal matrix remains `experiments/experiment_catalog.json`; do not treat the state file as a static profile definition.
- Keep the saved `benchmark_args` payload normalized once at the boundary: the current-format dict should drive JSON persistence, semantic plan matching, and resumed execution itself, and stale saved fields or removed graph methods should require a fresh formal run instead of reviving compatibility restore code.
- `experiments/recipes.py` owns formal profile alias normalization and catalog resolution. Keep formal-profile normalization in one cached pass there instead of regrowing tiny wrapper helpers around matrix normalization, override normalization, alias indexing, or profile-name derivation.
- Canonical recipe-name filtering also belongs in `experiments/recipes.py` now. Callers such as `quick_validate.py` should reuse `recipe_names(include_aliases=False)` instead of reopening the catalog and reimplementing alias filtering locally.
- Semantic formal-run plan matching should be derived from the normalized saved `benchmark_args` payload after excluding runtime-only override fields such as device, MLflow routing, and batch execution labels.
- `formal-run` now separates semantic protocol identity from execution identity: `--profile` selects a catalog bundle from `experiments/experiment_catalog.json`, while the persisted `profile_name` is a deterministic slug derived from that bundle and `batch_id` is only the resumable execution label stored in SQLite and MLflow.
- Formal profiles should own the full thesis matrix and support-parameter bundle directly in the catalog. The formal wrapper should stay thin and should not reintroduce parallel definitions for epochs, learning rate, sampled interactions, or fallback behavior.
- The current default formal profile is development-focused: it runs only the `ucagnn` preset, disables early stopping, keeps the full 60-epoch budget, uses learned fused scoring, and makes the mainline branch asymmetry explicit with `interest_gnn_layers=1`, `conformity_gnn_layers=2`, `num_neighbors=[10, 5]`, and `loss_schedule="baseline"` so fused BPR stays active from epoch 0.
- A second formal profile is reserved for the end-stage matched comparison pass, where `lightgcn` and `dice_like` re-enter under the same evaluator and logging stack.
- The semantic formal matrix remains `dataset * preset`; profile-owned `scoring_weight_modes` can add score-mix comparisons when explicitly requested, but the default day-to-day `ucagnn` profile now keeps learned fusion only. The benchmark/formal execution order should keep datasets as the innermost loop so a dataset-specific failure appears during the first preset sweep instead of after one dataset has already exhausted every preset/score-mix combination.
- Benchmark and `formal-run` no longer expose seed as a public orchestration flag. Keep the thesis matrix on `dataset * preset`, and treat any score-mix comparison as an explicit profile-owned support choice rather than the default mainline sweep.
- Benchmark, ablation, and `formal-run` now share the same OOM policy: keep the original OOM row as thesis evidence and continue without an automatic fallback retry.
- Keep the ablation runner's resume lookup direct at its only call site; do not hide a single `ExperimentLogger.find_latest_batch_experiment()` call behind an extra wrapper helper.
- SQLite experiment review now has three convenience views: completed runs, attention-required runs, and strict error runs. Prefer `query-results --view completed|attention|errors` over manual SQL when triaging long batches.
- The supported query view registry now lives in `ExperimentLogger.VIEW_TABLES`; keep `scripts/query_results.py` as a thin CLI over that logger-owned mapping instead of re-declaring view names.
- Keep one-use logger write-path details local to the public methods that own them: config serialization belongs in `ExperimentLogger.log_experiment()`, and per-epoch profiler-stage aggregation belongs in `ExperimentLogger.log_epoch()` rather than regrowing private wrapper helpers for those single call sites.
- `scripts/query_results.py` already uses `sqlite3.Row`; keep detail/profiling rendering keyed by selected column names rather than positional `row[0]`-style access so SQL column reordering cannot silently corrupt CLI output.
- Formal runs now persist `profile_name` alongside status, batch id, and hardware metadata, so query and MLflow inspection can distinguish the scientific profile from the operational batch.
- Checkpoint payload loading is shared between runtime auto-resume, quick validation observability probes, and same-checkpoint scoring-mode evaluation through `experiments/run_experiment.py::load_checkpoint_payload()`. Invalid checkpoint payloads should not be treated as resumable state.
- Runtime dataset/graph/model reconstruction helpers now live in `experiments/run_experiment.py`, so `evaluate-scoring-modes` / `scripts/evaluate_scoring_modes.py` should reuse them instead of carrying a parallel reconstruction path.
- `experiments/run_experiment.py::build_config()` should normalize namespace-like inputs once at the boundary and keep plain mapping access internally instead of layering generic field-access wrapper helpers on top of both `argparse.Namespace` and dict inputs.
- Experiment-level CLI parsers live in `experiments/cli_parsers.py` (`build_run_experiment_parser`, `build_benchmark_parser`, `build_formal_run_parser`, `build_ablation_parser`). Utility-script parsers live in `src/utils/cli_parsers.py`. Command files (`run_experiment.py`, `run_benchmark.py`, `run_ablation.py`) import their parser directly from `experiments/cli_parsers.py`; there are no thin `build_parser()` facade wrappers. Prefer extending the appropriate centralized module over adding local `add_argument(...)` blocks, and invoke the experiment commands through the packaged entry points (`uv run experiment`, `uv run formal-run`, `uv run ablation`) instead of relying on repo-root `sys.path` bootstrapping from direct file execution.

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
- Validation and test evaluation logs the thesis-facing PyG metrics: `NDCG@20`, `Recall@20`, `AveragePopularity@20`, `HitRatio@20`, `Personalization@20`, `NDCG@40`, `Recall@40`, `AveragePopularity@40`, `HitRatio@40`, and `Personalization@40`.
- Validation and test ranking must mask all observed non-target interactions before `topk`; otherwise train/val positives can occupy the ranked list and collapse held-out Recall/NDCG.
- The evaluator builds those six metrics via `LinkPredMetricCollection`. MetricCollection removes per-metric runtime update loops; construction still needs one metric instance per metric and cutoff.
- The evaluator rematerializes `edge_index`, `edge_sign`, `edge_norm`, and `popularity` on the evaluation device for each validation/test call instead of holding a persistent full-graph cache across epochs. When AMP is enabled on CUDA, the full-graph propagation and scoring path also runs under bf16 autocast and casts the evaluation embedding bundle to bf16 to reduce validation memory pressure on large datasets.
- `src/training/evaluator.py` is also the source of truth for the thesis-primary metric subset and lower-is-better metric polarity, so downstream benchmark, ablation, reporting, and scoring-mode scripts should import those constants instead of re-declaring them.
- Treat `AveragePopularity@20` and `AveragePopularity@40` as debiasing readouts where lower values are better. Summary aggregation, delta interpretation, and heatmap colors should reflect that polarity.
- The preferred mechanism comparison is same-checkpoint evaluation under `default`, `interest_only`, and `conformity_suppressed`. Leave `conformity_only` and `counterfactual_only` available for debugging rather than thesis headline tables.
- Use `uv run evaluate-scoring-modes --checkpoint-path ...` to run the thesis mechanism table from one saved checkpoint without retraining separate runs.
- PyG 2.7 also exposes Diversity and Personalization. The runtime evaluator now keeps `Personalization@20/40` as part of the thesis-facing metric set and defines the degenerate tiny-split case with fewer than two evaluated users as `0.0` so smoke validation stays finite. Diversity remains audit-only until the runtime contract grows category metadata for it.
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
- `quick_validate.py` still owns its tiny-run dataset caps and inline timing around `run_experiment()`, but its CLI definition now comes from the shared `src/utils/cli_parsers.py` module rather than a local parser block.
- Feature usage now follows the formal config default (`use_features=True`, `feature_policy="thesis_default"`), so tiny validation covers the same feature-aware model path as formal runs. The capped dataset loader path stays practical by reusing cached capped loads across repeated recipe cases.
- Treat tiny validation as row-scaled, not schema-changed: the intended invariant is that formal and tiny runs share the same canonical fields and feature-engineering path, while `loader_max_rows`, `sample_interactions`, `epochs`, `batch_size`, and semantic-eval caps control runtime.
- Use category filters such as `--categories recipes` or `--categories observability` only when debugging a specific surface; the default command is intended to be the single broad post-change validation entry point.
- Quick validation keeps MLflow disabled by default, so `uv run quick-validate` does not create MLflow tables or artifact files unless `--mlflow` is passed explicitly.
- Use `query-results` as the supported SQLite inspection path after runs. The repository currently does not expose a supported plotting command in the main workflow.
- `query-results` now supports `--view`, `--batch-id`, and `--status` filters for resumable benchmark and ablation inspection.
- Keep one-command runtime details local even though parser definitions are centralized. `scripts/_workflow_helpers.py` should stay limited to genuinely cross-command helpers such as batch-id generation and shared metric fallback logic, while one-command concerns like tiny dataset caps, inline timing, and same-checkpoint JSON writing remain local to their scripts.


