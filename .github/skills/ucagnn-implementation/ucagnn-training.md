# U-CaGNN Training Skill

Use this skill when working on training loop, evaluation, checkpoints, profiling, or experiment logging.

## Key Files
- `docs/ucagnn_implementation/training.md` - Training pipeline with paper cross-references
- `src/utils/trainer_runtime.py` - Shared trainer setup, checkpointing, profiling helpers, and early-stopping state
- `src/training/trainer.py` - Trainer class
- `src/training/evaluator.py` - Evaluator (PyG link-prediction metrics at K)
- `src/profiling/gpu_profiler.py` - GPUProfiler and profile_stage
- `src/utils/experiment_logger.py` - SQLite ExperimentLogger

## Paper Sources
| Feature | Source |
|---------|--------|
| Adam lr=1e-3 | LightGCN |
| Gradient clipping max_norm=1.0 | DICE |
| Early stopping patience=10 | DDCE |
| 60 epochs default | thesis_plan.md |

## Training Pipeline
```python
from src.training.trainer import Trainer
from src.utils.experiment_logger import ExperimentLogger

experiment_logger = ExperimentLogger()
exp_id = experiment_logger.log_experiment(config.dataset, config)

trainer = Trainer(model, loss_suite, data, config, profiler,
                  experiment_logger=experiment_logger, exp_id=exp_id)
history = trainer.train()

# Test metrics
test_metrics = trainer.evaluator.evaluate(model, data, data.test_mask)
for metric, value in test_metrics.items():
    experiment_logger.log_metric(exp_id, metric, value, split="test")

experiment_logger.close()
trainer.save_checkpoint("results/checkpoints/ucagnn_best.pt")
```

## Ownership Notes
- The three trainer modes keep separate epoch loops for readability and behavior isolation.
- `src/utils/trainer_runtime.py` owns only duplicated lifecycle machinery through `TrainerRuntime`: module/device setup, optimizer creation, checkpoint save/load, profiling toggles, resume history, and early-stopping helpers.
- Keep mode-specific semantics local to the concrete trainers: full-graph non-finite batch skipping, cached propagation `retain_graph` handling, and mini-batch subgraph/local-index behavior should not be hidden behind a generic batch callback.
- Treat `batch_size` as an interaction-loss knob, not a full-graph VRAM knob. In `full_graph`, each batch still propagates over the same full `edge_index`; move to `cached_propagation` or `mini_batch` before tuning `batch_size` for memory.
- `num_neighbors` is only meaningful in `mini_batch` because it controls `SubgraphSampler` fan-out. It does not change memory usage for `full_graph` or `cached_propagation`.
- Named recipes own matrix-defining fields such as `training_mode` and `graph_method`; conflicting CLI flags should be rejected rather than silently treated as an auto-scaling mechanism.
- Formal batch runners expose `--batch-id` and `--resume-batch` so benchmark and ablation sweeps can skip terminal rows (`completed`, `oom`, `failed`) without hiding failed feasibility cases.
- `main.py` is now the simple formal orchestration entry point via `uv run formal-run`. It persists `results/formal_run_state.json` and is the preferred path when the user wants one command that resumes formal experiments after interruption.
- `formal-run` now separates semantic protocol identity from execution identity: `--profile` selects a predefined catalog entry from `experiments/experiment_catalog.json`, while `batch_id` is only the resumable execution label stored in SQLite and MLflow.
- Formal profiles should own the full thesis matrix and support-parameter bundle directly in the catalog. The formal wrapper should stay thin and should not reintroduce parallel definitions for epochs, learning rate, sampled interactions, or fallback behavior.
- `--fallback-on-oom` is orchestration-level only: keep the original OOM row as thesis evidence, then launch a second explicit fallback run under `cached_propagation` or `mini_batch` rather than mutating the original run identity.
- SQLite experiment review now has three convenience views: completed runs, attention-required runs, and strict error runs. Prefer `query-results --view completed|attention|errors` over manual SQL when triaging long batches.
- Formal runs now persist `profile_name` alongside status, batch id, and hardware metadata, so query and MLflow inspection can distinguish the scientific profile from the operational batch.

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
- Validation and test evaluation logs only the thesis-facing PyG metrics: `NDCG@20`, `Recall@20`, `AveragePopularity@20`, `NDCG@50`, `Recall@50`, and `AveragePopularity@50`.
- The evaluator builds those six metrics via `LinkPredMetricCollection`. MetricCollection removes per-metric runtime update loops; construction still needs one metric instance per metric and cutoff.
- `src/training/evaluator.py` is also the source of truth for the thesis-primary metric subset and lower-is-better metric polarity, so downstream benchmark, ablation, reporting, and scoring-mode scripts should import those constants instead of re-declaring them.
- Treat `AveragePopularity@20` and `AveragePopularity@50` as debiasing readouts where lower values are better. Summary aggregation, delta interpretation, and heatmap colors should reflect that polarity.
- The preferred mechanism comparison is same-checkpoint evaluation under `default`, `interest_only`, and `conformity_suppressed`. Leave `conformity_only` and `counterfactual_only` available for debugging rather than thesis headline tables.
- Use `scripts/evaluate_scoring_modes.py` to run the thesis mechanism table from one saved checkpoint without retraining separate runs.
- PyG 2.7 also exposes Diversity and Personalization. They are allowed by the repo's metric audit, but the current evaluator does not log them because they require extra category inputs or additional pairwise recommendation computation.
- External implementation audits may discuss non-PyG causal-uplift evaluators such as PropCare's semi-simulated `CPrec` or `CDCG` pipeline, but those remain reference analyses unless the runtime data contract is extended with treatment, propensity, and causal-effect labels.

## MLflow Routing
- `experiments/run_experiment.py` uses `--mlflow-tracking-uri` first, then `MLFLOW_TRACKING_URI`, then the project default `results/mlflow.db`.
- `results/mlflow.db` is the MLflow backend store and `mlruns/` holds artifacts.
- Benchmark runs default to MLflow experiment `ucagnn-benchmark`; ablations default to `ucagnn-ablation`.
- Checkpoints are logged to MLflow under the `checkpoints/` artifact subpath.
- Use `uv run reset-experiment-db` to delete the repository-local thesis SQLite database and its sidecars.
- Use `uv run cleanup-experiment-artifacts` to delete `results/mlflow.db`, `mlruns/`, and `results/checkpoints/` in one step.
- Matrix fields should not be duplicated across MLflow params and tags; ordering/provenance should use explicit params such as `run_started_at_utc`, `project_version`, and `git_commit`.

## Fast Validation Workflow
- Use `quick-validate` as the default ultra-fast post-change validation path across all six datasets.
- Use `list-commands --command <name>` when you want the repository's short workflow summary before opening a command's full `--help` output.
- `quick_validate.py` is now the unified tiny-scale pipeline validator. By default it exercises the canonical recipe matrix, all ablation variants, representative observability probes (profiling, checkpoint/resume, feature path), and evaluation scoring modes with aggressive row caps and sampled interactions.
- Feature usage now follows the formal config default (`use_features=True`, `feature_policy="thesis_default"`), so tiny validation covers the same feature-aware model path as formal runs. The capped dataset loader path stays practical by reusing cached capped loads across repeated recipe cases.
- Treat tiny validation as row-scaled, not schema-changed: the intended invariant is that formal and tiny runs share the same canonical fields and feature-engineering path, while `loader_max_rows`, `sample_interactions`, `epochs`, `batch_size`, and semantic-eval caps control runtime.
- Use category filters such as `--categories recipes` or `--categories observability` only when debugging a specific surface; the default command is intended to be the single broad post-change validation entry point.
- Quick validation keeps MLflow disabled by default, so `uv run quick-validate` does not create MLflow tables or artifact files unless `--mlflow` is passed explicitly.
- Use `scripts/preflight_experiments.py --profile fast` when you want a single smoke/preflight-style pass without the paired ablation check.
- Use `query-results` as the supported SQLite inspection path after runs. The repository currently does not expose a supported plotting command in the main workflow.
- `query-results` now supports `--view`, `--batch-id`, and `--status` filters for resumable benchmark and ablation inspection.
- Keep the tiny validation CLIs locally explicit. Shared helpers should stay limited to low-level utilities such as dataset-limit lookup, timed `run_experiment()` execution, and JSON report writing; config and namespace wiring belongs in each script.

## Feature Probe Workflow
- Use `feature-probes` when you want tiny thesis-oriented feature screening rather than full-matrix validation.
- The script separates two questions:
- utility probes run `id_only` vs `thesis_default` across all feature-bearing thesis datasets;
- policy probes run `thesis_default` vs `all_optional` only on datasets where those policies actually differ today (`kuairec_v2`, `kuairand1k`).
- This keeps the thesis narrative clean: every feature-bearing dataset gets a utility check, while leakage-sensitive policy comparisons are restricted to datasets where broader optional scans would materially change the canonical inputs.
- The probe summary is written to `results/feature_policy_probes.json` and compares only PyG-backed metric names.
- Use `uv run audit-metrics` to print the allowed PyG metric families and scan the implementation source, SQLite metrics table, and MLflow test metrics for non-PyG names without adding runtime guard code.
