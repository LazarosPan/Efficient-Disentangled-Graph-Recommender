# Running Experiments

Use `uv run formal-run` as the default formal experiment entry point. Use `uv run experiment` and `uv run benchmark` only when you deliberately want lower-level control.

The runnable examples below are smoke-sized forms that were checked against the current CLI surface.

Before changing training code, use `uv run quick-validate` as the single repository validation command. For formal thesis runs, the normal workflow should start from `formal-run`. For resets, diagnostics, and result inspection, see `docs/usage/scripts.md`.

## Paths

```text
results/thesis_experiments.db   Thesis SQLite record
results/mlflow.db               MLflow backend DB
mlruns/                         MLflow artifacts
results/checkpoints/            Local checkpoints
```

## Terms

- `ablation`: run variants with parts removed to measure impact
- `recipe`: named experiment setup
- `preset`: model configuration family
- `sample-interactions`: smaller sampled run for testing
- `--no-mlflow`: disable MLflow for that run
- `--no-auto-resume`: ignore an existing checkpoint and start fresh

## Single Experiments

## Primary Formal Workflow

```bash
uv run formal-run --profile v1
uv run formal-run --list-profiles
uv run formal-run --resume-latest
uv run formal-run --new-run --profile v1
```

- `formal-run` is the easiest way to launch the full formal matrix.
- `--profile` is the semantic experiment bundle. It is loaded directly from `experiments/experiment_catalog.json`, alongside the rest of the repository's predefined experiment definitions.
- Each formal profile freezes both the matrix and the support parameters for that thesis run: dataset tier, presets, training modes, graph methods, seeds, epochs, learning rate, batch size, and mini-batch fan-out.
- `--version` is still accepted as a compatibility alias for `--profile`, but the scientific meaning is now the profile bundle rather than the execution batch slug.
- Re-running the same `--profile` resumes the saved batch for that profile when `results/formal_run_state.json` already points at it.
- `--resume-latest` reuses the last saved formal-run state from `results/formal_run_state.json`.
- `--new-run` forces a fresh batch even if a resumable state file already exists.
- `--restart` starts the selected profile from the beginning under a fresh batch id, instead of reusing the old one.
- `formal-run` always uses the full datasets. It does not expose sampled-interaction or loader-cap overrides, because those are smoke-test controls rather than formal thesis settings.
- The wrapper still uses the benchmark runner underneath, so failed runs are logged and the matrix continues to the next item.
- Formal OOM behavior is log-and-continue by default. The failing run is kept in SQLite and MLflow as thesis evidence, and the runner moves to the next experiment without an automatic fallback retry.
- `batch_id` is now purely operational. It identifies one resumable execution of a formal profile and is kept separate from the semantic profile name in SQLite and MLflow.

## Low-Level Single Experiments

```bash
uv run experiment --list-recipes
uv run experiment --dataset movielens1m --recipe full --sample-interactions 100 --loader-max-rows 100 --epochs 1 --device cpu --no-mlflow
uv run experiment --dataset movielens1m --recipe cached --sample-interactions 100 --loader-max-rows 100 --epochs 1 --batch-size 128 --device cpu --no-mlflow
uv run experiment --dataset amazonbook --recipe mini_batch --sample-interactions 100 --loader-max-rows 100 --epochs 1 --device cpu --no-mlflow
uv run experiment --dataset kuairec_v2 --recipe mini_batch --num-neighbors 25 10 --epochs 1 --no-mlflow
uv run experiment --dataset kuairec_v2 --preset full --training-mode mini_batch --graph-method knn --num-neighbors 25 10 --epochs 1 --no-mlflow
uv run experiment --dataset movielens1m --recipe full --sample-interactions 100 --loader-max-rows 100 --epochs 1 --device cpu --no-auto-resume --no-mlflow
uv run experiment --dataset movielens1m --recipe full --sample-interactions 100 --loader-max-rows 100 --epochs 1 --device cpu --mlflow-experiment-name ucagnn-debug
uv run experiment --dataset movielens1m --recipe full --sample-interactions 100 --loader-max-rows 100 --epochs 1 --device cpu --mlflow-tracking-uri "sqlite:///$PWD/results/mlflow.db"
```

- `--recipe` is the normal way to choose a known experiment path.
- Recipes own the matrix-defining fields they declare. If you select `--recipe full`, conflicting `--training-mode` or `--graph-method` flags are rejected instead of being silently ignored.
- `--sample-interactions` is for smoke checks and preflight-style runs, not thesis metrics.
- `--loader-max-rows` is useful when you want the dataset loader itself to stay in smoke-test territory.
- `--no-auto-resume` forces a fresh run even if a matching checkpoint already exists.

## Large Graph Guidance

- `batch_size` is not the main VRAM control for `full_graph`. The full graph is propagated on every forward pass, so shrinking `batch_size` does not remove the propagation spike.
- For large datasets, use an explicit scalable path: `--recipe mini_batch`, `--recipe cached`, or `--preset ... --training-mode ... --graph-method ...`.
- `--num-neighbors` only affects `mini_batch` mode. It is ignored by `full_graph` and `cached_propagation`.
- For thesis-scale comparisons, keep `training_mode` explicit in the command, recipe, and resulting canonical run name rather than relying on hardware-specific auto-tuning.

## Benchmark Matrix

`benchmark` still exists, but it is now the lower-level interface behind `formal-run`.

```bash
uv run benchmark --tier small --dry-run  # Preview plan
uv run benchmark --tier small --presets full --training-modes full_graph --graph-methods dense --seeds 42 --epochs 1 --sample-interactions 100 --device cpu --no-mlflow
uv run benchmark --tier small --presets full --training-modes full_graph --graph-methods dense --seeds 42 --epochs 1 --sample-interactions 100 --device cpu --batch-id smoke-bench --resume-batch --no-mlflow
```

`benchmark` uses the same tracking defaults as `experiment`.
Default MLflow experiment name: `ucagnn-benchmark`.

- Use `--dry-run` first when you are changing orchestration flags.
- The formal matrix is `dataset × preset × training_mode × graph_method × seed`.
- `--profile-name` is the optional semantic label recorded by higher-level wrappers such as `formal-run`; it is separate from `--batch-id`.
- `--batch-id` groups a long benchmark run into one resumable batch record in SQLite.
- `--resume-batch` skips matrix items that already reached a terminal state (`completed`, `oom`, or `failed`) for that batch.
- `--fallback-on-oom cached_propagation` or `--fallback-on-oom mini_batch` records the original OOM run first, then launches a second explicit fallback run instead of mutating the failing run in place.

## Ablations

`ablation` is a secondary study command, not the main thesis workflow.

```bash
uv run ablation --dataset movielens1m --dry-run
uv run ablation --dataset movielens1m --variants full --epochs 1 --sample-interactions 100 --loader-max-rows 100 --device cpu --no-mlflow
uv run ablation --dataset movielens1m --variants full --epochs 1 --sample-interactions 100 --loader-max-rows 100 --device cpu --batch-id smoke-ablation --resume-batch --no-mlflow
```

`ablation` uses the same tracking defaults as `experiment`.
Default MLflow experiment name: `ucagnn-ablation`.

- Use `--dry-run` to inspect planned variants before starting the run.
- `--batch-id`, `--resume-batch`, and `--fallback-on-oom` behave the same way as in `benchmark`, but at the ablation-variant level.

## Long-Run Resume Policy

- Batch resume is orchestration-level, not checkpoint-level mutation. A run that OOMs remains logged as `status=oom` so the thesis record keeps the feasibility evidence.
- Explicit fallback is a lower-level benchmark or ablation option. The default formal workflow does not use it; it logs the OOM row and continues.
- `formal-run` persists the latest formal batch configuration to `results/formal_run_state.json`, including the semantic `profile_name` and the current execution `batch_id`, so a restart after a stopped process or powered-off machine can reuse the same batch plan.
- Use `uv run query-results --view completed` for the clean finished list, `uv run query-results --view attention` for anything that still needs attention, and `uv run query-results --view errors` for the strict failed/OOM list.
- Add `--batch-id <id>` when you want the same exploration flow scoped to one benchmark or ablation batch.

## MLflow Tracking

Default MLflow resolution:

1. `--mlflow-tracking-uri`
2. `MLFLOW_TRACKING_URI`
3. `results/mlflow.db`

Use this shell setting if you want all MLflow commands to point at the repo DB:

```bash
set -x MLFLOW_TRACKING_URI "sqlite:///$PWD/results/mlflow.db"
```

Preferred UI:

```bash
uv run mlflow server --backend-store-uri "sqlite:///$PWD/results/mlflow.db" --host 127.0.0.1 --port 9090
```

Optional direct UI:

```bash
uv run mlflow ui --backend-store-uri "sqlite:///$PWD/results/mlflow.db" --port 5002
```

`mlflow.db` is the backend store. `mlruns/` holds artifacts. Local checkpoints remain under `results/checkpoints/` and are logged to MLflow under the `checkpoints/` artifact subpath.

## Cleanup

For reset and cleanup commands, use the workflow in `docs/usage/scripts.md`.

## Result Inspection

After runs finish, inspect the SQLite record with:

```bash
uv run query-results
uv run query-results --view completed
uv run query-results --view attention
uv run query-results --view errors
uv run query-results --batch-id smoke-bench
uv run query-results --status completed
uv run query-results --status oom
uv run query-results --exp 12
uv run query-results --metrics 12
uv run query-results --profiling 12
```

The supported workflow is SQLite-first inspection through `query-results`. The SQLite database now exposes convenience views for completed runs, attention-required runs, and strict error runs, and both SQLite and MLflow now record the semantic `profile_name` separately from the execution `batch_id`. There is currently no supported plotting command in the main workflow.

### Run Identification

New runs expose these columns explicitly:

- `run_started_at_utc`: exact run start timestamp
- `project_version`: project version from `pyproject.toml`
- `git_commit`: short git revision
- `canonical_name`: deterministic run identity from dataset, preset, training mode, graph method, and key settings

Sort by `Created at` or `run_started_at_utc` to see execution order.