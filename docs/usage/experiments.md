# Running Experiments

Use `uv run experiment` for one run, `uv run benchmark` for the formal matrix, and `uv run ablation` for component-removal studies.

Before longer experiment runs, use `uv run quick-validate` as the single repository validation command. For resets, diagnostics, and result inspection, see `docs/usage/scripts.md`.

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

```bash
uv run experiment --list-recipes
uv run experiment --dataset movielens1m --recipe full_full_graph_cagra
uv run experiment --dataset taobao --recipe cached_cagra --epochs 30 --batch-size 512
uv run experiment --dataset amazonbook --recipe full --sample-interactions 10000 --epochs 1
uv run experiment --dataset kuairec_v2 --recipe mini_batch_knn --no-mlflow
uv run experiment --dataset movielens1m --recipe full --no-auto-resume
uv run experiment --dataset movielens1m --recipe full --mlflow-experiment-name ucagnn-debug
uv run experiment --dataset movielens1m --recipe full --mlflow-tracking-uri "sqlite:///$PWD/results/mlflow.db"
```

- `--recipe` is the normal way to choose a known experiment path.
- `--sample-interactions` is for smoke checks and preflight-style runs, not thesis metrics.
- `--no-auto-resume` forces a fresh run even if a matching checkpoint already exists.

## Benchmark Matrix

```bash
uv run benchmark --tier small --dry-run  # Preview plan
uv run benchmark --tier small            # Execute
uv run benchmark --tier small --presets full --training-modes full_graph cached_propagation --seeds 42
```

`benchmark` uses the same tracking defaults as `experiment`.
Default MLflow experiment name: `ucagnn-benchmark`.

- Use `--dry-run` first when you are changing orchestration flags.
- The formal matrix is `dataset × preset × training_mode × graph_method × seed`.

## Ablations

```bash
uv run ablation --dataset movielens1m
uv run ablation --dataset movielens1m --dry-run
```

`ablation` uses the same tracking defaults as `experiment`.
Default MLflow experiment name: `ucagnn-ablation`.

- Use `--dry-run` to inspect planned variants before starting the run.

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
uv run query-results --exp 12
uv run query-results --metrics 12
uv run query-results --profiling 12
```

The supported workflow is SQLite-first inspection through `query-results`. There is currently no supported plotting command in the main workflow.

### Run Identification

New runs expose these columns explicitly:

- `run_started_at_utc`: exact run start timestamp
- `project_version`: project version from `pyproject.toml`
- `git_commit`: short git revision
- `canonical_name`: deterministic run identity from dataset, preset, training mode, graph method, and key settings

Sort by `Created at` or `run_started_at_utc` to see execution order.