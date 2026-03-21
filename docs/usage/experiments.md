# Running Experiments

Use `uv run experiment` to run single experiments or `uv run benchmark` for matrix sweeps.

## Single Experiments

List available recipes:
```bash
uv run experiment --list-recipes
```

Run a specific recipe:
```bash
uv run experiment --dataset movielens1m --recipe full_full_graph_cagra
```

Override parameters:
```bash
uv run experiment --dataset taobao --recipe cached_cagra --epochs 30 --batch-size 512
```

Sampled runs (for testing):
```bash
uv run experiment --dataset amazonbook --recipe full --sample-interactions 10000 --epochs 1
```

Disable MLflow or resume:
```bash
uv run experiment --dataset kuairec_v2 --recipe mini_batch_knn --no-mlflow
uv run experiment --dataset movielens1m --recipe full --no-auto-resume
```

## Benchmark Matrix

Run formal matrix across datasets, presets, training modes, graph methods, and seeds:
```bash
uv run benchmark --tier small --dry-run  # Preview plan
uv run benchmark --tier small            # Execute
```

Customize matrix:
```bash
uv run benchmark --tier small --presets full --training-modes full_graph cached_propagation --seeds 42
```

## Ablations

Run ablation studies:
```bash
uv run ablation --dataset movielens1m
```

## MLflow Tracking

Experiments log to SQLite (`results/thesis_experiments.db`) and optionally to MLflow (`results/mlflow.db`).

`uv run experiment` writes to `results/mlflow.db` by default. Plain `mlflow ui` or `mlflow server` without `--backend-store-uri` will use the separate default file `mlflow.db` in the repository root, which is not the project tracking database.

In the MLflow table, the empty capitalized `Dataset` column is MLflow's own evaluation-dataset field. The lowercase `dataset` column is this project's actual run parameter.

The preferred endpoint for this repository is the tracking server on port 9090. Use the direct UI only if you want a quick local browser over the SQLite file.

If you want one shell setting for all MLflow commands in this repository, set:
```bash
set -x MLFLOW_TRACKING_URI "sqlite:///$PWD/results/mlflow.db"
```

### Preferred: Tracking Server

Use this as the main UI and API endpoint:
```bash
uv run mlflow server --backend-store-uri "sqlite:///$PWD/results/mlflow.db" --host 127.0.0.1 --port 9090
```

Python clients can then talk to the same endpoint:
```bash
set -x MLFLOW_TRACKING_URI "http://127.0.0.1:9090"
```

### Optional: Direct Local UI

To view results locally:
```bash
uv run mlflow ui --backend-store-uri "sqlite:///$PWD/results/mlflow.db" --port 5002
```

This reads the same database as 9090, but without the tracking-server API layer.

### Run Identification

New runs expose these columns explicitly:

- `run_started_at_utc`: exact run start timestamp
- `project_version`: project version from `pyproject.toml`
- `git_commit`: short git revision
- `canonical_name`: deterministic run identity from dataset, preset, training mode, graph method, and key settings

Sort by `Created at` or `run_started_at_utc` to see execution order.