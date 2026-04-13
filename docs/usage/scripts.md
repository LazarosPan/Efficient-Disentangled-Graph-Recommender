# Running Scripts

Run these from the repository root with `uv`. Experiment-launch commands live in [docs/usage/experiments.md](docs/usage/experiments.md).

The runnable examples below are checked forms of the current CLI. Heavier diagnostics use smoke-sized arguments where possible.

## Paths

```text
results/thesis_experiments.db   Thesis SQLite record
results/mlflow.db               MLflow backend DB
results/formal_run_state.json   Formal-run resume state
mlruns/                         MLflow artifacts
results/checkpoints/            Local checkpoints
```

Use `uv run <command> --help` when you need the full option surface for a specific command.

## Validation And Maintenance

```bash
uv run quick-validate
uv run quick-validate --mlflow
uv run reset-experiment-db
uv run cleanup-experiment-artifacts
```

- `quick-validate`: default post-change validator. It runs tiny recipe, ablation, observability, and evaluation checks.
- `quick-validate --mlflow`: same validator, but also checks the optional MLflow logging path.
- `reset-experiment-db`: deletes only `results/thesis_experiments.db` and its SQLite sidecars.
- `cleanup-experiment-artifacts`: deletes the repo-local MLflow database, `results/formal_run_state.json`, `mlruns/`, and local checkpoints.

The older direct script path still works:

```bash
uv run scripts/quick_validate.py
```

## Analysis And Inspection

```bash
uv run query-results
uv run query-results --view completed
uv run query-results --view attention
uv run query-results --view errors
uv run evaluate-scoring-modes --checkpoint-path results/checkpoints/<checkpoint>.pt
uv run query-results --batch-id smoke-bench
uv run query-results --status completed
uv run query-results --exp 12
uv run query-results --metrics 12
uv run query-results --profiling 12
uv run query-results --alpha 12
uv run query-results --bottleneck 12
```

- `query-results`: inspect the thesis SQLite database. Use the base command for the run list, then add one focused flag when drilling into a run.
- `query-results --view completed`: show only finished runs via the SQLite completed-run view.
- `query-results --view attention`: show anything not yet cleanly completed, including running, unknown, OOM, and failed rows.
- `query-results --view errors`: show only the failed and OOM rows.
- `evaluate-scoring-modes --checkpoint-path ...`: reload one saved checkpoint and compare the thesis metrics under multiple evaluation-time scoring modes without retraining.
- `query-results --batch-id smoke-bench`: list only the runs that belong to one benchmark or ablation batch.
- `query-results --status completed`: filter the list to one terminal status such as `completed`, `oom`, or `failed`.
- `query-results --exp 12`: show the stored config and metadata for experiment 12.
- `query-results --metrics 12`: show train, validation, and test metrics for experiment 12.
- `query-results --profiling 12`: show per-stage runtime and VRAM summary for experiment 12.
- `query-results --alpha 12`: inspect alpha drift for sign-aware runs.
- `query-results --bottleneck 12`: rank the slowest profiling stages for experiment 12.

There is currently no supported plotting command in the main workflow. Use `query-results`, `evaluate-scoring-modes`, and the convenience SQLite views for inspection.

## Data

```bash
uv run download-datasets
```

- `download-datasets`: bootstrap the small set of PyG-managed datasets that the repository can fetch automatically.
