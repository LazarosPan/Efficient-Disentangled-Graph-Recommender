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
uv run reset-experiment-db
uv run cleanup-experiment-artifacts
```

- `quick-validate`: default post-change validator. It runs tiny recipe, ablation, observability, and evaluation checks.
- `quick-validate` is now a zero-argument smoke command with one fixed tiny runtime shape, so validation follows the active recipe/ablation semantics instead of ad-hoc CLI overrides.
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
uv run query-results --view comparison
uv run python src/data_exploration/explore_all_datasets.py
uv run python src/data_exploration/explore_all_datasets.py --output-dir results/dataset_visualizations
```

- `query-results`: inspect the thesis SQLite database. The base command renders the default thesis summary for full-data formal and ablation runs, ordered by dataset and then by `CRRU@20` / `CRRU@40`, with the full test metric suite, inline `CRRU@20` and `CRRU@40` columns framed as Causal Resource-aware Recommendation Utility at K, dataset-local min-max normalization for those CRRU columns, a per-run `Resources:` line (`training_time_s`, `completed_train_epochs`, `peak_vram_mb`, `avg_gpu_utilization_pct`), and full canonical experiment names, then writes that report to `results/query_results.md`.
- `query-results --view completed`: show only finished runs via the SQLite completed-run view.
- `query-results --view attention`: show anything not yet cleanly completed, including running, unknown, OOM, and failed rows.
- `query-results --view errors`: show only the failed and OOM rows.
- `query-results --view comparison`: align same-config runs across code versions for side-by-side inspection.
- `python src/data_exploration/explore_all_datasets.py`: load the six benchmark datasets through the canonical loader path, always use the full selected datasets, and rewrite the fixed benchmark/profile PNGs in `results/dataset_visualizations/`.

## Data

```bash
uv run download-datasets
```

- `download-datasets`: bootstrap the small set of PyG-managed datasets that the repository can fetch automatically.
