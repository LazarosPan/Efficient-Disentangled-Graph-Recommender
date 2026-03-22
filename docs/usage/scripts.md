# Running Scripts

Run these from the repository root with `uv`.

## Paths

```text
results/thesis_experiments.db   Thesis SQLite record
results/mlflow.db               MLflow backend DB
mlruns/                         MLflow artifacts
results/checkpoints/            Local checkpoints
```

## Help

```bash
uv run list-commands
```

Short glossary:

- `--keep-db`: keep the temporary verification DB

## Quick Validation

```bash
uv run scripts/quick_validate.py
uv run scripts/quick_validate.py --mlflow
```

The default command is the single post-change validator and leaves MLflow untouched. Add `--mlflow` only when you want the MLflow observability probe.

## Preflight

```bash
uv run preflight --dry-run
uv run preflight
uv run preflight --dataset taobao --sample-interactions 50000
uv run preflight --reset-sqlite-after
```

`--reset-sqlite-after` clears SQLite rows only.

## Inspect Results

```bash
uv run query-results
uv run query-results --exp 12
uv run query-results --metrics 12
uv run query-results --profiling 12
uv run query-results --alpha 12
uv run query-results --bottleneck 12
```

## Reset SQLite

```bash
uv run reset-experiment-db
```

Deletes `results/thesis_experiments.db` and SQLite sidecars. Does not touch `results/mlflow.db`, `mlruns/`, or checkpoints.

## Reset MLflow And Checkpoints

```bash
uv run cleanup-experiment-artifacts
```

Deletes `results/mlflow.db`, `mlruns/`, and `results/checkpoints/`.

## Verification

```bash
uv run scripts/quick_validate.py
uv run verify-setup
uv run verify-sqlite --keep-db
uv run verify-sqlite
```

Use `uv run scripts/quick_validate.py` as the actual code-change validation command.

`verify-setup` is narrower: it checks environment and dependency readiness, and `verify-setup --all` adds a short `quick_validate.py --categories evaluation` probe instead of a separate pipeline-sanity script.

`verify-pipeline` remains only as a legacy alias to `quick_validate.py` for compatibility and should not be the main documented workflow.

Verification DBs are temporary unless `--keep-db` is used.

## Data And Figures

```bash
uv run download-datasets
uv run visualize-results
```