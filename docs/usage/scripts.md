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

- `dry-run`: preview only, changes nothing
- `--yes`: execute a destructive command
- `preflight`: short safety check before long runs
- `--keep-db`: keep the temporary verification DB

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

## Reset SQLite Rows

```bash
uv run reset-experiment-db
uv run reset-experiment-db --yes
uv run reset-experiment-db --tables profiling metrics experiments --yes
uv run reset-experiment-db --drop-and-recreate --yes
```

Keeps the DB file. Does not touch `results/mlflow.db`, `mlruns/`, or checkpoints.

## Delete Files

Dry-run by default. Add `--yes` to execute.

```bash
uv run cleanup-experiment-artifacts --sqlite
uv run cleanup-experiment-artifacts --sqlite --yes
uv run cleanup-experiment-artifacts --mlflow-db --mlflow-artifacts --yes
uv run cleanup-experiment-artifacts --checkpoints --yes
uv run cleanup-experiment-artifacts --all --yes
```

`--all` deletes the thesis DB, MLflow DB, `mlruns/`, and `results/checkpoints/`.

## Verification

```bash
uv run verify-setup
uv run verify-pipeline
uv run verify-pipeline --keep-db
uv run verify-sqlite --keep-db
uv run verify-sqlite
```

Verification DBs are temporary unless `--keep-db` is used.

## Data And Figures

```bash
uv run download-datasets
uv run visualize-results
```