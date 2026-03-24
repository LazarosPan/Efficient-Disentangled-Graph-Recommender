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

## Canonical Workflow

Use these as the default day-to-day commands:

```bash
uv run quick-validate
uv run quick-validate --mlflow
uv run reset-experiment-db
uv run cleanup-experiment-artifacts
```

- `quick-validate` is the single post-change validator.
- `quick-validate --mlflow` adds the MLflow observability probe.
- `reset-experiment-db` clears the thesis SQLite database and sidecars only.
- `cleanup-experiment-artifacts` clears the repository-local MLflow database, artifacts, and checkpoints.

The older direct script path still works:

```bash
uv run scripts/quick_validate.py
```

## Analysis And Inspection

```bash
uv run query-results
uv run query-results --exp 12
uv run query-results --metrics 12
uv run query-results --profiling 12
uv run query-results --alpha 12
uv run query-results --bottleneck 12
uv run visualize-results
uv run audit-metrics
```

Use `query-results` to inspect SQLite records, `visualize-results` to generate figures, and `audit-metrics` to check that logged metric names stay within the allowed PyG metric families.

## Specialized Diagnostics

These commands remain useful, but they are not the main post-change workflow:

```bash
uv run verify-setup
uv run verify-setup --all
uv run verify-sqlite
uv run verify-sqlite --keep-db
uv run preflight --dry-run
uv run preflight
uv run feature-probes
```

- `verify-setup` checks environment and import readiness. Use it when setup problems are suspected.
- `verify-setup --all` adds a short evaluation-only quick validation probe.
- `verify-sqlite` is a targeted SQLite and ExperimentLogger diagnostic. Verification DBs are temporary unless `--keep-db` is used.
- `preflight` is a representative smoke harness for a small set of experiment paths before longer runs.
- `feature-probes` runs thesis-facing utility and policy checks for optional feature usage.

## Compatibility

```bash
uv run verify-pipeline
```

`verify-pipeline` remains a compatibility alias to `quick-validate`. Keep using `quick-validate` for the main workflow.

## Data

```bash
uv run download-datasets
```

## Terms

- `--keep-db`: keep the temporary verification DB