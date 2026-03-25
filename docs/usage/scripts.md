# Running Scripts

Run these from the repository root with `uv`.

## Paths

```text
results/thesis_experiments.db   Thesis SQLite record
results/mlflow.db               MLflow backend DB
mlruns/                         MLflow artifacts
results/checkpoints/            Local checkpoints
```

Use `uv run <command> --help` when you need the full option surface for a specific command.

If you want a repository-specific summary first, use:

```bash
uv run list-commands
uv run list-commands --group "Canonical Workflow"
uv run list-commands --command quick-validate
```

`list-commands` is a curated command reference for the repo. It is useful when you want a smaller, workflow-oriented overview before dropping into each command's full `--help` output.

## Canonical Workflow

Use these as the default day-to-day commands. Each one owns a distinct part of the workflow.

```bash
uv run quick-validate
uv run quick-validate --mlflow
uv run reset-experiment-db
uv run cleanup-experiment-artifacts
```

- `quick-validate`: default post-change validator. It runs tiny recipe, ablation, observability, and evaluation checks.
- `quick-validate --mlflow`: same validator, but also checks the optional MLflow logging path.
- `reset-experiment-db`: deletes only `results/thesis_experiments.db` and its SQLite sidecars.
- `cleanup-experiment-artifacts`: deletes the repo-local MLflow database, `mlruns/`, and local checkpoints.

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
uv run audit-metrics
```

- `query-results`: inspect the thesis SQLite database. Use the base command for the run list, then add one focused flag when drilling into a run.
- `query-results --exp 12`: show the stored config and metadata for experiment 12.
- `query-results --metrics 12`: show train, validation, and test metrics for experiment 12.
- `query-results --profiling 12`: show per-stage runtime and VRAM summary for experiment 12.
- `query-results --alpha 12`: inspect alpha drift for sign-aware runs.
- `query-results --bottleneck 12`: rank the slowest profiling stages for experiment 12.
- `audit-metrics`: check that source code and stored metrics stay within the allowed PyG metric families.

There is currently no supported plotting command in the main workflow. Use `query-results` for result inspection until a smaller reporting path replaces the removed plotting script.

## Specialized Diagnostics

These commands remain useful, but they are not the main post-change workflow:

```bash
uv run verify-setup
uv run verify-setup --all
uv run verify-sqlite
uv run verify-sqlite --keep-db
uv run preflight --dry-run
uv run preflight
uv run preflight --profile fast
uv run feature-probes
```

- `verify-setup`: environment and import readiness check. Use it when setup problems are suspected, not as the default post-change validator.
- `verify-setup --all`: adds `verify-sqlite` and a narrow evaluation-only quick validation probe.
- `verify-sqlite`: targeted SQLite and `ExperimentLogger` diagnostic. Verification DBs are temporary unless `--keep-db` is used.
- `preflight --dry-run`: preview the representative smoke plan before running it.
- `preflight`: representative smoke harness for a few realistic experiment paths, including checkpoint creation and resume behavior.
- `preflight --profile fast`: smallest retained preflight path when you want one very light representative run.
- `feature-probes`: thesis-facing utility and feature-policy checks for optional side-feature usage.

## Compatibility

```bash
uv run verify-pipeline
```

`verify-pipeline` is a compatibility alias to `quick-validate`. Keep using `quick-validate` for the real workflow and treat `verify-pipeline` as legacy.

## Data

```bash
uv run download-datasets
```

- `download-datasets`: bootstrap the small set of PyG-managed datasets that the repository can fetch automatically.

## Terms

- `--keep-db`: keep the temporary verification DB
- `--mlflow`: enable MLflow logging for commands that keep it off by default
- `--no-auto-resume`: ignore an existing checkpoint and start fresh
- `sample-interactions`: run against a smaller canonical interaction sample for smoke testing