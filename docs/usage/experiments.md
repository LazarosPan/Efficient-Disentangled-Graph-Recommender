# Running Experiments

Use `uv run formal-run` for thesis/formal runs. Use `uv run experiment` and `uv run ablation` for low-level single-run and ablation study control.

The examples below are smoke-sized forms that were checked against the current CLI surface.

## Primary Formal Workflow

```bash
uv run formal-run
uv run formal-run --list-profiles
uv run formal-run --resume-latest
uv run formal-run --new-run
uv run formal-run --profile <slug-from-list-profiles>
```

- `formal-run` launches the selected formal profile from `experiments/experiment_catalog.json`.
- `--profile` selects the semantic experiment bundle.
- `--list-profiles` prints the current deterministic profile slugs.
- `--resume-latest` reuses `results/formal_run_state.json`.
- `--new-run` forces a fresh batch even if a resumable state already exists.
- `--restart` starts the selected profile from the beginning under a fresh batch id.
- Formal runs always use the full datasets and keep OOM rows as thesis evidence while continuing to the next item.
- The current default formal profile is development-focused: it runs only the `ucagnn` preset, disables early stopping, keeps the full 60-epoch budget, uses learned fused scoring, and makes the mainline branch asymmetry explicit with `interest_gnn_layers=1`, `conformity_gnn_layers=2`, `num_neighbors=[10, 5]`, and `loss_schedule="baseline"` so BPR is active from the start.
- A second formal profile is reserved for the final matched comparison pass, where `lightgcn` and `dice_like` re-enter under the same pipeline.

## Low-Level Single Experiments

```bash
uv run experiment --list-recipes
uv run experiment --dataset movielens1m --recipe ucagnn --sample-interactions 100 --loader-max-rows 100 --epochs 1 --no-mlflow
uv run experiment --dataset amazonbook --recipe ucagnn --sample-interactions 100 --loader-max-rows 100 --epochs 1 --no-mlflow
uv run experiment --dataset kuairec_v2 --recipe ucagnn_knn --num-neighbors 25 10 --epochs 1 --no-mlflow
uv run experiment --dataset kuairec_v2 --preset ucagnn --graph-method knn --num-neighbors 25 10 --epochs 1 --no-mlflow
uv run experiment --dataset movielens1m --recipe ucagnn --sample-interactions 100 --loader-max-rows 100 --epochs 1 --no-auto-resume --no-mlflow
```

- `--recipe` is the normal way to choose a known experiment path.
- Use `ucagnn` as the main preset/recipe name for the thesis model.
- Recipes own the matrix-defining fields they declare.
- `--sample-interactions` and `--loader-max-rows` are smoke controls, not thesis settings.
- `--no-auto-resume` forces a fresh run even if a matching checkpoint already exists.

## Ablations

```bash
uv run ablation --dataset movielens1m --dry-run
uv run ablation --dataset movielens1m --variants mainline --epochs 1 --sample-interactions 100 --loader-max-rows 100 --no-mlflow
uv run ablation --dataset movielens1m --variants mainline --epochs 1 --sample-interactions 100 --loader-max-rows 100 --batch-id smoke-ablation --resume-batch --no-mlflow
```

- `ablation` is a secondary study command, not the main thesis workflow.
- `--batch-id` and `--resume-batch` behave the same way as in `benchmark`, but at the ablation-variant level.