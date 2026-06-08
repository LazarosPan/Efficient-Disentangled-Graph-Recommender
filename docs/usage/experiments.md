# Running Experiments

Use `uv run formal-run` for thesis/formal runs. Use `uv run experiment` and `uv run ablation` only for lower-level debugging or focused follow-up runs.

The examples below are smoke-sized forms that were checked against the current CLI surface.

## Primary Formal Workflow

```bash
uv run formal-run
uv run formal-run --list-profiles
uv run formal-run --profile <slug-from-list-profiles>
uv run formal-run --profile <slug-from-list-profiles> --overwrite-checkpoint
```

- `formal-run` launches the selected formal profile from `experiments/experiment_catalog.json`.
- `--profile` selects the semantic experiment bundle.
- `--list-profiles` prints the current deterministic profile slugs.
- `--overwrite-checkpoint` forces fresh training for each resolved run.
- `results/formal_run_state.json` is a strict generated resume pointer. `formal-run` automatically resumes the saved profile when the persisted semantic plan still matches the current catalog/profile definition; otherwise it starts a fresh batch for the selected profile instead of reviving compatibility fallbacks.
- Formal profiles may set `config_overrides.graph_policy` to either one value or a list like `["observed", "cagra_augmented"]`; a list expands into separate runs, and each concrete run still uses exactly one graph policy.
- Formal profiles may also set `config_overrides.preprocessing_preset` to either one preset or a list. The benchmark planner expands that sweep internally so each concrete run still carries one preprocessing preset.
- Formal runs always use the full datasets and keep OOM rows as thesis evidence while continuing to the next item. Smoke-only caps such as `sample_interactions` and `loader_max_rows` are ignored by `formal-run` even if they appear in a profile override block.
- The current default formal profile is `core-ucagnn-mainline`: it runs only the `ucagnn` preset, uses early stopping with `patience=10`, keeps the 200-epoch budget, uses learned fused scoring, and makes the mainline branch asymmetry explicit with `interest_gnn_layers=1`, `conformity_gnn_layers=2`, and `num_neighbors={"small": [[6, 3], [4, 2]], "medium": [[10, 5], [16, 8]]}` while leaving `loss_schedule` at the baseline default so BPR is active from the start.
- A second formal profile is reserved for the final matched comparison pass, where `lightgcn_paper` and `dice_paper` run beside U-CaGNN under the same canonical data and evaluation pipeline. The sampled `lightgcn` and legacy `dice_like` presets remain ablations, not paper-default baselines.

## Low-Level Single Experiments

```bash
uv run experiment --list-recipes
uv run experiment --dataset movielens1m --recipe ucagnn
uv run experiment --dataset amazonbook --recipe ucagnn --overwrite-checkpoint
uv run experiment --dataset kuairec_v2 --preset ucagnn
```

- `--recipe` is the normal way to choose a known experiment path.
- Use `ucagnn` as the main preset/recipe name for the thesis model.
- Recipes and presets own the training semantics they declare.
- The single-run CLI is intentionally selection-focused: dataset, recipe/preset, checkpointing, and lightweight tracking metadata stay public; training/runtime overrides stay inside presets, recipes, and catalog-owned profiles.
- In the current runtime, `cagra_augmented` means **observed train-interaction graph plus ANN edges**. It is not just a cuVS speed toggle over the same exact graph semantics.
- The current bootstrap path for `cagra_augmented` still requires item features; otherwise the ANN edges would be built from untrained ID-only embeddings before training starts.

## Ablations

```bash
uv run ablation --datasets movielens1m
uv run ablation --datasets amazonbook kuairec_v2 movielens1m --variants mainline no_features
uv run ablation --datasets movielens1m --variants mainline --overwrite-checkpoint
```

- `ablation` is a secondary study command, not the main thesis workflow.
- Ablation configs own their runtime semantics; the CLI only selects datasets, variants, and whether existing checkpoints should be replaced.
