# Running Experiments

Use `uv run formal-run` for thesis/formal runs. Use `uv run experiment` and `uv run ablation` only for lower-level debugging or focused follow-up runs.

The examples below are smoke-sized forms that were checked against the current CLI surface.

## Primary Formal Workflow

```bash
uv run formal-run
uv run formal-run --list-profiles
uv run formal-run --profile edgrec-compact-search-prior
uv run formal-run --profile amazonbook-edgrec-compact-candidate
uv run formal-run --profile amazonbook-edgrec-deep-features-candidate
uv run formal-run --profile <slug-from-list-profiles>
uv run formal-run --profile <slug-from-list-profiles> --overwrite-checkpoint
```

- `formal-run` launches the selected formal profile from `experiments/experiment_catalog.json`.
- `--profile` selects the semantic experiment bundle.
- `--list-profiles` prints the current deterministic profile slugs.
- `--overwrite-checkpoint` forces fresh training for each resolved run.
- `results/formal_run_state.json` is a strict generated resume pointer. `formal-run` automatically resumes the saved profile when the persisted semantic plan still matches the current catalog/profile definition; otherwise it starts a fresh batch for the selected profile instead of reviving compatibility fallbacks.
- Formal profiles use the observed train-interaction graph. `graph_policy` is retained for compatibility but only `observed` is supported.
- Formal profiles may also set `config_overrides.preprocessing_preset` to either one preset or a list. The benchmark planner expands that sweep internally so each concrete run still carries one preprocessing preset.
- Formal runs always use the full datasets and keep OOM rows as thesis evidence while continuing to the next item. Smoke-only caps such as `sample_interactions` and `loader_max_rows` are ignored by `formal-run` even if they appear in a profile override block.
- The current default candidate profile is `edgrec-compact-search-prior`: it runs only the `edgrec` preset on KuaiRec_v2, MovieLens1M, and KuaiRand1K, uses early stopping with `patience=8`, keeps the 200-epoch budget, keeps side features disabled, and fixes the lightweight `interest_gnn_layers=1`, `conformity_gnn_layers=2` branch path with `num_neighbors={"small": [[8, 4], [10, 5]], "medium": [[8, 4], [10, 5]]}`.
- `edgrec-compact-search-prior` is a compact low-risk search prior and formal candidate, not the final thesis selection. Current evidence supports it most strongly for KuaiRec_v2, treats MovieLens1M as near-parity/speed evidence with weaker popularity diagnostics, and keeps KuaiRand1K as a stress-test diagnostic rather than an accuracy headline.
- AmazonBook is intentionally separate from the shared compact default queue, but it is not excluded from EDGRec. Use `amazonbook-edgrec-compact-candidate` and `amazonbook-edgrec-deep-features-candidate` for the dataset-specific compact-vs-deep comparison, and keep `lightgcn_paper` as the current accuracy baseline until formal reruns decide the final thesis profile.
- Dedicated paper-baseline profiles own `lightgcn_paper` and `dice_paper` runs under the same canonical data and evaluation pipeline. The sampled `lightgcn` and legacy `dice_like` presets remain ablations, not paper-default baselines.

## Optuna Search

```bash
uv run search-experiments --list-spaces
uv run search-experiments --space edgrec-mechanism-coarse --dataset kuairec_v2 --trials 120
uv run search-experiments --space edgrec-mechanism-coarse --dataset movielens1m --trials 50
uv run search-experiments --space edgrec-mechanism-coarse --dataset kuairand1k --trials 25
uv run search-experiments --space amazonbook-edgrec-candidate-search --dataset amazonbook --trials 12
uv run search-experiments --space edgrec-core-optimization --trials 12
uv run search-experiments --space edgrec-core-optimization --dataset amazonbook --trials 12
uv run search-experiments --space edgrec-core-optimization --trials 1 --dry-run
uv run search-experiments --space edgrec-core-optimization --dataset amazonbook --trials 5 --mlflow
uv run report-optuna-optimization
uv run export-optuna-figures
uv run optuna-dashboard sqlite:///results/optuna_studies.db
```

- `search-experiments` is EDGRec-only. It resolves `experiments/search_spaces.json` entries via each `base_profile`, samples existing `EDGRecConfig` fields, and still executes each trial through `build_config()` and `run_experiment()`.
- `search-experiments --space a,b` runs search spaces sequentially, matching `formal-run` queue syntax. `--dataset` and `--trials` apply to every queued space. Omit `--study-name` for queues so each space keeps its normal study naming.
- `edgrec-mechanism-coarse` is the coarse mechanism discovery space. It samples profile labels for score fusion, item-branch capacity, explicit context/features, loss family, and graph depth; `experiments/search_spaces.json` owns both those labels and the sibling `profile_overrides` mappings to concrete config overrides. Resolved trials log both `sampled_params` and `resolved_config_overrides`. Use KuaiRec_v2 for the main 100-150 trial pass, MovieLens1M for a 40-60 trial sanity pass, and KuaiRand1K only for narrowed transfer checks. AmazonBook is excluded from this broad shared pass, not from EDGRec optimization.
- `amazonbook-edgrec-candidate-search` is the AmazonBook-specific Optuna comparison for the `compact` and `deep_features` EDGRec profile families under one `ValidationOnlineCRRU@20_40` objective/revision. Use it before promoting either AmazonBook EDGRec candidate into a thesis-facing formal profile.
- `edgrec-mechanism-coarse` keeps the scalar knobs coarse, with 40 startup TPE trials before exploitation. Dataset-local overrides widen only known useful regions: MovieLens1M adds `lr=0.005`, `score_mix_min_weight=0.1`, and margins `[20,40,80]`; KuaiRand1K uses margins `[50,70,80]`; KuaiRec_v2 keeps margins `[10,20,40]`.
- `edgrec-mechanism-coarse` uses validation `ValidationAccuracy@20_40`: `0.50*NDCG@20 + 0.25*Recall@20 + 0.15*NDCG@40 + 0.10*Recall@40`. CRRU remains logged when the full metric family is available, but it is not the broad-discovery objective for this space.
- CAGRA graph augmentation and CAGRA-specific Optuna spaces were removed. They do not address the current training-time VRAM/epoch bottleneck.
- `edgrec-core-optimization` is the CRRU-oriented compact EDGRec tuning space. It inherits `edgrec-compact-search-prior`, keeps side features, IPW, contrastive, and DirectAU disabled by default, and samples graph profiles for `(1,1)/[8]`, `(1,2)/[8,4]`, and `(2,2)/[10,5]`. Without `--dataset`, the controller expands the space into one independent dataset-local study per listed dataset. With `--dataset`, it runs only that dataset's study. In both cases, `--trials N` means N fresh informative trials for each selected dataset.
- The CRRU objective `ValidationOnlineCRRU@20_40` is an online proxy with the same component/exponent structure as report CRRU, including VRAM and seconds-per-epoch efficiency penalties; exact report CRRU still uses dataset-local section-row min-max after rows exist. Search runs skip final test evaluation; test metrics remain reserved for promoted confirmation profiles.
- Search runs do not save or resume checkpoints. They also keep MLflow disabled by default to avoid large exploratory artifacts; pass `--mlflow` only when you explicitly want MLflow mirroring for a short search.
- Optuna search spaces are configured separately from formal profiles: use `experiments/search_spaces.json` for tuning spaces and `experiments/experiment_catalog.json` for deterministic formal profiles/recipes.
- Trial runs are grouped with `batch_id=optuna-<study>-trial-<number>` and `profile_name=<search-space>`. The sampled values are present in each experiment `config_json`, while trial-level search metadata, importances, and failures are owned by Optuna RDB storage (`results/optuna_studies.db`), not the thesis SQLite database.
- `uv run report-optuna-optimization` writes `results/optuna_optimization.md`; `uv run export-optuna-figures` writes PNG diagnostics; `uv run optuna-dashboard sqlite:///results/optuna_studies.db` opens the interactive dashboard.
- Search spaces are exploratory. Promote selected candidates manually into named formal profiles before running full 200-epoch confirmation and reporting test results.
- Historical/internal Optuna labels such as `no_context_no_features` are search profile labels, not public ablation variants.

## Checkpoint Retention

```bash
uv run prune-checkpoints
uv run prune-checkpoints --keep 3 --execute
```

- `prune-checkpoints` ranks mapped checkpoints from `results/thesis_experiments.db` and keeps the top N per dataset and model family (`edgrec`, `dice`, `lightgcn`).
- The command defaults to a dry run. Add `--execute` only after reviewing the deletion plan.
- Existing Optuna-search checkpoints are deletion candidates by default because search results and parameters live in SQLite, not checkpoint files.

## Low-Level Single Experiments

```bash
uv run experiment --list-recipes
uv run experiment --dataset movielens1m --recipe edgrec
uv run experiment --dataset amazonbook --recipe edgrec --overwrite-checkpoint
uv run experiment --dataset kuairec_v2 --preset edgrec
```

- `--recipe` is the normal way to choose a known experiment path.
- Use `edgrec` as the main preset/recipe name for the thesis model.
- Recipes and presets own the training semantics they declare.
- The single-run CLI is intentionally selection-focused: dataset, recipe/preset, checkpointing, and lightweight tracking metadata stay public; training/runtime overrides stay inside presets, recipes, and catalog-owned profiles.
- The current runtime trains on the observed train-interaction graph only. CAGRA graph augmentation and CAGRA evaluation filtering are not supported.

## Ablations

```bash
uv run ablation --datasets movielens1m
uv run ablation --datasets amazonbook kuairec_v2 movielens1m --variants mainline with_features
uv run ablation --datasets kuairand1k --variants mainline with_ipw
uv run ablation --datasets movielens1m --variants mainline --overwrite-checkpoint
```

- `ablation` is a secondary study command, not the main thesis workflow.
- Ablation configs own their runtime semantics; the CLI only selects datasets, variants, and whether existing checkpoints should be replaced.
- Public ablations are positive opt-ins or removals around `preset_full()`: `mainline`, `with_features`, `with_contrastive`, `with_ipw`, `no_popularity_head`, and `no_independence`.
