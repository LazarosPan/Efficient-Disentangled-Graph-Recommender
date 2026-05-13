# U-CaGNN Configuration Skill

Use this skill when working on hyperparameters, config presets, or experiment configuration.

## Key Files
- `docs/ucagnn_implementation/config-reference.md` - All config fields with paper cross-references
- `src/utils/config.py` - UCaGNNConfig dataclass

## Config Presets
```python
from src.utils.config import UCaGNNConfig

# LightGCN baseline (non-causal)
config = UCaGNNConfig().preset_lightgcn()

# DICE-like (dual branch + independence only)
config = UCaGNNConfig().preset_dice_like()

# Full U-CaGNN mainline (fused scoring + sign-aware + IPW)
config = UCaGNNConfig().preset_full()
```

Public experiment naming note: use `ucagnn` as the main CLI preset/recipe name. The internal config method remains `preset_full()`.

## Key Hyperparameters (with paper sources)
| Parameter | Default | Source |
|-----------|---------|--------|
| embed_dim | 64 | LightGCN, DICE, MGCE, FMMRec, DDCE |
| single_branch_gnn_layers | 2 | Dedicated LightGCN / non-dual-branch depth |
| interest_gnn_layers | 1 | MGCE-style asymmetric interest-branch depth; mainline `ucagnn` sets this to 1 |
| conformity_gnn_layers | 2 | MGCE-style asymmetric conformity-branch depth; mainline `ucagnn` sets this to 2 |
| dropout | 0.1 | Reserved config knob; surfaced in formal profiles but not yet consumed by the model |
| lr | 1e-3 | LightGCN |
| weight_decay | 1e-5 | DDCE |
| batch_size | 4096 | Fallback training batch size when auto probing is disabled or running off-CUDA |
| epochs | 60 | thesis_plan.md |
| patience | 10 | DDCE |
| use_early_stopping | True | Repository policy toggle; default formal profile overrides to False |
| use_torch_compile | False | Opt-in only; dynamic mini-batch subgraphs currently recompile often enough to hurt throughput |
| grad_clip_norm | 1.0 | DICE |
| lr_scheduler | "none" | PyTorch ReduceLROnPlateau; opt-in |
| lr_scheduler_factor | 0.5 | PyTorch default halving |
| lr_scheduler_patience | 5 | Half of early-stopping patience |
| temperature | 0.1 | SimCLR, DCCL |
| propensity_clip_min | 0.01 | Surveys S1-S4 |
| propensity_clip_max | 0.99 | Surveys S1-S4 |
| use_ema | False | torch.optim.swa_utils EMA; opt-in for smoother generalization |
| ema_decay | 0.999 | Standard EMA decay rate |

## Loss Lambda Defaults
| Parameter | Default | Rationale |
|-----------|---------|--------|
| lambda_rec | 1.0 | BPR ranking signal; dominant |
| lambda_interest_bpr | 0.02 | Keep the interest branch individually rankable |
| lambda_conformity_bpr | 0.02 | Keep the conformity branch individually rankable |
| lambda_independence | 0.005 | Mild branch disentanglement without overpowering BPR |
| lambda_align | 0.02 | DirectAU positive-pair alignment inside active branches |
| lambda_uniform | 0.02 | DirectAU uniformity pressure on batch embeddings |
| lambda_pop | 0.02 | Supervise the scorer-owned popularity head |

## Evaluation Controls
- `scoring_weight_mode` defaults to `fixed` on the base config and switches to `learned` inside `preset_full()` so the mainline U-CaGNN path uses an adaptive fusion gate over interest, conformity, and popularity scores.
- `train_scoring_mode` defaults to `default` and controls which score view feeds the ranking loss.
- `eval_scoring_mode` defaults to `default` and controls which score view is used for validation/test metrics.
- The old semantic-evaluation config placeholders have been removed; the current logged evaluator uses only PyG link-prediction metrics.
- In `mini_batch` mode, `num_neighbors` must match the effective maximum active depth across `single_branch_gnn_layers`, `interest_gnn_layers`, and `conformity_gnn_layers`. The base dual-branch default is now the explicit two-hop shape `[10, 5]` because the default branch depths are `interest_gnn_layers=1` and `conformity_gnn_layers=2`.
- Each preset now rewrites its preset-owned fields explicitly before later profile/CLI overrides are applied, so switching presets on the same `UCaGNNConfig` instance does not preserve stale values. Explicit profile/CLI overrides still apply afterward, so branch-depth and `num_neighbors` changes become part of the resolved config and checkpoint identity instead of being reset back to preset values.
- `experiments/run_experiment.py` is now the shared ownership point for config input fields: `build_config()` normalizes namespace-like inputs through one `CONFIG_OVERRIDE_FIELDS` contract, and formal benchmark runs rebuild each per-run config through `build_benchmark_config_inputs(...)` before handing it back to `build_config()`.
- Formal profiles may now declare `num_neighbors_options` as a JSON-safe nested list such as `[[10, 5], [5, 3]]`. The benchmark runner expands those support-parameter variants into separate runs while each resolved run still carries one concrete `config.num_neighbors` vector, and shallower presets such as `lightgcn` consume only the prefix needed by their active depth.
- `batch_size` now defaults to `4096` for fixed-batch runs, but CUDA runs can opt into `auto_batch_size=True` to probe the largest feasible dataset-aware candidate before canonical naming, checkpoint hashing, and logging are frozen. The probe now follows the same epoch-0 shuffle used by training instead of probing an easier sequential slice, and the default candidate ladder extends down to `256` so dense sampled subgraphs can still land on a feasible size.
- `use_features` now defaults to `True`, so formal runs use canonical side-feature usage whenever a dataset provides item features; the current implementation consumes item features in Module A and falls back to ID-only embeddings on featureless datasets.
- `feature_policy` now defaults to `thesis_default`, which promotes only `safe_pre_treatment` columns from the structured feature registry on datasets with risky optional scans; switch to `all_optional` only for explicit leakage-sensitive ablations.
- `derived_split_mode` now defaults to `per_user_temporal`; set it to `global_temporal` only when you explicitly want the alternative global-temporal derived split behavior.
- `preprocessing_preset` keeps dataset-specific causal-loading choices inside the existing loader registry. Use it for cases like `kuairec_fullobs` rather than inventing new dataset names or wrapper scripts.
- `popularity_window_seconds` is optional and, when set, restricts item popularity to the trailing window inside the already train-only split. Leave it unset for the default full-train popularity summary.
- `use_early_stopping` defaults to `True` for general runs, but the current default formal profile overrides it to `False` so the development-focused `ucagnn` sweep completes all 60 epochs.
- `use_torch_compile` is now opt-in. The current mini-batch runtime feeds dynamic sampled subgraphs into `DualBranchGCN`, and observed `torch.compile(dynamic=True)` recompiles have outweighed the expected kernel-fusion win on formal runs.

## IPW Weight Range
Default `propensity_clip_min` is `0.01` (yields max weight 100*) for the base config.
`preset_full()` overrides this to `0.1` (max weight 10*) to prevent gradient explosion from poorly calibrated propensity estimates early in training.

## Preset Eval Scoring Modes
`preset_full()` now uses the fused `"default"` scorer for both training and evaluation. `preset_dice_like()` also uses the fixed `"default"` score so the baseline follows the DICE-style interest+conformity ranking path, while `preset_lightgcn()` keeps the default single-branch score. Both baseline presets now disable side features, and `preset_dice_like()` also disables sign-aware propagation to stay closer to the original repository semantics.

## Curriculum Schedule (CaDCR-Inspired)
Default: `auxiliary_losses_start_epoch=15, popularity_supervision_start_epoch=30, epochs=60` (these are the auxiliary-loss and popularity-supervision start epochs, respectively).
Phase 1 (epochs 0-14): fused BPR + branch BPR. Phase 2 (15-29): + independence, within-branch contrastive, and optional DirectAU geometry. Phase 3 (30+): + popularity supervision.

## Loss Schedule
`loss_schedule` is fixed to `"baseline"` for supported runs. Fused BPR stays active from epoch 0, and only auxiliary terms phase in via `auxiliary_loss_schedule`, `auxiliary_losses_start_epoch` / `auxiliary_losses_start_epoch`, and `popularity_supervision_start_epoch` / `popularity_supervision_start_epoch`.
Disable: set both to 0 for joint training from epoch 0.
