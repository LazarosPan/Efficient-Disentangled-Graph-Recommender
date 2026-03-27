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

# DICE-like (dual branch + orthogonality)
config = UCaGNNConfig().preset_dice_like()

# Full U-CaGNN (all modules + sign-aware + IPW)
config = UCaGNNConfig().preset_full()
```

## Key Hyperparameters (with paper sources)
| Parameter | Default | Source |
|-----------|---------|--------|
| embed_dim | 64 | LightGCN, DICE, MGCE, FMMRec, DDCE |
| n_gnn_layers | 2 | LightGCN, MCLN, CaDSI |
| interest_gnn_layers | None -> `n_gnn_layers` | MGCE-style asymmetric depth |
| conformity_gnn_layers | None -> `n_gnn_layers` | MGCE-style asymmetric depth |
| lr | 1e-3 | LightGCN |
| weight_decay | 1e-5 | DDCE |
| epochs | 60 | thesis_plan.md |
| patience | 10 | DDCE |
| grad_clip_norm | 1.0 | DICE |
| lr_scheduler | "none" | PyTorch ReduceLROnPlateau; opt-in |
| lr_scheduler_factor | 0.5 | PyTorch default halving |
| lr_scheduler_patience | 5 | Half of early-stopping patience |
| temperature | 0.1 | SimCLR, DCCL |
| propensity_clip_min | 0.01 | Surveys S1-S4 |
| propensity_clip_max | 0.99 | Surveys S1-S4 |

## Evaluation Controls
- `scoring_weight_mode` defaults to `fixed` and can be switched to `learned` to let the model learn simplex-constrained weights over interest, conformity, and counterfactual scores.
- `eval_scoring_mode` defaults to `default` and can switch evaluation to `interest_only`, `conformity_only`, `counterfactual_only`, or `conformity_suppressed` without changing training losses.
- The old semantic-evaluation config placeholders have been removed; the current logged evaluator uses only PyG link-prediction metrics.
- In `mini_batch` mode, `num_neighbors` must now match the effective maximum branch depth, not only `n_gnn_layers`.
- `use_features` now defaults to `True`, so formal runs use canonical side-feature usage whenever a dataset provides item features; the current implementation consumes item features in Module A and falls back to ID-only embeddings on featureless datasets.
- `feature_policy` now defaults to `thesis_default`, which enforces the thesis-safe item-feature allowlist on datasets with risky optional scans; switch to `all_optional` only for explicit leakage-sensitive ablations.

## IPW Weight Range
With default clip bounds [0.01, 0.99], IPW weights range from 1.01 to 100.0.
If training unstable, increase `propensity_clip_min` (e.g., 0.1 yields max weight 10.0).

## Curriculum Schedule (E12)
For curriculum training: `curriculum_phase1_end=20, curriculum_phase2_end=40, epochs=60`
