# U-CaGNN Loss Functions Skill

Use this skill when working on loss functions, multi-task learning, curriculum scheduling, or IPW weighting.

## Key Files
- `.github/skills/ucagnn-implementation/ucagnn-losses.md` - Routed loss summary for the current implementation
- `src/losses/loss_suite.py` - LossSuite orchestrator
- `experiments/ablation_configs.py` - Thesis-facing ablation variants built from `preset_full()`

## Ownership Notes
- `LossSuite` is the only public loss-layer surface.
- The individual loss implementations now live as private helpers inside `src/losses/loss_suite.py`; keep them there unless an external caller with a real separate lifecycle appears.
- The current mainline uses a batch-safe DCCL-style contrastive auxiliary with `auxiliary_loss_schedule="linear_ramp"`; keep DirectAU alignment/uniformity available as optional ablations rather than the default path.
- The base dataclass keeps `lambda_align=lambda_uniform=0.02`, but `preset_full()` overrides both to `0.0`; DirectAU is currently present as an optional diagnostic path, not part of the mainline preset.
- Keep popularity supervision attached to the scorer-owned popularity head rather than reintroducing a separate predictor inside the loss suite.
- `LossSuite` receives scores from `UCaGNN.build_training_output()`, which now respects `config.train_scoring_mode`; keep the ranking loss semantics aligned with that model-owned scoring contract.
- Keep the public ablation matrix focused on headline U-CaGNN components: `mainline`, `fixed_score_mix`, `no_popularity_head`, `no_ipw`, `no_contrastive`, `no_independence`, and `no_features`. Treat curriculum or DirectAU geometry toggles as local diagnostics, not thesis headline ablations.

## Loss Formula
```
L_total = lambda_rec * L_rec
        + lambda_interest_bpr * L_interest_bpr
        + lambda_conformity_bpr * L_conformity_bpr
        + lambda_independence * L_independence
        + lambda_contrastive * L_contrastive
        + lambda_align * L_align
        + lambda_uniform * L_uniform + lambda_pop * L_pop
```

## Paper Sources
| Loss | Lambda | Source |
|------|--------|--------|
| L_rec (BPR) | 1.0 | Rendle 2009 |
| L_interest_bpr / L_conformity_bpr | 0.02 | Branch-local BPR auxiliaries keep each branch predictive |
| L_independence | 0.005 | Cosine-squared branch separation |
| L_contrastive | 0.02 | Sum of popularity-aware `L_interest_contrastive` and `L_conformity_contrastive` over batch-local other-user negatives |
| L_align / L_uniform | 0.02 | DirectAU-style branch-local geometry regularization (optional) |
| L_pop | 0.02 | DDCE-style popularity supervision routed through the scorer |

Current preset deltas:
- `preset_lightgcn()`: only `L_rec` remains active.
- `preset_dice_like()`: `L_rec + 0.1 * L_interest_bpr + 0.1 * L_conformity_bpr + 0.01 * L_independence`.
- `preset_full()`: `L_rec + 0.02 * L_interest_bpr + 0.02 * L_conformity_bpr + 0.005 * L_independence + 0.02 * L_contrastive + 0.02 * L_pop`, with `L_align` and `L_uniform` disabled unless you opt in manually.

## Curriculum Scheduling
Config field `auxiliary_loss_schedule` controls how auxiliary weights activate:
- `"linear_ramp"`: mainline path; fused BPR and branch BPR are active from epoch 0, while the other auxiliaries ramp from 0 using `auxiliary_ramp_rate` (`L_interest_bpr`, `L_conformity_bpr`, `L_contrastive`, `L_align`, `L_uniform`, `L_pop`) and `independence_ramp_rate` (`L_independence`).
- `"phased"`: stage auxiliaries with `auxiliary_losses_start_epoch` and `popularity_supervision_start_epoch` while fused BPR still stays on from epoch 0.

When the phased schedule is active, `auxiliary_losses_start_epoch` and `popularity_supervision_start_epoch` control when auxiliary losses activate:
- Base-config default: `auxiliary_losses_start_epoch=15, popularity_supervision_start_epoch=30` (CaDCR-inspired staged curriculum)
- Disable curriculum: set both to 0 (joint training from epoch 0)

`loss_schedule` is fixed to `"baseline"` for supported runs. Do not reintroduce delayed-BPR schedules; the mainline contract is fused BPR from epoch 0 with only the auxiliary terms phased or ramped.

## Quick Reference
```python
from src.losses.loss_suite import LossSuite

loss_suite = LossSuite(config)
losses = loss_suite(model_output, item_popularity, pos_item_ids, epoch=current_epoch)
# Returns dict: {"total", "rec", "interest_bpr", "conformity_bpr", "independence", "interest_contrastive", "conformity_contrastive", "contrastive", "align", "uniform", "pop"}
```
