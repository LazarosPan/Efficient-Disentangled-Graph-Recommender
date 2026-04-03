# U-CaGNN Loss Functions Skill

Use this skill when working on loss functions, multi-task learning, curriculum scheduling, or IPW weighting.

## Key Files
- `docs/ucagnn_implementation/losses.md` - Loss components with paper cross-references
- `src/losses/loss_suite.py` - LossSuite orchestrator

## Ownership Notes
- `LossSuite` is the only public loss-layer surface.
- The individual loss implementations now live as private helpers inside `src/losses/loss_suite.py`; keep them there unless an external caller with a real separate lifecycle appears.
- The current wave-1 mainline uses a batch-safe within-branch contrastive auxiliary with `auxiliary_loss_schedule="linear_ramp"`; keep DirectAU alignment/uniformity available as optional ablations rather than the default path.
- Keep popularity supervision attached to the scorer-owned popularity head rather than reintroducing a separate predictor inside the loss suite.
- `LossSuite` receives scores from `UCaGNN.build_training_output()`, which now respects `config.train_scoring_mode`; keep the ranking loss semantics aligned with that model-owned scoring contract.

## Loss Formula
```
L_total = lambda_rec * L_rec + lambda_interest_bpr * L_int + lambda_conformity_bpr * L_conf
        + lambda_independence * L_indep + lambda_contrastive * L_contrastive
        + lambda_align * L_align
        + lambda_uniform * L_uniform + lambda_pop * L_pop
```

## Paper Sources
| Loss | Lambda | Source |
|------|--------|--------|
| L_rec (BPR) | 1.0 | Rendle 2009 |
| L_int / L_conf | 0.02 | Branch-local BPR auxiliaries keep each branch predictive |
| L_indep | 0.005 | Cosine-squared branch separation |
| L_contrastive | 0.02 | Within-branch InfoNCE on aligned positive user-item pairs |
| L_align / L_uniform | 0.02 | DirectAU-style branch-local geometry regularization (optional) |
| L_pop | 0.02 | DDCE-style popularity supervision routed through the scorer |

## Curriculum Scheduling
Config field `auxiliary_loss_schedule` controls how auxiliary weights activate:
- `"linear_ramp"`: wave-1 mainline path; auxiliaries ramp from 0 using `auxiliary_ramp_rate` and `independence_ramp_rate` while fused BPR stays on from epoch 0.
- `"phased"`: stage auxiliaries with `curriculum_phase1_end` and `curriculum_phase2_end` while fused BPR still stays on from epoch 0.

When the phased schedule is active, `curriculum_phase1_end` and `curriculum_phase2_end` control when auxiliary losses activate:
- Default: `curriculum_phase1_end=15, curriculum_phase2_end=30` (CaDCR-inspired staged curriculum)
- Disable curriculum: set both to 0 (joint training from epoch 0)

`loss_schedule` is fixed to `"baseline"` for supported runs. Do not reintroduce delayed-BPR schedules; the mainline contract is fused BPR from epoch 0 with only the auxiliary terms phased or ramped.

## Quick Reference
```python
from src.losses.loss_suite import LossSuite

loss_suite = LossSuite(config)
losses = loss_suite(model_output, pos_items, neg_items, item_pop, epoch=current_epoch)
# Returns dict: {"total", "rec", "interest_bpr", "conformity_bpr", "independence", "contrastive", "align", "uniform", "pop"}
```
