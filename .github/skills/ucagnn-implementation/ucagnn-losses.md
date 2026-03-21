# U-CaGNN Loss Functions Skill

Use this skill when working on loss functions, multi-task learning, curriculum scheduling, or IPW weighting.

## Key Files
- `docs/ucagnn_implementation/losses.md` - Loss components with paper cross-references
- `src/losses/loss_suite.py` - LossSuite orchestrator
- `src/losses/bpr.py` - L_rec (BPR with optional IPW)
- `src/losses/orthogonality.py` - L_ortho (cosine similarity squared)
- `src/losses/contrastive.py` - L_contr (NT-Xent)
- `src/losses/counterfactual.py` - L_cf (KL divergence)
- `src/losses/popularity.py` - L_pop (learned MLP predictor)

## Loss Formula
```
L_total = lambda_rec * L_rec + lambda_ortho * L_ortho + lambda_contr * L_contr
        + lambda_cf * L_cf + lambda_pop * L_pop
```

## Paper Sources
| Loss | Lambda | Source |
|------|--------|--------|
| L_rec (BPR) | 1.0 | Rendle 2009 |
| L_ortho | 0.02 | FMMRec 2023 |
| L_contr | 0.1 | DCCL 2023 |
| L_cf | 0.08 | MCLN 2023 |
| L_pop | 0.15 | DDCE 2023 |

## Curriculum Scheduling
Config fields `curriculum_phase1_end` and `curriculum_phase2_end` control when losses activate:
- Default (both=0): All losses from epoch 0 (joint training)
- E12 recommended: `curriculum_phase1_end=20, curriculum_phase2_end=40` with `epochs=60`

## Quick Reference
```python
from src.losses.loss_suite import LossSuite

loss_suite = LossSuite(config)
losses = loss_suite(model_output, pos_items, neg_items, item_pop, epoch=current_epoch)
# Returns dict: {"total", "bpr", "ortho", "contr", "cf", "pop"}
```
