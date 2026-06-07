# U-CaGNN Configuration

Use this file for the config contract: presets, grouped knobs, and how runtime configs are assembled from defaults, recipes, profiles, and explicit overrides.

## Key files

- `.github/skills/ucagnn-implementation/ucagnn-config.md`
- `src/utils/config.py`
- `experiments/run_experiment.py`
- `experiments/recipes.py`
- `experiments/ablation_configs.py`

## Build order

```mermaid
flowchart LR
    A[Defaults] --> B[Preset]
    B --> C[Recipe overrides]
    C --> D[Explicit overrides]
    D --> E[validate]
```

The diagram shows the only supported precedence order. `build_config()` starts from `UCaGNNConfig()` defaults, applies the chosen preset, then recipe-owned overrides, then explicit overrides, and finally calls `validate()`.

For paper baselines, `build_config()` re-applies the paper-owned contract after recipe and explicit overrides. This prevents shared formal-profile knobs such as `dropout`, `num_neighbors`, or `graph_policy` from silently changing the LightGCN or DICE paper architecture.

## Preset contract

| Preset | Current behavior |
| --- | --- |
| `lightgcn` preset (`UCaGNNConfig.preset_lightgcn()`) | Scalable sampled-neighborhood LightGCN approximation: single branch with fixed interest-only mixing, no sign-aware weighting, no IPW, no popularity head, no side features, only `L_rec` active. |
| `lightgcn_paper` preset (`UCaGNNConfig.preset_lightgcn_paper()`) | Paper-faithful LightGCN adapter: `PaperLightGCN`, single branch, full-graph propagation during training, observed graph only, no dropout, Adam optimizer, explicit ego-embedding L2 with `weight_decay=1e-4`, `lr=0.001`, `batch_size=2048`, and no sampled-neighbor fan-out in the optimizer step. |
| `dice_like` preset (`UCaGNNConfig.preset_dice_like()`) | Legacy DICE-like ablation: dual branch with fixed interest+conformity mixing, no sign-aware weighting, no IPW, no popularity head, no side features, branch BPR plus independence active. It is not the official DICE paper baseline. |
| `dice_paper` preset (`UCaGNNConfig.preset_dice_paper()`) | Paper-faithful GCN-DICE adapter: `PaperGCNDICE`, dual self-looped LightGCN backbone branches, full-graph propagation during training, observed graph only, DICE popularity-conditioned negative sampling, DICE branch losses, Adam with DICE betas `(0.5, 0.99)` and AMSGrad, `n_negatives=4`, `dropout=0.2`, `batch_size=128`, and DICE adaptive decay enabled. `lgndice_paper` remains a compatibility alias. |
| `ucagnn` preset (`UCaGNNConfig.preset_full()`) | Dual branch with learned score mixing over interest, conformity, and the item-only context head, sign-aware propagation, calibrated-IPW support disabled by default, item features used when available, DICE-conditioned popularity negatives with one negative per positive, DICE-style causal branch supervision, a small learned-mix floor to prevent branch collapse, `linear_ramp` schedule, and contrastive/DirectAU auxiliaries implemented but off by default. |

## Build rules

1. Recipe-owned fields are strict: conflicting explicit overrides raise instead of being silently merged.
2. If `preprocessing_preset` is unset, `build_config()` fills it from `src/data/loaders/_registry.py`.
3. `num_neighbors` must match `max_gnn_layers`, which is `single_branch_gnn_layers` in single-branch mode and `max(interest_gnn_layers, conformity_gnn_layers)` in dual-branch mode.
4. `auxiliary_loss_schedule` is the live auxiliary-schedule switch. The separate legacy field `loss_schedule` remains checkpoint-compatible, but the only supported value is `baseline`.
5. `lightgcn_paper` and `dice_paper` lock architecture-, sampler-, optimizer-, and batch-size-owned fields after shared overrides. Use separate tuned baseline presets if you need a fair-tuned table with intentionally changed paper defaults.
6. `validate()` is the final authority for config shape and contract checks.

## Key config groups

| Group | Fields | Current meaning |
| --- | --- | --- |
| Graph build | `graph_policy`, `cagra_k`, `cagra_out_degree`, `cagra_initial_degree`, `cagra_team_size`, `cagra_metric`, `cagra_itopk_size` | Controls observed-vs-augmented training graph construction. |
| Eval prefilter | `cagra_candidate_k` | Optional evaluation-only ANN candidate filter; `0` means full-catalog scoring. |
| Model depth | `single_branch_gnn_layers`, `interest_gnn_layers`, `conformity_gnn_layers`, `num_neighbors` | Couples propagation depth to sampled fan-out. |
| Score fusion | `score_weight_interest`, `score_weight_conformity`, `score_weight_popularity`, `score_mix_min_weight`, `use_popularity_head` | Sets preset-owned default priors; baselines keep fixed mixing while `preset_full()` keeps learned `score_mix_weights`, and `score_mix_min_weight` applies only to learned components available by the model/data contract. |
| Loss and schedule | `loss_weight_*`, `branch_loss_mode`, `recommendation_loss_mode`, `auxiliary_loss_schedule`, `auxiliary_ramp_rate`, `independence_ramp_rate`, `distance_correlation_max_pairs`, `uniformity_max_pairs`, `loss_weight_propensity_calibration` | Enables auxiliaries, selects symmetric-vs-DICE branch supervision, caps quadratic auxiliary estimators, and controls how weights activate over time. |
| Training mode | `training_graph_mode`, `negative_sampling_strategy`, `n_negatives`, `dice_sampler_margin`, `dice_sampler_pool`, `dice_branch_margin`, `dice_loss_decay`, `dice_margin_decay`, `dice_adaptive_decay` | Selects sampled-subgraph vs full-graph training and standard vs DICE popularity-conditioned negative sampling. |
| Propensity | `use_ipw`, `propensity_hidden`, `propensity_clip_min`, `propensity_clip_max` | Controls the item-side propensity estimator; `use_ipw=True` requires positive `loss_weight_propensity_calibration`. |
| Runtime | `batch_size`, `auto_batch_size`, `batch_size_candidates`, `epochs`, `patience`, `use_early_stopping`, `use_amp`, `use_torch_compile`, `use_ema`, `lr_scheduler`, `eval_ks` | Controls optimization and execution behavior. CUDA runs default to `bfloat16` AMP; the experiment CLIs do not expose a separate public AMP mode. |
| Data | `dataset`, `preprocessing_preset`, `feature_policy`, `derived_split_mode`, `sample_interactions`, `loader_max_rows`, `seed` | Controls loader behavior, split derivation, and tiny-run caps. |

## Defaults worth remembering

- `graph_policy="observed"` is the default thesis path.
- `cagra_candidate_k=0` means evaluation scores the full catalog.
- The dataclass default schedule is `phased`, but `preset_full()` switches to `linear_ramp`.
- The dataclass default `propensity_clip_min` is `0.01`; `preset_full()` raises it to `0.1`.
- `use_ipw=False` is the dataclass and preset default. IPW must be explicitly enabled with `loss_weight_propensity_calibration > 0`.
- `use_features=True` is the dataclass default, but the non-causal presets disable side features.
- `score_mix_min_weight=0.0` is the dataclass default; `preset_full()` sets it to `0.05` so learned fusion cannot collapse to interest-only when conformity/context are available, even if a current batch gives one component zero-valued scores.
- `negative_sampling_strategy="standard"` is the dataclass default; `preset_full()` switches to DICE popularity-conditioned negatives with `n_negatives=1` and a stable `dice_branch_margin == dice_sampler_margin`.
- `use_amp=True` is the default runtime path, and `amp_dtype` is fixed to `bfloat16`.
- `loss_weight_propensity_calibration=0.0` is opt-in and stays inactive unless model outputs, dataset targets, and explicit IPW/calibration config exist.
- `distance_correlation_max_pairs=1024` and `uniformity_max_pairs=2048` cap quadratic auxiliary estimators while preserving deterministic hash-sampled coverage across epochs.

## Experiment-facing contract

- The formal experiment grid is **dataset x preset**.
- Support parameters such as `batch_size`, `num_neighbors`, `graph_policy`, and `lr_scheduler` are profile-owned runtime choices, not thesis axes.
- Formal profiles may sweep `num_neighbors`, `graph_policy`, or preprocessing presets as lists, but each resolved run still receives one concrete value in the final `UCaGNNConfig`.
- The default formal profiles target the practical core datasets: `amazonbook`, `movielens1m`, `kuairec_v2`, and `kuairand1k`. `taobao` and `movielens20m` remain explicit stress/optional runs instead of default catalog entries.
- Public ablation variants start from `preset_full()`: `mainline`, `no_popularity_head`, `no_independence`, and `no_features`.
