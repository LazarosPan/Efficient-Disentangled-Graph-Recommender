# EDGRec Result Analysis

Use this file for current thesis result interpretation. Truth source: `results/thesis_experiments.db`; required readable surfaces: `results/query_results.md`, `results/optuna_optimization.md`, and `results/feature_analysis/`.

## Update Contract

| Requirement | Contract |
| --- | --- |
| Refresh trigger | Whenever `results/query_results.md`, `results/optuna_optimization.md`, or `results/feature_analysis/` is regenerated, re-check this file before using it for thesis writing. |
| Report layout | `results/query_results.md` now uses one unified test-set leaderboard for completed full-data rows. `DatasetRank` is dataset-local display order, `ExpID` is the SQLite experiment id, and `Evidence` distinguishes thesis profile, supporting, and ablation rows. Do not call `DatasetRank` a run id. |
| Test-result source | Use `results/query_results.md` for test-set rows, paper baseline status, runtime probes, CRRU definitions, speed/VRAM values, popularity-diversity diagnostics, and row-level narratives. |
| Search-result source | Use `results/optuna_optimization.md` for validation objective semantics, trial accounting, search-space revision status, importances, promotion candidates, and hyperparameter-response explanations. |
| Feature-result source | Use `results/feature_analysis/feature_subset_best_by_dataset.md` for feature-subset interpretation. These are validation/search results unless a matching full-data test row exists. |
| Conflict rule | If this file disagrees with a generated report, the generated report wins; update this file rather than carrying stale interpretation forward. |
| Thesis wording | Every result explanation must tie a metric pattern to one generated report, then state the evidence role: full-data test row, runtime probe, validation search candidate, imported trial, or diagnostic-only evidence. |
| Optuna caution | Do not use mixed, imported, or unrevisioned Optuna rows as strong thesis evidence; use fresh same-revision importances for strong search claims and full-data reruns for final test claims. |
| Optuna figures | Default Optuna figures aggregate loaded source studies by dataset using runtime-aware `ValidationOnlineCRRU@20_40`; thesis-facing plots call it the validation CRRU selection score. Gray importance cells mean no detected association, and branch-depth cells marked `n*` have fewer than 10 completed trials. |

Evidence roles:

| Role | Use |
| --- | --- |
| validation search candidate | Optuna validation evidence only; promote into a named test profile before making test claims. |
| full-data test row | Completed full-data row with test metrics in `results/thesis_experiments.db`. |
| runtime probe | Resource/feasibility evidence; accuracy is diagnostic only. |
| diagnostic-only evidence | Score-mix, branch-rank, contribution, and popularity diagnostics; not causal proof. |

## Evidence Status

| Evidence family | Current status | Thesis use |
| --- | --- | --- |
| `edgrec-global-top-*` | Full-data test rows now exist for `amazonbook`, `kuairec_v2`, `kuairand1k`, and `movielens1m`. | Main current promoted EDGRec test evidence. These profiles came from dataset-local validation CRRU candidates, but the unified test leaderboard may rank other completed rows above them. |
| `lightgcn_paper` | Full-data test rows: `amazonbook`, `kuairec_v2`, `movielens1m`; runtime probes: `kuairand1k`, `kuairec_v2`. | Paper-faithful accuracy/resource comparison where full rows exist. Runtime-only rows are feasibility evidence. |
| `dice_paper` | Runtime probes only: `amazonbook`, `movielens1m`. | Resource feasibility only; do not claim final accuracy against DICE yet. |
| feature-subset EDGRec search | Completed validation/search coverage for all current feature-subset datasets. | Feature evidence only unless a named full-data test rerun matches the feature profile. |
| sampled `lightgcn` / `dice_like` | Supporting fast ablation and legacy mechanism rows. | Useful for engineering comparison, not paper-faithful baseline claims. |

## Feature-Subset Search Boundary

Feature-subset conclusions use completed, non-probe rows from `edgrec-feature-subset-search` only. They do not override full-data test rankings by themselves.

| Dataset | Current feature-subset evidence | Thesis boundary |
| --- | --- | --- |
| `amazonbook` | Best completed profile is `graph_only` with validation OnlineCRRU@20_40 0.142527. | No side-feature claim; AmazonBook remains graph-only. |
| `kuairand1k` | Best completed profile is `triple_item_author_music__item_upload_time__item_category` with OnlineCRRU@20_40 0.061002; side-feature gain is +0.019579 and `item_category` is the strongest single group (+0.017831). | Side features help validation CRRU in the subset search, but current promoted test rows still have weak ranking metrics. |
| `kuairec_v2` | Best completed profile is `single_item_resolution` with OnlineCRRU@20_40 0.204999; side-feature gain is +0.003935, with strong single-group gains for `item_resolution` and `item_video_metadata`. | Side features are promising validation evidence; the strongest current full-data test rows are ablation/supporting rows, not the promoted global-top EDGRec profiles. |
| `movielens1m` | Best completed profile is `none` with OnlineCRRU@20_40 0.256069; genre side features hurt (`side_feature_gain` -0.012320). | Do not frame MovieLens genre features as useful under the current EDGRec search basin. |

Global profile promotion is separate from feature evidence: `edgrec-global-top-<dataset>-r<rank>` profiles are selected across loaded studies by dataset-local validation `ValidationOnlineCRRU@20_40`, then tested through full-data rows.

## Interpretation Rules

| Rule | Reason |
| --- | --- |
| Compare accuracy only on same dataset, split, and full-data status. | Runtime probes and full-data test rows have different evidence roles. |
| Treat CRRU as parameterized utility, not causal effect. | CRRU combines accuracy, popularity-diversity, time, and VRAM under task-specific weights. |
| Treat inverse AvgPop carefully. | Lower average popularity means lower popularity concentration, not guaranteed fairness or causal debiasing. |
| Keep KuaiRec `kuairec_small_matrix_full_observation` separate from `kuairec_big_matrix_watch_ratio_threshold_0_5`. | `small_matrix` is near-oracle dense sensitivity; default sparse story is `big_matrix` with `watch_ratio >= 0.5`. |
| Report DICE paper speed as probe evidence. | No full-data DICE paper accuracy rows yet. |
| Distinguish `lightgcn_paper` from sampled `lightgcn`. | Paper fidelity vs scalable approximation. |
| Explain speed using code paths, not model simplicity. | EDGRec has more components but often uses a cheaper sampled-training path. |

## Current Headline Comparisons

Rows below use the current unified full-data test leaderboard unless marked as a probe. CRRU values are current leaderboard-scope values and will change if the comparison set changes.

| Dataset | EDGRec row | Comparator | EDGRec improves | EDGRec does not improve | Thesis reading |
| --- | --- | --- | --- | --- | --- |
| `amazonbook` | ExpID 17200, `edgrec-global-top-amazonbook-r2` | Best LightGCN paper row ExpID 8712 | Time/epoch 3.3s vs 36.2s; AvgPop@20 0.0995 vs 0.1387. | CRRU@20 0.6120 vs 0.8623; NDCG@20 0.0189 vs 0.0246; Recall@20 0.0234 vs 0.0315; Hit@20 0.1546 vs 0.1958; peak VRAM 9380MB vs 931MB. | EDGRec is a speed and lower-popularity-concentration trade-off, not the AmazonBook leaderboard winner. |
| `kuairec_v2` | ExpID 17213, `edgrec-global-top-kuairec-v2-r3` | LightGCN paper ExpID 8701; leaderboard winner is ablation ExpID 4047 | Against LightGCN paper: CRRU@20 0.3747 vs 0.0443; NDCG@20 0.1181 vs 0.0484; Recall@20 0.0268 vs 0.0108; Hit@20 0.7299 vs 0.4164; AvgPop@20 0.3068 vs 0.5754; time/epoch 4.1s vs 226.3s. | Peak VRAM is higher than LightGCN paper: 3265MB vs 1288MB. The top test row is the public ablation `mainline` with CRRU@20 0.7155 and NDCG@20 0.9195, so the promoted global-top row is not the dataset winner. | KuaiRec remains the strongest positive EDGRec dataset, but the best current full-data row is an ablation/supporting-family result that should be treated as a promotion candidate before thesis headline wording. |
| `movielens1m` | ExpID 17228, `edgrec-global-top-movielens1m-r1` | Best LightGCN paper row ExpID 8711 | Time/epoch 2.0s vs 3.2s; AvgPop@20 0.2972 vs 0.3351; Personalization@20 0.9447 vs 0.9329. | CRRU@20 0.7343 vs 0.8781; NDCG@20 0.0913 vs 0.0997; Recall@20 0.1282 vs 0.1398; Hit@20 0.4762 vs 0.4977; peak VRAM 2105MB vs 551MB. | EDGRec is a popularity/speed alternative. LightGCN remains the current CRRU and raw-accuracy winner. |
| `kuairand1k` | ExpID 17226, `edgrec-global-top-kuairand1k-r5` | Best full-data supporting row ExpID 1094; LightGCN paper is probe-only P2 | Recall@20 0.0056 vs 0.0007; AvgPop@20 0.2380 vs 0.3642; time/epoch 0.074s vs 25.3s; peak VRAM 2045MB vs 12008MB. | CRRU@20 0.3173 vs 0.4574; NDCG@20 0.0026 vs 0.0202; Hit@20 0.0110 vs 0.3000. | KuaiRand remains stress-test evidence. The promoted EDGRec rows are extremely cheap under a compact randomized-exposure training contract, but ranking utility is not competitive. |

## KuaiRand Timing Interpretation

The `Time/Ep` value in `results/query_results.md` is the CRRU runtime source: it prefers logged `avg_epoch_time_s`, then runtime-probe seconds/epoch, then `training_time_s / completed_train_epochs`. For the promoted KuaiRand global-top rows, the source is logged average epoch time, not checkpoint load or evaluation-only timing.

| Evidence | Interpretation |
| --- | --- |
| ExpID 17226 logs 50 epochs with `avg_epoch_time_s=0.0737s`, batch size 1,048,576, `train_edge_count=7,088`, `item_universe_policy=random_exposure_items_only`, and `train_edge_keep_prob=0.6`. | The sub-second value is plausible for a tiny sampled train graph and one very large batch. |
| ExpIDs 17221/17223/17225 use batch size 2,097,152 with the same 7,088 train edges and 0.073-0.074s/epoch. ExpID 17219 uses 4,790 train edges with `train_edge_keep_prob=0.4`. | The compact graph and huge batch are the direct reason these rows are fast. |
| Older/non-compact KuaiRand rows report much larger epoch times: ExpID 1094 at 25.3s, ExpID 8698 at 46.1s, and mainline EDGRec ExpIDs 11006/11007 at 77.0-98.2s. | Do not compare the global-top KuaiRand timing as if it were full standard-view KuaiRand throughput. Report it as randomized-exposure compact-regime efficiency. |

## Dataset-Conditioned Profile Policy

| Dataset | Policy |
| --- | --- |
| `kuairec_v2` | Use EDGRec as the current positive dataset, but distinguish the promoted global-top rows from the stronger public-ablation/supporting rows now visible in the unified leaderboard. |
| `movielens1m` | Use EDGRec as a resource/popularity-aware alternative. Keep LightGCN's higher CRRU, raw accuracy, and lower VRAM visible. |
| `amazonbook` | Use EDGRec as an efficiency/popularity-concentration trade-off. Keep LightGCN paper as the accuracy and CRRU baseline; do not call EDGRec the ranking winner. |
| `kuairand1k` | Keep as diagnostic and stress-test evidence. Current promoted rows are compact randomized-exposure runs; ranking utility remains weak and timing is not full standard-view throughput. |

## DICE Paper Runtime Evidence

| Dataset | EDGRec reference | DICE probe | Speed evidence | Resource note | Accuracy status |
| --- | --- | --- | --- | --- | --- |
| `amazonbook` | ExpID 17200 | Probe P1 | 3.3s/epoch vs 3426.2s/epoch: about 1038x faster. | EDGRec peak VRAM is higher: 9380MB vs 5197MB. | DICE NDCG@20 is one-epoch diagnostic only. |
| `movielens1m` | ExpID 17228 | Probe P4 | 2.0s/epoch vs 578.8s/epoch: about 289x faster. | EDGRec peak VRAM is lower: 2105MB vs 2899MB. | DICE NDCG@20 is one-epoch diagnostic only. |

Thesis-safe DICE statement: "Paper-faithful DICE is orders of magnitude slower per epoch under current profiles; final DICE ranking comparison remains open until full rows exist."

## Why EDGRec Can Be Faster

| Mechanism | EDGRec path | Paper baseline path | Effect |
| --- | --- | --- | --- |
| Training graph | sampled subgraph around batch seeds | full-graph propagation per optimizer batch for paper adapters | EDGRec cost scales with sampled neighborhood, not full edge set each step. |
| Batch size | auto-batch often resolves to large batches | LightGCN paper locks 2048; DICE paper locks 128 | EDGRec amortizes optimizer overhead and uses fewer batches. |
| Negative sampling | vectorized DICE-style routing; `n_negatives=1` default in many profiles | DICE paper locks `n_negatives=4` and exact per-user pool correction | EDGRec keeps bias-aware signal but avoids the expensive paper sampler path. |
| Quadratic auxiliaries | hash-sampled caps for dCor/uniformity; contrastive is explicit | DICE discrepancy and branch training are paper-faithful | EDGRec bounds expensive terms and keeps optional terms configurable. |
| CUDA staging | sampled graph/negative sampling can run on device; CPU fallback exists | full-graph tensors cached for baseline propagation | EDGRec better uses GPU on datasets where full graph is too expensive. |
| Paper fidelity | EDGRec optimizes a practical training contract | paper baselines preserve original contracts | Speedup is partly a systems contribution, not a like-for-like algorithmic simplification. |

## Dataset-Level Interpretation

| Dataset | Current pattern | Likely explanation | Thesis wording |
| --- | --- | --- | --- |
| `amazonbook` | EDGRec global-top rows lower AvgPop and train about 10x faster than LightGCN paper, but LightGCN has higher CRRU, NDCG, Recall, Hit, and much lower VRAM. Several EDGRec top rows also have high branch-cosine warnings. | Sparse graph-only data gives limited side/context signal; the branch split can reduce popularity concentration without enough relevance gain. | "On AmazonBook, EDGRec is an efficiency and popularity-concentration trade-off, not a ranking-accuracy win in current full-data rows." |
| `kuairec_v2` | Promoted EDGRec beats LightGCN paper on accuracy, AvgPop, CRRU, and speed, while the best current test row is the public ablation `mainline` rather than a promoted global-top profile. | Watch-ratio video data has strong exposure/popularity structure; EDGRec's sampled training and branch/context terms find useful signal while full-graph LightGCN is costly. The ablation result may reflect a stronger profile than the currently promoted candidates. | "KuaiRec is current evidence for EDGRec-family benefit, but headline claims should identify whether they use the promoted profile or the stronger ablation row." |
| `movielens1m` | EDGRec global-top rows improve AvgPop and speed but remain below LightGCN paper on CRRU and raw accuracy and above it on VRAM. | Dense explicit ratings make LightGCN strong; EDGRec's fixed/balanced score mix trades some accuracy for lower popularity concentration and speed. | "MovieLens shows a resource/popularity-aware alternative, with the accuracy and VRAM cost reported explicitly." |
| `kuairand1k` | EDGRec global-top rows are extremely cheap because they use compact randomized-exposure item-universe training with huge batches and 4,790-7,088 train edges, but ranking metrics remain low. Paper LightGCN evidence is still probe-only. | Randomized exposure and sparse positives make target relevance hard; CRRU can be dominated by efficiency and popularity terms when accuracy is weak. | "KuaiRand remains unresolved; use it as compact-regime stress-test evidence, not a positive accuracy headline." |

## Why Accuracy Can Improve or Degrade

| Driver | Improves when... | Degrades when... | Diagnostic |
| --- | --- | --- | --- |
| Interest/conformity split | popularity-biased interactions hide real preference and branch losses separate useful signals | branches collapse or conformity dominates relevance | `test_interest_conformity_cosine_*`, branch rank metrics |
| Context head | train-only popularity/recency/feature context matches real exposure effects | context encodes popularity without enough relevance correction | context contribution and final popularity Spearman |
| Score mix | learned/fixed mix keeps interest primary while preserving useful bias controls | mix shifts too heavily to conformity/context or floor keeps weak branches active | `score_mix_*_mean/std`, contribution ratios, branch-collapse warnings |
| Side features | features are pre-treatment and predictive | features are weak, noisy, or dataset has graph-only semantics | `with_features` and feature-subset validation reports against graph-only default |
| Fan-out/depth | sampled neighborhood captures enough signal cheaply | fan-out too small loses structure; too deep oversmooths/costs more | neighbor profile rows, branch cosine, time/epoch |
| DICE losses | popularity-conditioned negatives identify conformity pressure | active masks are sparse or branch loss scale overwhelms recommendation loss | DICE mask rates, weighted losses |

## Contribution Decision Table

| Result case | Contribution framing |
| --- | --- |
| Accuracy >= LightGCN paper and time/epoch much lower | Main contribution: practical bias-aware/disentangled recommender with better efficiency and no accuracy loss. |
| Accuracy slightly lower but time/epoch much lower | Secondary contribution: resource-aware alternative; report accuracy and VRAM cost explicitly. |
| Accuracy lower and popularity concentration lower | Bias-control trade-off; not a recommendation-accuracy win. |
| DICE full run infeasible | Systems feasibility finding; paper-faithful causal baselines may be impractical on current large/profiled datasets. |
| DICE probe faster after future optimization | Revisit contribution; current speed claim depends on present paper-faithful adapter/runtime profile. |

## Evidence Still Needed

| Need | Reason |
| --- | --- |
| Full or bounded `dice_paper` accuracy rows | Required before ranking-accuracy claims against DICE. |
| Multi-seed confirmation for best rows | Current global-top rows are still mostly one-seed evidence. |
| Full-data reruns for promising feature-subset profiles | Needed before KuaiRec/KuaiRand side-feature validation gains become test-set claims. |
| Per-dataset branch diagnostic writeup | Explains why score mix, branch rank, and branch cosine differ by dataset. |
| Pareto/frontier view for accuracy, AvgPop, time, and VRAM | Avoids relying on one scalar CRRU when the thesis claim is trade-off based. |
