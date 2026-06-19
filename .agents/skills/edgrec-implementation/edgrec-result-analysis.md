# EDGRec Result Analysis

Use this file for current thesis result interpretation. Truth source: `results/thesis_experiments.db`; required readable surfaces: `results/query_results.md` and `results/optuna_optimization.md`.

## Update Contract

| Requirement | Contract |
| --- | --- |
| Refresh trigger | Whenever `results/query_results.md` or `results/optuna_optimization.md` is regenerated, re-check this file before using it for thesis writing. |
| Formal-result source | Use `results/query_results.md` for final test rows, paper baseline status, runtime probes, CRRU definitions, speed/VRAM values, popularity-diversity diagnostics, and row-level narratives. |
| Search-result source | Use `results/optuna_optimization.md` for validation objective semantics, trial accounting, search-space revision status, importances, formal-promotion candidates, and hyperparameter-response explanations. |
| Conflict rule | If this file disagrees with either generated report, the generated report wins; update this file rather than carrying stale interpretation forward. |
| Thesis wording | Every result explanation must tie a metric pattern to one of the two reports, then state the boundary: formal row, runtime probe, search candidate, imported trial, or diagnostic-only evidence. |
| Optuna caution | Do not use mixed, imported, or unrevisioned Optuna rows as strong thesis evidence; use fresh same-revision importances for strong search claims and formal reruns for final test claims. |
| Optuna figures | Default Optuna figures aggregate all loaded source studies by dataset using runtime-aware `ValidationOnlineCRRU@20_40`; thesis-facing plots call it the validation CRRU selection score, gold stars mark selected trials, black diamonds mark fan-out medians, gray importance cells mean no detected association, and branch-depth cells marked `n*` have fewer than 10 completed trials. The exporter writes PNG figures only and removes stale generated PNG/HTML artifacts before each run. |

## Evidence Status

| Baseline | Current status | Thesis use |
| --- | --- | --- |
| `lightgcn_paper` | Full formal rows: `amazonbook`, `kuairec_v2`, `movielens1m`; runtime probes: `kuairand1k`, `kuairec_v2`. | Accuracy/resource comparison where full rows exist. Runtime-only rows are feasibility evidence. |
| `dice_paper` | Runtime probes only: `amazonbook`, `movielens1m`. | Resource feasibility only; do not claim final accuracy against DICE yet. |
| sampled `lightgcn` | Supporting fast ablation rows. | Useful for engineering comparison, not paper-faithful baseline. |
| `dice_like` | Legacy sampled DICE-like ablation rows. | Mechanism/fallback comparison, not paper DICE. |
| `edgrec` | Full rows across core datasets and ablations. | Main thesis model; compare by dataset/profile. |

## Interpretation Rules

| Rule | Reason |
| --- | --- |
| Compare accuracy only on same dataset, split, and full-data status. | Runtime probes and full formal rows have different evidence roles. |
| Treat CRRU as parameterized utility, not causal effect. | CRRU combines accuracy, popularity-diversity, time, and VRAM under task-specific weights. |
| Treat inverse AvgPop carefully. | Lower average popularity means lower popularity concentration, not guaranteed fairness or causal debiasing. |
| Keep KuaiRec `fullobs` separate from `watchratio`. | `fullobs` is near-oracle dense sensitivity; default sparse story is `watchratio`. |
| Report DICE paper speed as probe evidence. | No full DICE formal accuracy rows yet. |
| Distinguish `lightgcn_paper` from sampled `lightgcn`. | Paper fidelity vs scalable approximation. |
| Explain speed using code paths, not model simplicity. | EDGRec has more components but cheaper training path. |

## Current Headline Comparisons

Pairs use current SQLite rows from `core-paper-architecture-comparison` when available.

| Dataset | Comparison | Speed evidence | Accuracy evidence | Popularity-diversity note | Status |
| --- | --- | --- | --- | --- | --- |
| `amazonbook` | EDGRec row 8695 vs LightGCN paper row 8699 | 3.8s/epoch vs 35.8s/epoch: 9.4x faster | NDCG@20 0.0185 vs 0.0241: EDGRec 0.77x LightGCN | AvgPop@20 0.1581 vs 0.1493: not less popular in this pair | LightGCN wins accuracy; EDGRec wins speed. |
| `kuairec_v2` | EDGRec row 8697 vs LightGCN paper row 8701 | 6.0s/epoch vs 226.3s/epoch: 37.9x faster | NDCG@20 0.0868 vs 0.0484: EDGRec 1.79x LightGCN | AvgPop@20 0.3599 vs 0.5754: EDGRec less popularity-heavy | Strongest current EDGRec result. |
| `movielens1m` | EDGRec row 8696 vs LightGCN paper row 8700 | 2.0s/epoch vs 3.1s/epoch: 1.5x faster | NDCG@20 0.0990 vs 0.0983: near parity/slightly higher | AvgPop@20 0.4235 vs 0.3643: more popularity-heavy in this pair | Accuracy parity, modest speed win, weaker popularity profile. |
| `kuairand1k` | EDGRec row 8698 vs LightGCN paper probe row 11250 | 46.1s/epoch vs 633.5s/epoch: 13.7x faster | NDCG@20 0.0055 vs 0.0081, but LightGCN row is runtime-probe accuracy | AvgPop@20 0.7214 vs 0.3824: EDGRec architecture row is popularity-heavy | Resource comparison only; final accuracy unresolved. |

## DICE Paper Runtime Evidence

| Dataset | EDGRec row | DICE probe row | Speed evidence | Resource note | Accuracy status |
| --- | ---: | ---: | --- | --- | --- |
| `amazonbook` | 8695 | 11251 | 3.8s/epoch vs 3426.6s/epoch: 895.2x faster | 1322MB vs 5197MB peak VRAM | DICE NDCG@20 is one-epoch diagnostic only. |
| `movielens1m` | 8696 | 10999 | 2.0s/epoch vs 578.9s/epoch: 282.5x faster | 594MB vs 2899MB peak VRAM | DICE NDCG@20 is one-epoch diagnostic only. |

Thesis-safe DICE statement: "Paper-faithful DICE is orders of magnitude slower per epoch under current profiles; final DICE ranking comparison remains open until full rows exist."

## Why EDGRec Can Be Faster

| Mechanism | EDGRec path | Paper baseline path | Effect |
| --- | --- | --- | --- |
| Training graph | sampled subgraph around batch seeds | full-graph propagation per optimizer batch for paper adapters | EDGRec cost scales with sampled neighborhood, not full edge set each step. |
| Batch size | auto-batch often resolves to large batches such as 32768 | LightGCN paper locks 2048; DICE paper locks 128 | EDGRec amortizes optimizer overhead and uses fewer batches. |
| Negative sampling | vectorized DICE high/low routing; `n_negatives=1` default | DICE paper locks `n_negatives=4` and exact per-user pool correction | EDGRec keeps DICE signal but avoids expensive paper sampler path. |
| Quadratic auxiliaries | hash-sampled caps for dCor/uniformity; contrastive off by default | DICE discrepancy and DICE branch training are paper-faithful | EDGRec bounds expensive terms and keeps optional terms explicit. |
| CUDA staging | sampled graph/negative sampling can run on device; CPU fallback exists | full-graph tensors cached for baseline propagation | EDGRec better uses GPU on datasets where full graph is too expensive. |
| Paper fidelity | EDGRec optimizes practical training contract | paper baselines preserve original contracts | Speedup is partly a systems contribution, not a like-for-like algorithmic simplification. |

## Dataset-Level Interpretation

| Dataset | Current pattern | Likely explanation | Thesis wording |
| --- | --- | --- | --- |
| `amazonbook` | LightGCN paper has higher NDCG/Recall; EDGRec is much faster. Some mainline EDGRec rows reduce AvgPop strongly but accuracy drops. | Sparse graph-only data gives limited side/context signal; causal branch constraints can move away from popularity without enough relevance signal. | "On AmazonBook, EDGRec is an efficiency trade-off, not an accuracy win in current rows." |
| `kuairec_v2` | EDGRec beats LightGCN paper in NDCG/Recall/Hit and is far faster; AvgPop is lower. | Watch-ratio video data has stronger exposure/popularity structure; branch/context modeling has useful signal, while full-graph LightGCN is costly. | "KuaiRec is current evidence for combined accuracy, debiasing, and efficiency benefit." |
| `movielens1m` | EDGRec paper-comparison row matches LightGCN accuracy and is faster; mainline rows lower popularity but lose accuracy. | Dense explicit ratings make LightGCN already strong; EDGRec needs score-mix/profile choice to avoid over-regularizing or over-shifting toward bias controls. | "MovieLens shows near-parity accuracy with modest speedup, but mechanism settings govern the popularity trade-off." |
| `kuairand1k` | EDGRec full rows have weak Recall/NDCG; LightGCN paper evidence is runtime-probe only. Mainline rows lower AvgPop but accuracy remains low. | Randomized/standard mix and sparse positives make target signal hard; architecture row can become popularity-heavy, while mainline bias controls can suppress relevance. | "KuaiRand remains unresolved; use it as stress-test evidence, not a positive headline." |

## Why Accuracy Can Improve or Degrade

| Driver | Improves when... | Degrades when... | Diagnostic |
| --- | --- | --- | --- |
| Interest/conformity split | popularity-biased interactions hide real preference and branch losses separate useful signals | branches collapse or conformity dominates relevance | `test_interest_conformity_cosine_*`, branch rank metrics |
| Context head | train-only popularity/recency/feature context matches real exposure effects | context encodes popularity without enough relevance correction | context contribution and final popularity Spearman |
| Score mix | learned/fixed mix keeps interest primary while preserving useful bias controls | mix shifts too heavily to conformity/context or floor keeps weak branches active | `score_mix_*_mean/std`, contribution stats |
| Side features | features are pre-treatment and predictive | features are weak, noisy, or dataset has graph-only semantics | `no_features` ablations |
| Fan-out/depth | sampled neighborhood captures enough signal cheaply | fan-out too small loses structure; too deep oversmooths/costs more | neighbor profile rows, branch cosine, time/epoch |
| DICE losses | popularity-conditioned negatives identify conformity pressure | active masks are sparse or branch loss scale overwhelms recommendation loss | DICE mask rates, weighted losses |

## Contribution Decision Table

| Result case | Contribution framing |
| --- | --- |
| Accuracy >= LightGCN paper and time/epoch much lower | Main contribution: practical causal-branch recommender with better efficiency and no accuracy loss. |
| Accuracy slightly lower but time/epoch much lower | Secondary contribution: resource-aware alternative; report accuracy cost explicitly. |
| Accuracy lower and popularity lower | Bias-control trade-off; not a recommendation-accuracy win. |
| DICE full run infeasible | Systems feasibility finding; paper-faithful causal baselines may be impractical on current large/profiled datasets. |
| DICE probe faster after future optimization | Revisit contribution; current speed claim depends on present paper-faithful adapter/runtime profile. |

## Evidence Still Needed

| Need | Reason |
| --- | --- |
| Full or bounded `dice_paper` formal accuracy rows | Required before ranking-accuracy claims against DICE. |
| Multi-seed confirmation for best rows | Current headline rows are mostly seed 13. |
| Fresh revisioned Optuna summaries | Avoid using mixed search-space history as thesis evidence. |
| Per-dataset branch diagnostic writeup | Explain why score mix and branch rank differ by dataset. |
| Runtime-normalized comparison table | Separate per-epoch speed, total training time, VRAM, and accuracy. |
| Pareto/frontier view | Use the selection-frontier PNG and component-response PNG before relying on one scalar utility. |
