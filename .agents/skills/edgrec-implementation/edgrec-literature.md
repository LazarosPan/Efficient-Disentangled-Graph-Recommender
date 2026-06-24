# EDGRec Literature Evidence

Use this file for thesis-facing rationale: which literature-backed ideas justify the current EDGRec design. Keep implementation details in the owner docs. Internal code package remains `EDGRec`.

## Source Hierarchy

| Source | Role |
| --- | --- |
| `.agents/skills/existing-work/full_summary.md` | Theory and paper synthesis. Primary literature map. |
| `.agents/skills/existing-work/existing_technical_synthesis.md` | Code-first mechanics from audited prior implementations. |
| `docs/paper_summaries/summary_by_paper_10.md` | Paper-level details for DICE, CausE, DCCL, CaDCR, DDCE, etc. |
| `docs/paper_summaries/gcn_models.md` | LightGCN, LayerGCN, LightGCN++ mechanics. |
| `docs/paper_summaries/summary_performance_papers.md` | Full-graph vs mini-batch GNN training, CAGRA, PANORAMA. |
| `results/query_results.md` and `results/thesis_experiments.db` | Current empirical evidence. Not literature. |

## Paper Evidence Ledger

| Paper or family | Evidence used | EDGRec use | Do not overclaim |
| --- | --- | --- | --- |
| LightGCN, SIGIR 2020 | Sparse normalized propagation, no feature transform/nonlinearity/self-loop/dropout, dot-product ranking, BPR-style training. | Backbone family and paper-faithful `lightgcn_paper` baseline. | LightGCN is not causal by itself. |
| LightGCN++ / LayerGCN | Small kernel/readout changes affect depth, normalization, and over-smoothing behavior. | Shallow asymmetric branch depths and explicit deeper-profile diagnostics. | EDGRec is not currently a LightGCN++ implementation. |
| DICE, 2020 | Interest/conformity embeddings, additive click score, popularity-conditioned triplets, branch BPR, discrepancy loss, macro-cause limitation. | Dual branches, DICE sampler masks, branch losses, independence regularization, paper-DICE baseline. | Do not claim EDGRec reproduces DICE exactly; `dice_paper` owns that. |
| CausE, 2018 | Treatment/control representation learning needs biased plus randomized exposure evidence; randomized logging is costly/sparse. | Propensity/IPW is optional and gated; randomized KuaiRand evidence is treated carefully. | Observational ranking gains are not ITE estimates. |
| PropCare | Propensity/relevance split and capped propensity-style evaluation need explicit treatment/propensity/effect columns. | Propensity calibration target path exists where data supports it. | Default EDGRec is not a PropCare evaluator. |
| DDCE / MGCE / MCLN | Popularity, quality, conformity, multimodal, and counterfactual signals can help, but noise/depth/counterfactual overhead are risks. | Item-only context head, safe feature policy, feature gates initialized near zero. | Side features are not assumed universally causal. |
| DCCL / DirectAU | Contrastive and geometry objectives can help representation quality but add `O(B^2 d)` or pairwise-distance cost. | Contrastive/DirectAU terms are implemented, capped, and disabled by default. | They are not part of the default mainline contribution unless enabled. |
| Full-graph vs mini-batch GNN training | Full graph is a mini-batch limit; tuned batch/fan-out can improve throughput and trade off generalization. | Sampled subgraph EDGRec default; full-graph paper baselines remain fidelity references. | Faster training is partly a training-protocol contribution. |
| CAGRA / PANORAMA | ANN/retrieval systems can give large graph/search speedups but have memory/layout constraints. | Background only for current EDGRec training; CAGRA graph augmentation was removed. | ANN indexing does not reduce LightGCN message-passing cost when materialized as extra training edges. |

## Thesis-Safe Framing

| Claim type | Safe wording |
| --- | --- |
| Causal scope | EDGRec is a causal-recommendation-inspired debiasing architecture; current ranking metrics are not causal-effect estimates. |
| Core mechanism | EDGRec synthesizes LightGCN propagation, DICE-style interest/conformity separation, popularity-aware negative sampling, bounded independence losses, safe side features, and optional propensity calibration. |
| Efficiency contribution | EDGRec tests whether causal-branch recommendation can retain accuracy while replacing paper-baseline full-graph and expensive sampler/loss paths with sampled neighborhoods, vectorized sampling, and bounded auxiliaries. |
| Baseline comparison | `lightgcn_paper` and `dice_paper` are paper-faithful contracts; sampled `lightgcn` and `dice_like` are fast ablations, not paper baselines. |
| Avoid | "EDGRec proves causality", "CRRU is causal", "DICE is less accurate" before full DICE formal rows exist. |

## Evidence to Design Map

| Literature evidence | EDGRec design choice | Current owner |
| --- | --- | --- |
| Recommendation literature models exposure as treatment and interaction as outcome; PO/SCM, IPS, clipped IPS, SNIPS, and doubly robust estimators recur. | `use_ipw` and `L_prop_calib` exist, but are opt-in and gated by dataset targets; default ranking does not use uncalibrated IPW. | `edgrec-data-pipeline.md`, `edgrec-losses.md` |
| Surveys warn that exposure, selection, popularity, conformity, position, and feedback-loop bias can distort observed interactions. | Default loaders use train-only graph/popularity; feature policy excludes post-treatment aggregates from thesis-default features. | `edgrec-data-pipeline.md` |
| DICE defines an additive score `s_click = s_interest + s_conformity`, uses interest/conformity embeddings, popularity-conditioned triplets, BPR branch losses, and discrepancy regularization. | EDGRec has dual interest/conformity branches, DICE negative masks, branch-local BPR, and independence regularization. | `edgrec-architecture.md`, `edgrec-losses.md` |
| DICE reports dCor as stronger but expensive; audited summaries identify dCor as a scaling bottleneck. | EDGRec caps quadratic independence rows with deterministic hash sampling and keeps weights smaller than paper DICE. | `edgrec-losses.md` |
| DCCL-style contrastive causal learning has `O(B^2 d)` batch cost; DirectAU uniformity uses `torch.pdist`. | Contrastive, align, and uniform losses are implemented but off by default; pair/row counts are capped when enabled. | `edgrec-losses.md` |
| LightGCN removes feature transforms, nonlinearities, self-connections, and dropout, keeping normalized sparse propagation and dot-product scoring. | Paper LightGCN adapter preserves this contract; EDGRec keeps LightGCN-style propagation as backbone but adds branches and scorer. | `edgrec-architecture.md`, `edgrec-config.md` |
| LightGCN++ and LayerGCN show that simple LightGCN-family kernel changes can affect accuracy and depth sensitivity; over-smoothing remains a recurring depth limit. | EDGRec keeps shallow asymmetric branch depths by default: interest 1, conformity 2; deeper runs are explicit diagnostics. | `edgrec-config.md` |
| Full-graph training is a special mini-batch limit; literature reports that tuned mini-batch/fan-out can improve throughput and sometimes trade off accuracy/generalization better than full graph. | EDGRec uses sampled subgraph training by default; paper baselines lock full-graph training to preserve fidelity. | `edgrec-training.md` |
| For sparse graphs, the performance summary recommends fan-out under 15 as a practical threshold. | Default EDGRec `num_neighbors=[10,5]`; formal comparisons use small explicit fan-out sweeps. | `edgrec-config.md` |
| CAGRA reports high GPU ANN graph construction/query throughput, but is memory-bandwidth and device-memory constrained. | Not part of EDGRec training/search spaces; graph augmentation was removed after OOM evidence. | `edgrec-data-pipeline.md`, `edgrec-config.md` |
| Propensity methods such as CausE/PropCare require randomized or propensity/effect evidence; surveys warn propensity correctness is hard to validate. | KuaiRand `show_cnt` can calibrate propensity targets; default scorer zero-fills propensity context unless explicit calibrated IPW is active. | `edgrec-data-pipeline.md`, `edgrec-architecture.md` |
| DDCE/MGCE/MCLN/FMMRec split interest, conformity, popularity, quality, modality, or fairness signals; several papers report multimodal noise and depth sensitivity. | EDGRec includes safe item-feature gates initialized near zero; side features start as weak optional evidence, not dominant signal. | `edgrec-architecture.md`, `edgrec-config.md` |
| Causal-rec surveys use standard ranking metrics plus causal metrics; they warn ranking accuracy can improve while causal quality degrades. | Thesis reports PyG ranking metrics plus bias/resource diagnostics; CRRU is explicitly resource-aware utility, not causal effect. | `edgrec-training.md`, `edgrec-result-analysis.md` |

## What Is New in This Implementation

| Area | Literature anchor | EDGRec synthesis |
| --- | --- | --- |
| Architecture | LightGCN + DICE + DDCE/MGCE-style explicit popularity path | Dual LightGCN-style branches plus item-only context head and learned bounded score mixing. |
| Causal supervision | DICE branch triplets and discrepancy; DCCL/DirectAU geometry as optional add-ons | DICE branch losses are default; contrastive/DirectAU terms are available but disabled unless explicitly tested. |
| Leakage control | Survey warnings about post-treatment features and exposure bias | `thesis_default` feature policy; train-only popularity/recency; explicit propensity-gate rules. |
| Systems path | Full-graph vs mini-batch GNN literature; ANN systems papers | Sampled subgraphs, auto-batch, vectorized negative sampling, bounded pairwise losses. |
| Evaluation | Ranking metrics plus popularity/resource diagnostics | NDCG/Recall/Hit/raw PyG AveragePopularity/Personalization plus absolute per-run CRRU that internally normalizes raw ARP; no causal-effect metric claim. |

## Contribution Hypotheses

| Hypothesis | Evidence needed |
| --- | --- |
| H1: Dual-branch causal supervision can match or beat LightGCN accuracy on exposure-biased datasets. | Full formal rows where EDGRec >= `lightgcn_paper` on NDCG/Recall/Hit. Currently strongest on `kuairec_v2`; mixed elsewhere. |
| H2: Sampled causal-branch training is much faster than paper-faithful full-graph causal baselines. | Per-epoch time from SQLite/query report. Current DICE evidence is runtime-probe only. |
| H3: Learned score mixing can trade accuracy against popularity sensitivity. | Score-mix, raw PyG AveragePopularity, branch-rank, and Spearman diagnostics from final test rows. |
| H4: Safe side features help only when dataset semantics support them. | Explicit `with_features` ablations and preprocessing-sweep rows; avoid cross-dataset generalization from one view. |

## Claim Boundaries

| If result shows... | Thesis wording |
| --- | --- |
| EDGRec better accuracy and faster than `lightgcn_paper` | "EDGRec improves both ranking and training efficiency on this dataset under the tested protocol." |
| EDGRec equal/slightly worse accuracy but much faster | "EDGRec offers a resource-efficiency trade-off; contribution is systems/practical, not accuracy SOTA." |
| EDGRec lower popularity but lower accuracy | "The model shifted recommendations away from popular items, but the trade-off hurt ranking utility." |
| DICE only has runtime probes | "DICE paper-faithful training appears computationally impractical under current hardware/profile; accuracy comparison remains open." |
| CAGRA improves EDGRec training speed | "CAGRA is not used as an EDGRec training accelerator; materialized ANN edges increase the message-passing graph." |

## Open Thesis Evidence Gaps

| Gap | Needed before strong claim |
| --- | --- |
| DICE final accuracy | Full or constrained paper-DICE formal rows; runtime probes are resource-only. |
| Multi-seed robustness | Repeat best EDGRec and LightGCN paper rows across seeds or state single-seed limitation. |
| Causal validity | Use randomized/exposure-aware datasets carefully; ranking metrics alone do not validate causal effects. |
| Score-mix mechanism | Explain dataset-local score-mix and branch diagnostics instead of claiming universal disentanglement. |
| Search provenance | Use only revisioned fresh Optuna trials for tuning claims. |
