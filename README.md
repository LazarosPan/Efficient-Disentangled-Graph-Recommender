# Efficient Disentangled Graph Recommender

## Abstract

This repository hosts a thesis-oriented implementation of EDGRec for bias-aware recommendation.  
The codebase is organized as a reproducible research implementation rather than a production service, with explicit modeling, training, and reporting contracts used for dissertation experiments.

The central focus is a decomposed ranking model that combines preference disentanglement with resource-aware evaluation.  
Modeling and evaluation choices are constrained to be methodologically explicit, especially with respect to utility aggregation and comparability across datasets.

## Project scope

- The repository contains an implementation track for EDGRec and paper-faithful baselines (`lightgcn_paper`, `dice_paper`).
- The implementation is benchmarked as a ranking system, not as a causal-effect estimation system.
- The resource-efficiency utility is used for experiment selection and reporting, not as a universal recommendation objective.

## Method summary (high level)

The EDGRec path includes:

- Dual-branch graph representation modeling (interest branch and conformity branch).
- Context branch that injects item-level signals such as popularity and optional calibrated propensity factors.
- Learned score mixing that combines branch outputs and context adjustment.
- Sampled mini-batch training in the mainline EDGRec path, with full-graph contracts preserved for paper baselines.

See [edgrec-architecture](.agents/skills/edgrec-implementation/edgrec-architecture.md) for a detailed architecture map.

## Data and graph pipeline

The thesis pipeline is built around canonicalized interaction loading and graph construction rules:

- Graph construction is `observed` by default.
- Edge construction follows strict train-only conventions for the main evaluation contract.
- The item universe can be configured, including observed-only, full catalog, and KuaiRand random-exposure diagnostic variants.
- Default feature policy (`thesis_default`) is aligned to avoid post-treatment leakage.
- The compact EDGRec search-prior path is graph-only; side features are opt-in ablations or explicit dataset/profile candidates.

See [edgrec-data-pipeline](.agents/skills/edgrec-implementation/edgrec-data-pipeline.md) for the dataset and loader contract.

## Configuration and experiment contracts

Build precedence is:

- defaults
- profile preset
- recipe
- explicit CLI overrides
- runtime validation

Primary profiles documented in the implementation notes are:

- `edgrec` (`preset_full()`): mainline dual-branch model configuration.
- `edgrec-compact-search-prior`: default compact graph-only search prior and formal candidate, not a final thesis selection.
- `amazonbook-edgrec-compact-candidate`: explicit AmazonBook graph-only compact EDGRec candidate outside the shared compact default queue.
- `amazonbook-edgrec-deep-features-candidate`: explicit AmazonBook candidate preserving prior Optuna evidence for deeper/use_features EDGRec.
- `lightgcn`: scalable baseline path.
- `dice_like`: DICE-like ablation.
- `lightgcn_paper`: paper-faithful LightGCN baseline contract.
- `dice_paper`: paper-faithful DICE baseline contract.

See [edgrec-config](.agents/skills/edgrec-implementation/edgrec-config.md).

## Objective and losses

`LossSuite` composes several terms for recommendation quality and branch consistency, including:

- ranking loss for recommendation targets,
- branch-specific supervision,
- independence-style regularization,
- optional contrastive or auxiliary terms when explicitly enabled.

See [edgrec-losses](.agents/skills/edgrec-implementation/edgrec-losses.md).

## Training and evaluation protocol

End-to-end experimental flow:

- build and validate experiment configuration;
- load and canonicalize data;
- build the configured graph runtime;
- train with `MiniBatchTrainer` (sampled subgraphs for EDGRec, full-graph for paper baselines);
- evaluate standard ranking and diversity-related metrics;
- persist results and metadata to SQLite (MLflow logging remains optional).

Evaluated metrics include:

- NDCG@K, Recall@K, Hit@K, raw PyG AveragePopularity@K, Personalization@K at K = 20 and K = 40.

See [edgrec-training](.agents/skills/edgrec-implementation/edgrec-training.md).

## CRRU utility family

The utility used for thesis-style model selection is a deterministic post-hoc
weighted geometric utility for one completed run:

$$
\mathrm{RankingAccuracy}_K
=\mathrm{NDCG@K}^{0.50}\cdot
\mathrm{Recall@K}^{0.35}\cdot
\mathrm{HitRatio@K}^{0.15}
$$

$$
\mathrm{PopularityAwarePersonalization}_K
=\mathrm{Personalization@K}^{0.40}\cdot
\mathrm{InverseRecommendationPopularity@K}^{0.60}
$$

$$
\mathrm{InverseRecommendationPopularity@K}
=1-\mathrm{CRRUNormalizedAveragePopularity@K}
$$

where

$$
\mathrm{CRRUNormalizedAveragePopularity@K}
=\frac{\log(1+\mathrm{AveragePopularity@K})}
{\log(1+\mathrm{LargestTrainingItemInteractionCount})}
$$

$$
\mathrm{TrainingResourceUtility}
=\mathrm{PeakGpuMemoryCapacityScore}^{0.50}\cdot
\mathrm{EpochDurationEfficiencyScore}^{0.50}
$$

$$
\mathrm{CRRU}_K
=\mathrm{RankingAccuracy}_K^{0.55}\cdot
\mathrm{PopularityAwarePersonalization}_K^{0.30}\cdot
\mathrm{TrainingResourceUtility}^{0.15}
$$

Interpretation rules:

- CRRU is higher-is-better, bounded in `[0,1]`, and parameterized by explicit weights.
- CRRU is an absolute per-run utility; adding/removing report rows must not change an existing run's CRRU.
- PyG AveragePopularity is logged from raw train-only item interaction counts.
- CRRU log-normalizes raw AveragePopularity internally with the logged or reconstructed largest train item count before inversion.
- Peak GPU memory is a capacity cost; epoch duration is a throughput cost.
- Invalid or missing CRRU inputs raise errors; `CRRU_EPSILON` only prevents exact-zero collapse under fractional powers.
- CRRU is not a causal estimator, fairness metric, debiasing proof, standard recommender metric, or universal cross-dataset quality score.

See [edgrec-result-analysis](.agents/skills/edgrec-implementation/edgrec-result-analysis.md).

## Reporting artifacts

Primary thesis evidence artifacts:

- `results/thesis_experiments.db` (main experiment store)
- `results/mlflow.db` (optional mirror)
- `results/query_results.md` (generated report-style tables)
- `results/optuna_optimization.md` (search summaries)
- `results/optuna_figures/` (supporting plots)

Evidence note: current Optuna evidence supports compact EDGRec candidates for KuaiRec_v2, a near-parity/speed candidate for MovieLens1M, and KuaiRand1K as a stress-test diagnostic. AmazonBook is not part of the shared compact default queue, but it is still part of EDGRec optimization: compare the AmazonBook compact and deep/features candidates against the LightGCN-paper accuracy baseline before any thesis-profile promotion.

## Claim boundaries

- Causal wording is intentionally constrained to modeling assumptions and structural claims.
- Ranking gains are presented as empirical results, not causal effects.
- Paper-faithful baselines are evaluated under separate contracts to avoid conflating scalable approximations with faithful reproduction.

## Literature and implementation map

- Method rationale and model references: [edgrec-literature](.agents/skills/edgrec-implementation/edgrec-literature.md)
- Full implementation map and cross-cutting assumptions: [edgrec_full](.agents/skills/edgrec-implementation/edgrec_full.md)
