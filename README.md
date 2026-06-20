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

- NDCG@K, Recall@K, Hit@K, AveragePopularity@K, Personalization@K at K = 20 and K = 40.

See [edgrec-training](.agents/skills/edgrec-implementation/edgrec-training.md).

## CRRU utility family

The utility used for thesis-style model selection is:

$$
\mathrm{CRRU}_K(m;\theta)
=\mathrm{Accuracy}_K(m)^{\lambda_A}\cdot
\mathrm{PopularityDiversity}_K(m)^{\lambda_P}\cdot
\mathrm{Efficiency}(m)^{\lambda_E}
$$

with $\lambda_A+\lambda_P+\lambda_E=1$ and non-negative component weights.

$$
\mathrm{Accuracy}_K
=\mathrm{NDCG@K}^{0.50}\cdot
\mathrm{Recall@K}^{0.35}\cdot
\mathrm{Hit@K}^{0.15}
$$

$$
\mathrm{PopularityDiversity}_K
=\mathrm{Personalization@K}^{0.40}\cdot
(1-\mathrm{AvgPop@K}_n)^{0.60}
$$

$$
\mathrm{Efficiency}
=(1-\log(1+\mathrm{VRAM})_n)^{0.50}\cdot
(1-\log(1+\mathrm{time/epoch})_n)^{0.50}
$$

$$
\mathrm{CRRU}_K
=\mathrm{Accuracy}_K^{0.55}\cdot
\mathrm{PopularityDiversity}_K^{0.30}\cdot
\mathrm{Efficiency}^{0.15}
$$

Interpretation rules:

- CRRU is higher-is-better but not a universal recommendation metric.
- Component inputs are normalized per dataset/report section using dataset-local min-max normalization on report rows.
- Relative comparisons are affected by the set of rows included in a table.
- Lower average popularity is interpreted as lower concentration and is not a direct fairness or causal debiasing claim.

See [edgrec-result-analysis](.agents/skills/edgrec-implementation/edgrec-result-analysis.md).

## Reporting artifacts

Primary thesis evidence artifacts:

- `results/thesis_experiments.db` (main experiment store)
- `results/mlflow.db` (optional mirror)
- `results/query_results.md` (generated report-style tables)
- `results/optuna_optimization.md` (search summaries)
- `results/optuna_figures/` (supporting plots)

## Claim boundaries

- Causal wording is intentionally constrained to modeling assumptions and structural claims.
- Ranking gains are presented as empirical results, not causal effects.
- Paper-faithful baselines are evaluated under separate contracts to avoid conflating scalable approximations with faithful reproduction.

## Literature and implementation map

- Method rationale and model references: [edgrec-literature](.agents/skills/edgrec-implementation/edgrec-literature.md)
- Full implementation map and cross-cutting assumptions: [edgrec_full](.agents/skills/edgrec-implementation/edgrec_full.md)
