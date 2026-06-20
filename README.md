# Efficient Disentangled Graph Embeddings for Bias-Aware Recommendation

This repository implements EDGRec, a resource-aware disentangled graph recommender inspired by causal embedding methods such as DICE.

## Overview

EDGRec synthesizes techniques from multiple causal recommendation papers (CausE, DICE, MCLN, SIGformer) into a single framework. It uses dual-branch graph convolution to learn interest and conformity channels, DICE-style branch supervision, popularity-aware diagnostics, and optional propensity calibration. The current implementation supports causal-style ablations and disentanglement diagnostics, but it does not claim a fully identified treatment/control effect estimator.

Key features:
- Config-driven architecture with explicit EDGRec, paper LightGCN, and paper GCN-DICE model adapters
- Mini-batch EDGRec training plus full-graph training for paper baselines
- Graph construction: observed train-interaction graphs
- Formal experiment matrix across datasets and presets, with paper baselines kept on observed interaction graphs
- SQLite primary logging with MLflow secondary tracking
- Automatic checkpointing and resume from crashes

## Thesis Contribution

EDGRec is evaluated as a resource-aware graph recommender, not as a fully identified causal-effect estimator. The thesis contribution has two parts:

- Model side: a practical disentangled graph recommender with interest and conformity branches, DICE-style branch supervision, bounded auxiliary losses, safe item features, and optional propensity calibration.
- Evaluation side: Composite Resource-aware Recommendation Utility (CRRU), a parameterized utility family for comparing recommenders under accuracy, popularity-diversity, and computational-efficiency trade-offs.

## CRRU Utility Family

CRRU is defined as a configurable weighted geometric utility:

$$
\operatorname{CRRU}_K(m; \theta)
=
\operatorname{Accuracy}_K(m)^{\lambda_A}
\cdot
\operatorname{PopularityDiversity}_K(m)^{\lambda_P}
\cdot
\operatorname{Efficiency}(m)^{\lambda_E}
$$

with $\lambda_A + \lambda_P + \lambda_E = 1$ and non-negative component weights.

Thesis instantiation:

$$
\operatorname{Accuracy}_K =
\operatorname{NDCG@K}^{0.50}
\cdot
\operatorname{Recall@K}^{0.35}
\cdot
\operatorname{Hit@K}^{0.15}
$$

$$
\operatorname{PopularityDiversity}_K =
\operatorname{Personalization@K}^{0.40}
\cdot
(1 - \operatorname{AvgPop@K}_n)^{0.60}
$$

$$
\operatorname{Efficiency} =
(1 - \log(1+\operatorname{VRAM})_n)^{0.50}
\cdot
(1 - \log(1+\operatorname{time/epoch})_n)^{0.50}
$$

$$
\operatorname{CRRU}_K =
\operatorname{Accuracy}_K^{0.55}
\cdot
\operatorname{PopularityDiversity}_K^{0.30}
\cdot
\operatorname{Efficiency}^{0.15}
$$

Important interpretation rules:

- CRRU is higher-is-better, but it is a utility scalarization, not a universal recommender metric.
- The thesis default is accuracy-dominant: `0.55` accuracy, `0.30` popularity-diversity, `0.15` efficiency.
- Average popularity, VRAM, and time per epoch are inverted after dataset-local section-row min-max normalization.
- CRRU is relative within a dataset/report section. Adding or removing comparison rows can change values.
- The inverse popularity term measures lower popularity concentration, not guaranteed fairness or causal debiasing.

## Datasets

Supported datasets include MovieLens 1M/20M, Amazon Book, Taobao, KuaiRec v2, and KuaiRand-1K.

## Usage

See [docs/usage/](docs/usage/) for detailed usage guides on running experiments and scripts.

## Results

Experiment results are logged to SQLite (`results/thesis_experiments.db`) and to MLflow (`results/mlflow.db`). A generated summary view lives in `results/query_results.md`. The CRRU@K reporting metric is defined in [.agents/skills/edgrec-implementation/edgrec-training.md](.agents/skills/edgrec-implementation/edgrec-training.md); Optuna search diagnostics live in `results/optuna_optimization.md` with a compact paper-ready figure set in `results/optuna_figures/`.
