# Causal Embeddings for Recommendations

This repository implements U-CaGNN, a resource-aware causal-disentangled graph recommendation framework that separates user interest from popularity conformity.

## Overview

U-CaGNN synthesizes techniques from multiple causal recommendation papers (CausE, DICE, MCLN, SIGformer) into a single framework. It uses dual-branch graph convolution to separate genuine user preferences from conformity-driven behavior, with multi-task losses and inverse propensity weighting driven by an item-side propensity proxy rather than a fully identified treatment/control model.

Key features:
- Config-driven architecture with explicit U-CaGNN, paper LightGCN, and paper GCN-DICE model adapters
- Mini-batch U-CaGNN training plus full-graph training for paper baselines
- Graph construction methods: observed train-interaction graphs by default, with optional CAGRA ANN augmentation
- Formal experiment matrix across datasets and presets, with paper baselines kept on observed interaction graphs
- SQLite primary logging with MLflow secondary tracking
- Automatic checkpointing and resume from crashes

## Datasets

Supported datasets include MovieLens 1M/20M, Amazon Book, Taobao, KuaiRec v2, and KuaiRand-1K.

## Usage

See [docs/usage/](docs/usage/) for detailed usage guides on running experiments and scripts.

## Results

Experiment results are logged to SQLite (`results/thesis_experiments.db`) and to MLflow (`results/mlflow.db`). A generated summary view lives in `results/query_results.md`.
