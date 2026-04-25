---
name: ucagnn-implementation
description: Implementation guide for U-CaGNN covering architecture, configuration, data pipeline, losses, and training workflow.
---

# U-CaGNN Implementation Skills

Use this skill set when working on the U-CaGNN recommendation system implementation.

Prefer changing an existing routed file before adding a new script or module; add new entry points only when reuse would make the current path less clear. When cleaning CLI/orchestration code, remove no-op wrappers, collapse repeated validation/config scaffolding, let centralized parsers enforce valid CLI choices, remove redundant repo-root `sys.path` bootstrapping instead of suppressing import-order lint, and prefer direct named-field access over schema-order-dependent indexing.

## Read the Most Relevant File

- [ucagnn-architecture.md](ucagnn-architecture.md) — Model architecture, module layout, embeddings, LightGCN backbone, scoring, and propensity components.
- [ucagnn-config.md](ucagnn-config.md) — Configuration presets, key hyperparameters, curriculum schedule, and stability guidance for IPW.
- [ucagnn-data-pipeline.md](ucagnn-data-pipeline.md) — Dataset loading, canonical interaction format, graph construction methods, and negative sampling.
- [ucagnn-losses.md](ucagnn-losses.md) — Loss components, total loss formula, lambda defaults, IPW weighting, and curriculum activation.
- [ucagnn-training.md](ucagnn-training.md) — Training loop, evaluation, checkpointing, profiling, and experiment logging.

## Quick Routing Guide

- Use [ucagnn-architecture.md](ucagnn-architecture.md) when changing the model structure or adding modules.
- Use [ucagnn-config.md](ucagnn-config.md) when tuning experiments or selecting presets.
- Use [ucagnn-data-pipeline.md](ucagnn-data-pipeline.md) when modifying graph building, splits, or samplers.
- Use [ucagnn-losses.md](ucagnn-losses.md) when adjusting objectives or curriculum phases.
- Use [ucagnn-training.md](ucagnn-training.md) when working on trainer behavior, metrics, or logging.
