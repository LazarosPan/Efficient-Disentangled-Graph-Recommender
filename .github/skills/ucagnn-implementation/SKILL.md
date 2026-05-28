---
name: ucagnn-implementation
description: Implementation guide for U-CaGNN covering architecture, configuration, data pipeline, losses, and training workflow.
---

# U-CaGNN Implementation Skill

Use this routed doc set for the **current** U-CaGNN codebase. These files are live implementation notes, not theory archives. Keep them aligned to the code, keep them short, and store each fact in one owner file only.

## Read in this order

1. [ucagnn-architecture.md](ucagnn-architecture.md) - model components, score views, and public `UCaGNN` surfaces.
2. [ucagnn-data-pipeline.md](ucagnn-data-pipeline.md) - loaders, canonical schema, graph construction, and samplers.
3. [ucagnn-config.md](ucagnn-config.md) - defaults, presets, override precedence, and experiment-facing knobs.
4. [ucagnn-losses.md](ucagnn-losses.md) - `LossSuite` terms, schedule behavior, and optional auxiliaries.
5. [ucagnn-training.md](ucagnn-training.md) - runtime flow, trainer, evaluator, checkpoints, and logging.
6. [ucagnn_full.md](ucagnn_full.md) - navigation, end-to-end integration flow, and source map.

## File ownership

| File | Owns |
| --- | --- |
| `ucagnn-architecture.md` | embedding, propagation, scoring, and propensity components; score views; and `UCaGNN` public API |
| `ucagnn-data-pipeline.md` | loader boundary, feature policy, canonical schema, graph build, and samplers |
| `ucagnn-config.md` | defaults, presets, precedence, validation rules, and experiment-facing config |
| `ucagnn-losses.md` | `LossSuite` terms, weighting, and schedule semantics |
| `ucagnn-training.md` | runtime orchestration, trainer, evaluator, checkpointing, and tracking |
| `ucagnn_full.md` | reading map, integration flow, and code source map |

## Editing rules

1. Put a fact in exactly one owner file and link to that file elsewhere.
2. Prefer current names and current behavior; do not keep stale or legacy wording unless the code still supports it.
3. Keep `ucagnn_full.md` as an integration map, not a second copy of the slice docs.
4. Keep Mermaid blocks simple: standard diagram types, plain ASCII labels, and no nested fences.
