---
name: ucagnn-implementation
description: Implementation guide for U-CaGNN covering architecture, configuration, data pipeline, losses, and training workflow.
---

# U-CaGNN Implementation Skill

Live implementation notes for the current U-CaGNN codebase. Dense by design: many facts, few words, no long prose.

## Read in this order

1. [ucagnn-literature.md](ucagnn-literature.md) - literature-backed rationale, contribution hypotheses, claim boundaries.
2. [ucagnn-architecture.md](ucagnn-architecture.md) - model modules, scorer, public `UCaGNN` surfaces.
3. [ucagnn-data-pipeline.md](ucagnn-data-pipeline.md) - loaders, schema, feature policy, graph build, samplers.
4. [ucagnn-config.md](ucagnn-config.md) - defaults, presets, profiles, search spaces, override rules.
5. [ucagnn-losses.md](ucagnn-losses.md) - `LossSuite`, loss terms, schedules, IPW/calibration gates.
6. [ucagnn-training.md](ucagnn-training.md) - runtime, trainer, evaluator, checkpoints, tracking, reports.
7. [ucagnn-result-analysis.md](ucagnn-result-analysis.md) - current result interpretation from `results/query_results.md` + `results/optuna_optimization.md`.
8. [ucagnn_full.md](ucagnn_full.md) - integration map and source map only.

## File ownership

| File | Owns |
| --- | --- |
| `ucagnn-literature.md` | source-backed design rationale, thesis claim boundaries, and contribution hypotheses |
| `ucagnn-architecture.md` | embedding, propagation, scoring, and propensity components; the refined scorer contract; and `UCaGNN` public API |
| `ucagnn-data-pipeline.md` | loader boundary, feature policy, canonical schema, graph build, and samplers |
| `ucagnn-config.md` | defaults, presets, precedence, validation rules, and experiment-facing config |
| `ucagnn-losses.md` | `LossSuite` terms, weighting, and schedule semantics |
| `ucagnn-training.md` | runtime orchestration, trainer, evaluator, checkpointing, and tracking |
| `ucagnn-result-analysis.md` | generated-report-backed interpretation from `results/query_results.md` and `results/optuna_optimization.md`, comparison rules, and dataset-level result narratives |
| `ucagnn_full.md` | reading map, integration flow, and code source map |

## Style contract

| Rule | Contract |
| --- | --- |
| Fact ownership | Put each fact in one owner file; elsewhere link or name owner. |
| Token shape | Prefer tables, bullets, and short sentences. Avoid long paragraphs. |
| Currency | Use live code names/behavior. Keep legacy wording only when code supports it. |
| Scope | Implementation owner files are code notes; literature/result files are thesis-facing evidence summaries and must stay source-backed. |
| Result analysis | Before using or editing `ucagnn-result-analysis.md`, compare it with `results/query_results.md` and `results/optuna_optimization.md`; refresh stale speed, accuracy, CRRU, Optuna, and promotion-candidate explanations. |
| Overview file | `ucagnn_full.md` maps flow and sources; it must not copy slice details. |
| Diagrams | Simple Mermaid only: standard diagram types, plain ASCII labels, no nested fences. |
