---
name: edgrec-implementation
description: Implementation guide for EDGRec covering architecture, configuration, data pipeline, losses, and training workflow.
---

# EDGRec Implementation Skill

Live implementation notes for the current EDGRec codebase. Dense by design: many facts, few words, no long prose.

## Read in this order

1. [edgrec-literature.md](edgrec-literature.md) - literature-backed rationale, contribution hypotheses, claim boundaries.
2. [edgrec-architecture.md](edgrec-architecture.md) - model modules, scorer, public `EDGRec` surfaces.
3. [edgrec-data-pipeline.md](edgrec-data-pipeline.md) - loaders, schema, feature policy, graph build, samplers.
4. [edgrec-config.md](edgrec-config.md) - defaults, presets, profiles, search spaces, override rules.
5. [edgrec-losses.md](edgrec-losses.md) - `LossSuite`, loss terms, schedules, IPW/calibration gates.
6. [edgrec-training.md](edgrec-training.md) - runtime, trainer, evaluator, checkpoints, tracking, reports.
7. [edgrec-result-analysis.md](edgrec-result-analysis.md) - current result interpretation from `results/query_results.md` + `results/optuna_optimization.md`.
8. [edgrec_full.md](edgrec_full.md) - integration map and source map only.

## File ownership

| File | Owns |
| --- | --- |
| `edgrec-literature.md` | source-backed design rationale, thesis claim boundaries, and contribution hypotheses |
| `edgrec-architecture.md` | embedding, propagation, scoring, and propensity components; the refined scorer contract; and `EDGRec` public API |
| `edgrec-data-pipeline.md` | loader boundary, feature policy, canonical schema, graph build, and samplers |
| `edgrec-config.md` | defaults, presets, precedence, validation rules, and experiment-facing config |
| `edgrec-losses.md` | `LossSuite` terms, weighting, and schedule semantics |
| `edgrec-training.md` | runtime orchestration, trainer, evaluator, checkpointing, and tracking |
| `edgrec-result-analysis.md` | generated-report-backed interpretation from `results/query_results.md` and `results/optuna_optimization.md`, comparison rules, and dataset-level result narratives |
| `edgrec_full.md` | reading map, integration flow, and code source map |

## Style contract

| Rule | Contract |
| --- | --- |
| Fact ownership | Put each fact in one owner file; elsewhere link or name owner. |
| Token shape | Prefer tables, bullets, and short sentences. Avoid long paragraphs. |
| Currency | Use live code names/behavior. Keep legacy wording only when code supports it. |
| Scope | Implementation owner files are code notes; literature/result files are thesis-facing evidence summaries and must stay source-backed. |
| Result analysis | Before using or editing `edgrec-result-analysis.md`, compare it with `results/query_results.md` and `results/optuna_optimization.md`; refresh stale speed, accuracy, CRRU, Optuna, and promotion-candidate explanations. |
| Overview file | `edgrec_full.md` maps flow and sources; it must not copy slice details. |
| Diagrams | Simple Mermaid only: standard diagram types, plain ASCII labels, no nested fences. |
