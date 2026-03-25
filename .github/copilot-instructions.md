# Workspace Instructions

You are working on a thesis project about causal embeddings for recommendation. These instructions describe how to use the docs, codebase, and workflow.

## Iteration Workflow (Mandatory)

Before each implementation iteration:

1. Read `.github/skills/ucagnn-implementation/`  
   – Fast routing layer for implementation work; use it to decide what parts of the system matter.

2. Read `docs/ucagnn_implementation/`  
   – Authoritative implementation reference; follow it to keep behavior, defaults, and workflow aligned.

After each iteration, update both so they match the code:

- `.github/skills/ucagnn-implementation/`
- `docs/ucagnn_implementation/`

Keep changes minimal and concise.

## Repository Layout (Guidance)

Use this structure to place and find instructions:

- `.github/copilot-instructions.md` – Always-on router; keep short and pointer-based.
- `.github/instructions/` – File- or topic-scoped guidance.
- `.github/prompts/` – Reusable prompt templates.
- `.github/agents/` – Custom agents for specialized workflows.
- `.github/skills/<skill-name>/SKILL.md` – Entry point for a skill and its supporting files.

Core project folders for code:

- `src/training/` – Training loops, checkpointing, evaluation.
- `src/models/` – Model architecture and scoring.
- `src/data/` – Dataset loaders and graph builders.
- `src/losses/` – Loss functions and multi-task orchestration.
- `experiments/` – Experiment runners and catalog definitions.
- `scripts/` – Preflight and maintenance utilities.

## Project Rules

- SQLite is the primary experiment record; MLflow is a secondary UI and artifact tracker.
- The formal experiment grid is `dataset × preset × training_mode × graph_method`.
- Treat `batch_size`, `num_neighbors`, etc. as support parameters to validate via preflight, not as thesis axes.
- One experiment = one training run = one checkpoint; evaluation should reuse checkpoints, not retrain.

- Prefer editing existing files over adding new ones; avoid new scripts/CLIs unless clearly justified.
- Keep diffs small and auditable, especially for policy, reporting, config, and wiring changes.
- When a change exposes duplication, remove or merge redundant paths instead of leaving parallel logic.

## Engineering Principles

- Use KISS, YAGNI, DRY, and SOLID pragmatically:
  - KISS: prefer simple, clear designs.
  - YAGNI: implement only what is actually needed.
  - DRY: avoid duplicating behavior and logic.
  - SOLID: keep responsibilities narrow and components extensible.
- Prefer test-driven development: write a failing test, then the minimal implementation.
- Maintain consistent abstraction levels within a file/function; do not mix very high- and low-level concerns.
- Reuse existing code whenever possible; do not reimplement utilities elsewhere.

## Coding Standards

- Every function must be type-annotated and have a short docstring describing purpose, arguments, and return values.
- Prefer vectorized operations and efficient data structures over explicit Python loops where feasible.
- Follow PEP 8. Run `ruff format` to format code.
- After any change, run `uv run scripts/quick_validate.py` to validate the project.