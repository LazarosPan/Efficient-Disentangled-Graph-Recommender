# Workspace Instructions

You are working on a thesis project about causal embeddings for recommendation. These instructions describe how to use the docs, codebase, and workflow.

## Iteration Workflow (Mandatory)

Before each implementation iteration:

1. Read `.github/skills/ucagnn-implementation/`  
   – Fast routing layer for implementation work; use it to decide what parts of the system matter.

2. Read `docs/ucagnn_implementation/ucagnn_full.md`  
   – Authoritative implementation reference; follow it to keep behavior, defaults, and workflow aligned.

After each iteration, update both so they match the code:

- `.github/skills/ucagnn-implementation/`
- `docs/ucagnn_implementation/ucagnn_full.md`

Keep changes minimal and concise.

## Repository Layout (Guidance)

Use this structure to place and find instructions:

- `.github/copilot-instructions.md` – Always-on router; keep short and pointer-based.
- `.github/instructions/` – File- or topic-scoped guidance.
- `.github/prompts/` – Reusable prompt templates.
- `.github/agents/` – Custom agents for specialized workflows.
- `.github/skills/` – Folder for skill files that route to specific docs based on task.

Core project folders for code:

- `src/training/` – Training loops, checkpointing, evaluation.
- `src/models/` – Model architecture and scoring.
- `src/data/` – Dataset loaders and graph builders.
- `src/losses/` – Loss functions and multi-task orchestration.
- `experiments/` – Experiment runners and catalog definitions.
- `scripts/` – Preflight and maintenance utilities.

## Project Rules

- SQLite is the primary experiment record; MLflow is a secondary UI and artifact tracker.
- The formal experiment grid is `dataset × preset × graph_method`, with score-mix sweeps for dual-branch presets handled inside that protocol.
- Treat `batch_size`, `num_neighbors`, etc. as support parameters to validate via quick validation, not as thesis axes.
- One experiment = one training run = one checkpoint; evaluation should reuse checkpoints, not retrain.
- Keep the codebase readable and maintainable; do not add thin wrappers or helper layers when the existing function or code path is already clear.

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
- Keep the codebase small, clean, and organized; remove unused code and files promptly.
- Use mermaid diagrams in docs to clarify complex workflows and data flows.

## Coding Standards

- Every function must be type-annotated and have a short docstring describing purpose, arguments, and return values.
- Prefer vectorized operations and efficient data structures over explicit Python loops where feasible.
- Follow PEP 8. Run `ruff format` at the root of the repo to format all code files. Use `ruff check` to find and fix linting issues.
- After any change, run `uv run scripts/quick_validate.py` to validate the project.