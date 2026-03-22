# Workspace Instructions

You are a team of researchers and engineers working on a thesis project about causal embeddings for recommendations. This file provides instructions on how to navigate the codebase, documentation, and workflow for implementation iterations.

## Mandatory Iteration Workflow

Before each implementation iteration, read all files in:

1. `.github/skills/ucagnn-implementation/`
   Fast routing layer for the implementation area. Read this first to decide what part of the system is relevant.
2. `docs/ucagnn_implementation/`
   Authoritative implementation reference. Read this before changing code so behavior, defaults, and workflow stay aligned.

After every iteration, update the relevant files in the same two folders so the implementation instructions stay current with the codebase:

1. `.github/skills/ucagnn-implementation/`
2. `docs/ucagnn_implementation/`

## .github Layout

Use this repository structure when deciding where instructions belong:

- `.github/copilot-instructions.md`
  Always-on router. Keep it concise and pointer-based.
- `.github/instructions/`
  File-scoped or topic-scoped instructions.
- `.github/prompts/`
  Reusable prompt templates.
- `.github/agents/`
  Custom agents for specialized workflows.
- `.github/skills/<skill-name>/SKILL.md`
  Entry point for a skill; supporting files stay in the same skill folder.

## What To Read Next

Start with the core documentation folders:

- `.github/skills/ucagnn-implementation/`
  Fast routing skill layer for implementation decisions; load these first.
- `docs/ucagnn_implementation/`
  Authoritative implementation docs; use these to understand architecture, config, data, losses, and training.

After the documentation is understood, use topic folders for code exploration:

- `src/training/` for training loops, checkpointing, and evaluation
- `src/models/` for model architecture and scoring
- `src/data/` for dataset loaders and graph builders
- `src/losses/` for loss functions and multi-task orchestration
- `experiments/` for runner orchestration and catalog definitions
- `scripts/` for preflight and maintenance utilities

## Project Rules

- SQLite is the primary thesis record. MLflow is secondary UI and artifact tracking.
- The formal experiment matrix is `dataset × preset × training_mode × graph_method`.
- Treat `batch_size`, `num_neighbors`, and similar values as support parameters to validate via preflight, not as frozen thesis axes.
- One experiment should map to one training run and one saved checkpoint, with later evaluation possible without retraining.
- Updates to `.github/skills/ucagnn-implementation/` and `docs/ucagnn_implementation/` should be concise and minimal; avoid excessive verbosity.
- Prefer concise, auditable changes. Keep this file focused on routing to the right folders and files, not on embedding large code instructions here.
- Choose the correct data structures and algorithms to make code efficient.
- Apply KISS, YAGNI, DRY, and SOLID where appropriate:
  - KISS: prefer clear, minimal designs over complexity.
  - YAGNI: you ain’t gonna need it — build only what is proven needed.
  - DRY: keep behavior and logic in one place, avoid duplicate code.
  - SOLID: build single-responsibility, extensible, interface-driven components.
- WET: write everything twice before abstracting; avoid premature abstraction, refactor only once duplication reaches 3x.
- TDD: test-driven design—write a failing test first, then implement minimal code to pass it.
- Maintain consistent levels of abstraction in each function/class/file: avoid mixing low-level and high-level details in the same block.
- Write code that is easy to maintain, readable, and self-explanatory.
- Reuse existing code; avoid creating new duplicates in other files unless absolutely necessary.
- The codebase should be as simple as possible while still meeting the requirements; avoid over-engineering.
- Favor test-driven development (TDD) and full-cycle product thinking: code + tests + eval + iteration.
- After every change and update in the code, run `uv run scripts/quick_validate.py` to check if everything works fine.