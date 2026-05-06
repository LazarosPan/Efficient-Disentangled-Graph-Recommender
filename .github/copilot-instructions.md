# Workspace Instructions

You are working on a thesis project about causal embeddings for recommendation. Follow these rules to keep implementation quality high and diffs small.

## Core Working Principles

### 1. Think Before Coding

- State assumptions explicitly.
- If multiple interpretations exist, present them instead of choosing silently.
- If a simpler approach exists, call it out and push back when warranted.
- If requirements are unclear, stop and ask clarifying questions.

### 2. Simplicity First

- Implement the minimum code needed to solve the requested problem.
- Do not add speculative features, configurability, or abstractions for single-use paths.
- Do not add error handling for impossible scenarios.
- If a solution is overcomplicated, simplify it.

### 3. Surgical Changes

- Touch only what is required for the request.
- Do not refactor or "clean up" adjacent code unless it is necessary for correctness.
- Match existing style in the edited area.
- Remove only the unused code/imports introduced by your own changes.
- If unrelated dead code is found, mention it but do not remove it unless asked.

### 4. Goal-Driven Execution

- Translate requests into verifiable success criteria.
- Prefer test-first validation for bug fixes, validation changes, and refactors.
- For multi-step work, keep a short plan with a verification check per step.
- Loop until criteria are verified; avoid vague "looks good" completion.

## Iteration Workflow (Mandatory)

Before each implementation iteration read the following files:

-  `.github/skills/` - Useful skills for implementation, project structure, and existing work (see the detailed list below).

- Target system / development tracking files:

1. `.github/skills/ucagnn-implementation/`.
2. `docs/ucagnn_implementation/ucagnn_full.md`.

After each iteration, update the target system files so they match the code; do not treat them as the primary theory source:

- `.github/skills/ucagnn-implementation/`
- `docs/ucagnn_implementation/ucagnn_full.md`
- `pyproject.toml` (for dependencies and versioning if applicable)

## Project Rules

- SQLite is the primary experiment record; MLflow is a secondary UI and artifact tracker.
- The formal experiment grid is `dataset * preset`, with score-mix sweeps for dual-branch presets handled inside that protocol.
- Treat `batch_size`, `num_neighbors`, and similar settings as support parameters to validate quickly, not thesis axes.
- One experiment = one training run = one checkpoint; evaluation should reuse checkpoints, not retrain.
- Prefer editing existing files over adding new ones; avoid new scripts/CLIs unless clearly justified.
- Keep diffs small and auditable, especially for policy, reporting, config, and wiring changes.
- When a change exposes duplication, merge redundant paths instead of leaving parallel logic.

## Engineering Standards

- Apply KISS, YAGNI, DRY, and SOLID pragmatically.
- Maintain consistent abstraction levels within a file/function.
- Reuse existing code and utilities whenever possible.
- Keep the codebase clean and organized; remove unused or duplicated code.
- Use mermaid diagrams in docs for complex workflows or data flows.

## Coding Standards

- Every function must be type-annotated and include a short docstring describing purpose, arguments, and return values.
- Prefer vectorized operations (NumPy/CuPy/Torch), efficient data structures and algorithms over explicit Python loops to optimize performance.
- Follow PEP 8 via the project's Ruff configuration.
- Before considering an iteration complete:
    1. Run `ruff check --fix .` to apply all safe and unsafe fixes.
    2. Run `ruff format .` and `ruff check --statistics .`, then fix any remaining errors manually.
    3. Run `ruff format .` again to ensure consistent styling.
- After any code change, run `uv run scripts/quick_validate.py` (document updates do not require this).

## Repository Layout (Guidance)

### Instructions:

- `.github/copilot-instructions.md` - always-on router; keep short and pointer-based.
- `.github/instructions/` - file- or topic-scoped guidance.
- `.github/prompts/` - reusable prompt templates.
- `.github/agents/` - custom agents for specialized workflows.

### Skills:

Read the SKILL.md files and choose which files that they point to are relevant for the task at hand.

- `.github/skills/cuvs/SKILL.md` - cuVS Python API reference.
- `.github/skills/existing-work/SKILL.md` - prior causal recommendation implementation synthesis.
- `.github/skills/mlflow/SKILL.md` - MLflow/PyTorch tracking guidance.
- `.github/skills/project-structure/SKILL.md` - project layout reference.
- `.github/skills/pytorch/SKILL.md` - PyTorch module and runtime guide.
- `.github/skills/pytorch-geometric/SKILL.md` - PyG reference and performance tips.
- `.github/skills/ucagnn-implementation/SKILL.md` - U-CaGNN implementation guide.

### Core project folders:

- `src/utils/` - utilities and helpers for all other code.
- `src/training/` - training loops, checkpointing, evaluation.
- `src/models/` - model architecture and scoring.
- `src/data/` - dataset loaders and graph builders.
- `src/losses/` - loss functions and multi-task orchestration.
- `experiments/` - experiment runners and catalog definitions.
- `scripts/` - preflight and maintenance utilities.
