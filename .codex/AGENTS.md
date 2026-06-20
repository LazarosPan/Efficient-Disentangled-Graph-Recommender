# Workspace Instructions

You are working on a thesis project about efficient disentangled graph recommender. Follow these rules to keep implementation quality high and diffs small.

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
- Read `.agents/skills/python-patterns/SKILL.md`.
- Remove unused code, do not keep legacy features. Keep the codebase clean.

### 4. Goal-Driven Execution

- Translate requests into verifiable success criteria.
- Prefer test-first validation for bug fixes, validation changes, and refactors.
- For multi-step work, keep a short plan with a verification check per step.
- Loop until criteria are verified; avoid vague "looks good" completion.

## Iteration Workflow (Mandatory)

Before each implementation iteration read the following files:

- `.agents/skills/` - Useful skills for implementation, project structure, and existing work (see the detailed list below).
- `.agents/skills/existing-work/full_summary.md` - This file is very important, as it is the scientific source of truth for the project; it synthesizes all the relevant papers and prior implementations that the project builds on. It is not a routing target, but it should be read before any other files to understand the scientific context and motivation for the implementation. The goal of this project is to synthesize the best ideas from the existing work into a single, unified codebase that implements the EDGRec model and its training protocol.

- Target system / development tracking files (not source of truth):

1. `.agents/skills/edgrec-implementation/`.
2. `.agents/skills/edgrec-implementation/edgrec_full.md`.

After each iteration, update the target system files so they match the code; do not treat them as the primary theory source:

- `.agents/skills/edgrec-implementation/` - These files should contain as much information as possible about the project, with the least amount of words possible.
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

- Apply KISS (Keep It Simple, Stupid), YAGNI (You Aren't Gonna Need It), DRY (Don’t Repeat Yourself), and SOLID (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion) pragmatically.
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

- `.codex/AGENTS.md` - always-on router.

### Skills:

Read the SKILL.md files and choose which files that they point to are relevant for the task at hand.

- `.agents/skills/cuvs/SKILL.md` - cuVS Python API reference.
- `.agents/skills/existing-work/SKILL.md` - prior causal recommendation implementation synthesis.
- `.agents/skills/optuna/SKILL.md` - key Optuna optimization workflows and their implementation mechanics.
- `.agents/skills/mlflow/SKILL.md` - MLflow/PyTorch tracking guidance.
- `.agents/skills/python-patterns/SKILL.md` - Python coding patterns.
- `.agents/skills/project-structure/SKILL.md` - project layout reference.
- `.agents/skills/pytorch/SKILL.md` - PyTorch module and runtime guide.
- `.agents/skills/pytorch-geometric/SKILL.md` - PyG reference and performance tips.
- `.agents/skills/memory-optimizations/SKILL.md` - memory optimization techniques for Python and PyTorch.
- `.agents/skills/numba/SKILL.md` - Numba JIT compilation guide and best practices.
- `.agents/skills/causal-inference/SKILL.md` - key causal inference techniques and their implementation mechanics.
- `.agents/skills/recsys/SKILL.md` - key recommendation techniques and their implementation mechanics.
- `.agents/skills/graph-neural-networks/SKILL.md` - GNN architecture, training and optimization methods.
- `.agents/skills/edgrec-implementation/SKILL.md` - EDGRec implementation guide.

### Core project folders:

- `src/utils/` - utilities and helpers for all other code.
- `src/training/` - training loops, checkpointing, evaluation.
- `src/models/` - model architecture and scoring.
- `src/data/` - dataset loaders and graph builders.
- `src/losses/` - loss functions and multi-task orchestration.
- `experiments/` - experiment runners and catalog definitions.
- `scripts/` - preflight and maintenance utilities.

<!-- rtk-instructions v2 -->
# RTK — Token-Optimized CLI

**rtk** is a CLI proxy that filters and compresses command outputs, saving 60-90% tokens.

## Rule

Always prefix shell commands with `rtk`:

```bash
# Instead of:              Use:
git status                 rtk git status
git log -10                rtk git log -10
git tree                   rtk git tree
git diff                   rtk git diff
```

## Meta commands (use directly)

```bash
rtk gain              # Token savings dashboard
rtk gain --history    # Per-command savings history
rtk discover          # Find missed rtk opportunities
rtk proxy <cmd>       # Run raw (no filtering) but track usage
```
<!-- /rtk-instructions -->
