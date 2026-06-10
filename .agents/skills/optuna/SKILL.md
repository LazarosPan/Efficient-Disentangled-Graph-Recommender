---
name: optuna
description: Summarizes key Optuna workflows and optimization mechanics.
---

This skill is pointer to Optuna optimization resources:

- [cli.md](cli.md): Optuna CLI commands and a quick entry point for running ask/tell and related study operations.
- [ask-and-tell-interface.md](ask-and-tell-interface.md): interactive ask/tell workflow for optimization when objectives need external control or batching.
- [saving-resuming-study-with-rdb-backend.md](saving-resuming-study-with-rdb-backend.md): persistent studies and resume patterns with RDB storage.
- [easy-parallelization.md](easy-parallelization.md): single-process, multi-process, and multi-node parallel optimization patterns and storage choices.
- [efficient-optimization-algorithms.md](efficient-optimization-algorithms.md): sampler and pruner options, switching strategies, and default algorithm behavior.
- [pythonic-search-space.md](pythonic-search-space.md): `suggest_*` APIs, conditional/search-space definitions, and loops/branches in trial definitions.
- [user-defined-sampler.md](user-defined-sampler.md): implementing custom samplers from `BaseSampler`.
- [user-defined-pruner.md](user-defined-pruner.md): implementing custom pruners with `BasePruner`.
- [callback-for-study.optimize.md](callback-for-study.optimize.md): study-level callbacks and custom early-stop behavior during `optimize`.
- [specify-hyperparameters-manually.md](specify-hyperparameters-manually.md): using `enqueue_trial` and `add_trial` for manual/historical parameter injection.
- [early-stopping-independent-evaluations-by-wilcoxon-pruner.md](early-stopping-independent-evaluations-by-wilcoxon-pruner.md): Wilcoxon-based pruning for noisy, batch-evaluated objectives.
- [multi-objective-optimization.md](multi-objective-optimization.md): multi-objective studies, Pareto-front workflows, and objective handling.
- [quick-visualization-for-hyperparameter-optimization-analysis.md](quick-visualization-for-hyperparameter-optimization-analysis.md): plotting and interpreting optimization diagnostics.
- [reuse-best-trial.md](reuse-best-trial.md): re-running the best trial or best Pareto trials under alternate objective functions.
