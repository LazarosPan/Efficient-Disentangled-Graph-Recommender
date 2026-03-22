---
name: mlflow
description: Documentation files for MLflow integration with PyTorch and repository-local experiment tracking.
---

# MLflow Skill Files

This folder contains the following MLflow skill documentation files:

- [mlflow-pytorch.md](mlflow-pytorch.md): Core API reference pulled from MLflow docs for `mlflow.pytorch`. Includes class and function signatures, parameter details, and usage examples for model log/load/autolog/checkpoint workflows.
- [mlflow-pytorch-integration.md](mlflow-pytorch-integration.md): Integration guide and tutorial covering MLflow + PyTorch best practices, autologging, manual logging, system metrics, model signatures, checkpoint tracking, model registry integration, and advanced recipes.
- [tracking-experiments-local.md](tracking-experiments-local.md): Local experiment tracking tutorial using SQLite backend store for MLflow tracking. Includes setup (`MLFLOW_TRACKING_URI`), logging flows, MLflow UI launch, and change-over recommendations.

Each file is maintained independently and provides different levels of documentation:

- Reference-style API details (mlflow-pytorch.md)
- User-friendly integration tutorial (mlflow-pytorch-integration.md)
- Local experiment tracking onboarding (tracking-experiments-local.md)
