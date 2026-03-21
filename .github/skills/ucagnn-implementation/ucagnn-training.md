# U-CaGNN Training Skill

Use this skill when working on training loop, evaluation, checkpoints, profiling, or experiment logging.

## Key Files
- `docs/ucagnn_implementation/training.md` - Training pipeline with paper cross-references
- `src/training/trainer.py` - Trainer class
- `src/training/evaluator.py` - Evaluator (Recall@K, NDCG@K)
- `src/profiling/gpu_profiler.py` - GPUProfiler and profile_stage
- `src/utils/experiment_logger.py` - SQLite ExperimentLogger

## Paper Sources
| Feature | Source |
|---------|--------|
| Adam lr=1e-3 | LightGCN |
| Gradient clipping max_norm=1.0 | DICE |
| Early stopping patience=10 | DDCE |
| 60 epochs default | thesis_plan.md |

## Training Pipeline
```python
from src.training.trainer import Trainer
from src.utils.experiment_logger import ExperimentLogger

experiment_logger = ExperimentLogger()
exp_id = experiment_logger.log_experiment(config.dataset, config)

trainer = Trainer(model, loss_suite, data, config, profiler,
                  experiment_logger=experiment_logger, exp_id=exp_id)
history = trainer.train()

# Test metrics
test_metrics = trainer.evaluator.evaluate(model, data, data.test_mask)
for metric, value in test_metrics.items():
    experiment_logger.log_metric(exp_id, metric, value, split="test")

experiment_logger.close()
trainer.save_checkpoint("results/checkpoints/ucagnn_best.pt")
```

## What Gets Logged Automatically
| Data | SQLite Table | Split |
|------|--------------|-------|
| Training loss | metrics | train |
| Validation metrics | metrics | val |
| Profiling stages | profiling | -- |
| Alpha values (sign-aware) | metrics | train |

## MLflow Routing
- `experiments/run_experiment.py` uses `--mlflow-tracking-uri` first, then `MLFLOW_TRACKING_URI`, then the project default `results/mlflow.db`.
- Plain `mlflow ui` or `mlflow server` without `--backend-store-uri` use MLflow's default root-level `mlflow.db`, not the thesis database under `results/`.
- Matrix fields should not be duplicated across MLflow params and tags; ordering/provenance should use explicit params such as `run_started_at_utc`, `project_version`, and `git_commit`.
