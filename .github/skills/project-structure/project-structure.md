# Run: `tree -I 'latex|.venv|external|results/checkpoints|mlruns|*/__pycache__/'`

```
├── causal_embeddings_for_recommendations.egg-info
│   ├── dependency_links.txt
│   ├── entry_points.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── data
│   ├── all_datasets_feature_audit.json
│   ├── all_datasets_information.md
│   ├── AmazonBook
│   │   ├── processed
│   │   └── raw
│   │       ├── item_list.txt
│   │       ├── test.txt
│   │       ├── train.txt
│   │       └── user_list.txt
│   ├── datasets_information.md
│   ├── KuaiRand-1K
│   │   ├── data
│   │   │   ├── log_random_4_22_to_5_08_1k.csv
│   │   │   ├── log_standard_4_08_to_4_21_1k.csv
│   │   │   ├── log_standard_4_22_to_5_08_1k.csv
│   │   │   ├── user_features_1k.csv
│   │   │   ├── video_features_basic_1k.csv
│   │   │   └── video_features_statistic_1k.csv
│   │   ├── figs
│   │   │   ├── KuaiRand.png
│   │   │   ├── kuaishou-app.png
│   │   │   └── three-version.png
│   │   ├── LICENSE
│   │   ├── load_data_1k.py
│   │   └── README.md
│   ├── KuaiRand_SIGformer
│   │   └── raw
│   │       ├── info.txt
│   │       ├── test.txt
│   │       ├── train.txt
│   │       └── valid.txt
│   ├── KuaiRec_SIGformer
│   │   └── raw
│   │       ├── info.txt
│   │       ├── test.txt
│   │       ├── train.txt
│   │       └── valid.txt
│   ├── KuaiRec_v2
│   │   ├── data
│   │   │   ├── big_matrix.csv
│   │   │   ├── item_categories.csv
│   │   │   ├── item_daily_features.csv
│   │   │   ├── kuairec_caption_category.csv
│   │   │   ├── README.md
│   │   │   ├── small_matrix.csv
│   │   │   ├── social_network.csv
│   │   │   ├── user_features.csv
│   │   │   └── video_raw_categories_multi.csv
│   │   ├── figs
│   │   │   ├── colab-badge.svg
│   │   │   └── KuaiRec.png
│   │   ├── LICENSE
│   │   ├── loaddata.py
│   │   └── Statistics_KuaiRec.ipynb
│   ├── MovieLens1M
│   │   ├── processed
│   │   └── raw
│   │       ├── movies.dat
│   │       ├── ratings.dat
│   │       ├── README.md
│   │       └── users.dat
│   ├── MovieLens20M
│   │   └── raw
│   │       ├── genome-scores.csv
│   │       ├── genome-tags.csv
│   │       ├── links.csv
│   │       ├── movies.csv
│   │       ├── ratings.csv
│   │       ├── README.md
│   │       └── tags.csv
│   └── Taobao
│       └── raw
│           ├── README.md
│           ├── UserBehavior.csv
│           └── UserBehavior.csv.zip.md5
├── docs
│   ├── existing_implementations
│   │   ├── CaDSI_audit.md
│   │   ├── CausE_audit.md
│   │   ├── DICE_audit.md
│   │   ├── DirectAU_audit.md
│   │   ├── FMMRec_audit.md
│   │   ├── LayerGCN_audit.md
│   │   ├── LightGCNpp_audit.md
│   │   ├── MCLN_audit.md
│   │   ├── MGCE_audit.md
│   │   ├── PropCare_audit.md
│   │   └── SIGformer_audit.md
│   ├── guidelines
│   │   ├── env_setup.md
│   │   ├── profile_plan.md
│   │   └── thesis_plan.md
│   ├── notes
│   │   ├── manus_research_report.md
│   │   ├── progress_ideas.md
│   │   ├── recsys_improvements.md
│   │   ├── ucagnn_consolidated_recommendations.md
│   │   ├── UCaGNN_updates_implementation_focused.md
│   │   ├── UCaGNN_updates.md
│   │   └── useful_commands.md
│   ├── paper_summaries
│   │   ├── full_summary_detailed.md
│   │   ├── gcn_models.md
│   │   ├── methematical_formulations.md
│   │   ├── notes_by_paper_10.md
│   │   ├── summary_by_paper_10.md
│   │   ├── summary_hybrid_transGNN.md
│   │   ├── summary_per_ai_recommendation.md
│   │   ├── summary_performance_papers.md
│   │   ├── summary_propcore.md
│   │   └── summary_survey_papers_4.md
│   ├── ucagnn_implementation
│   │   ├── architecture.md
│   │   ├── config-reference.md
│   │   ├── data-pipeline.md
│   │   ├── losses.md
│   │   ├── models.md
│   │   ├── README.md
│   │   ├── theoretical_justifications.md
│   │   ├── training.md
│   │   └── ucagnn_full.md
│   └── usage
│       ├── experiments.md
│       └── scripts.md
├── experiments
│   ├── ablation_configs.py
│   ├── cli_parsers.py
│   ├── experiment_catalog.json
│   ├── __init__.py
│   ├── recipes.py
│   ├── run_ablation.py
│   ├── run_benchmark.py
│   └── run_experiment.py
├── LICENCE
├── pyproject.toml
├── README.md
├── results
│   ├── dataset_visualizations
│   │   ├── amazonbook_profile.png
│   │   ├── benchmark_overview.png
│   │   ├── benchmark_summary.json
│   │   ├── benchmark_summary.md
│   │   ├── kuairand1k_profile.png
│   │   ├── kuairec_v2_profile.png
│   │   ├── movielens1m_profile.png
│   │   ├── movielens20m_profile.png
│   │   └── taobao_profile.png
│   ├── experiments.db
│   ├── formal_run_state.json
│   ├── mlflow.db
│   └── thesis_experiments.db
├── scripts
│   ├── cleanup_experiment_artifacts.py
│   ├── download_pyg_datasets.py
│   ├── evaluate_scoring_modes.py
│   ├── __init__.py
│   ├── query_results.py
│   ├── quick_validate.py
│   ├── reset_experiment_db.py
│   └── _workflow_helpers.py
├── src
│   ├── data
│   │   ├── canonical.py
│   │   ├── feature_policy.py
│   │   ├── graph_builder.py
│   │   ├── __init__.py
│   │   ├── loaders
│   │   │   ├── amazonbook.py
│   │   │   ├── _explicit_ratings.py
│   │   │   ├── __init__.py
│   │   │   ├── kuairand1k.py
│   │   │   ├── kuairec_v2.py
│   │   │   ├── movielens1m.py
│   │   │   ├── movielens20m.py
│   │   │   ├── _registry.py
│   │   │   └── taobao.py
│   │   ├── negative_sampler.py
│   │   └── subgraph_sampler.py
│   ├── data_exploration
│   │   ├── data_exploration.ipynb
│   │   ├── data_exploration.py
│   │   ├── data_information.py
│   │   └── explore_all_datasets.py
│   ├── __init__.py
│   ├── losses
│   │   ├── __init__.py
│   │   └── loss_suite.py
│   ├── models
│   │   ├── embeddings.py
│   │   ├── __init__.py
│   │   ├── lightgcn.py
│   │   ├── propensity.py
│   │   ├── scoring.py
│   │   └── ucagnn.py
│   ├── profiling
│   │   ├── gpu_profiler.py
│   │   └── __init__.py
│   ├── training
│   │   ├── evaluator.py
│   │   ├── __init__.py
│   │   └── mini_batch_trainer.py
│   └── utils
│       ├── cli_parsers.py
│       ├── config.py
│       ├── csv_features.py
│       ├── dataset_loader_utils.py
│       ├── experiment_logger.py
│       ├── __init__.py
│       ├── interaction_indexing.py
│       ├── project_paths.py
│       ├── reproducibility.py
│       └── trainer_runtime.py
├── tests
│   ├── sqlite_queries
│   │   └── failure_reasons.sql
│   ├── test_cli_parsers.py
│   ├── test_data_and_reproducibility.py
│   ├── test_experiment_logger.py
│   ├── test_formal_training_policy.py
│   └── test_split_safety.py
└── uv.lock
```
