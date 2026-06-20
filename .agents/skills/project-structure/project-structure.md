# Run: `tree -I 'latex|.venv|external|results/checkpoints|mlruns|*/__pycache__/'`

```
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
│   │   ├── gpt-pro.md
│   │   ├── gpt-xhigh.md
│   │   ├── profile_plan.md
│   │   └── thesis_plan.md
│   ├── notes
│   │   ├── edgrec_consolidated_recommendations.md
│   │   ├── EDGRec_updates_implementation_focused.md
│   │   ├── EDGRec_updates.md
│   │   ├── exa.md
│   │   ├── manus_research_report.md
│   │   ├── progress_ideas.md
│   │   ├── recsys_improvements.md
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
│   └── usage
│       ├── experiments.md
│       └── scripts.md
├── efficient_disentangled_graph_recommender.egg-info
│   ├── dependency_links.txt
│   ├── entry_points.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── experiments
│   ├── ablation_configs.py
│   ├── benchmark_resolvers.py
│   ├── cli_parsers.py
│   ├── experiment_catalog.json
│   ├── __init__.py
│   ├── recipes.py
│   ├── run_ablation.py
│   ├── run_benchmark.py
│   ├── run_experiment.py
│   ├── run_search.py
│   └── search_spaces.json
├── LICENCE
├── Papers_Efficient_Disentangled_Graph_Recommender
│   ├── Causal_Embeddings_for_RecSys
│   │   ├── Causal Embeddings for Recommendation.pdf
│   │   └── Causal Variational Inference for Deconfounded Multi-Behavior Recommendation.pdf
│   ├── Causal_Surveys
│   │   ├── A Survey on Causal Inference for Recommendation.pdf
│   │   ├── Causal Inference for Recommendation_ Foundations, Methods, and Applications.pdf
│   │   ├── Causal Inference in Recommender Systems - A Survey and Future Directions.pdf
│   │   └── Causal Inference in Recommender Systems_ A Survey of Strategies for Bias Mitigation, Explanation, and Generalization.pdf
│   ├── Datasets
│   │   ├── KuaiRand_ An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos.pdf
│   │   ├── KuaiRec_ A Fully-observed Dataset and Insights for Evaluating Recommender Systems.pdf
│   │   ├── RecFlow_ An Industrial Full Flow Recommendation Dataset.pdf
│   │   └── The MovieLens Datasets: History and Context.pdf
│   ├── Disentanglement
│   │   ├── causal-augmented-disentanglement-for-contrastive-recommendation.pdf
│   │   ├── Causal_Disentanglement_for_Semantic-Aware_Intent_Learning_in_Recommendation.pdf
│   │   ├── Disentangled Causal Embedding With Contrastive Learning For Recommender System.pdf
│   │   ├── Disentangling User Interest and Conformity for Recommendation with Causal Embedding.pdf
│   │   └── Dual disentanglement of user–item interaction for recommendation with causal embedding.pdf
│   ├── GNN_GCN
│   │   ├── Graph Convolutional Neural Networks for Web-Scale Recommender Systems.pdf
│   │   ├── Graph Embedding Techniques, Applications, and Performance_ A Survey.pdf
│   │   ├── Layer-refined Graph Convolutional Networks for_Recommendation.pdf
│   │   ├── LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.pdf
│   │   ├── Revisiting LightGCN: Unexpected Inflexibility, Inconsistency, and A Remedy Towards Improved Recommendation.pdf
│   │   ├── SIGformer_ Sign-aware Graph Transformer for Recommendation.pdf
│   │   └── TransGNN_ Harnessing the Collaborative Power of Transformers and Graph Neural Networks for Recommender Systems.pdf
│   ├── Losses_Validation-metrics
│   │   ├── A Comprehensive Survey of Evaluation Techniques for Recommendation Systems.pdf
│   │   ├── BPR_ Bayesian Personalized Ranking from Implicit Feedback.pdf
│   │   ├── NeurIPS-2023-estimating-propensity-for-causality-based-recommendation-without-exposure-data-Paper-Conference.pdf
│   │   ├── Scaling Limit of Neural Networks with the Xavier Initialization and Convergence to a Global Minimum.pdf
│   │   └── Towards Representation Alignment and Uniformity in Collaborative Filtering.pdf
│   ├── Performance
│   │   ├── CAGRA_ Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs.pdf
│   │   ├── FULL-GRAPH VS. MINI-BATCH TRAINING_ COMPREHENSIVE ANALYSIS FROM A BATCH SIZE AND FAN- OUT SIZE PERSPECTIVE.pdf
│   │   ├── Optuna_ A Next-generation Hyperparameter Optimization Framework.pdf
│   │   └── PANORAMA_ FAST-TRACK NEAREST NEIGHBORS.pdf
│   └── Recommendation_Systems
│       ├── A comprehensive review of recommender systems_ Transitioning from theory to practice.pdf
│       ├── A Survey of Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions.pdf
│       └── Graph Neural Networks in Recommender Systems: A Survey.pdf
├── pyproject.toml
├── README.md
├── results
│   ├── archive
│   │   └── query_results_20260527.md
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
│   ├── edgrec_config_audit.md
│   ├── experiments.db
│   ├── formal_run_state.json
│   ├── mlflow.db
│   ├── optuna_figures
│   │   ├── optuna_crru_branch_depth_heatmaps.png
│   │   ├── optuna_crru_component_correlations_by_dataset.png
│   │   ├── optuna_crru_components_by_dataset.png
│   │   ├── optuna_crru_fanout_runtime_tradeoffs.png
│   │   ├── optuna_crru_importance_by_dataset.png
│   │   ├── optuna_crru_lr_branchmix_landscape.png
│   │   └── optuna_crru_selection_frontier_by_dataset.png
│   ├── optuna_optimization.md
│   ├── optuna_studies.db
│   ├── query_results.md
│   └── thesis_experiments.db
├── scripts
│   ├── cleanup_experiment_artifacts.py
│   ├── download_pyg_datasets.py
│   ├── export_optuna_figures.py
│   ├── __init__.py
│   ├── prune_checkpoints.py
│   ├── query_results.py
│   ├── quick_validate.py
│   ├── report_optuna_optimization.py
│   ├── reset_experiment_db.py
│   └── _workflow_helpers.py
├── src
│   ├── data
│   │   ├── canonical.py
│   │   ├── feature_policy.py
│   │   ├── graph_builder.py
│   │   ├── __init__.py
│   │   ├── interaction_masks.py
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
│   │   ├── baselines
│   │   │   ├── common.py
│   │   │   ├── dice.py
│   │   │   ├── __init__.py
│   │   │   └── lightgcn.py
│   │   ├── common.py
│   │   ├── edgrec.py
│   │   ├── embeddings.py
│   │   ├── __init__.py
│   │   ├── lightgcn.py
│   │   ├── propensity.py
│   │   └── scoring.py
│   ├── profiling
│   │   ├── gpu_profiler.py
│   │   └── __init__.py
│   ├── training
│   │   ├── evaluator.py
│   │   ├── __init__.py
│   │   └── mini_batch_trainer.py
│   └── utils
│       ├── benchmark_datasets.py
│       ├── cli_parsers.py
│       ├── config.py
│       ├── crru.py
│       ├── csv_features.py
│       ├── dataset_loader_utils.py
│       ├── experiment_logger.py
│       ├── experiment_naming.py
│       ├── __init__.py
│       ├── interaction_indexing.py
│       ├── method_naming.py
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
│   ├── test_prune_checkpoints.py
│   ├── test_search_experiments.py
│   └── test_split_safety.py
└── uv.lock
```
