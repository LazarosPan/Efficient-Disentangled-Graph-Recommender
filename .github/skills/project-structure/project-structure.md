# Run: `tree -I 'latex|.venv|external'`

```
├── causal_embeddings_for_recommendations.egg-info
│   ├── dependency_links.txt
│   ├── entry_points.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── data
│   ├── AmazonBook
│   │   ├── processed
│   │   └── raw
│   │       ├── item_list.txt
│   │       ├── test.txt
│   │       ├── train.txt
│   │       └── user_list.txt
│   ├── AmazonCDs
│   │   └── raw
│   │       ├── info.txt
│   │       ├── test.txt
│   │       ├── train.txt
│   │       └── valid.txt
│   ├── AmazonMusic
│   │   └── raw
│   │       ├── info.txt
│   │       ├── test.txt
│   │       ├── train.txt
│   │       └── valid.txt
│   ├── AmazonProducts
│   │   └── raw
│   │       ├── adj_full.npz
│   │       ├── class_map.json
│   │       ├── feats.npy
│   │       └── role.json
│   ├── datasets_feature_audit.json
│   ├── datasets_information.md
│   ├── Douban
│   │   └── raw
│   │       └── training_test_dataset.mat
│   ├── Douban_Book
│   │   ├── author.txt
│   │   ├── item_list.txt
│   │   ├── location.txt
│   │   ├── publisher.txt
│   │   ├── s_adj_mat.npz
│   │   ├── s_mean_adj_mat.npz
│   │   ├── s_norm_adj_mat.npz
│   │   ├── s_pre_adj_mat.npz
│   │   ├── test.txt
│   │   ├── train.txt
│   │   ├── user_list.txt
│   │   ├── user.txt
│   │   └── year.txt
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
│   ├── KuaiSAR_v2
│   │   ├── item_features.csv
│   │   ├── README.md
│   │   ├── rec_inter.csv
│   │   ├── social_network.csv
│   │   ├── src_inter.csv
│   │   └── user_features.csv
│   ├── MovieLens
│   │   └── raw
│   │       └── ml-latest-small
│   │           ├── links.csv
│   │           ├── movies.csv
│   │           ├── ratings.csv
│   │           ├── README.txt
│   │           └── tags.csv
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
│   ├── netflix
│   │   └── raw
│   │       └── output
│   │           ├── coo_record.npz
│   │           ├── item_reindex.json
│   │           ├── popularity_all.npy
│   │           ├── popularity_blend.npy
│   │           ├── popularity.npy
│   │           ├── popularity_skew.npy
│   │           ├── record.csv
│   │           ├── test_coo_record.npz
│   │           ├── test_record.csv
│   │           ├── train_blend_coo_adj_graph.npz
│   │           ├── train_coo_adj_graph.npz
│   │           ├── train_coo_record.npz
│   │           ├── train_record.csv
│   │           ├── train_skew_coo_adj_graph.npz
│   │           ├── train_skew_coo_record.npz
│   │           ├── train_skew_record.csv
│   │           ├── user_reindex.json
│   │           ├── val_coo_record.npz
│   │           └── val_record.csv
│   ├── Taobao
│   │   └── raw
│   │       ├── README.md
│   │       ├── UserBehavior.csv
│   │       └── UserBehavior.csv.zip.md5
│   └── Yelp
│       └── raw
│           ├── adj_full.npz
│           ├── class_map.json
│           ├── feats.npy
│           └── role.json
├── docs
│   ├── existing_implementations
│   │   ├── CaDSI_audit.md
│   │   ├── CausE_audit.md
│   │   ├── DICE_audit.md
│   │   ├── FMMRec_audit.md
│   │   ├── MCLN_audit.md
│   │   ├── MGCE_audit.md
│   │   ├── PropCare_audit.md
│   │   ├── SIGformer_audit.md
│   │   └── U-CaGNN_Synthesis_Report.md
│   ├── guidelines
│   │   ├── env_setup.md
│   │   ├── profile_plan.md
│   │   └── thesis_plan.md
│   ├── notes
│   │   ├── manus_research_report.md
│   │   ├── progress_ideas.md
│   │   └── useful_commands.md
│   ├── paper_summaries
│   │   ├── full_summary.md
│   │   ├── lightgcn.md
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
│   │   └── training.md
│   └── usage
│       ├── experiments.md
│       └── scripts.md
├── experiments
│   ├── ablation_configs.py
│   ├── experiment_catalog.json
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── ablation_configs.cpython-313.pyc
│   │   ├── __init__.cpython-313.pyc
│   │   ├── recipes.cpython-313.pyc
│   │   ├── run_ablation.cpython-313.pyc
│   │   ├── run_benchmark.cpython-313.pyc
│   │   └── run_experiment.cpython-313.pyc
│   ├── recipes.py
│   ├── run_ablation.py
│   ├── run_benchmark.py
│   └── run_experiment.py
├── LICENCE
├── main.py
├── pyproject.toml
├── README.md
├── results
│   └── feature_policy_probes.json
├── scripts
│   ├── audit_metrics.py
│   ├── cleanup_experiment_artifacts.py
│   ├── download_pyg_datasets.py
│   ├── evaluate_scoring_modes.py
│   ├── feature_policy_probes.py
│   ├── __init__.py
│   ├── list_commands.py
│   ├── preflight_experiments.py
│   ├── __pycache__
│   │   ├── audit_metrics.cpython-313.pyc
│   │   ├── cleanup_experiment_artifacts.cpython-313.pyc
│   │   ├── download_pyg_datasets.cpython-313.pyc
│   │   ├── feature_policy_probes.cpython-313.pyc
│   │   ├── __init__.cpython-313.pyc
│   │   ├── list_commands.cpython-313.pyc
│   │   ├── preflight_experiments.cpython-313.pyc
│   │   ├── query_results.cpython-313.pyc
│   │   ├── quick_validate.cpython-313.pyc
│   │   ├── reset_experiment_db.cpython-313.pyc
│   │   ├── verify_pipeline.cpython-313.pyc
│   │   ├── verify_setup.cpython-313.pyc
│   │   ├── verify_setup.cpython-314.pyc
│   │   └── verify_sqlite.cpython-313.pyc
│   ├── query_results.py
│   ├── quick_validate.py
│   ├── reset_experiment_db.py
│   ├── verify_pipeline.py
│   ├── verify_setup.py
│   ├── verify_sqlite.py
│   └── _workflow_helpers.py
├── src
│   ├── data
│   │   ├── canonical.py
│   │   ├── feature_policy.py
│   │   ├── graph_builder.py
│   │   ├── __init__.py
│   │   ├── loaders
│   │   │   ├── amazonbook.py
│   │   │   ├── __init__.py
│   │   │   ├── kuairand1k.py
│   │   │   ├── kuairec_v2.py
│   │   │   ├── movielens1m.py
│   │   │   ├── movielens20m.py
│   │   │   ├── __pycache__
│   │   │   │   ├── amazonbook.cpython-313.pyc
│   │   │   │   ├── _feature_utils.cpython-313.pyc
│   │   │   │   ├── __init__.cpython-313.pyc
│   │   │   │   ├── kuairand1k.cpython-310.pyc
│   │   │   │   ├── kuairand1k.cpython-313.pyc
│   │   │   │   ├── kuairec_v2.cpython-310.pyc
│   │   │   │   ├── kuairec_v2.cpython-313.pyc
│   │   │   │   ├── movielens1m.cpython-310.pyc
│   │   │   │   ├── movielens1m.cpython-313.pyc
│   │   │   │   ├── movielens20m.cpython-310.pyc
│   │   │   │   ├── movielens20m.cpython-313.pyc
│   │   │   │   └── taobao.cpython-313.pyc
│   │   │   └── taobao.py
│   │   ├── negative_sampler.py
│   │   ├── __pycache__
│   │   │   ├── canonical.cpython-310.pyc
│   │   │   ├── canonical.cpython-313.pyc
│   │   │   ├── feature_policy.cpython-313.pyc
│   │   │   ├── graph_builder.cpython-310.pyc
│   │   │   ├── graph_builder.cpython-313.pyc
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── __init__.cpython-313.pyc
│   │   │   ├── negative_sampler.cpython-313.pyc
│   │   │   └── subgraph_sampler.cpython-313.pyc
│   │   └── subgraph_sampler.py
│   ├── data_exploration
│   │   ├── data_exploration.ipynb
│   │   ├── data_exploration.py
│   │   ├── data_information.py
│   │   ├── explore_all_datasets.py
│   │   └── __pycache__
│   │       ├── data_exploration.cpython-313.pyc
│   │       ├── data_information.cpython-313.pyc
│   │       └── explore_all_datasets.cpython-313.pyc
│   ├── evaluation
│   │   └── __init__.py
│   ├── __init__.py
│   ├── losses
│   │   ├── bpr.py
│   │   ├── contrastive.py
│   │   ├── counterfactual.py
│   │   ├── __init__.py
│   │   ├── loss_suite.py
│   │   ├── orthogonality.py
│   │   ├── popularity.py
│   │   └── __pycache__
│   │       ├── bpr.cpython-313.pyc
│   │       ├── contrastive.cpython-313.pyc
│   │       ├── counterfactual.cpython-313.pyc
│   │       ├── __init__.cpython-313.pyc
│   │       ├── loss_suite.cpython-313.pyc
│   │       ├── orthogonality.cpython-313.pyc
│   │       └── popularity.cpython-313.pyc
│   ├── models
│   │   ├── embeddings.py
│   │   ├── __init__.py
│   │   ├── lightgcn.py
│   │   ├── propensity.py
│   │   ├── __pycache__
│   │   │   ├── embeddings.cpython-310.pyc
│   │   │   ├── embeddings.cpython-313.pyc
│   │   │   ├── __init__.cpython-313.pyc
│   │   │   ├── lightgcn.cpython-310.pyc
│   │   │   ├── lightgcn.cpython-313.pyc
│   │   │   ├── propensity.cpython-313.pyc
│   │   │   ├── scoring.cpython-313.pyc
│   │   │   └── ucagnn.cpython-313.pyc
│   │   ├── scoring.py
│   │   └── ucagnn.py
│   ├── profiling
│   │   ├── gpu_profiler.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── gpu_profiler.cpython-313.pyc
│   │       └── __init__.cpython-313.pyc
│   ├── __pycache__
│   │   ├── feature_policy.cpython-313.pyc
│   │   └── __init__.cpython-313.pyc
│   ├── training
│   │   ├── cached_trainer.py
│   │   ├── evaluator.py
│   │   ├── __init__.py
│   │   ├── mini_batch_trainer.py
│   │   ├── __pycache__
│   │   │   ├── cached_trainer.cpython-313.pyc
│   │   │   ├── evaluator.cpython-313.pyc
│   │   │   ├── __init__.cpython-313.pyc
│   │   │   ├── mini_batch_trainer.cpython-313.pyc
│   │   │   └── trainer.cpython-313.pyc
│   │   └── trainer.py
│   └── utils
│       ├── config.py
│       ├── csv_features.py
│       ├── experiment_logger.py
│       ├── __init__.py
│       ├── interaction_indexing.py
│       └── __pycache__
│           ├── config.cpython-310.pyc
│           ├── config.cpython-313.pyc
│           ├── csv_features.cpython-313.pyc
│           ├── experiment_logger.cpython-313.pyc
│           ├── __init__.cpython-310.pyc
│           ├── __init__.cpython-313.pyc
│           └── interaction_indexing.cpython-313.pyc
└── uv.lock
```