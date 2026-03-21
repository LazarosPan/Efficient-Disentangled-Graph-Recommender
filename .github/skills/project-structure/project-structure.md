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
│   │   ├── SIGformer_audit.md
│   │   └── U-CaGNN_Synthesis_Report.md
│   ├── guidelines
│   │   ├── env_setup.md
│   │   ├── profile_plan.md
│   │   └── thesis_plan.md
│   ├── notes
│   │   ├── dataset_analysis.md
│   │   ├── dataset_plan.md
│   │   └── manus_research_report.md
│   ├── paper_summaries
│   │   ├── full_summary.md
│   │   ├── lightgcn.md
│   │   ├── notes_by_paper_10.md
│   │   ├── summary_by_paper_10.md
│   │   ├── summary_hybrid_transGNN.md
│   │   ├── summary_per_ai_recommendation.md
│   │   ├── summary_performance_papers.md
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
│   │   ├── run_benchmark.cpython-313.pyc
│   │   └── run_experiment.cpython-313.pyc
│   ├── recipes.py
│   ├── run_ablation.py
│   ├── run_benchmark.py
│   └── run_experiment.py
├── external
│   ├── CaDSI
│   │   ├── CaDSI
│   │   │   ├── CaDSI.py
│   │   │   └── utility
│   │   │       ├── batch_test.py
│   │   │       ├── helper.py
│   │   │       ├── load_data.py
│   │   │       ├── metrics.py
│   │   │       ├── parser.py
│   │   │       ├── __pycache__
│   │   │       │   ├── batch_test.cpython-36.pyc
│   │   │       │   ├── helper.cpython-36.pyc
│   │   │       │   ├── load_data.cpython-36.pyc
│   │   │       │   ├── metrics.cpython-36.pyc
│   │   │       │   └── parser.cpython-36.pyc
│   │   │       └── README.md
│   │   ├── Data
│   │   │   └── Douban_Book
│   │   │       ├── author.txt
│   │   │       ├── item_list.txt
│   │   │       ├── location.txt
│   │   │       ├── publisher.txt
│   │   │       ├── s_adj_mat.npz
│   │   │       ├── s_mean_adj_mat.npz
│   │   │       ├── s_norm_adj_mat.npz
│   │   │       ├── s_pre_adj_mat.npz
│   │   │       ├── test.txt
│   │   │       ├── train.txt
│   │   │       ├── user_list.txt
│   │   │       ├── user.txt
│   │   │       └── year.txt
│   │   └── README.md
│   ├── CausE
│   │   ├── LICENSE.txt
│   │   ├── README.md
│   │   └── src
│   │       ├── causal_prod2vec2i.py
│   │       ├── causal_prod2vec.py
│   │       ├── Data
│   │       │   └── dataset_loading.py
│   │       ├── models.py
│   │       └── utils.py
│   ├── DICE
│   │   ├── data
│   │   │   ├── ml10m.zip
│   │   │   └── netflix.zip
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── src
│   │   │   ├── app.py
│   │   │   ├── candidate_generator.py
│   │   │   ├── config
│   │   │   │   ├── const.py
│   │   │   │   ├── ml10m_cause.cfg
│   │   │   │   ├── ml10m_dice.cfg
│   │   │   │   ├── ml10m_ips.cfg
│   │   │   │   ├── ml10m_lgncause.cfg
│   │   │   │   ├── ml10m_lgn.cfg
│   │   │   │   ├── ml10m_lgndice.cfg
│   │   │   │   ├── ml10m_lgnips.cfg
│   │   │   │   ├── ml10m_mf.cfg
│   │   │   │   ├── nf_cause.cfg
│   │   │   │   ├── nf_dice.cfg
│   │   │   │   ├── nf_ips.cfg
│   │   │   │   ├── nf_lgncause.cfg
│   │   │   │   ├── nf_lgn.cfg
│   │   │   │   ├── nf_lgndice.cfg
│   │   │   │   ├── nf_lgnips.cfg
│   │   │   │   └── nf_mf.cfg
│   │   │   ├── data.py
│   │   │   ├── data_utils
│   │   │   │   ├── __init__.py
│   │   │   │   ├── loader.py
│   │   │   │   ├── sampler.py
│   │   │   │   └── transformer.py
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py
│   │   │   ├── model.py
│   │   │   ├── recommender.py
│   │   │   ├── tester.py
│   │   │   ├── trainer.py
│   │   │   └── utils.py
│   │   └── viz
│   │       ├── data
│   │       │   ├── dice_ml.txt
│   │       │   ├── dice_nf.txt
│   │       │   ├── dice_nopop.txt
│   │       │   ├── dice_pop.txt
│   │       │   ├── ml_popularity.txt
│   │       │   └── nf_popularity.txt
│   │       ├── embedding_viz.m
│   │       └── viz.py
│   ├── FMMRec
│   │   ├── data
│   │   │   ├── microlens
│   │   │   │   ├── audio_feat.npy
│   │   │   │   ├── biased_a_feat_epoch=70.npy
│   │   │   │   ├── biased_t_feat_epoch=40.npy
│   │   │   │   ├── biased_v_feat_epoch=100.npy
│   │   │   │   ├── DRAGON_item_representation.npy
│   │   │   │   ├── DRAGON_user_representation.npy
│   │   │   │   ├── filtered_a_feat_epoch=70.npy
│   │   │   │   ├── filtered_t_feat_epoch=40.npy
│   │   │   │   ├── filtered_v_feat_epoch=100.npy
│   │   │   │   ├── image_feat.npy
│   │   │   │   ├── text_feat.npy
│   │   │   │   └── user_graph_dict.npy
│   │   │   └── ml1m
│   │   │       ├── audio_feat.npy
│   │   │       ├── biased_a_feat_epoch=80.npy
│   │   │       ├── biased_t_feat_epoch=70.npy
│   │   │       ├── biased_v_feat_epoch=90.npy
│   │   │       ├── filtered_a_feat_epoch=80.npy
│   │   │       ├── filtered_t_feat_epoch=70.npy
│   │   │       ├── filtered_v_feat_epoch=90.npy
│   │   │       ├── image_feat.npy
│   │   │       ├── LATTICE_item_representation.npy
│   │   │       ├── LATTICE_user_representation.npy
│   │   │       ├── ml1m.inter
│   │   │       ├── ml1m_u_gender_u_age_u_occupation.attacker.test.tsv
│   │   │       ├── ml1m_u_gender_u_age_u_occupation.attacker.train.tsv
│   │   │       ├── text_feat.npy
│   │   │       └── user_graph_dict.npy
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── scripts
│   │   │   └── ml1m
│   │   │       └── preprocessing
│   │   │           ├── 0rating2inter.py
│   │   │           ├── 1splitting.py
│   │   │           ├── 2reindex-feat.py
│   │   │           ├── 3feat-encoder.py
│   │   │           └── 4sensitive-feat.py
│   │   └── src
│   │       ├── BMMF_filters.py
│   │       ├── BMMF.py
│   │       ├── BMMF_runner.py
│   │       ├── BMMF_trainer.py
│   │       ├── common
│   │       │   ├── abstract_recommender.py
│   │       │   ├── discriminators.py
│   │       │   ├── disc_trainer.py
│   │       │   ├── disc_trainer_random.py
│   │       │   ├── init.py
│   │       │   ├── loss.py
│   │       │   ├── predictors.py
│   │       │   └── trainer.py
│   │       ├── configs
│   │       │   ├── dataset
│   │       │   │   ├── microlens.yaml
│   │       │   │   └── ml1m.yaml
│   │       │   ├── model
│   │       │   │   ├── fairness_tuning
│   │       │   │   │   ├── DRAGON.yaml
│   │       │   │   │   └── LATTICE.yaml
│   │       │   │   └── pretrain
│   │       │   │       ├── DRAGON.yaml
│   │       │   │       └── LATTICE.yaml
│   │       │   └── overall.yaml
│   │       ├── log
│   │       │   ├── DRAGON_BFMMR_k=7_filter=shared_prompt=concat.log
│   │       │   └── LATTICE_BFMMR_k=10_filter=shared_prompt=concat.log
│   │       ├── main.py
│   │       ├── models
│   │       │   ├── fairness_models
│   │       │   │   ├── bfmmr.py
│   │       │   │   └── __pycache__
│   │       │   │       └── bfmmr.cpython-310.pyc
│   │       │   └── recommendation_models
│   │       │       ├── dragon.py
│   │       │       └── lattice.py
│   │       └── utils
│   │           ├── configurator.py
│   │           ├── dataloader.py
│   │           ├── dataset.py
│   │           ├── data_utils.py
│   │           ├── logger.py
│   │           ├── metrics.py
│   │           ├── misc.py
│   │           ├── quick_start.py
│   │           ├── topk_evaluator.py
│   │           └── utils.py
│   ├── MCLN
│   │   ├── Data
│   │   │   └── README.md
│   │   ├── Model-art
│   │   │   ├── load_data.py
│   │   │   ├── model-art.py
│   │   │   └── __pycache__
│   │   │       ├── load_data_addci.cpython-36.pyc
│   │   │       └── load_data.cpython-36.pyc
│   │   ├── Model-beauty
│   │   │   ├── load_data.py
│   │   │   ├── model-beauty.py
│   │   │   └── __pycache__
│   │   │       ├── load_data_addci_2.cpython-36.pyc
│   │   │       ├── load_data_addci_2_three.cpython-36.pyc
│   │   │       ├── load_data_addci.cpython-36.pyc
│   │   │       ├── load_data_addci_int.cpython-36.pyc
│   │   │       └── load_data.cpython-36.pyc
│   │   ├── Model-taobao
│   │   │   ├── load_data.py
│   │   │   ├── model-taobao.py
│   │   │   └── __pycache__
│   │   │       ├── load_data_addci.cpython-36.pyc
│   │   │       └── load_data.cpython-36.pyc
│   │   └── README.md
│   ├── MGCE
│   │   ├── Data
│   │   │   └── README.md
│   │   ├── Model-art
│   │   │   ├── load_data.py
│   │   │   └── model-art.py
│   │   ├── Model-beauty
│   │   │   ├── load_data.py
│   │   │   └── model_beauty.py
│   │   ├── Model-taobao
│   │   │   ├── load_data.py
│   │   │   └── model-taobao.py
│   │   └── README.md
│   └── SIGformer
│       ├── code
│       │   ├── dataloader.py
│       │   ├── main.py
│       │   ├── model.py
│       │   ├── parse.py
│       │   └── utils.py
│       ├── data
│       │   ├── amazon-cds
│       │   │   ├── info.txt
│       │   │   ├── test.txt
│       │   │   ├── train.txt
│       │   │   └── valid.txt
│       │   ├── amazon-music
│       │   │   ├── info.txt
│       │   │   ├── test.txt
│       │   │   ├── train.txt
│       │   │   └── valid.txt
│       │   ├── epinions
│       │   │   ├── info.txt
│       │   │   ├── test.txt
│       │   │   ├── train.txt
│       │   │   └── valid.txt
│       │   ├── KuaiRand
│       │   │   ├── info.txt
│       │   │   ├── test.txt
│       │   │   ├── train.txt
│       │   │   └── valid.txt
│       │   └── KuaiRec
│       │       ├── info.txt
│       │       ├── test.txt
│       │       ├── train.txt
│       │       └── valid.txt
│       ├── README.md
│       └── requirements.txt
├── latex
│   ├── examples
│   │   ├── cover_page_samples
│   │   │   ├── latex_sources
│   │   │   │   ├── cover_compile.sh
│   │   │   │   ├── No_cc_license.tex
│   │   │   │   ├── One_author_one_degree.tex
│   │   │   │   ├── One_author_one_degree_two_departments.tex
│   │   │   │   ├── One_author_two_degrees_from_one_department.tex
│   │   │   │   ├── One_author_two_degrees.tex
│   │   │   │   ├── README_cover_page_sample_sources.txt
│   │   │   │   ├── Two_authors_one_degree.tex
│   │   │   │   └── Two_authors_two_degrees.tex
│   │   │   ├── No_cc_license.pdf
│   │   │   ├── One_author_one_degree.pdf
│   │   │   ├── One_author_one_degree_two_departments.pdf
│   │   │   ├── One_author_two_degrees_from_one_department.pdf
│   │   │   ├── One_author_two_degrees.pdf
│   │   │   ├── Two_authors_one_degree.pdf
│   │   │   └── Two_authors_two_degrees.pdf
│   │   ├── design_examples
│   │   │   ├── latex_sources
│   │   │   │   ├── compile-design-samples.sh
│   │   │   │   ├── MIT-Thesis_libertinus_headings_UA2.tex
│   │   │   │   ├── MIT-Thesis_redsans_headings_UA2.tex
│   │   │   │   ├── mydesign_libertinus_headings.tex
│   │   │   │   ├── mydesign_redsans_headings.tex
│   │   │   │   └── README_design_sample_sources.txt
│   │   │   ├── MIT-Thesis_libertinus_headings_UA2.pdf
│   │   │   └── MIT-Thesis_redsans_headings_UA2.pdf
│   │   └── font_samples
│   │       ├── Defaultfonts_sample.pdf
│   │       ├── Fira_Newtxsf_sample.pdf
│   │       ├── Heros-Stix2_sample.pdf
│   │       ├── latex_sources
│   │       │   ├── compile-font-samples.sh
│   │       │   ├── Defaultfonts_sample.tex
│   │       │   ├── Fira_Newtxsf_sample.tex
│   │       │   ├── Heros-Stix2_sample.tex
│   │       │   ├── Libertinus_sample.tex
│   │       │   ├── Lmodern_sample.tex
│   │       │   ├── Lucida_sample.tex
│   │       │   ├── Newtx_sample.tex
│   │       │   ├── Newtx-sans-text_sample.tex
│   │       │   ├── README_font_sample_sources.txt
│   │       │   ├── README_font_sample_sources-ua2.txt
│   │       │   ├── Stix2_sample.tex
│   │       │   ├── Termes_sample.tex
│   │       │   └── Termes-stix2_sample.tex
│   │       ├── Libertinus_sample.pdf
│   │       ├── Lmodern_sample.pdf
│   │       ├── Lucida_sample.pdf
│   │       ├── Newtx_sample.pdf
│   │       ├── Newtx-sans-text_sample.pdf
│   │       ├── Stix2_sample.pdf
│   │       ├── Termes_sample.pdf
│   │       └── Termes-stix2_sample.pdf
│   ├── mitthesis.cls
│   ├── mitthesis-doc
│   │   ├── mitthesis-doc.pdf
│   │   ├── mitthesis-doc-style.css
│   │   └── mitthesis-doc.tex
│   ├── MIT-Thesis.pdf
│   ├── MIT-thesis-template
│   │   ├── abstract.tex
│   │   ├── acknowledgments.tex
│   │   ├── appendixa.tex
│   │   ├── appendixb.tex
│   │   ├── biography.tex
│   │   ├── build
│   │   │   ├── acknowledgments.aux
│   │   │   ├── appendixa.aux
│   │   │   ├── appendixb.aux
│   │   │   ├── biography.aux
│   │   │   ├── chapter1.aux
│   │   │   ├── fontsets
│   │   │   ├── MIT-Thesis.aux
│   │   │   ├── MIT-Thesis.bbl
│   │   │   ├── MIT-Thesis.bcf
│   │   │   ├── MIT-Thesis.blg
│   │   │   ├── MIT-Thesis.fdb_latexmk
│   │   │   ├── MIT-Thesis.fls
│   │   │   ├── MIT-Thesis.lof
│   │   │   ├── MIT-Thesis.log
│   │   │   ├── MIT-Thesis.lot
│   │   │   ├── MIT-Thesis.pdf
│   │   │   ├── MIT-Thesis.run.xml
│   │   │   ├── MIT-Thesis.synctex.gz
│   │   │   └── MIT-Thesis.toc
│   │   ├── chapter1.tex
│   │   ├── fontsets
│   │   │   ├── mitthesis-defaultfonts.tex
│   │   │   ├── mitthesis-fira-newtxsf.tex
│   │   │   ├── mitthesis-heros-stix2.tex
│   │   │   ├── mitthesis-libertinus.tex
│   │   │   ├── mitthesis-lmodern.tex
│   │   │   ├── mitthesis-lucida.tex
│   │   │   ├── mitthesis-newtx-sans-text.tex
│   │   │   ├── mitthesis-newtx.tex
│   │   │   ├── mitthesis-stix2.tex
│   │   │   ├── mitthesis-termes-stix2.tex
│   │   │   └── mitthesis-termes.tex
│   │   ├── mitthesis.cls
│   │   ├── mitthesis-sample.bib
│   │   ├── mitthesis-style.css
│   │   ├── MIT-Thesis.tex
│   │   └── mydesign.tex
│   └── README.md
├── LICENCE
├── main.py
├── mlflow.db
├── mlruns
│   ├── 1
│   │   ├── 34b43a8804ca45279a79e329a5c4c715
│   │   │   └── artifacts
│   │   │       └── movielens1m_full_mini_batch_knn_ep2_bs2048_dim64_layers2_nbr10-10_sample20000_preflight_seed13.pt
│   │   ├── dc637e1794ae4dbaaeeaf9a7d17aa8fe
│   │   │   └── artifacts
│   │   │       └── movielens1m_full_full_graph_dense_ep2_bs2048_dim64_layers2_sample20000_preflight_seed13.pt
│   │   └── f3fde9080a8645a3a4931deb41de6a37
│   │       └── artifacts
│   │           └── movielens1m_full_cached_propagation_cagra_ep2_bs2048_dim64_layers2_sample20000_preflight_seed13.pt
│   ├── 2
│   │   ├── aaae2f01f03d4003a14dbeb8b0e2d4fe
│   │   │   └── artifacts
│   │   │       └── movielens1m_full_full_graph_dense_ep5_bs2048_dim64_layers2_seed13.pt
│   │   ├── b2ba57fe30364183bca6fa597a4a2a41
│   │   │   └── artifacts
│   │   │       └── preflight_amazonbook_full_full_graph_dense.pt
│   │   ├── c4c24207348d4e35b197160c74497d7b
│   │   │   └── artifacts
│   │   │       └── preflight_amazonbook_full_mini_batch_knn.pt
│   │   └── e074c6ed52994e91a87b7415cf8d216d
│   │       └── artifacts
│   │           └── preflight_amazonbook_full_cached_propagation_cagra.pt
│   └── 3
│       ├── 4332b1f61b0e4fc394735eef3ee7dd2b
│       │   └── artifacts
│       │       └── amazonbook_full_mini_batch_knn_ep1_bs128_dim64_layers2_nbr10-10_sample3600_preflight_seed13.pt
│       ├── bea83f8f93f54ee4828b33848514942d
│       │   └── artifacts
│       │       └── amazonbook_full_full_graph_dense_ep1_bs128_dim64_layers2_sample3600_preflight_seed13.pt
│       └── c9a3b9491d9a48308650a45520441fa5
│           └── artifacts
│               └── amazonbook_full_cached_propagation_cagra_ep1_bs128_dim64_layers2_sample3600_preflight_seed13.pt
├── pyproject.toml
├── README.md
├── results
│   ├── checkpoints
│   │   ├── amazonbook_full_cached_propagation_cagra_ep1_bs128_dim64_layers2_sample3600_preflight_seed13.pt
│   │   ├── amazonbook_full_full_graph_dense_ep1_bs128_dim64_layers2_sample3600_preflight_seed13.pt
│   │   ├── amazonbook_full_mini_batch_knn_ep1_bs128_dim64_layers2_nbr10-10_sample3600_preflight_seed13.pt
│   │   ├── movielens1m_full_cached_propagation_cagra_ep2_bs2048_dim64_layers2_sample20000_preflight_seed13.pt
│   │   ├── movielens1m_full_full_graph_dense_ep2_bs2048_dim64_layers2_sample20000_preflight_seed13.pt
│   │   ├── movielens1m_full_full_graph_dense_ep5_bs2048_dim64_layers2_seed13.pt
│   │   ├── movielens1m_full_mini_batch_knn_ep2_bs2048_dim64_layers2_nbr10-10_sample20000_preflight_seed13.pt
│   │   ├── preflight_amazonbook_full_cached_propagation_cagra.pt
│   │   ├── preflight_amazonbook_full_full_graph_dense.pt
│   │   ├── preflight_amazonbook_full_mini_batch_knn.pt
│   │   ├── preflight_movielens1m_full_cached_propagation_cagra.pt
│   │   ├── preflight_movielens1m_full_full_graph_dense.pt
│   │   └── preflight_movielens1m_full_mini_batch_knn.pt
│   ├── figures
│   ├── mlflow.db
│   ├── thesis_experiments.db
│   ├── thesis_experiments.db-shm
│   └── thesis_experiments.db-wal
├── scripts
│   ├── download_pyg_datasets.py
│   ├── __init__.py
│   ├── preflight_experiments.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   ├── preflight_experiments.cpython-313.pyc
│   │   ├── verify_pipeline.cpython-313.pyc
│   │   ├── verify_setup.cpython-313.pyc
│   │   └── verify_setup.cpython-314.pyc
│   ├── query_results.py
│   ├── reset_experiment_db.py
│   ├── verify_pipeline.py
│   ├── verify_setup.py
│   ├── verify_sqlite.py
│   └── visualize_results.py
├── src
│   ├── baselines
│   ├── data
│   │   ├── canonical.py
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
│   ├── interventions
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
│       ├── experiment_logger.py
│       ├── __init__.py
│       └── __pycache__
│           ├── config.cpython-310.pyc
│           ├── config.cpython-313.pyc
│           ├── experiment_logger.cpython-313.pyc
│           ├── __init__.cpython-310.pyc
│           └── __init__.cpython-313.pyc
└── uv.lock
```