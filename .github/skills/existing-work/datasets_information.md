# Dataset Information Report

- Generated at: 2026-05-05T23:02:52+00:00
- Scan root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data`
- Selected datasets: AmazonBook, KuaiRand-1K, KuaiRec_v2, MovieLens20M, Taobao, MovieLens1M
- Datasets scanned: 6
- Files scanned: 44
- Total size: 10,792,435,396 B (10,292.47 MiB | 10.05 GiB)

## Dataset Summary

| Dataset | Files | Total Size |
| --- | ---: | ---: |
| KuaiRand-1K | 12 | 4,404.30 MiB |
| Taobao | 3 | 3,502.23 MiB |
| KuaiRec_v2 | 14 | 1,507.51 MiB |
| MovieLens20M | 7 | 835.03 MiB |
| MovieLens1M | 4 | 23.75 MiB |
| AmazonBook | 4 | 19.65 MiB |

## KuaiRand-1K

- Files: 12
- Total size: 4,618,247,460 B (4,404.30 MiB | 4.30 GiB)

| Path | Extension | Size (MiB) | Column Names & DTypes |
| --- | --- | ---: | --- |
| data/log_random_4_22_to_5_08_1k.csv | .csv | 2.94 MiB | user_id:Int64, video_id:Int64, date:Int64, hourmin:Int64, time_ms:Int64, is_click:Int64, is_like:Int64, is_follow:Int64, is_comment:Int64, is_forward:Int64, is_hate:Int64, long_view:Int64, play_time_ms:Int64, duration_ms:Int64, profile_stay_time:Int64, comment_stay_time:Int64, is_profile_enter:Int64, is_rand:Int64, tab:Int64 |
| data/log_standard_4_08_to_4_21_1k.csv | .csv | 356.36 MiB | user_id:Int64, video_id:Int64, date:Int64, hourmin:Int64, time_ms:Int64, is_click:Int64, is_like:Int64, is_follow:Int64, is_comment:Int64, is_forward:Int64, is_hate:Int64, long_view:Int64, play_time_ms:Int64, duration_ms:Int64, profile_stay_time:Int64, comment_stay_time:Int64, is_profile_enter:Int64, is_rand:Int64, tab:Int64 |
| data/log_standard_4_22_to_5_08_1k.csv | .csv | 469.50 MiB | user_id:Int64, video_id:Int64, date:Int64, hourmin:Int64, time_ms:Int64, is_click:Int64, is_like:Int64, is_follow:Int64, is_comment:Int64, is_forward:Int64, is_hate:Int64, long_view:Int64, play_time_ms:Int64, duration_ms:Int64, profile_stay_time:Int64, comment_stay_time:Int64, is_profile_enter:Int64, is_rand:Int64, tab:Int64 |
| data/user_features_1k.csv | .csv | 0.12 MiB | user_id:Int64, user_active_degree:String, is_lowactive_period:Int64, is_live_streamer:Int64, is_video_author:Int64, follow_user_num:Int64, follow_user_num_range:String, fans_user_num:Int64, fans_user_num_range:String, friend_user_num:Int64, friend_user_num_range:String, register_days:Int64, register_days_range:String, onehot_feat0:Int64, onehot_feat1:Int64, onehot_feat2:Int64, onehot_feat3:Int64, onehot_feat4:Float64, onehot_feat5:Int64, onehot_feat6:Int64, onehot_feat7:Int64, onehot_feat8:Int64, onehot_feat9:Int64, onehot_feat10:Int64, onehot_feat11:Int64, onehot_feat12:Float64, onehot_feat13:Float64, onehot_feat14:Float64, onehot_feat15:Float64, onehot_feat16:Float64, onehot_feat17:Float64 |
| data/video_features_basic_1k.csv | .csv | 359.43 MiB | video_id:Int64, author_id:Int64, video_type:String, upload_dt:String, upload_type:String, visible_status:Float64, video_duration:Float64, server_width:Float64, server_height:Float64, music_id:Int64, music_type:Float64, tag:String |
| data/video_features_statistic_1k.csv | .csv | 3,214.48 MiB | video_id:Int64, counts:Int64, show_cnt:Float64, show_user_num:Float64, play_cnt:Float64, play_user_num:Float64, play_duration:Float64, complete_play_cnt:Float64, complete_play_user_num:Float64, valid_play_cnt:Float64, valid_play_user_num:Float64, long_time_play_cnt:Float64, long_time_play_user_num:Float64, short_time_play_cnt:Float64, short_time_play_user_num:Float64, play_progress:Float64, comment_stay_duration:Float64, like_cnt:Float64, like_user_num:Float64, click_like_cnt:Float64, double_click_cnt:Float64, cancel_like_cnt:Float64, cancel_like_user_num:Float64, comment_cnt:Float64, comment_user_num:Float64, direct_comment_cnt:Float64, reply_comment_cnt:Float64, delete_comment_cnt:Float64, delete_comment_user_num:Float64, comment_like_cnt:Float64, comment_like_user_num:Float64, follow_cnt:Float64, follow_user_num:Float64, cancel_follow_cnt:Float64, cancel_follow_user_num:Float64, share_cnt:Float64, share_user_num:Float64, download_cnt:Float64, download_user_num:Float64, report_cnt:Float64, report_user_num:Float64, reduce_similar_cnt:Float64, reduce_similar_user_num:Float64, collect_cnt:Float64, collect_user_num:Float64, cancel_collect_cnt:Float64, cancel_collect_user_num:Float64, direct_comment_user_num:Float64, reply_comment_user_num:Float64, share_all_cnt:Float64, share_all_user_num:Float64, outsite_share_all_cnt:Float64 |
| figs/KuaiRand.png | .png | 0.09 MiB | - |
| figs/kuaishou-app.png | .png | 1.03 MiB | - |
| figs/three-version.png | .png | 0.31 MiB | - |
| LICENSE | <no_ext> | 0.02 MiB | - |
| load_data_1k.py | .py | 0.00 MiB | - |
| README.md | .md | 0.02 MiB | - |

## Taobao

- Files: 3
- Total size: 3,672,350,226 B (3,502.23 MiB | 3.42 GiB)

| Path | Extension | Size (MiB) | Column Names & DTypes |
| --- | --- | ---: | --- |
| raw/README.md | .md | 0.00 MiB | - |
| raw/UserBehavior.csv | .csv | 3,502.22 MiB | user_id:Int64, item_id:Int64, category_id:Int64, behavior_type:String, timestamp:Int64 |
| raw/UserBehavior.csv.zip.md5 | .md5 | 0.00 MiB | - |

## KuaiRec_v2

- Files: 14
- Total size: 1,580,742,417 B (1,507.51 MiB | 1.47 GiB)

| Path | Extension | Size (MiB) | Column Names & DTypes |
| --- | --- | ---: | --- |
| data/big_matrix.csv | .csv | 1,033.33 MiB | user_id:Int64, video_id:Int64, play_duration:Int64, video_duration:Int64, time:String, date:Int64, timestamp:Float64, watch_ratio:Float64 |
| data/item_categories.csv | .csv | 0.11 MiB | video_id:Int64, feat:String |
| data/item_daily_features.csv | .csv | 81.88 MiB | video_id:Int64, date:Int64, author_id:Int64, video_type:String, upload_dt:String, upload_type:String, visible_status:String, video_duration:Float64, video_width:Int64, video_height:Int64, music_id:Int64, video_tag_id:Int64, video_tag_name:String, show_cnt:Int64, show_user_num:Int64, play_cnt:Int64, play_user_num:Int64, play_duration:Int64, complete_play_cnt:Int64, complete_play_user_num:Int64, valid_play_cnt:Int64, valid_play_user_num:Int64, long_time_play_cnt:Int64, long_time_play_user_num:Int64, short_time_play_cnt:Int64, short_time_play_user_num:Int64, play_progress:Float64, comment_stay_duration:Int64, like_cnt:Int64, like_user_num:Int64, click_like_cnt:Int64, double_click_cnt:Int64, cancel_like_cnt:Int64, cancel_like_user_num:Int64, comment_cnt:Int64, comment_user_num:Int64, direct_comment_cnt:Int64, reply_comment_cnt:Int64, delete_comment_cnt:Int64, delete_comment_user_num:Int64, comment_like_cnt:Int64, comment_like_user_num:Int64, follow_cnt:Int64, follow_user_num:Int64, cancel_follow_cnt:Int64, cancel_follow_user_num:Int64, share_cnt:Int64, share_user_num:Int64, download_cnt:Int64, download_user_num:Int64, report_cnt:Int64, report_user_num:Int64, reduce_similar_cnt:Int64, reduce_similar_user_num:Int64, collect_cnt:Float64, collect_user_num:Float64, cancel_collect_cnt:Float64, cancel_collect_user_num:Float64 |
| data/kuairec_caption_category.csv | .csv | 1.87 MiB | video_id:Int64, manual_cover_text:String, caption:String, topic_tag:String, first_level_category_id:Int64, first_level_category_name:String, second_level_category_id:Int64, second_level_category_name:String, third_level_category_id:Int64, third_level_category_name:String |
| data/README.md | .md | 0.02 MiB | - |
| data/small_matrix.csv | .csv | 387.34 MiB | user_id:Int64, video_id:Int64, play_duration:Int64, video_duration:Int64, time:String, date:Float64, timestamp:Float64, watch_ratio:Float64 |
| data/social_network.csv | .csv | 0.01 MiB | user_id:Int64, friend_list:String |
| data/user_features.csv | .csv | 0.71 MiB | user_id:Int64, user_active_degree:String, is_lowactive_period:Int64, is_live_streamer:Int64, is_video_author:Int64, follow_user_num:Int64, follow_user_num_range:String, fans_user_num:Int64, fans_user_num_range:String, friend_user_num:Int64, friend_user_num_range:String, register_days:Int64, register_days_range:String, onehot_feat0:Int64, onehot_feat1:Int64, onehot_feat2:Int64, onehot_feat3:Int64, onehot_feat4:Int64, onehot_feat5:Int64, onehot_feat6:Int64, onehot_feat7:Int64, onehot_feat8:Int64, onehot_feat9:Int64, onehot_feat10:Int64, onehot_feat11:Int64, onehot_feat12:Int64, onehot_feat13:Int64, onehot_feat14:Int64, onehot_feat15:Int64, onehot_feat16:Int64, onehot_feat17:Int64 |
| data/video_raw_categories_multi.csv | .csv | 1.64 MiB | video_id:Int64, category_name:String, category_id:Int64, category_level:Int64, prob:Float64, source:Int64, upload_date:Int64, category_online:Int64, root_id:Int64, root_name:String, parent_id:Int64, parent_name:String, type:Int64 |
| figs/colab-badge.svg | .svg | 0.00 MiB | - |
| figs/KuaiRec.png | .png | 0.29 MiB | - |
| LICENSE | <no_ext> | 0.02 MiB | - |
| loaddata.py | .py | 0.00 MiB | - |
| Statistics_KuaiRec.ipynb | .ipynb | 0.30 MiB | - |

## MovieLens20M

- Files: 7
- Total size: 875,588,784 B (835.03 MiB | 0.82 GiB)

| Path | Extension | Size (MiB) | Column Names & DTypes |
| --- | --- | ---: | --- |
| raw/genome-scores.csv | .csv | 308.56 MiB | movieId:Int64, tagId:Int64, relevance:Float64 |
| raw/genome-tags.csv | .csv | 0.02 MiB | tagId:Int64, tag:String |
| raw/links.csv | .csv | 0.54 MiB | movieId:Int64, imdbId:Int64, tmdbId:Int64 |
| raw/movies.csv | .csv | 1.33 MiB | movieId:Int64, title:String, genres:String |
| raw/ratings.csv | .csv | 508.73 MiB | userId:Int64, movieId:Int64, rating:Float64, timestamp:Int64 |
| raw/README.md | .md | 0.01 MiB | - |
| raw/tags.csv | .csv | 15.83 MiB | userId:Int64, movieId:Int64, tag:String, timestamp:Int64 |

## MovieLens1M

- Files: 4
- Total size: 24,905,384 B (23.75 MiB | 0.02 GiB)

| Path | Extension | Size (MiB) | Column Names & DTypes |
| --- | --- | ---: | --- |
| raw/movies.dat | .dat | 0.16 MiB | movie_id:Int64, title:String, genres:String |
| raw/ratings.dat | .dat | 23.45 MiB | user_id:Int64, movie_id:Int64, rating:Float64, timestamp:Int64 |
| raw/README.md | .md | 0.01 MiB | - |
| raw/users.dat | .dat | 0.13 MiB | user_id:Int64, gender:String, age:Int64, occupation:Int64, zip_code:String |

## AmazonBook

- Files: 4
- Total size: 20,601,125 B (19.65 MiB | 0.02 GiB)

| Path | Extension | Size (MiB) | Column Names & DTypes |
| --- | --- | ---: | --- |
| raw/item_list.txt | .txt | 1.47 MiB | org_id:String, remap_id:Int64 |
| raw/test.txt | .txt | 3.67 MiB | user_id:Int64, item_ids...:list[Int64] |
| raw/train.txt | .txt | 13.47 MiB | user_id:Int64, item_ids...:list[Int64] |
| raw/user_list.txt | .txt | 1.03 MiB | org_id:String, remap_id:Int64 |
