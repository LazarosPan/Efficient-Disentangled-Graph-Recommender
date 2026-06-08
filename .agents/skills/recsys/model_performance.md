Measuring Model Performance (https://apxml.com/courses/building-ml-recommendation-system/chapter-5-evaluating-recommendation-systems/practice-measuring-model-performance)

We will implement main offline evaluation metrics. A dataset is loaded, and two different models—a neighborhood-based KNNBasic and a matrix factorization-based SVD—are trained. Their performance is then systematically measured. This direct comparison highlights how different models excel at different tasks and why a single metric rarely tells the whole story.

Our goal is to create a reusable evaluation workflow that you can adapt for your own projects. We will move from calculating simple prediction accuracy to implementing the more involved but highly informative ranking metrics from scratch.
Setting Up the Environment and Data

First, we need to import the necessary tools and prepare our dataset. We will use the surprise library, which simplifies loading data, splitting it for evaluation, and training standard recommender algorithms. We'll work with the widely-used MovieLens 100k dataset.

```python
import pandas as pd
from collections import defaultdict

from surprise import Dataset, Reader
from surprise import SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Split the data into a training set and a test set.
# test_size=0.25 means we'll use 25% of the data for testing.
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
```

With our data split, we have a trainset for model training and a testset containing user-item interactions that the model has not seen. We will use this testset as our ground truth to evaluate the models' predictions.
Measuring Prediction Accuracy: RMSE and MAE

We'll start with the most straightforward evaluation task: measuring how accurately our models predict explicit ratings. This is useful in systems where showing an accurate star rating is important. Let's train both an SVD and a user-based k-NN model and see how they perform.

```python
# --- Train and Evaluate SVD Model ---
svd_model = SVD(random_state=42)
svd_model.fit(trainset)
svd_predictions = svd_model.test(testset)

# Calculate RMSE and MAE for SVD
print("SVD Model Performance:")
accuracy.rmse(svd_predictions)
accuracy.mae(svd_predictions)

# --- Train and Evaluate k-NN Model ---
# Using user-based collaborative filtering
knn_model = KNNBasic(sim_options={'user_based': True})
knn_model.fit(trainset)
knn_predictions = knn_model.test(testset)

# Calculate RMSE and MAE for k-NN
print("\nk-NN Model Performance:")
accuracy.rmse(knn_predictions)
accuracy.mae(knn_predictions)
```

Running this code will produce output similar to this:

```
SVD Model Performance:
RMSE: 0.9348
MAE:  0.7371

k-NN Model Performance:
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.9791
MAE:  0.7725
```

From these results, the SVD model achieves a lower Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) than the k-NN model. This indicates that SVD is better at predicting the exact rating a user would give to a movie. An RMSE of 0.93 means its predictions are, on average, off by just under one star on the 1-to-5 rating scale.
Evaluating Ranking Quality

While prediction accuracy is useful, most modern recommenders are judged by their ability to generate a good ranked list of items. For this, we need ranking metrics. We will implement functions to calculate Precision@k and Recall@k, which measure the relevance of the top-kk recommendations.

Our process will be:

    For each user, get the top kk recommended items from the model.
    Identify the set of items the user actually liked in the test set (our "ground truth"). We'll define "liked" as having a rating of 4.0 or higher.
    Compare the recommended set with the ground truth set to see how many recommendations were relevant.

Here is a function that automates this for any set of predictions from the surprise library.

```python
def calculate_precision_recall_at_k(predictions, k=10, rating_threshold=4.0):
    """
    Return precision and recall at k for each user.
    """
    # First, map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value in descending order
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items in top k
        n_rel = sum((true_r >= rating_threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= rating_threshold) for (est, _) in user_ratings[:k])

        # Number of relevant AND recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= rating_threshold) and (est >= rating_threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@k: Proporation of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@k: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    # Average across all users
    avg_precision = sum(p for p in precisions.values()) / len(precisions)
    avg_recall = sum(r for r in recalls.values()) / len(recalls)

    return avg_precision, avg_recall
```

Now, let's apply this function to the predictions generated by our SVD and k-NN models. We'll set k=10k=10 to evaluate the top 10 recommendations.

```python
# Calculate Precision and Recall for SVD
svd_precision, svd_recall = calculate_precision_recall_at_k(svd_predictions, k=10)
print(f"SVD Precision@10: {svd_precision:.4f}")
print(f"SVD Recall@10: {svd_recall:.4f}")

# Calculate Precision and Recall for k-NN
knn_precision, knn_recall = calculate_precision_recall_at_k(knn_predictions, k=10)
print(f"\nk-NN Precision@10: {knn_precision:.4f}")
print(f"k-NN Recall@10: {knn_recall:.4f}")
```

This might produce the following output:

```
SVD Precision@10: 0.8384
SVD Recall@10: 0.5891

k-NN Precision@10: 0.8421
k-NN Recall@10: 0.5301
```

These results tell a different story. The k-NN model has slightly higher precision, meaning that when it recommends an item in the top 10, it's slightly more likely to be an item the user actually likes. However, the SVD model has a significantly higher recall, suggesting it is better at finding a larger proportion of the total set of items the user would like.
Measuring Ranking Order with NDCG

Precision and recall treat all items in the top-kk list equally. Normalized Discounted Cumulative Gain (NDCG) improves on this by giving more credit to relevant items that appear higher up in the recommendation list.

Let's write a function to calculate NDCG. The logic involves computing the Discounted Cumulative Gain (DCG) of the model's recommended list and normalizing it by the Ideal DCG (IDCG), which represents the best possible ranking.

```python
import numpy as np

def calculate_ndcg_at_k(predictions, k=10, rating_threshold=4.0):
    """
    Return the average NDCG at k.
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    ndcgs = []
    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Get relevance scores for the top k recommended items
        relevance_scores = [(1 if true_r >= rating_threshold else 0) for (_, true_r) in user_ratings[:k]]

        # Calculate DCG for the model's ranking
        dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores)])

        # Create the ideal ranking for IDCG calculation
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance)])

        if idcg == 0:
            continue # Skip users with no relevant items in the test set

        ndcgs.append(dcg / idcg)

    return np.mean(ndcgs)
```

Now, let's compute the NDCG for both models.

```python
# Calculate NDCG for SVD
svd_ndcg = calculate_ndcg_at_k(svd_predictions, k=10)
print(f"SVD NDCG@10: {svd_ndcg:.4f}")

# Calculate NDCG for k-NN
knn_ndcg = calculate_ndcg_at_k(knn_predictions, k=10)
print(f"k-NN NDCG@10: {knn_ndcg:.4f}")
```

The output will be similar to:
```
SVD NDCG@10: 0.8953
k-NN NDCG@10: 0.8879
```
Here, SVD scores slightly higher on NDCG. This suggests that not only does SVD find a good number of relevant items (high recall), but it also tends to place them higher in the top-10 list than the k-NN model does.
Synthesis and Final Comparison

We've calculated a suite of metrics for our two models. Let's gather them to make a final comparison.
```
Metric 	SVD 	k-NN 	Winner
RMSE 	0.9348 	0.9791 	SVD
MAE 	0.7371 	0.7725 	SVD
Precision@10 	0.8384 	0.8421 	k-NN
Recall@10 	0.5891 	0.5301 	SVD
NDCG@10 	0.8953 	0.8879 	SVD
```
    Comparison of SVD and k-NN models across main evaluation metrics. Lower is better for RMSE, while higher is better for Precision and NDCG.

This practical exercise demonstrates a critical lesson in building recommenders: the "best" model depends on your objective.

    If your primary goal is to predict user ratings accurately, SVD is the clear winner due to its superior RMSE and MAE.
    If your goal is to present a highly relevant top-10 list, the choice is less clear. k-NN shows slightly better precision, but SVD has much better recall and a small edge in NDCG.

This evaluation framework gives you the tools to move past simple accuracy and measure what truly matters for your application. By combining multiple metrics, you can gain a more complete picture of model performance and make informed decisions when selecting and tuning your recommendation algorithms.