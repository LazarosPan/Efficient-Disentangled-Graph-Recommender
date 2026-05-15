A Weighted Hybrid Movie Recommender (https://apxml.com/courses/building-ml-recommendation-system/chapter-6-constructing-a-hybrid-recommendation-system/practice-weighted-hybrid-recommender)

A weighted hybrid recommender combines the outputs of content-based and collaborative filtering models. Combining these different recommendation signals creates a more balanced and effective system. This hands-on exercise demonstrates how to construct such a weighted hybrid recommender.

Our goal is to create a final recommendation score for each movie by calculating a weighted sum of its scores from two different models. The formula we will implement is:
Scorehybrid=α⋅Scorecontent+(1−α)⋅Scorecollab
Scorehybrid​=α⋅Scorecontent​+(1−α)⋅Scorecollab​

Here, αα is a parameter we can tune, allowing us to control the influence of each recommender. A higher αα gives more weight to the content-based suggestions, while a lower αα favors the collaborative filtering results.
Setting Up the Environment

We begin by loading our required libraries and preparing the data. For this exercise, we will assume you have already generated and saved two sets of predictions for a specific user: one from a content-based model (perhaps using TF-IDF on movie plots) and another from a model-based collaborative filter (like SVD).

Let's assume these predictions are stored in pandas DataFrames. Each DataFrame should contain at least two columns: movieId and score.

```python
import pandas as pd
import numpy as np

# Load pre-computed scores from our previous models
# In a real application, these would be generated on-the-fly or from a database.

# Sample scores from a content-based model (e.g., cosine similarity)
content_data = {'movieId': [101, 102, 103, 104, 105],
                'title': ['Action Movie A', 'Sci-Fi Epic', 'Action Movie B', 'Thriller X', 'Drama Y'],
                'score': [0.95, 0.88, 0.76, 0.65, 0.21]}
content_df = pd.DataFrame(content_data)

# Sample scores from a collaborative filtering model (e.g., predicted SVD rating)
collab_data = {'movieId': [101, 103, 106, 107, 102],
               'title': ['Action Movie A', 'Action Movie B', 'Comedy Z', 'Romance Q', 'Sci-Fi Epic'],
               'score': [4.8, 4.5, 4.2, 3.9, 3.5]}
collab_df = pd.DataFrame(collab_data)

print("Content-Based Recommendations:")
print(content_df)
print("\nCollaborative Filtering Recommendations:")
print(collab_df)
```

Notice that the scores are on different scales. The content-based scores are similarity values between 0 and 1, while the collaborative filtering scores are predicted ratings, possibly from 1 to 5. Combining them directly would give unfair weight to the collaborative filter. Our first task is to normalize them.
Step 1: Normalize Recommendation Scores

To combine scores from different models fairly, they must be on the same scale. A common technique is min-max normalization, which scales values to a specific range, typically [0, 1].

The formula for min-max normalization is:
Scorenorm=x−min(x)max(x)−min(x)
Scorenorm​=max(x)−min(x)x−min(x)​

We will apply this to the score column of both DataFrames.

```python
# Min-max normalization for the content-based scores
content_min = content_df['score'].min()
content_max = content_df['score'].max()
content_df['norm_score'] = (content_df['score'] - content_min) / (content_max - content_min)

# Min-max normalization for the collaborative filtering scores
collab_min = collab_df['score'].min()
collab_max = collab_df['score'].max()
collab_df['norm_score'] = (collab_df['score'] - collab_min) / (collab_max - collab_min)

print("Normalized Content-Based Scores:")
print(content_df[['movieId', 'title', 'norm_score']])
print("\nNormalized Collaborative Filtering Scores:")
print(collab_df[['movieId', 'title', 'norm_score']])
```

With both sets of scores now scaled between 0 and 1, we can proceed with combining them.
Step 2: Combine Scores with a Weighted Sum

Next, we'll merge the two sets of recommendations into a single DataFrame. We need to handle items that appear in one list but not the other. An outer join is perfect for this, as it keeps all movies from both lists. We'll fill any missing scores with 0, assuming that if a model didn't recommend an item, its score for that item is effectively zero.

```python
# Merge the two dataframes on movieId and title
hybrid_df = pd.merge(content_df[['movieId', 'title', 'norm_score']],
                     collab_df[['movieId', 'title', 'norm_score']],
                     on=['movieId', 'title'],
                     how='outer')

# Rename the score columns for clarity
hybrid_df.rename(columns={'norm_score_x': 'content_score', 'norm_score_y': 'collab_score'}, inplace=True)

# Fill missing values with 0
hybrid_df.fillna(0, inplace=True)

print("Merged and Cleaned DataFrame:")
print(hybrid_df)
```

Now we can apply our weighted hybrid formula. Let's start by choosing an alpha of 0.5, giving equal weight to both models.

```python
# Define the weight alpha
alpha = 0.5

# Calculate the hybrid score
hybrid_df['hybrid_score'] = (alpha * hybrid_df['content_score']) + ((1 - alpha) * hybrid_df['collab_score'])

# Sort the recommendations by the new hybrid score
hybrid_df_sorted = hybrid_df.sort_values(by='hybrid_score', ascending=False)

print("\nHybrid Recommendations (alpha = 0.5):")
print(hybrid_df_sorted)
```

The diagram below illustrates the flow of our hybridization process. Scores from the two independent models are normalized and then combined using a weighted formula to produce a final, unified list of recommendations.
Content-Based Scores Normalize Scores (0 to 1) Collaborative Filtering Scores Normalize Scores (0 to 1) Weighted Hybridization α ⋅ S_content + (1-α) ⋅ S_collab α 1-α Final Hybrid Recommendations

    The process of creating a weighted hybrid recommender.

Step 3: Analyze the Hybrid Results

Let's examine the top 5 recommendations from our sorted list.

    Action Movie A (movieId 101): This movie scored highest in both original models and naturally tops our hybrid list.
    Action Movie B (movieId 103): This was ranked lower by the content model but very high by the collaborative model. The hybrid approach gives it a significant boost, placing it second.
    Sci-Fi Epic (movieId 102): This movie was strong in the content model but weaker in the collaborative one. Its final rank is a balance of the two.
    Comedy Z (movieId 106): This is an interesting case. The content model gave it a score of 0 (it wasn't in the top results), but it was highly rated by the collaborative filter. The hybrid model surfaces this item, adding novelty that a pure content-based system would have missed.
    Thriller X (movieId 104): Recommended by the content filter but not the collaborative one. It still makes the list, but its position is moderated by its zero score from the other model.

This hybrid list is a compelling blend. It preserves the strong "safe" recommendations that both models agree on, but it also introduces items that one model discovered and the other missed. This helps mitigate the over-specialization ("filter bubble") problem of content-based systems by incorporating collaborative signals.
Tuning the Alpha Parameter

The choice of alpha=0.5 was arbitrary. In a production system, this value should be carefully tuned. How do you find the best alpha? You can treat it as a hyperparameter and optimize it using a validation set.

The process would look like this:

    Hold out a set of user-item interactions as a validation set.
    Iterate through a range of alpha values (e.g., from 0.0 to 1.0 in increments of 0.1).
    For each alpha, generate hybrid recommendations for users in your validation set.
    Measure the performance using an offline metric like NDCG or MAP (as discussed in Chapter 5).
    Select the alpha value that yields the best performance on the validation metric.

This data-driven approach ensures that the balance between content and collaborative signals is optimized for your specific dataset and user base.

By completing this practical exercise, you have successfully constructed a hybrid recommendation system. You've learned to normalize scores from disparate models, combine them using a weighted formula, and analyze the resulting recommendations. This technique is a powerful and widely-used method for building more resilient and accurate recommenders that overcome the limitations of any single algorithm.