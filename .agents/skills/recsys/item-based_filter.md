Implementing an Item-Based Filter (https://apxml.com/courses/building-ml-recommendation-system/chapter-3-neighborhood-based-collaborative-filtering/practice-implementing-item-based-filter)

Implementing an item-based collaborative filter from scratch demonstrates the practical application of neighborhood-based collaborative filtering. This approach is often preferred in practice over a user-based one because item similarities tend to be more stable over time than user similarities. A user's tastes can change, but the relationship between two movies, for example, is relatively static.

We will use a subset of the popular MovieLens dataset. Our goal is to create a function that, given a movie title, returns a list of the most similar movies based on user rating patterns.
Preparing the Data

First, we need to load our data and structure it into the user-item interaction matrix discussed earlier. Let's assume we have two CSV files: movies.csv containing movieId and title, and ratings.csv containing userId, movieId, and rating.

We'll start by loading these into pandas DataFrames and merging them.
```python
import pandas as pd
import numpy as np

# Load the datasets
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Merge them to have movie titles alongside ratings
df = pd.merge(ratings_df, movies_df, on='movieId')

print(df.head())
```

This gives us a tidy format, but it's not the user-item matrix we need. We can create this matrix by using the pivot_table function, with users as the index, movie titles as the columns, and ratings as the values.
```python
# Create the user-item interaction matrix
user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating')

# Fill missing values with 0
user_item_matrix.fillna(0, inplace=True)

print(user_item_matrix.head())
```

Filling missing values with 0 is a simplification. It implies that if a user hasn't rated an item, their preference is neutral. While more advanced techniques exist for imputation, this is a common and effective starting point. The resulting matrix is very sparse, meaning most of its cells are zero, which is typical for this kind of data.
Calculating Item-Item Similarity

With our matrix in place, we can now compute the similarity between every pair of movies. We'll use cosine similarity, which measures the cosine of the angle between two non-zero vectors. In our context, each movie is a vector where the components are the ratings given by each user.

The formula for cosine similarity between two items, ii and jj, is:
similarity(i,j)=cos⁡(θ)=vi⋅vj∥vi∥∥vj∥
similarity(i,j)=cos(θ)=∥vi​∥∥vj​∥vi​⋅vj​​

Here, vivi​ is the vector of ratings for item ii.

While we could compute this manually, it's more efficient to use libraries like scikit-learn. cosine_similarity expects samples in rows, but our items (movies) are currently columns. We must first transpose the matrix.

```python
from sklearn.metrics.pairwise import cosine_similarity

# Transpose the matrix so items are in rows
item_user_matrix = user_item_matrix.T

# Calculate the cosine similarity matrix
item_similarity_matrix = cosine_similarity(item_user_matrix)

# Convert the result into a DataFrame for better readability
item_similarity_df = pd.DataFrame(item_similarity_matrix, 
                                  index=user_item_matrix.columns, 
                                  columns=user_item_matrix.columns)

print(item_similarity_df.head())
```

The item_similarity_df is a square matrix where each row and column represents a movie. The value at item_similarity_df.loc['Movie A', 'Movie B'] is the cosine similarity score between Movie A and Movie B. Naturally, the diagonal will consist of ones, as every movie is perfectly similar to itself.
Building the Recommender Function

Now for the final step: creating a function that generates recommendations. This function will take a movie title and find the most similar movies from our similarity matrix.
Input Similarity Lookup Ranking & Selection Output Seed Movie: 'Star Wars (1977)' Item Similarity Matrix Look up scores Sorted Similarity Scores: 1. Star Wars (1.0) 2. Empire Strikes Back (0.78) 3. Return of the Jedi (0.74) 4. Raiders of the Lost Ark (0.54) ... Retrieve similarities Select Top-N (excluding self) Sort and filter Recommendations: 1. Empire Strikes Back 2. Return of the Jedi Generate list

    The process of generating item-based recommendations, from selecting a seed item to producing a final sorted list.

Here is the Python function that implements this logic.

```python
def get_item_based_recommendations(movie_title, similarity_matrix_df, n_recommendations=5):
    """
    Generates item-based collaborative filtering recommendations.

    Args:
        movie_title (str): The movie to get recommendations for.
        similarity_matrix_df (pd.DataFrame): The item-item similarity matrix.
        n_recommendations (int): The number of recommendations to return.

    Returns:
        A list of recommended movie titles.
    """
    # Get the similarity scores for the movie
    similar_scores = similarity_matrix_df[movie_title]

    # Sort the movies based on similarity score (descending)
    similar_scores = similar_scores.sort_values(ascending=False)

    # Exclude the movie itself (similarity is 1.0)
    similar_scores = similar_scores.drop(movie_title)

    # Return the top N movie titles
    return similar_scores.head(n_recommendations).index.tolist()
```

Testing Our Recommender

Let's test our function with a few examples to see the results. We'll find recommendations for a classic sci-fi film and a famous crime drama.

```python
# Get recommendations for 'Star Wars: Episode IV - A New Hope (1977)'
sw_recommendations = get_item_based_recommendations(
    'Star Wars: Episode IV - A New Hope (1977)', 
    item_similarity_df, 
    n_recommendations=5
)
print("Recommendations for 'Star Wars: Episode IV - A New Hope (1977)':")
print(sw_recommendations)

print("\n" + "="*50 + "\n")

# Get recommendations for 'Godfather, The (1972)'
godfather_recommendations = get_item_based_recommendations(
    'Godfather, The (1972)', 
    item_similarity_df, 
    n_recommendations=5
)
print("Recommendations for 'The Godfather (1972)':")
print(godfather_recommendations)
```

You should see results that are quite intuitive. For Star Wars, the recommendations will likely include other films in the series like The Empire Strikes Back and Return of the Jedi, as well as other contemporary sci-fi blockbusters. For The Godfather, you would expect to see The Godfather: Part II and other acclaimed crime films.

These results are generated without any knowledge of genre, director, or plot. They are derived purely from the collective behavior of thousands of users. This demonstrates the power of collaborative filtering: it uncovers relationships between items based on shared taste.
Summary and What's Next

In this practical exercise, you have successfully built a complete item-based collaborative filtering recommender. You transformed raw user-item interaction data into a structured matrix, calculated item-item similarities using cosine distance, and developed a function to generate ranked recommendations.

However, this neighborhood-based approach has its limitations. It struggles with scalability as the number of items grows, since calculating the full similarity matrix can be computationally expensive. Furthermore, it cannot recommend items that have no ratings. In the next chapter, we will explore model-based techniques, specifically matrix factorization, which can help address these challenges and often produce more accurate and personalized recommendations.