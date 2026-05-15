Generating Recommendations with SVD (https://apxml.com/courses/building-ml-recommendation-system/chapter-4-model-based-collaborative-filtering/practice-recommendations-with-svd)

We will build a complete recommendation engine using the surprise library. The goal is to train an SVD model on the MovieLens dataset and use it to generate a ranked list of movie recommendations for a specific user.
Setting Up the Environment and Data

First, ensure you have the necessary libraries installed. We'll be using pandas for data manipulation and surprise for implementing our SVD model. We will work with the widely-used MovieLens 100k dataset.

Let's begin by loading the user ratings and movie titles into pandas DataFrames.

```python
import pandas as pd
from surprise import Dataset, Reader, SVD

# Load the movies and ratings datasets
movies_df = pd.read_csv('ml-latest-small/movies.csv')
ratings_df = pd.read_csv('ml-latest-small/ratings.csv')

# Display the first few rows of each dataframe to inspect them
print("Movies DataFrame:")
print(movies_df.head())

print("\nRatings DataFrame:")
print(ratings_df.head())
```

The ratings_df contains our user-item interaction data in the format (userId, movieId, rating). This is the primary input for our collaborative filtering model. The movies_df will be useful later for mapping the movieIds in our recommendations back to human-readable movie titles.
Preparing Data for the Surprise Library

The surprise library requires data to be in a specific format. We need to define a Reader object to parse our ratings data, specifying the scale of the ratings (in this case, 0.5 to 5.0 for the MovieLens dataset). Then, we load our DataFrame into a surprise Dataset object.

```python
# The Reader object helps parse the file or dataframe
reader = Reader(rating_scale=(0.5, 5.0))

# Load the data from the pandas dataframe
# The columns must be in the order: user, item, rating
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
```

By loading the data this way, surprise can now understand its structure and use it for model training and prediction.
Training the SVD Model

With our data prepared, we can now instantiate and train our SVD model. When generating final recommendations, it's a common practice to train the model on the entire dataset to capture as many user-item interaction patterns as possible. The surprise library makes this straightforward with the build_full_trainset() method.

```python
# Build a training set from the entire dataset
trainset = data.build_full_trainset()

# Instantiate the SVD algorithm
svd_model = SVD(n_factors=100, n_epochs=20, random_state=42)

# Train the algorithm on the trainset
print("Training the SVD model...")
svd_model.fit(trainset)
print("Training complete.")
```

Here, we've initialized the SVD algorithm with 100 latent factors (n_factors) and set it to run for 20 iterations (n_epochs) over the data during optimization. After the fit method completes, svd_model is our trained model, ready to make predictions.
Generating Top-N Recommendations for a User

The primary function of a recommender is not just to predict a rating for a single item, but to produce a ranked list of items a user might like. To do this, we need to predict ratings for all the items a user has not yet seen and then sort these predictions to find the best ones.

The following diagram illustrates the workflow for generating recommendations.
Inputs Recommendation Logic Output Target User ID 1. Identify Unrated Items for User Trained SVD Model 2. Predict Rating for Each Unrated Item Set of All Items 3. Sort Items by Predicted Rating Top-N Recommended Items

    The process begins with a target user and a trained model. It identifies items the user hasn't interacted with, predicts a score for each, and finally returns a sorted list of the top N recommendations.

Let's implement this logic in a Python function. This function will take a user ID and our trained model as input and return the top 10 recommended movies.

```python
def get_top_n_recommendations(user_id, model, ratings_df, movies_df, n=10):
    """
    Generates top-N recommendations for a given user using a trained SVD model.
    """
    # Get a list of all movie IDs
    all_movie_ids = ratings_df['movieId'].unique()

    # Get a list of movie IDs that the user has rated
    rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()

    # Get a list of movie IDs that the user has NOT rated
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]

    # Predict ratings for all unrated movies
    predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movie_ids]

    # Sort the predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get the top N movie IDs
    top_n_predictions = predictions[:n]
    top_n_movie_ids = [pred.iid for pred in top_n_predictions]

    # Get the movie titles for the top N movie IDs
    recommended_movies = movies_df[movies_df['movieId'].isin(top_n_movie_ids)]

    return recommended_movies

# Let's generate recommendations for user with ID 50
user_id_to_recommend = 50
recommendations = get_top_n_recommendations(user_id_to_recommend, svd_model, ratings_df, movies_df)

print(f"\nTop 10 movie recommendations for user {user_id_to_recommend}:")
print(recommendations)
```

Let's examine the output for user 50.

```
Top 10 movie recommendations for user 50:
	movieId 	title 	genres
14 	15 	As Good as It Gets (1997) 	Comedy, Drama, Romance
27 	28 	Persuasion (1995) 	Drama, Romance
163 	194 	Stargate (1994) 	Action, Adventure, Sci-Fi
176 	208 	Waterworld (1995) 	Action, Adventure, Sci-Fi
227 	266 	While You Were Sleeping (1995) 	Comedy, Romance
315 	357 	Four Weddings and a Funeral (1994) 	Comedy, Romance
451 	515 	Remains of the Day, The (1993) 	Drama, Romance
465 	529 	My Life as a Dog (Mitt liv som hund) (1985) 	Comedy, Drama
470 	534 	Sense and Sensibility (1995) 	Drama, Romance
520 	608 	Fargo (1996) 	Comedy, Crime, Drama, Thriller
```

Our function successfully generated a list of personalized recommendations. The model has learned this user's preferences from their past ratings and identified other movies they are likely to enjoy. This demonstrates the power of matrix factorization; even without knowing anything about the movies' content, the model can infer compatibility based on the latent factors discovered in the user-item interaction matrix.

With this, you have successfully built and used a model-based collaborative filtering system. The next step is to rigorously evaluate its performance, which we will cover in the following chapter.