Building a Movie Recommender (https://apxml.com/courses/building-ml-recommendation-system/chapter-2-content-based-filtering/practice-building-movie-recommender)

To construct a fully functional content-based movie recommender, this practical exercise will apply the concepts of item profiles, TF-IDF, and cosine similarity. We will use a popular dataset from The Movie Database (TMDB) that contains rich metadata for thousands of movies, including plot overviews, genres, keywords, cast, and crew. This exercise will walk you through every step, from raw data to personalized recommendations.

1. Setting Up and Loading the Data

First, we need to import the necessary libraries and load our datasets. We will use pandas for data manipulation and scikit-learn for our TF-IDF vectorization and similarity calculations.

The dataset is split into two CSV files: movies.csv contains movie metadata, and credits.csv contains cast and crew information. Let's load them and inspect the first few rows.

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Display the first few rows of each dataframe
print("Movies DataFrame:")
print(movies.head(2))
print("\nCredits DataFrame:")
print(credits.head(2))
```

To work with this data effectively, we'll merge these two dataframes into a single one based on the movie title.

```python
# Merge the two dataframes on the 'title' column
movies = movies.merge(credits, on='title')
```

2. Feature Engineering and Preprocessing

Our goal is to create a single text field for each movie that encapsulates its important content features. We will select the following columns: genres, keywords, overview, cast, and crew.

These columns are not in a friendly format. genres, keywords, cast, and crew are stored as JSON-like strings. We need to parse them to extract the information we need. For instance, from cast, we might only take the names of the top three actors, and from crew, we'll want to find the director.

Let's define a few helper functions to handle this parsing. We'll use Python's ast library to safely evaluate the string representations of lists and dictionaries.

```python
import ast

def extract_names(text):
    """Extracts names from a list of dictionaries, e.g., genres or keywords."""
    names = []
    for i in ast.literal_eval(text):
        names.append(i['name'])
    return names

def extract_top_actors(text, n=3):
    """Extracts the names of the top n actors from the cast."""
    actors = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < n:
            actors.append(i['name'])
            counter += 1
        else:
            break
    return actors

def extract_director(text):
    """Extracts the director's name from the crew list."""
    directors = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            directors.append(i['name'])
    return directors
```

Now, we apply these functions to our dataframe and clean up the resulting text by removing spaces, which helps the vectorizer treat "James Cameron" as a single entity rather than two separate words.

```python
# Apply the helper functions
movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)
movies['cast'] = movies['cast'].apply(extract_top_actors)
movies['crew'] = movies['crew'].apply(extract_director)

# Clean up text data by removing spaces
def remove_spaces(text_list):
    return [i.replace(" ","") for i in text_list]

movies['genres'] = movies['genres'].apply(remove_spaces)
movies['keywords'] = movies['keywords'].apply(remove_spaces)
movies['cast'] = movies['cast'].apply(remove_spaces)
movies['crew'] = movies['crew'].apply(remove_spaces)
```        

With our features extracted and cleaned, we can combine them into a single "tags" column. This column will serve as the document for each movie that our TF-IDF vectorizer will process.

```python
# Combine features into a single 'tags' column
movies['tags'] = (
    movies['overview'].fillna('').apply(lambda x: x.split()) +
    movies['genres'] +
    movies['keywords'] +
    movies['cast'] +
    movies['crew']
)

# Convert the list of tags back into a single string
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# Create a new, cleaner dataframe with only the necessary columns
recommender_df = movies[['movie_id', 'title', 'tags']].copy()
```

Let's look at the tags for the movie 'Avatar':

    In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting the home he feels is his home. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron

This combined string provides a rich description of the movie's content.
3. Vectorizing the Content

We now have a text corpus where each document is the tags string for a movie. The next step is to convert this text into numerical vectors using TF-IDF. This will create a matrix where each row represents a movie and each column represents a word from our vocabulary. The value in each cell will be the TF-IDF score of that word for that movie.

We will use TfidfVectorizer from scikit-learn. To keep the matrix manageable, we'll limit the number of features (words) to the 5,000 most frequent ones and remove common English stop words.

```python
# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the 'tags' column
feature_vectors = tfidf.fit_transform(recommender_df['tags']).toarray()

print(f"Shape of our feature vectors: {feature_vectors.shape}")
```

The output shape, likely (4806, 5000), confirms that we have a vector of 5,000 dimensions for each of the 4,806 movies in our dataset.

The following diagram shows the workflow we have just completed.
Data Preparation Model Building Movie Metadata (overview, genres, cast...) Feature Engineering (parse, clean, combine) Combined 'Tags' Field TF-IDF Vectorization Movie Feature Vectors (Numerical Representation) Cosine Similarity Calculation Similarity Matrix Recommendation Function Top-N Recommended Movies

    The process of building our content-based recommender, from raw metadata to the final similarity matrix.

4. Computing Similarity

With our movies represented as vectors, we can now calculate the cosine similarity between every pair of movies. The cosine_similarity function from scikit-learn is perfect for this. It takes our matrix of feature vectors and returns a square matrix where each entry (i, j) is the similarity score between movie i and movie j.
similarity(A,B)=A⋅B∥A∥∥B∥
similarity(A,B)=∥A∥∥B∥A⋅B​

This operation is computationally efficient on the entire matrix.

```python
# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(feature_vectors)

print(f"Shape of the similarity matrix: {similarity_matrix.shape}")
print("\nSample of the similarity matrix (first 5x5):")
print(similarity_matrix[:5, :5])
```

The resulting (4806, 4806) matrix contains all the pairwise similarity scores. Notice the diagonal is all 1s, as every movie is perfectly similar to itself.

5. Creating the Recommendation Function

We have all the pieces in place. The final step is to build a function that uses the similarity_matrix to provide recommendations. This function will take a movie title as input and perform the following steps:

    Find the index of the input movie in our dataframe.
    Retrieve the corresponding row from the similarity matrix.
    Sort the similarity scores in descending order.
    Select the indices of the top 5 most similar movies.
    Return the titles of these movies.

```python
def recommend_movies(movie_title):
    """
    Recommends movies similar to the input movie_title.
    """
    # Find the index of the movie that matches the title
    try:
        movie_index = recommender_df[recommender_df['title'] == movie_title].index[0]
    except IndexError:
        return f"Movie '{movie_title}' not found in the dataset."

    # Get the pairwise similarity scores for the given movie
    distances = similarity_matrix[movie_index]

    # Sort the movies based on the similarity scores
    # We use enumerate to keep track of the original index
movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

# Get the top 5 most similar movies, skipping the first one (itself)
recommended = []
for i in movies_list[1:6]:
    recommended.append(recommender_df.iloc[i[0]].title)

return recommended


### 6. Testing the Recommender

Let's test our function with a few examples to see how well it performs.

**Example 1: A superhero movie**

recommend_movies('The Dark Knight Rises')
```

Output:
```
['The Dark Knight', 'Batman Begins', 'Batman', 'Batman Returns', 'Batman: The Dark Knight Returns, Part 2']
```

The recommendations are excellent. The system correctly identifies other Batman films, demonstrating its ability to connect movies based on characters (from the cast and keywords) and genre.

Example 2: A science fiction epic
```python
recommend_movies('Avatar')
```

Output:
```
['Aliens vs Predator: Requiem', 'Aliens', 'Falcon Rising', 'Independence Day', 'Titan A.E.']
```
These recommendations are also strong. They share themes of science fiction, aliens, and action, which were prominent in the tags we created for 'Avatar'.

Example 3: A classic animated film
```python
recommend_movies('Toy Story')
```
                
Output:
```
['Toy Story 3', 'Toy Story 2', 'Cars 2', 'Monsters, Inc.', 'Finding Nemo']
```
Here, the recommender correctly identifies other films by the same studio (Pixar) and in the same genre (animation), which often share similar keywords and even crew members (like director John Lasseter).
Summary and What's Next

In this hands-on section, you successfully built a content-based recommendation system from the ground up. You performed feature engineering on raw movie metadata, transformed text into numerical vectors using TF-IDF, calculated similarity scores with cosine similarity, and used these scores to generate relevant movie recommendations.

This type of system is powerful because it doesn't require any user interaction data. It can recommend items immediately, which helps with the "new item" problem. However, it has its limitations. It tends to recommend items that are very similar to what a user has already seen, which can lead to a lack of discovery, sometimes called a "filter bubble." Furthermore, its quality is entirely dependent on the richness and accuracy of the item metadata.

In the next chapter, we will explore a different approach: collaborative filtering. Instead of looking at item content, we will analyze user behavior to find patterns and make recommendations, which can help overcome some of the limitations we see here.