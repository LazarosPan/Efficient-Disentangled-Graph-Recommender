Loading and Inspecting a Dataset (https://apxml.com/courses/building-ml-recommendation-system/chapter-1-foundations-of-recommendation-systems/practice-loading-dataset)

Work with a classic dataset to understand user-item interaction data and perform initial exploratory steps that precede model building. Such exploration enhances understanding of user behavior data and prepares for the algorithms implemented in later chapters.
The MovieLens Dataset

For our exercises, we will use the popular MovieLens 100k dataset, collected and maintained by the GroupLens research group at the University of Minnesota. It is a standard benchmark dataset in the field of recommendation systems. This version contains 100,000 ratings from 610 users for 9,742 movies.

The dataset consists of a few files, but we are primarily interested in two:

    ratings.csv: Contains the user-item interactions. Each row represents a rating given by a user to a movie and includes the userId, movieId, rating, and timestamp.
    movies.csv: Contains metadata about the items. Each row represents a movie and includes the movieId, title, and pipe-separated genres.

Let's begin by loading this data into our environment using the pandas library.
Loading the Data with Pandas

Assuming you have downloaded the dataset and placed the files in your working directory, you can load them into pandas DataFrames with the following code. DataFrames are two-dimensional labeled data structures that are ideal for handling tabular data like ours.

```python
import pandas as pd

# Define the file paths
movies_path = 'movies.csv'
ratings_path = 'ratings.csv'

# Load the data into pandas DataFrames
movies_df = pd.read_csv(movies_path)
ratings_df = pd.read_csv(ratings_path)
```

Initial Inspection

A good first step in any data analysis task is to inspect the data to understand its structure, size, and content. The .head() method is perfect for previewing the first few rows of a DataFrame.

Let's look at the ratings_df:

```python
print("Ratings DataFrame:")
print(ratings_df.head())
```

Output:
```
Ratings DataFrame:
   userId  movieId  rating  timestamp
0       1        1     4.0  964982703
1       1        3     4.0  964981247
2       1        6     4.0  964982224
3       1       47     5.0  964983815
4       1       50     5.0  964982931
```

This output confirms what we expect: each row is an explicit rating from a specific user for a specific movie. The rating scale appears to be from 1 to 5.

Now let's inspect the movies_df:
```python
print("\nMovies DataFrame:")
print(movies_df.head())
```

Output:
```
Movies DataFrame:
   movieId                               title                                       genres
0        1                    Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy
1        2                      Jumanji (1995)                   Adventure|Children|Fantasy
2        3             Grumpier Old Men (1995)                               Comedy|Romance
3        4            Waiting to Exhale (1995)                         Comedy|Drama|Romance
4        5  Father of the Bride Part II (1995)                                       Comedy
```

Here, we see the movie metadata. The genres column contains multiple values separated by a pipe | character. This is a common format we will learn to handle when building content-based recommenders.

To understand the scale of our dataset, we can find the number of unique users and movies.

```python
n_users = ratings_df['userId'].nunique()
n_movies = ratings_df['movieId'].nunique()

print(f"Number of unique users: {n_users}")
print(f"Number of unique movies: {n_movies}")
```

Output:
```
Number of unique users: 610
Number of unique movies: 9724
```

Our dataset contains interactions from 610 users on 9,724 different movies.
Exploring the Ratings Distribution

Understanding how users rate items is significant. Are users generally positive or negative? Are certain rating values more common than others? We can get a statistical summary of the rating column using the .describe() method.
```python
print(ratings_df['rating'].describe())
```
                
Output:
```
count    100836.000000
mean          3.501557
std           1.042529
min           0.500000
max           5.000000
25%           3.000000
50%           3.500000
75%           4.000000
Name: rating, dtype: float64
```

The average rating is approximately 3.5, suggesting a slight positive bias in the ratings. The ratings range from 0.5 to 5.0. Let's visualize the distribution of these ratings to get a clearer picture.

    The histogram shows that whole-star ratings (like 3.0, 4.0, and 5.0) are much more common than half-star ratings. It also confirms the positive rating bias, with 4.0 being the most frequent rating.

Creating a Unified DataFrame

Working with two separate DataFrames is inconvenient. We can combine them into a single DataFrame using a merge operation, similar to a SQL join. We will join ratings_df and movies_df on the movieId column, which is common to both.

```python
# Merge the two DataFrames
merged_df = pd.merge(ratings_df, movies_df, on='movieId')

print("Merged DataFrame:")
print(merged_df.head())
```
                
Output:
```
Merged DataFrame:
   userId  movieId  rating   timestamp            title                                       genres
0       1        1     4.0   964982703  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy
1       5        1     4.0   847434962  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy
2       7        1     4.5  1106635946  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy
3      15        1     2.5  1510577970  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy
4      17        1     4.5  1305696483  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy
```

This combined DataFrame gives us all the information for each rating in a single structure, which will be much easier to work with.
A First Look at the User-Item Matrix

As discussed earlier in the chapter, many recommendation algorithms operate on a user-item interaction matrix. While we won't build a full model yet, we can create a simple version of this matrix to visualize its structure. The rows will represent users, the columns will represent movies, and the cell values will be the ratings.

We can use pandas' .pivot_table() method to transform our data into this format.
```python
# Create the user-item matrix
user_item_matrix = merged_df.pivot_table(index='userId', columns='title', values='rating')

print("Shape of the user-item matrix:", user_item_matrix.shape)
print("\nFirst 5 rows and 5 columns of the matrix:")
print(user_item_matrix.iloc[:5, :5])
```
                
Output:
```
Shape of the user-item matrix: (610, 9719)

First 5 rows and 5 columns of the matrix:
title   '71 (2014)  'Hellboy': The Seeds of Creation (2004)  'Round Midnight (1986)  'Salem's Lot (2004)  'Til There Was You (1997)
userId
1              NaN                                      NaN                     NaN                 NaN                         NaN
2              NaN                                      NaN                     NaN                 NaN                         NaN
3              NaN                                      NaN                     NaN                 NaN                         NaN
4              NaN                                      NaN                     NaN                 NaN                         NaN
5              NaN                                      NaN                     NaN                 NaN                         NaN
```

The most immediate observation is the prevalence of NaN (Not a Number) values. These represent movies that a user has not rated. This is a direct illustration of the sparsity problem. Most users have rated only a tiny fraction of the available movies. We can quantify this sparsity:
Sparsity=1−Number of RatingsNumber of Users×Number of Movies
Sparsity=1−Number of Users×Number of MoviesNumber of Ratings​

Let's calculate it for our dataset.

```python
# Calculate the number of non-NaN values (actual ratings)
num_ratings = user_item_matrix.notna().sum().sum()

# Calculate the total number of possible ratings
total_possible_ratings = user_item_matrix.shape[0] * user_item_matrix.shape[1]

# Calculate sparsity
sparsity = 1 - (num_ratings / total_possible_ratings)

print(f"Sparsity of the user-item matrix: {sparsity:.4f}")
```

Output:
```
Sparsity of the user-item matrix: 0.9830
```

A sparsity of over 98% means that more than 98% of the cells in our user-item matrix are empty. This is typical for recommendation datasets and is a central challenge that collaborative filtering algorithms are designed to handle.

This initial exploration has given us a solid feel for the data's structure, size, and characteristics. We have loaded the data, examined its components, and visualized the distribution of ratings. Most importantly, we have seen a practical example of the sparse user-item interaction matrix that forms the foundation of collaborative filtering. In the next chapter, we will use this prepared data to build our first recommender, a content-based system that uses movie genres to find similar items.