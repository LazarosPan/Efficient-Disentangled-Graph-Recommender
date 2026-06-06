# PyTorch-Geometric Datasets

Choose the most relevant ones for the thesis—those also used in SOTA papers.

## Homogeneous Datasets

### Yelp

```python
class Yelp(root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, force_reload: bool = False)[source]
```

> **Bases:** `InMemoryDataset`  
> The Yelp dataset from the “GraphSAINT: Graph Sampling Based Inductive Learning Method” paper, containing customer reviewers and their friendship.

**Parameters:**

- `root` (`str`) – Root directory where the dataset should be saved.  
- `transform` (`callable`, optional) – A function/transform that takes in a `Data` object and returns a transformed version. The data object will be transformed before every access. (default: `None`)  
- `pre_transform` (`callable`, optional) – A function/transform that takes in a `Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: `None`)  
- `force_reload` (`bool`, optional) – Whether to re-process the dataset. (default: `False`)

**Stats:**

| #nodes     | #edges        | #features | #tasks |
|------------|---------------|-----------|--------|
| 716,847    | 13,954,819    | 300       | 100    |

---

### AmazonProducts

```python
class AmazonProducts(root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, force_reload: bool = False)[source]
```

> **Bases:** `InMemoryDataset`  
> The Amazon dataset from the “GraphSAINT: Graph Sampling Based Inductive Learning Method” paper, containing products and their categories.

**Parameters:**

- `root` (`str`) – Root directory where the dataset should be saved.  
- `transform` (`callable`, optional) – A function/transform that takes in a `Data` object and returns a transformed version. The data object will be transformed before every access. (default: `None`)  
- `pre_transform` (`callable`, optional) – A function/transform that takes in a `Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: `None`)  
- `force_reload` (`bool`, optional) – Whether to re-process the dataset. (default: `False`)

**Stats:**

| #nodes      | #edges         | #features | #classes |
|-------------|----------------|-----------|----------|
| 1,569,960   | 264,339,468    | 200       | 107      |

---

## Heterogeneous Datasets

### MovieLens

```python
class MovieLens(root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, model_name: Optional[str] = 'all-MiniLM-L6-v2', force_reload: bool = False)[source]
```

> **Bases:** `InMemoryDataset`  
> A heterogeneous rating dataset, assembled by GroupLens Research from the MovieLens website, consisting of nodes of type `"movie"` and `"user"`. User ratings for movies are available as ground truth labels for the edges between the users and the movies (`("user", "rates", "movie")`).

**Parameters:**

- `root` (`str`) – Root directory where the dataset should be saved.  
- `transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before every access. (default: `None`)  
- `pre_transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: `None`)  
- `model_name` (`str`) – Name of model used to transform movie titles to node features. The model comes from the [Hugging Face SentenceTransformer](https://huggingface.co/sentence-transformers).  
- `force_reload` (`bool`, optional) – Whether to re-process the dataset. (default: `False`)

---

### MovieLens100K

```python
class MovieLens100K(root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, force_reload: bool = False)[source]
```

> **Bases:** `InMemoryDataset`  
> The MovieLens 100K heterogeneous rating dataset, assembled by GroupLens Research from the MovieLens website, consisting of movies (1,682 nodes) and users (943 nodes) with 100K ratings between them. User ratings for movies are available as ground truth labels. Features of users and movies are encoded according to the “Inductive Matrix Completion Based on Graph Neural Networks” paper.

**Parameters:**

- `root` (`str`) – Root directory where the dataset should be saved.  
- `transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before every access. (default: `None`)  
- `pre_transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: `None`)  
- `force_reload` (`bool`, optional) – Whether to re-process the dataset. (default: `False`)

**Stats:**

| Node/Edge Type | #nodes / #edges | #features | #tasks |
|----------------|------------------|-----------|--------|
| Movie          | 1,682            | 18        | —      |
| User           | 943              | 24        | —      |
| User–Movie     | 80,000           | 1         | 1      |

---

### MovieLens1M

```python
class MovieLens1M(root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, force_reload: bool = False)[source]
```

> **Bases:** `InMemoryDataset`  
> The MovieLens 1M heterogeneous rating dataset, assembled by GroupLens Research from the MovieLens website, consisting of movies (3,883 nodes) and users (6,040 nodes) with approximately 1 million ratings between them. User ratings for movies are available as ground truth labels. Features of users and movies are encoded according to the “Inductive Matrix Completion Based on Graph Neural Networks” paper.

**Parameters:**

- `root` (`str`) – Root directory where the dataset should be saved.  
- `transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before every access. (default: `None`)  
- `pre_transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: `None`)  
- `force_reload` (`bool`, optional) – Whether to re-process the dataset. (default: `False`)

**Stats:**

| Node/Edge Type | #nodes / #edges | #features | #tasks |
|----------------|------------------|-----------|--------|
| Movie          | 3,883            | 18        | —      |
| User           | 6,040            | 30        | —      |
| User–Movie     | 1,000,209        | 1         | 1      |

---

### Taobao

```python
class Taobao(root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, force_reload: bool = False)[source]
```

> **Bases:** `InMemoryDataset`  
> Taobao is a dataset of user behaviors from Taobao offered by Alibaba, provided by the Tianchi Alicloud platform.  
> Taobao is a heterogeneous graph for recommendation. Nodes represent users (with user IDs), items (with item IDs), and categories (with category IDs). Edges between users and items represent different types of user behaviors toward items (alongside timestamps). Edges between items and categories assign each item to its set of categories.

**Parameters:**

- `root` (`str`) – Root directory where the dataset should be saved.  
- `transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before every access. (default: `None`)  
- `pre_transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: `None`)  
- `force_reload` (`bool`, optional) – Whether to re-process the dataset. (default: `False`)

---

### IGMCDataset

```python
class IGMCDataset(root: str, name: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, force_reload: bool = False)[source]
```

> **Bases:** `InMemoryDataset`  
> The user-item heterogeneous rating datasets **"Douban"**, **"Flixster"**, and **"Yahoo-Music"** from the “Inductive Matrix Completion Based on Graph Neural Networks” paper.  
> Nodes represent users and items. Edges and features between users and items represent a (training) rating of the item given by the user.

**Parameters:**

- `root` (`str`) – Root directory where the dataset should be saved.  
- `name` (`str`) – The name of the dataset (`"Douban"`, `"Flixster"`, `"Yahoo-Music"`).  
- `transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before every access. (default: `None`)  
- `pre_transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: `None`)  
- `force_reload` (`bool`, optional) – Whether to re-process the dataset. (default: `False`)

---

### AmazonBook

```python
class AmazonBook(root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, force_reload: bool = False)[source]
```

> **Bases:** `InMemoryDataset`  
> A subset of the AmazonBook rating dataset from the “LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation” paper. This is a heterogeneous dataset consisting of 52,643 users and 91,599 books with approximately 2.9 million ratings between them. No labels or features are provided.

**Parameters:**

- `root` (`str`) – Root directory where the dataset should be saved.  
- `transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before every access. (default: `None`)  
- `pre_transform` (`callable`, optional) – A function/transform that takes in a `HeteroData` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: `None`)  
- `force_reload` (`bool`, optional) – Whether to re-process the dataset. (default: `False`)
