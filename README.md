# Movie Recommendation Engine

This project uses **collaborative filtering** to suggest movies to a user based on their past ratings and the behavior of other users.

It implements two popular collaborative filtering techniques:

- **User-Based Collaborative Filtering:** "Users who are similar to you also liked these movies."
- **Item-Based Collaborative Filtering:** "You liked this movie, so you'll probably like these similar ones too."

The script also evaluates the quality of its recommendations using the **Precision@K** metric.

---

## Dataset

This script is built for the **MovieLens 100K dataset**. You'll need to download it and have the following files **in the same directory** as the script:

- `u.data`: Contains 100,000 ratings from 943 users on 1682 movies.
- `u.item`: Contains information about the movies, like their titles.

---

## Setup

Before you run the script, you'll need to install a couple of libraries.

1.  **Prerequisites**:

    - Python 3.x
    - pip

2.  **Installation**:
    Open your terminal and run this command:
    ```bash
    pip install pandas scikit-learn
    ```

---

## Running in Jupyter Notebook:

1.  **Configure Parameters**:
    You can change these two variables at the top of the script to customize the output:

    ```python
    test_user_id = 10  # The ID of the user to get recommendations for
    k = 10             # The number of movies to recommend
    ```

2.  **Switch Recommendation Method** ( at the bottom of the script ):

    - For **User-Based** (default):
      ```python
      get_precision("user")
      ```
    - For **Item-Based**:
      ```python
      get_precision("item")
      ```

---

## Running in the Terminal:

### Command-Line Arguments

- `--user_id`: The ID of the user you want recommendations for. **(Default: 10)**
- `--k`: The number of movies you want to recommend. **(Default: 10)**
- `--m`: The recommendation method to use. **(Choices: [ user , item ], Default: user)**

### Examples

**To run with default settings (User-based, user_id = 10, K = 10):**

```bash
python Movie_Recommendation.py
```

**Or change settings with**

```bash
python Movie_Recommendation.py --user_id 1 --k 10 --m item
```

## How It Works

The script follows a few key steps to generate and evaluate recommendations.

1.  **Data Prep**: It loads the `u.data` and `u.item` files and merges them into a single DataFrame. It then splits this data into a training set (75%) and a testing set (25%).

2.  **Utility Matrix**: It creates a **user-item matrix** from the training data. This is a table where rows are users, columns are movies, and the cells contain the ratings. Any movies a user hasn't seen are given a rating of `0`.

3.  **Similarity Calculation**: Using `cosine_similarity` from scikit-learn, it calculates:

    - A **user-similarity matrix**: A score from 0 to 1 indicating how similar each user's tastes are to every other user.
    - An **item-similarity matrix**: A score from 0 to 1 indicating how similar each movie is to every other movie based on user ratings.

4.  **Recommendation Generation**:

    - The `get_user_based_recommendations` function finds the top K most similar users and recommends movies they rated highly (that the target user hasn't seen).
    - The `get_item_based_recommendations` function finds movies similar to the ones the target user has already rated highly and recommends those.

5.  **Evaluation**:
    The `get_precision` function takes the generated recommendations and compares them against the movies the user _actually_ liked in the test set. It then calculates **Precision@K**, which is the percentage of recommended items that were actually relevant.
