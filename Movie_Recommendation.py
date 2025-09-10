import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import argparse

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

m_cols = ["movie_id", "title", "release date", "video release date", "IMDb URL"] + [f"genre_{i}" for i in range(19)]

ratings_df = pd.read_csv("u.data", sep="\t", names=r_cols, encoding="latin-1")
movies_df = pd.read_csv("u.item", sep="|", names=m_cols, usecols=["movie_id", "title"], encoding="latin-1", index_col="movie_id")

df = pd.merge(movies_df, ratings_df, on="movie_id")


train_df, test_df = train_test_split(df, test_size= 0.25, random_state=42)
user_item_matrix = train_df.pivot_table(index="user_id", columns="title", values="rating")

user_item_matrix.fillna(0, inplace=True)

# user-based
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# item-based
item_user_matrix = user_item_matrix.T
item_similarity = cosine_similarity(item_user_matrix)
item_similarity_df = pd.DataFrame(item_similarity, item_user_matrix.index, item_user_matrix.index)


def get_user_based_recommendations(user_id, k):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:k+1]
    user_ratings = user_item_matrix.loc[user_id]
    movies_seen_by_user = user_ratings[user_ratings > 0].index
    
    recommendations = {}
    for similar_user, score in similar_users.items():
        similar_users_ratings = user_item_matrix.loc[similar_user]
        seen_by_similar_users = similar_users_ratings[similar_users_ratings > 0]
        high_ratings = seen_by_similar_users[seen_by_similar_users > 3]
       
        for movie, rating in high_ratings.items():
            if movie not in movies_seen_by_user:
                if movie not in recommendations:
                    recommendations[movie] = 0
                recommendations[movie] += score
                
    sorted_r = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)
    
    return pd.DataFrame(sorted_r, columns=['Movie', 'Score']).head(k)


def get_item_based_recommendations(user_id, k):
    user_ratings = user_item_matrix.loc[user_id]
    movies_seen = user_ratings[user_ratings > 0].index.tolist()
    top_movies = user_ratings[user_ratings >= 4].index.tolist()

    recommendations = {}
    for movie in top_movies:
        similar_movies = item_similarity_df[movie].sort_values(ascending=False)[1:k+1]

        for s_movie, score in similar_movies.items():
            if s_movie not in recommendations:
                recommendations[s_movie] = 0
            recommendations[s_movie] += score * user_ratings[movie]

    r = {movie: score for movie, score in recommendations.items() if movie not in movies_seen}    
    sorted_r = sorted(r.items(), key=lambda item: item[1], reverse=True)

    return pd.DataFrame(sorted_r, columns=['Movie', 'Score']).head(k)



def get_percision(test_user_id, k, user_based):
    if user_based in ["user", "user"]:
        r = get_user_based_recommendations(test_user_id, k)
        m = "User_based"
    else :    
        r = get_item_based_recommendations(test_user_id, k)
        m = "Item_based"
    recommended_list = r["Movie"].tolist()
    
    test_user_df = test_df[test_df["user_id"] == test_user_id]
    actual_list = test_user_df[test_user_df["rating"] > 3]["title"].tolist()
    
    hits = [movie for movie in recommended_list if movie in actual_list]
    num_hits = len(hits)
    
    precision = num_hits / k if k > 0 else 0.0 

    print(f"using {m}_Matrix : \n---------------------")
    print(f"Recs for User {test_user_id}: {recommended_list}")
    print(f"\nTrue Hits in Test Set: {actual_list}\n\nLen : {len(actual_list)}")
    print(f"\nHits: {num_hits}")
    print(f"\nPrecision@{k} for this user is: {precision:.0%}\n---------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a movie recommender and get its precision.")
    
    parser.add_argument("--user_id", type=int, default=10, help="ID of the user to get recommendations for.")
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations to generate.")
    parser.add_argument("--m", type=str, default="user", choices=["user", "item"], help="Recommendation method: 'user' or 'item'.")

    args = parser.parse_args()
    
    print(f"Running {args.m}-based CF for user_id={args.user_id} with K={args.k}\n")
    
    get_percision(test_user_id=args.user_id, k=args.k, user_based=args.m)