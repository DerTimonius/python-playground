import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


df_movies = pd.read_csv("~/documents/Coding/datasets/ml-latest/movies.csv")
df_ratings = pd.read_csv("~/documents/Coding/datasets/ml-latest/ratings.csv")

def clean_title(title):
  return re.sub("[^a-zA-Z0-9 ]", "", title)

df_movies["clean_title"] = df_movies["title"].apply(clean_title)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(df_movies["clean_title"])

def search(title):
  title = clean_title(title)
  query_vectorized = vectorizer.transform([title])
  similarity = cosine_similarity(query_vectorized, tfidf).flatten()
  indices = np.argpartition(similarity, -5)[-5:]
  results = df_movies.iloc[indices][::-1]
  return results

def find_similar_movies(movie_id):
  similar_users = df_ratings[(df_ratings["movieId"] == movie_id) & (df_ratings["rating"] > 4)]["userId"].unique()
  similar_users_recommendations = df_ratings[(df_ratings["userId"].isin(similar_users)) & (df_ratings["rating"] > 4)]["movieId"]

  similar_users_recommendations = similar_users_recommendations.value_counts() / len(similar_users)
  similar_users_recommendations = similar_users_recommendations[similar_users_recommendations > .10]

  all_users = df_ratings[(df_ratings["movieId"].isin(similar_users_recommendations.index)) & (df_ratings["rating"] > 4)]
  all_users_recommendations = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

  recommend_percentages = pd.concat([similar_users_recommendations, all_users_recommendations], axis=1)
  recommend_percentages.columns = ["similar", "all"]
  recommend_percentages["score"] = recommend_percentages["similar"] / recommend_percentages["all"]
  recommend_percentages = recommend_percentages.sort_values("score", ascending=False)
  return recommend_percentages.merge(df_movies, left_index=True, right_on="movieId")[["score", "title", "genres", "movieId"]]

def get_score(df, movie_id):
  recommend_percentages = df

  movie_score = recommend_percentages[(recommend_percentages["movieId"] == movie_id)]["score"]
  return movie_score

def get_movie(movie_id):
  return df_movies[df_movies["movieId"] == movie_id][["title", "genres", "movieId"]]

def get_best_movie_rec(movie_input_id_list):
  combined_df_list = []
  combined_top_list = []
  for movie in movie_input_id_list:
    combined_df_list.append(find_similar_movies(movie))
  for i in range(len(combined_df_list)):
    combined_top_list.append(combined_df_list[i][:10]["movieId"].tolist())
  total_scores = []
  for top_list in combined_top_list:
    for movie in top_list:
      total_score = 0
      if movie in movie_input_id_list:
        continue
      for df in combined_df_list:
        # print(get_score(df, movie))
        try:
          total_score += float(get_score(df, movie))
          # print(total_score)
        except TypeError:
          continue
      total_scores.append({"movie_id": movie, "score": total_score})
  top_movies = sorted(total_scores, key=lambda x: x["score"], reverse=True)
  return get_movie(top_movies[0]["movie_id"])

movie_list = [68157, 47, 88129, 140174, 1721, 160571]
print(get_best_movie_rec(movie_list))