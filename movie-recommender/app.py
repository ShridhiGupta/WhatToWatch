import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("🎬 Movie Recommender System")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("ratings.csv")

df = load_data()

# Create user-item matrix
user_movie_matrix = df.pivot_table(index='user_id', columns='movie', values='ratings').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_sim_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Input user ID
user_id = st.number_input("Enter User ID (1 - 4)", min_value=1, max_value=4, step=1)

if st.button("Get Recommendations"):
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:]
    most_similar_user = similar_users.idxmax()

    user_movies = df[df['user_id'] == user_id]['movie'].tolist()
    similar_user_movies = df[df['user_id'] == most_similar_user]

    recommendations = similar_user_movies[~similar_user_movies['movie'].isin(user_movies)]

    if not recommendations.empty:
        st.subheader("Recommended Movies:")
        for movie in recommendations.sort_values(by='ratings', ascending=False)['movie']:
            st.write(f"🎥 {movie}")
    else:
        st.info("No new movies to recommend from similar users.")
