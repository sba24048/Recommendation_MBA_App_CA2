import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer   
from sklearn.metrics.pairwise import cosine_similarity        

st.title("Recommendations & Market Basket", width="stretch")

# Streamlit App Repository:https://github.com/sba24048/Recommendation_MBA_App_CA2


basket_merged = pd.read_csv("video_games_for_viz.csv.gzip", compression="gzip")
mba = pd.read_csv("mba_sample.csv.gzip", compression="gzip")
rules_fp1 = pd.read_csv("rules_fp1.csv.gzip", compression="gzip")

# Games Recommendations

clean_games = basket_merged[["item_id", "title", "genres"]].drop_duplicates(subset="title").reset_index(drop=True)
clean_games["genres"] = clean_games["genres"].fillna("")

tfidf = TfidfVectorizer(stop_words = "english")                                     # Create TF-IDF vectorizer, removing common English words

tfidf_matrix = tfidf.fit_transform(clean_games["genres"])                              # Convert movie content text into TF-IDF feature matrix

content_sim = cosine_similarity(tfidf_matrix)                                     # Compute cosine similarity between all games

games_idx = pd.Series(clean_games.index,index = clean_games["title"]).drop_duplicates()


# For content based filtering based on Genres
def recommend_genre_based(title, top_n, games, games_idx, content_sim):   # Function definition with inputs
    if title not in games_idx:                                               # Check if movie exists
        raise ValueError(f"Game '{title}' not found.")                      # Raise error if not found

    idx = games_idx[title]                                                   # Get index of selected movie

    sim_scores = list(enumerate(content_sim[idx]))                           # Get similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)        # Sort movies by similarity (highest first), x[0] -> index score and x[1] -> similarity score

    sim_scores = sim_scores[1: top_n + 1]                                    # Skip itself and take top N

    games_indices = [i for i, score in sim_scores]                           # Extract indices
    scores = [score for i, score in sim_scores]                              # Extract scores

    recs = games.iloc[games_indices][["item_id", "title", "genres"]].copy() # Get movie details
    recs["similarity_score"] = scores                                        # Add similarity scores

    return recs.reset_index(drop=True)                                       # Return result

select_game = st.selectbox(label="Choose Game", options=(clean_games["title"].dropna().unique().tolist()))

top_n = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=20,
    value=10
)

recs = recommend_genre_based(title = select_game, top_n = top_n, games = clean_games, games_idx = games_idx, content_sim = content_sim)

st.dataframe(recs)

st.subheader("Choose A Game to Receive Recommendations")



















