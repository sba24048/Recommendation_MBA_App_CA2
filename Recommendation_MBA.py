import pandas as pd
import numpy as np
import streamlit as st

# Streamlit App Repository:https://github.com/sba24048/Recommendation_MBA_App_CA2


basket_merged = pd.read_parquet("video_games_for_viz.csv.gzip", compression="gzip")
mba = pd.read_parquet("mba_sample.csv.gzip", compression="gzip")
rules_fp1 = pd.read_csv("rules_fp1.csv.gzip", compression="gzip")

# Games Recommendations
clean_games = basket_merged[["item_id", "title", "genres"]].drop_duplicates(subset="title").reset_index(drop=True)
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

recs = recommend_genre_based(title = selected_game, top_n = 10, games = clean_games, games_idx = games_idx, content_sim = content_sim)

st.title("Recommendations & Market Basket", width="stretch")