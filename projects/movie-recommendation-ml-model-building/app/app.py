# app/app.py
import streamlit as st
import pandas as pd

from pathlib import Path
import sys
# Add src to sys.path for local imports when running app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommend import RecommenderArtifacts, recommendations_df

st.set_page_config(page_title="Movie Recommender (kNN Cosine)", layout="centered")
st.title("Movie Recommender (kNN Cosine)")

# Lazy load artifacts once
@st.cache_resource
def load_artifacts():
    return RecommenderArtifacts()

A = load_artifacts()

with st.sidebar:
    st.header("Controls")
    raw_user_id = st.number_input("User ID", min_value=min(A.uid_to_idx.keys()), max_value=max(A.uid_to_idx.keys()), value=42, step=1)
    k_neighbors = st.slider("k neighbors", min_value=5, max_value=100, value=20, step=5)
    topn = st.slider("Top-N", min_value=5, max_value=50, value=10, step=5)
    min_rating = st.slider("Neighbor min rating", min_value=1.0, max_value=5.0, value=4.0, step=0.5)

if st.button("Recommend"):
    df = recommendations_df(A, int(raw_user_id), k_neighbors=k_neighbors, topn=topn, min_neighbor_rating=min_rating)
    if df.empty:
        st.warning("No recommendations (user not found or insufficient data).")
    else:
        st.subheader("Top Recommendations")
        st.dataframe(df, use_container_width=True)
else:
    st.info("Pick a User ID and click 'Recommend' to see results.")