# app.py
import streamlit as st
import pandas as pd

from recommender import (
    load_user_song_matrix,
    build_model,
    recommend_for_user,
    top_listened_songs
)

st.set_page_config(page_title="Spotify Recommendation System", layout="wide")

st.title("Spotify Recommendation System (Collaborative Filtering)")
st.caption("Item–Item recommendations using cosine similarity (NearestNeighbors).")

DATA_PATH = "spotify (1).xls"  # keep file in the same repo folder


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return load_user_song_matrix(path)


@st.cache_resource(show_spinner=False)
def get_model(df: pd.DataFrame):
    # building NN model is heavier; cache it
    return build_model(df, n_neighbors=60)


with st.spinner("Loading data..."):
    df = load_data(DATA_PATH)

with st.spinner("Building recommendation model..."):
    model = get_model(df)

# Sidebar controls
st.sidebar.header("Controls")
user_id = st.sidebar.selectbox("Select user", model.users, index=0)
top_n = st.sidebar.slider("How many recommendations?", min_value=5, max_value=50, value=10, step=1)
min_interaction = st.sidebar.slider("Min interaction to treat as 'liked/listened'", 1.0, 20.0, 1.0, 1.0)
neighbors = st.sidebar.slider("Neighbors per seed song", min_value=10, max_value=100, value=30, step=5)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("User’s top listened songs")
    listened = top_listened_songs(model, user_id=user_id, top_k=15)
    if not listened:
        st.info("No interactions found for this user.")
    else:
        st.dataframe(
            pd.DataFrame(listened, columns=["song_id", "interaction"]),
            use_container_width=True
        )

with col2:
    st.subheader("Recommended songs")
    recos = recommend_for_user(
        model,
        user_id=user_id,
        top_n=top_n,
        per_seed_neighbors=neighbors,
        min_interaction=min_interaction
    )
    if not recos:
        st.warning("No recommendations (maybe this user has very low / zero interactions).")
    else:
        st.dataframe(
            pd.DataFrame(recos, columns=["song_id", "score"]),
            use_container_width=True
        )

st.markdown("---")
st.write(
    "Note: Your dataset has columns like `song_1`, `song_2`… (no track names). "
    "If you have a separate metadata file (track_name/artist/genre), you can map song_id → real info in the UI."
)
