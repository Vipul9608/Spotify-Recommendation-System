# recommender.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


@dataclass
class RecoModel:
    users: List[str]
    songs: List[str]
    user_item: csr_matrix          # shape: (n_users, n_songs)
    item_user: csr_matrix          # shape: (n_songs, n_users)
    nn: NearestNeighbors           # fitted on item_user


def load_user_song_matrix(path: str) -> pd.DataFrame:
    """
    Your file is CSV content with .xls extension.
    First column is user_id like user_1, user_2... (named 'Unnamed: 0').
    Remaining columns are song_1..song_5000 with integer interactions.
    """
    df = pd.read_csv(path)

    # Make the first column the user id index
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "user_id"}).set_index("user_id")

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Optional: if negatives exist, clamp
    df[df < 0] = 0

    return df


def build_model(df: pd.DataFrame, n_neighbors: int = 50) -> RecoModel:
    users = df.index.astype(str).tolist()
    songs = df.columns.astype(str).tolist()

    user_item = csr_matrix(df.values)       # (users x songs)
    item_user = user_item.T.tocsr()          # (songs x users)

    nn = NearestNeighbors(
        n_neighbors=min(n_neighbors, item_user.shape[0]),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )
    nn.fit(item_user)

    return RecoModel(
        users=users,
        songs=songs,
        user_item=user_item,
        item_user=item_user,
        nn=nn
    )


def recommend_for_user(
    model: RecoModel,
    user_id: str,
    top_n: int = 10,
    per_seed_neighbors: int = 30,
    min_interaction: float = 1.0
) -> List[Tuple[str, float]]:
    """
    Item-item CF:
    - Take user's interacted songs (value >= min_interaction)
    - For each seed song, find similar songs via NearestNeighbors
    - Aggregate scores = sum(similarity * user_interaction)
    - Filter songs already interacted with
    Returns: list of (song_id, score)
    """

    if user_id not in model.users:
        raise ValueError(f"Unknown user_id: {user_id}")

    uidx = model.users.index(user_id)
    user_vec = model.user_item[uidx]  # sparse row

    # songs the user has interacted with
    seed_song_indices = user_vec.indices
    seed_song_values = user_vec.data

    # Apply min_interaction threshold
    mask = seed_song_values >= min_interaction
    seed_song_indices = seed_song_indices[mask]
    seed_song_values = seed_song_values[mask]

    if len(seed_song_indices) == 0:
        return []

    scores: Dict[int, float] = {}

    # For each seed song, find similar songs
    k = min(per_seed_neighbors, model.item_user.shape[0])
    for song_idx, strength in zip(seed_song_indices, seed_song_values):
        distances, indices = model.nn.kneighbors(
            model.item_user[song_idx],
            n_neighbors=k
        )
        # cosine distance -> similarity
        sims = 1.0 - distances.ravel()
        neigh = indices.ravel()

        for j, sim in zip(neigh, sims):
            if j == song_idx:
                continue
            scores[j] = scores.get(j, 0.0) + float(sim * strength)

    # remove already listened songs
    listened = set(seed_song_indices.tolist())
    candidates = [(idx, sc) for idx, sc in scores.items() if idx not in listened]

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:top_n]

    return [(model.songs[idx], score) for idx, score in top]


def top_listened_songs(model: RecoModel, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
    if user_id not in model.users:
        raise ValueError(f"Unknown user_id: {user_id}")

    uidx = model.users.index(user_id)
    row = model.user_item[uidx]
    pairs = list(zip(row.indices.tolist(), row.data.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:top_k]
    return [(model.songs[i], float(v)) for i, v in pairs]
