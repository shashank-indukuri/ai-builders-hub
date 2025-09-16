# src/recommend.py
from typing import List, Tuple, Dict
import numpy as np
import joblib
import pandas as pd
from scipy.sparse import csr_matrix

from src.utils import ARTIFACTS_DIR, load_csr, load_pickle

R_PATH = ARTIFACTS_DIR / "R_sparse.npz"
KNN_PATH = ARTIFACTS_DIR / "knn.joblib"
UID2IDX_PATH = ARTIFACTS_DIR / "uid_to_idx.pkl"
IDX2UID_PATH = ARTIFACTS_DIR / "idx_to_uid.pkl"
IID2IDX_PATH = ARTIFACTS_DIR / "iid_to_idx.pkl"
IDX2IID_PATH = ARTIFACTS_DIR / "idx_to_iid.pkl"
ITEM_META_PATH = ARTIFACTS_DIR / "items_meta.pkl"

class RecommenderArtifacts:
    def __init__(self):
        self.R: csr_matrix = load_csr(R_PATH)
        self.knn = joblib.load(KNN_PATH)
        self.uid_to_idx: Dict[int, int] = load_pickle(UID2IDX_PATH)
        self.idx_to_uid: Dict[int, int] = load_pickle(IDX2UID_PATH)
        self.iid_to_idx: Dict[int, int] = load_pickle(IID2IDX_PATH)
        self.idx_to_iid: Dict[int, int] = load_pickle(IDX2IID_PATH)
        self.item_meta: Dict[int, str] = load_pickle(ITEM_META_PATH)

def recommend_for_user(
    A: RecommenderArtifacts,
    raw_user_id: int,
    k_neighbors: int = 20,
    topn: int = 10,
    min_neighbor_rating: float = 4.0,
) -> List[Tuple[int, float]]:
    """
    Returns a list of (item_id, score) recommended for raw_user_id.
    - Find nearest users via kNN (cosine similarity = 1 - distance).
    - Aggregate neighbors' high-rated items unseen by the target user.
    """
    if raw_user_id not in A.uid_to_idx:
        return []

    user_idx = A.uid_to_idx[raw_user_id]
    # Find neighbors (+1 includes self; then remove)
    max_neighbors = min(k_neighbors + 1, A.R.shape[0])  # Get maximum possible neighbors
    n_neighbors = int(max_neighbors)  # Ensure it's a single integer
    distances, indices = A.knn.kneighbors(A.R[user_idx], n_neighbors=n_neighbors)
    distances = distances.flatten()
    indices = indices.flatten()

    # Remove self
    keep = indices != user_idx
    neighbor_idxs = indices[keep]
    neighbor_sims = 1.0 - distances[keep]  # cosine similarity

    user_row = A.R[user_idx].toarray().ravel()
    seen_items = set(np.where(user_row > 0)[0])  # Get the first element of the tuple from np.where

    scores: Dict[int, float] = {}
    for n_idx, sim in zip(neighbor_idxs, neighbor_sims):
        n_row = A.R[n_idx].toarray().ravel()
        liked_items = np.where(n_row >= min_neighbor_rating)[0]  # Get the first element of the tuple
        for j in liked_items:
            if j in seen_items:
                continue
            scores[j] = scores.get(j, 0.0) + sim * n_row[j]

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topn]
    # Convert column indices back to raw item ids
    return [(A.idx_to_iid[j], float(score)) for j, score in ranked]

def recommendations_df(A: RecommenderArtifacts, raw_user_id: int, **kwargs) -> pd.DataFrame:
    """
    Convenience formatter with titles for UI/API.
    """
    recs = recommend_for_user(A, raw_user_id, **kwargs)
    rows = []
    for iid, score in recs:
        title = A.item_meta.get(iid, f"Movie {iid}")
        rows.append({"item_id": iid, "title": title, "score": score})
    return pd.DataFrame(rows)