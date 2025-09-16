# src/data_prep.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
from typing import Tuple, Dict

from src.utils import DATA_DIR, ARTIFACTS_DIR, save_csr, save_pickle

# Output artifact paths
R_PATH = ARTIFACTS_DIR / "R_sparse.npz"
UID2IDX_PATH = ARTIFACTS_DIR / "uid_to_idx.pkl"
IDX2UID_PATH = ARTIFACTS_DIR / "idx_to_uid.pkl"
IID2IDX_PATH = ARTIFACTS_DIR / "iid_to_idx.pkl"
IDX2IID_PATH = ARTIFACTS_DIR / "idx_to_iid.pkl"
ITEM_META_PATH = ARTIFACTS_DIR / "items_meta.pkl"

def load_movielens_100k() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads ML-100k ratings and items.
    u.data columns: user_id, item_id, rating, timestamp
    u.item columns: item_id | title | (other metadata)
    """
    # DATA_DIR already points to data/raw/ml-100k/
    ratings_path = DATA_DIR / "u.data"
    items_path = DATA_DIR / "u.item"

    # MovieLens 100K uses tab for u.data and pipe for u.item, with latin-1 encoding
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user", "item", "rating", "ts"],
        engine="python",
    )
    items = pd.read_csv(
        items_path,
        sep="|",
        header=None,
        encoding="latin-1",
        engine="python",
    )
    # Map columns minimally: 0=item_id, 1=title (ML-100K spec)
    items = items.rename(columns={0: "item", 1: "title"})
    items = items[["item", "title"]]
    return ratings, items

def build_matrix_and_mappings(ratings: pd.DataFrame) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Creates CSR matrix R (users x items) with explicit ratings,
    plus mappings both directions for users and items.
    """
    # Stable ordering of ids
    unique_users = np.sort(ratings["user"].unique())
    unique_items = np.sort(ratings["item"].unique())

    uid_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    idx_to_uid = {i: uid for uid, i in uid_to_idx.items()}
    iid_to_idx = {iid: j for j, iid in enumerate(unique_items)}
    idx_to_iid = {j: iid for iid, j in iid_to_idx.items()}

    rows = ratings["user"].map(uid_to_idx).values
    cols = ratings["item"].map(iid_to_idx).values
    vals = ratings["rating"].astype(float).values

    n_users = len(unique_users)
    n_items = len(unique_items)
    R = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    return R, uid_to_idx, idx_to_uid, iid_to_idx, idx_to_iid

def main():
    ratings, items = load_movielens_100k()
    R, uid_to_idx, idx_to_uid, iid_to_idx, idx_to_iid = build_matrix_and_mappings(ratings)

    # Save artifacts
    save_csr(R, R_PATH)
    save_pickle(uid_to_idx, UID2IDX_PATH)
    save_pickle(idx_to_uid, IDX2UID_PATH)
    save_pickle(iid_to_idx, IID2IDX_PATH)
    save_pickle(idx_to_iid, IDX2IID_PATH)

    # Save item metadata (title by item id for quick lookups)
    item_meta = dict(zip(items["item"].astype(int).tolist(), items["title"].astype(str).tolist()))
    save_pickle(item_meta, ITEM_META_PATH)

    print("Saved artifacts to:", ARTIFACTS_DIR)

if __name__ == "__main__":
    main()