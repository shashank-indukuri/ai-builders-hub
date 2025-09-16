# src/model_knn.py
from pathlib import Path
import joblib
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from src.utils import ARTIFACTS_DIR, load_csr

R_PATH = ARTIFACTS_DIR / "R_sparse.npz"
KNN_PATH = ARTIFACTS_DIR / "knn.joblib"

def fit_knn(R: csr_matrix, metric: str = "cosine", algorithm: str = "brute") -> NearestNeighbors:
    """
    Fit a NearestNeighbors model for user-based CF.
    Cosine distance works well on sparse rating vectors.
    """
    model = NearestNeighbors(metric=metric, algorithm=algorithm)
    model.fit(R)
    return model

def main():
    R = load_csr(R_PATH)
    knn = fit_knn(R)
    joblib.dump(knn, KNN_PATH)
    print(f"Saved kNN model to: {KNN_PATH}")

if __name__ == "__main__":
    main()