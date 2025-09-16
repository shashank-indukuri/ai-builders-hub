# src/utils.py
from pathlib import Path
import pickle
from scipy.sparse import csr_matrix, save_npz, load_npz
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "ml-100k"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def save_csr(mat: csr_matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(path, mat)

def load_csr(path: Path) -> csr_matrix:
    return load_npz(path)