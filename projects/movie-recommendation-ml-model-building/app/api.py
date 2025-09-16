"""
FastAPI service exposing the movie recommender.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import sys

# Add repo root to sys.path so `src` is importable when running via uvicorn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommend import RecommenderArtifacts, recommendations_df

app = FastAPI(title="Movie Recommender API", version="1.0.0")


class RecRequest(BaseModel):
    user_id: int
    k_neighbors: int = 20
    topn: int = 10
    min_neighbor_rating: float = 4.0


# Load artifacts at startup
A = None


@app.on_event("startup")
def _load():
    global A
    A = RecommenderArtifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecRequest):
    df = recommendations_df(
        A,
        req.user_id,
        k_neighbors=req.k_neighbors,
        topn=req.topn,
        min_neighbor_rating=req.min_neighbor_rating,
    )
    return {
        "user_id": req.user_id,
        "items": df.to_dict(orient="records"),
    }
