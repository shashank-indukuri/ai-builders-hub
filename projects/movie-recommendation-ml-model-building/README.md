# Run Movie Recommender Locally with Streamlit + FastAPI 🎬

An end-to-end, local movie recommendation demo powered by user-based kNN on the MovieLens 100K dataset. Try recommendations in a clean Streamlit UI and hit a production-style FastAPI for programmatic access.

## Why This Matters

- ⚡ Real-time recommendations: Interactive UI with instant results
- 🧩 End-to-end workflow: Data prep → model → UI/API → Docker
- 🧪 Reproducible artifacts: Deterministic build of sparse matrix + kNN
- 🧰 Practical baseline: Simple, interpretable user-based CF you can extend
- 🚀 Easy to run: Python 3.12 with `uv` or `pip`, Streamlit and FastAPI

## Key Features

- Streamlit UI to explore recommendations for any user id
- FastAPI endpoint for `/recommend` and `/health`
- Configurable neighbors (k), Top-N, and rating threshold
- Reusable artifacts for quick startup and reproducibility
- Dockerfile for containerized API deployment

## Installation and Setup

### Prerequisites
- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### 1) Install dependencies

Using uv (recommended):

```bash
uv sync
```

Using pip:

```bash
pip install -r requirements.txt
```

### 2) Prepare artifacts (optional — already included)

Artifacts for ML-100K are prebuilt under `artifacts/`. To regenerate them from raw data:

```bash
# Build sparse ratings matrix and id mappings
uv run python -m src.data_prep

# Fit kNN model on the sparse matrix
uv run python -m src.model_knn
```

### 3) Run the Streamlit app

```bash
uv run streamlit run app/app.py
# or: streamlit run app/app.py
```

The app runs at http://localhost:8501 (or 8502 if 8501 is busy).

### 4) Run the FastAPI service

```bash
uv run uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
# or: uvicorn app.api:app --reload
```

API docs will be at http://localhost:8000/docs

## Project Structure

```
Demos/
├── app/
│   ├── app.py            # Streamlit UI
│   └── api.py            # FastAPI service
├── src/
│   ├── data_prep.py      # Build CSR matrix + mappings
│   ├── model_knn.py      # Fit and save NearestNeighbors model
│   ├── recommend.py      # Recommendation logic + DataFrame formatter
│   └── utils.py          # Paths and IO helpers
├── artifacts/            # Prebuilt artifacts (matrix, mappings, kNN)
├── data/raw/ml-100k/     # MovieLens 100K dataset (included)
├── Dockerfile            # Container for FastAPI service
├── pyproject.toml        # uv/pep621 project config
├── requirements.txt      # pip fallback dependencies
└── README.md
```

## Usage

### Streamlit UI
1) Start the app and open the browser
2) Select a `User ID` (min/max shown in the sidebar)
3) Tune `k neighbors`, `Top-N`, and `Neighbor min rating`
4) Click “Recommend” to see the Top-N movies with scores

### API Examples

Health check:

```bash
curl http://localhost:8000/health
```

Get recommendations:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
        "user_id": 42,
        "k_neighbors": 20,
        "topn": 10,
        "min_neighbor_rating": 4.0
      }'
```

Response:

```json
{
  "user_id": 42,
  "items": [
    { "item_id": 50, "title": "Star Wars (1977)", "score": 12.34 },
    { "item_id": 100, "title": "Fargo (1996)", "score": 11.27 }
  ]
}
```

## Dataset

- MovieLens 100K (included under `data/raw/ml-100k/`).
- Artifacts are generated to `artifacts/` as:
  - `R_sparse.npz` (CSR ratings matrix)
  - `uid_to_idx.pkl`, `idx_to_uid.pkl` (user mappings)
  - `iid_to_idx.pkl`, `idx_to_iid.pkl` (item mappings)
  - `items_meta.pkl` (id → title)
  - `knn.joblib` (NearestNeighbors model)

## Docker (API)

Build and run the API as a container:

```bash
docker build -t movie-recs-api .
docker run --rm -p 8000:8000 movie-recs-api
```

Then open http://localhost:8000/docs

## Extending the Project

- Swap kNN for matrix factorization or ANN search
- Add user/item cold-start strategies
- Log interactions and track simple offline metrics
- Deploy the API behind a gateway and add auth/rate limits


## Contribution

Contributions are welcome! Fork the repo and open a PR with your improvements.
