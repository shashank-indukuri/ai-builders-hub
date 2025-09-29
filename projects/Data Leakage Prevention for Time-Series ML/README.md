Data Leakage Prevention for Time-Series ML ⏱️
-----------------------------------------------------------------------------
Forecast closure speed for NYC 311 requests the moment they arrive—without
accidentally training on tomorrow’s data. This project demonstrates how to keep
time-ordered datasets honest using Featuretools, scikit-learn, and walk-forward
evaluation.

Why This Matters
----------------
- 🕒 **As-of-time features:** Every feature is computed at the ticket’s timestamp,
  so the model never peeks into the future.
- 🧱 **No mock data:** Pulls live NYC 311 records straight from the public API.
- 📊 **Honest vs leaky comparison:** Side-by-side ROC-AUC showing the cost of
  shuffling time.
- 🧭 **Transparent signals:** Exports the top “as-of-now” drivers for operations
  teams.
- 💻 **Pure Python workflow:** No external services; everything runs locally with
  open-source tooling.

Key Features
------------
- **Time-aware feature engineering:** Featuretools EntitySet + `cutoff_time`
  ensures every aggregate respects the request’s creation moment.
- **Dual evaluations:** Walk-forward `TimeSeriesSplit` vs. deliberately leaky
  `StratifiedShuffleSplit` to highlight metric inflation.
- **Visual narrative:** Matplotlib bar chart comparing honest and shuffled
  ROC-AUC scores for stakeholders.
- **Explainability:** RandomForest feature importances saved to CSV for quick
  handoff.
- **Reusable scaffold:** Drop in any timestamped dataset to replicate the
  pipeline.

Installation and Setup
----------------------

### Prerequisites

- Python 3.12 or later
- `uv` (recommended) or `pip`

### 1. Install Dependencies

Using `uv` (recommended):

```
uv sync
```

Using `pip`:

```
pip install -r requirements.txt
```

### 2. Run the Script

```
uv run python main.py
# or with pip: python main.py
```

Project Structure
-----------------

```
data-leakage-prevention-for-time-series-ml/
├── main.py                     # Time-aware Featuretools + evaluation workflow
├── nyc311_48h_bar.png          # Honest vs shuffled ROC-AUC comparison chart
├── nyc311_top_features.csv     # Top “as-of-time” feature importances
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Deterministic dependency lockfile (uv)
└── README.md
```

Usage
-----

1. Run the script to download the latest NYC 311 data and build features.
2. Watch the console for class-balance diagnostics and evaluation summaries.
3. Open `nyc311_48h_bar.png` to show stakeholders how leakage inflates scores.
4. Inspect `nyc311_top_features.csv` for the most predictive time-aware signals.
5. Swap in your own timestamped dataset by updating the API fetch and entity
   definitions.

Contribution
------------

Contributions are welcome! Fork the repository and submit a pull request with
improvements, additional evaluations, or new deployment recipes.
