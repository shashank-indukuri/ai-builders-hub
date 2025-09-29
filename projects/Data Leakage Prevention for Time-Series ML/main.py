# Predict at report time: “Will this NYC 311 request be closed within 24 hours?”
# Libraries: Featuretools (DFS + cutoff_time), scikit-learn (TimeSeriesSplit), pandas/matplotlib
# No fallbacks, no mock rows. Real public CSV; multi-table EntitySet; time-aware features; time-based backtest.

import warnings
import os

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Could not infer format, so each element will be parsed individually, falling back to `dateutil`.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The provided callable <function (max|min|std).*",
    category=FutureWarning,
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

import featuretools as ft
from sklearn.model_selection import TimeSeriesSplit, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1) Load recent NYC 311 data (public, no sign-in). Adjust $limit if needed.
URL = (
    "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?"
    "$limit=60000&"
    "$select=unique_key,created_date,closed_date,agency,complaint_type,descriptor,"
    "incident_zip,city,borough,latitude,longitude&"
    "$order=created_date%20DESC"  # space encoded as %20
)
print("Downloading NYC 311 sample…")
df = pd.read_csv(URL)

# 2) Timestamps and basic fields
for col in ["created_date", "closed_date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")
df = df.sort_values("created_date").dropna(subset=["created_date"]).reset_index(drop=True)

# 3) Label at report time: closed within 48 hours from creation
threshold = pd.Timedelta(hours=24)
df = df.dropna(subset=["closed_date"])
df["closed_within_24h"] = ((df["closed_date"] - df["created_date"]) <= threshold).astype(int)
print("Label distribution after thresholding:")
print(df["closed_within_24h"].value_counts(dropna=False))

# 4) Minimal columns; drop closure timestamp from features (to avoid obvious leakage)
keep = [
    "unique_key","created_date","agency","complaint_type","descriptor",
    "incident_zip","borough","city","latitude","longitude","closed_within_24h"
]
df = df[keep].copy()

# 5) Clean identifiers for parents and fill NA to ensure stable keys
for col in ["agency","complaint_type","descriptor","incident_zip","borough","city"]:
    if col in df.columns:
        df[col] = df[col].astype("string").fillna("UNKNOWN")

# 6) Build multi-table EntitySet so DFS can aggregate history by group parents
es = ft.EntitySet(id="nyc311")
es = es.add_dataframe(
    dataframe_name="requests",
    dataframe=df,
    index="unique_key",
    time_index="created_date"
)

# Ensure the primary dataframe has a time index registered with Woodwork
es["requests"].ww.set_time_index("created_date")

# Normalize parents to unlock aggregations like “count/mean up to cutoff_time” per group
parents = ["incident_zip","complaint_type","agency","borough","city"]
for p in parents:
    if p in df.columns:
        # Skip normalization if cardinality explodes (still okay for demo sizes)
        es = es.normalize_dataframe(
            base_dataframe_name="requests",
            new_dataframe_name=p,
            index=p
        )

# Populate last_time_index on all dataframes to support training_window
es.add_last_time_indexes()

# 7) Build per-row cutoff_time at the report moment and use a recent training window
cutoff_df = df[["unique_key","created_date"]].rename(columns={"unique_key":"instance_id","created_date":"time"})

agg_primitives = ["count","mean","max","min","std"]  # applied across normalized parents
trans_primitives = ["weekday","month","day","time_since"]  # simple time transforms
CPU_WORKERS = 1  # Keeping dfs on a single worker avoids requiring Dask
# CPU_WORKERS = max(1, (os.cpu_count() or 2) - 1)
DFS_CHUNK_SIZE = 10000

print("Running DFS with cutoff_time (time-aware features)…")
X_honest, defs_honest = ft.dfs(
    entityset=es,
    target_dataframe_name="requests",
    agg_primitives=agg_primitives,
    trans_primitives=trans_primitives,
    max_depth=2,
    cutoff_time=cutoff_df,                 # per-row time boundary (as-of features)
    cutoff_time_in_index=True,             # keep cutoff timestamp alongside rows
    training_window="30d",                 # use only the last 30 days of history for each row
    verbose=False,
    n_jobs=CPU_WORKERS,                     # keep n_jobs=1 to avoid Dask requirement
    chunk_size=DFS_CHUNK_SIZE
)

def align_target(labels_df, feature_index, label="closed_within_24h"):
    base = labels_df.set_index("unique_key")[label].astype(int)
    if isinstance(feature_index, pd.MultiIndex):
        instance_ids = feature_index.get_level_values(0)
        aligned = base.reindex(instance_ids)
        if aligned.isna().any():
            missing = int(aligned.isna().sum())
            raise RuntimeError(f"{missing} target values missing after reindex. Check cutoff_time alignment.")
        aligned = aligned.astype(int)
        return pd.Series(aligned.to_numpy(), index=feature_index, name=label)
    aligned = base.reindex(feature_index)
    if aligned.isna().any():
        missing = int(aligned.isna().sum())
        raise RuntimeError(f"{missing} target values missing after reindex. Check target alignment.")
    return aligned.astype(int)


y = align_target(df, X_honest.index)

# 8) Shortcut version (“looks better on paper”): DFS without cutoff_time (peeks at future)
print("Running DFS without cutoff_time (shortcut)…")
X_shortcut, defs_shortcut = ft.dfs(
    entityset=es,
    target_dataframe_name="requests",
    agg_primitives=agg_primitives,
    trans_primitives=trans_primitives,
    max_depth=2,
    verbose=False,
    n_jobs=CPU_WORKERS,
    chunk_size=DFS_CHUNK_SIZE
)
y_short = align_target(df, X_shortcut.index)

# 9) Feature cleanup (drop any target remnants if present)
X_honest = X_honest.drop(columns=[c for c in X_honest.columns if c == "closed_within_24h"], errors="ignore")
X_shortcut = X_shortcut.drop(columns=[c for c in X_shortcut.columns if c == "closed_within_24h"], errors="ignore")

# Ensure numeric matrix for the model
X_honest_num = X_honest.select_dtypes(include=[np.number]).fillna(0.0)
X_shortcut_num = X_shortcut.select_dtypes(include=[np.number]).fillna(0.0)

# 10) Time-aware evaluation (honest) vs shuffled-time evaluation (shortcut)
def eval_time_series_auc(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for tr, te in tscv.split(X):
        # Skip folds that lack both classes (rare but possible)
        if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
            continue
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        proba = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], proba))
    return float(np.mean(aucs)) if aucs else np.nan

def eval_shuffled_auc(X, y):
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.25, random_state=42)
    aucs = []
    for tr, te in sss.split(X, y):
        if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
            continue
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        proba = clf.predict_proba(X.iloc[te])
        if proba.shape[1] < 2:
            continue
        aucs.append(roc_auc_score(y.iloc[te], proba[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan

print("Evaluating (time-aware, walk-forward)…")
honest_auc = eval_time_series_auc(X_honest_num, y)

print("Evaluating (shuffled across time)…")
shortcut_auc = eval_shuffled_auc(X_shortcut_num, y_short)

# 11) Visualization: simple bar chart for a non-technical audience
plt.figure(figsize=(7,4.5))
labels = ["Only info available that day", "Shuffled across time"]
scores = [honest_auc, shortcut_auc]
colors = ["#2E7D32", "#FF7043"]
plt.bar(labels, scores, color=colors)
plt.ylim(0.5, 1.0)
for i, v in enumerate(scores):
    if not np.isnan(v):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
plt.ylabel("ROC AUC (higher is better)")
plt.title("Using tomorrow’s info makes us look better on paper — not in real life")
plt.tight_layout()
plt.savefig("nyc311_48h_bar.png", dpi=170)
print("Saved chart: nyc311_48h_bar.png")

# 12) Top features from the time-aware run (single final model for illustration)
# Use the last fold’s fit by retraining once over the last 75% to get importances
cut = int(len(X_honest_num) * 0.25)
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_honest_num.iloc[cut:], y.iloc[cut:])
print("Label distribution in final training slice:")
print(y.iloc[cut:].value_counts(dropna=False))
imp = pd.Series(clf.feature_importances_, index=X_honest_num.columns).sort_values(ascending=False)
top10 = imp.head(10).reset_index()
top10.columns = ["feature_name", "importance"]
top10.to_csv("nyc311_top_features.csv", index=False)
print("Saved top features: nyc311_top_features.csv")
print(top10)
