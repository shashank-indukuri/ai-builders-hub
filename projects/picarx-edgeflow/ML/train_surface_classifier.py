"""
Author: Mohan Kandregula
Surface Classifier Training Script for testing initially before wiring it into the Dagster
Pulls features from Snowflake, trains a Random Forest, logs to MLflow
"""

import os
import mlflow
import pandas as pd
import snowflake.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# -- Config --
SNOWFLAKE_CONFIG = {
    "account": "<<YOUR_SNOWFLAKE_ACCOUNT_NAME>>",
    "user": "<<YOUR_SNOWFLAKE_USER_ID>>",
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "database": "PICARX_DB",
    "schema": "GOLD",
    "warehouse": "PICARX_WH",
}

FEATURE_COLUMNS = [
    "GS_MEAN",
    "GS_SPREAD",
    "GS_LEFT_ROLLING_STD_10",
    "GS_CENTER_ROLLING_STD_10",
    "GS_RIGHT_ROLLING_STD_10",
    "GS_LEFT_ROLLING_MEAN_10",
    "GS_CENTER_ROLLING_MEAN_10",
    "GS_RIGHT_ROLLING_MEAN_10",
    "GS_CENTER_TO_OUTER_RATIO",
    "GS_TEXTURE_SCORE",
]

LABEL_COLUMN = "SURFACE_LABEL"

# --- Step 1: Fetching features from the snowflake
print("Connecting to snowflake...")
conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
query = "SELECT * FROM PICARX_DB.GOLD.ML_SURFACE_FEATURES"
df = pd.read_sql(query, conn)
print(f"Columns: {df.columns.tolist()}")
conn.close()
print(f" Loaded {len(df)} rows from snowflake")
print(f"Label distribution: \n{df[LABEL_COLUMN].value_counts()}\n")

# --- Step 2: Preparing Features and labels
X = df[FEATURE_COLUMNS]
y = df[LABEL_COLUMN]

# Drop rows with any null features
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]
print(f" After dropping nulls: {len(X)} rows\n")

# --- Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y    # stratify=y ensures each surface has proportional representation in both train and test sets
)

print(f"Training set: {len(X_train)} rows")
print(f" Test set: {len(X_test)} rows")

# --- Step 4: Train and evaluate
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("surface-classifier")

with mlflow.start_run(run_name="random_forest_v1"):
    # Model parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "random_state": 42,
    }

    # Train
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.3f}\n")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # Feature importance - which features mattered most? MLFLOW's feature
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    print("\n Feature Importance:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")
    
    # Step 5: Log Everything to MLflow
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)

    # Log per-class metrics
    for label in y.unique():
        mask = y_test == label
        class_acc = accuracy_score(y_test[mask], y_pred[mask])
        mlflow.log_metric(f"accuracy_{label}", class_acc)

    # Log feature importances
    for feat, imp in importances.items():
        mlflow.log_metric(f"importance_{feat}", imp)
    
    # Log the model artifact
    mlflow.sklearn.log_model(model, "model")

    print(f"\nMLflow run logged, Experiment is : surface-classifier")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
