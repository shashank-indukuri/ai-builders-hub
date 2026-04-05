# Training the ML Model after feature selection is done in the DBT

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import snowflake.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from dagster import asset, AssetExecutionContext, MetadataValue
from dagster_pipelines.resources import SnowflakeResource

# Features Identified
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

MIN_NEW_ROWS_FOR_RETRAIN = 1500


@asset(
    deps=["ml_surface_features"],
    description="Trains surface classifier on latest feature data and skips if exists any insufficient new data",
)
def surface_classifier_model(
    context: AssetExecutionContext,
    snowflake: SnowflakeResource,
):
    # Pull all the features
    sf = snowflake.get_client()
    conn = sf.get_connection("GOLD")
    df = pd.read_sql("SELECT * FROM ML_SURFACE_FEATURES", conn)
    conn.close()
    
    # Making train and test sets for the ML Model Training
    X = df[FEATURE_COLUMNS]
    y = df["SURFACE_LABEL"]

    # pre-processing of data
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    total_rows = len(X)

    context.log.info(f"Total labeled rows: {total_rows}, {y.nunique()} classes")

    # check if enough new data arrived, since last training, then re training with the same params
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("surface-classifier")

    last_runs = mlflow.search_runs(
        experiment_names=["surface-classifier"],
        order_by=["start_time DESC"],
        max_results=1,
    )
    
    # Model retraining condition
    if len(last_runs) > 0 and "metrics.training_rows" in last_runs.columns:
        last_training_rows = int(last_runs.iloc[0]["metrics.training_rows"])
        last_total = int(last_training_rows / 0.8)  # 80/20 split means total was training/0.8
        new_rows = total_rows - last_total

        context.log.info(f"Last training used {last_total} rows. New rows since: {new_rows}")

        if new_rows < MIN_NEW_ROWS_FOR_RETRAIN:
            context.log.info(
                f"Only {new_rows} new rows (need {MIN_NEW_ROWS_FOR_RETRAIN}). Skipping retraining."
            )
            context.add_output_metadata({
                "status": MetadataValue.text(f"skipped — only {new_rows} new rows"),
                "total_rows": MetadataValue.int(total_rows),
                "new_rows_since_last_train": MetadataValue.int(new_rows),
                "threshold": MetadataValue.int(MIN_NEW_ROWS_FOR_RETRAIN),
            })
            return
    else:
        context.log.info("No previous training found. Training first model.")

    # Train the ML Model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    context.log.info(f"Training: {len(X_train)} rows, Testing: {len(X_test)} rows")
    
    # Log everything into MLFLOW for comparisions later
    with mlflow.start_run(run_name="dagster_triggered"):
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("training_rows", len(X_train))
        mlflow.log_metric("test_rows", len(X_test))
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id

    context.log.info(f"Accuracy: {accuracy:.3f}")
    context.log.info(f"MLflow run ID: {run_id}")
    context.log.info(f"\n{report}")

    context.add_output_metadata({
        "status": MetadataValue.text("trained"),
        "accuracy": MetadataValue.float(accuracy),
        "training_rows": MetadataValue.int(len(X_train)),
        "test_rows": MetadataValue.int(len(X_test)),
        "classes": MetadataValue.int(y.nunique()),
        "mlflow_run_id": MetadataValue.text(run_id),
    })