#### This script exports ML Model from MLFLOW to ONNX format , as scikit-learn is heavy on raspberry pi4, hence running with onnxruntime

import mlflow
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# load latest model from mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# finding the best run
runs = mlflow.search_runs(
    experiment_names=["surface-classifier"],
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)

if len(runs) == 0:
    print("No runs found. Train a model first")
    exit(1)

best_run_id = runs.iloc[0]["run_id"]
best_accuracy = runs.iloc[0]["metrics.accuracy"]
print(f"Best run: {best_run_id} (accuracy: {best_accuracy:.3f})")

# loading the sklearn model
model_uri = f"runs:/{best_run_id}/model"
model = mlflow.sklearn.load_model(model_uri)
print(f" Loaded model: {type(model).__name__}")

# Converting to ONNX
# 10 features, float input
initial_type = [("features", FloatTensorType([None, 10]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

#save
output_path = "ML/surface_classifier.onnx"
with open(output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Exported to: {output_path}")
print(f"File Size: {len(onnx_model.SerializeToString()) / 1024:.1f} KB")

# Testing inference with ONNX Runtime
import onnxruntime as rt

sess = rt.InferenceSession(output_path)
# just inputting 10 features test data
test_input = np.array([[900.0, 200.0, 150.0, 80.0, 160.0, 950.0, 820.0, 1000.0, 0.77, 130.0]], dtype=np.float32)
result = sess.run(None, {"features": test_input})
print(f"Test prediction: {result[0][0]}")
print(f"Probabilities: {result[1][0]}")