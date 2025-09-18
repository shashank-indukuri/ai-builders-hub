import pandas as pd
import joblib
import json
import numpy as np

# Load trained model
def load_model(path='xgboost_model.joblib'):
    return joblib.load(path)

def predict_spikes(model, future_df, features, threshold_multiplier=1.5):
    predictions = model.predict(future_df[features])
    future_df['yhat'] = predictions
    future_df['yhat_upper'] = predictions + np.std(predictions)  # Simple upper band for demo

    mean_demand = future_df['yhat'].mean()
    threshold = mean_demand * threshold_multiplier
    spikes = future_df[future_df['yhat_upper'] > threshold]
    return future_df, spikes, threshold

# Simulated action triggers
def generate_cft():
    cft = {
        "Resources": {
            "MyCluster": {
                "Type": "AWS::EC2::Instance",
                "Properties": {
                    "InstanceType": "t3.medium"
                }
            }
        }
    }
    with open('cft.json', 'w') as f:
        json.dump(cft, f)
    print('CloudFormation template generated at cft.json')

def create_k8s_pod_yaml():
    pod_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: spike-pod
spec:
  containers:
  - name: worker
    image: busybox
"""
    with open('pod.yaml', 'w') as f:
        f.write(pod_yaml)
    print('Kubernetes pod YAML created at pod.yaml')

def main():
    model = load_model()
    
    # Create future dataframe (e.g., next 24 hours with features)
    last_date = pd.to_datetime('now')  # Use current time for demo
    future_index = pd.date_range(start=last_date, periods=24, freq='H')
    future_df = pd.DataFrame(index=future_index)
    future_df = future_df.reset_index().rename(columns={'index': 'Datetime'})
    future_df['Datetime'] = pd.to_datetime(future_df['Datetime'])
    future_df = future_df.set_index('Datetime')
    
    # Engineer same features for future
    future_df['hour'] = future_df.index.hour
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['quarter'] = future_df.index.quarter
    future_df['month'] = future_df.index.month
    future_df['year'] = future_df.index.year
    future_df['dayofyear'] = future_df.index.dayofyear
    future_df['dayofmonth'] = future_df.index.day
    future_df['weekofyear'] = future_df.index.isocalendar().week
    
    # Dummy lags/rolling (in real, use last known values)
    future_df['lag1'] = np.mean(future_df['hour'])  # Placeholder
    future_df['lag24'] = future_df['lag1'].shift(24, fill_value=0)
    future_df['rolling_mean7'] = future_df['lag1'].rolling(window=7).mean().fillna(0)
    
    forecast, spikes, threshold = predict_spikes(model, future_df, features=['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'lag1', 'lag24', 'rolling_mean7'])
    print(f'Predicted forecast for next 24 periods')
    print(f'Using spike threshold: {threshold:.2f}')

    if not spikes.empty:
        print('Spike predicted! Taking actions...')
        generate_cft()
        create_k8s_pod_yaml()
    else:
        print('No spike detected in the forecast')

if __name__ == "__main__":
    main()