# Real-Time Electricity Spike Detection with XGBoost ⚡

Detect and predict electricity consumption spikes using machine learning. This project demonstrates an end-to-end solution for forecasting energy demand spikes, built with XGBoost and Streamlit.

## Why This Matters

⚡ **Proactive Energy Management**: Predict and prepare for electricity demand spikes before they occur

📊 **Data-Driven Insights**: Leverage historical consumption patterns for accurate forecasting

🤖 **Automated Alerts**: Get notified about potential spikes through an intuitive web interface

🌐 **Cloud-Ready**: Includes templates for AWS CloudFormation and Kubernetes deployment

## Key Features

- **Interactive Dashboard**: Real-time visualization of energy consumption and spike predictions
- **Customizable Thresholds**: Adjust sensitivity to detect spikes based on your needs
- **Temporal Feature Engineering**: Advanced time-based feature extraction for better predictions
- **Model Persistence**: Save and load trained models for consistent performance
- **Deployment Ready**: Includes infrastructure-as-code templates for cloud deployment

## Project Structure

```
electricity-spike-detecting-ml-model-demo/
├── app.py                # Streamlit web application
├── train_with_eda.py     # Model training with exploratory data analysis
├── predict_and_act.py    # Prediction and action triggers
├── xgboost_model.joblib  # Trained XGBoost model
├── data/                 # Data directory
│   └── PJME_hourly.csv   # Sample energy consumption data
└── README.md             # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip or uv package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohankandregula/ai-builders-hub.git
   cd ai-builders-hub/projects/electricity-spike-detecting-ml-model-demo
   ```

2. **Install dependencies**
   Using pip:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using uv (recommended):
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python train_with_eda.py
   python predict_and_act.py
   streamlit run app.py
   ```
   The app will be available at `http://localhost:8501`

## Usage

1. **Training the Model**
   Run the training script to train the XGBoost model:
   ```bash
   python train_with_eda.py
   ```
   This will:
   - Download the dataset (if not present)
   - Perform exploratory data analysis
   - Train and save the XGBoost model

2. **Using the Web Interface**
   - Adjust the forecast horizon using the slider
   - Set the spike detection threshold
   - View predicted spikes in the interactive chart

3. **Programmatic Usage**
   Use the `predict_and_act.py` script to integrate predictions into your workflow:
   ```python
   from predict_and_act import load_model, predict_spikes
   
   # Load the trained model
   model = load_model()
   
   # Make predictions
   future_df = prepare_future_data()  # Your data preparation function
   predictions, spikes, threshold = predict_spikes(model, future_df)
   ```

## Deployment

The project includes templates for deployment:

### AWS CloudFormation
Generate a CloudFormation template:
```python
python -c "from predict_and_act import generate_cft; generate_cft()"
```

### Kubernetes
Generate a Kubernetes pod specification:
```python
python -c "from predict_and_act import create_k8s_pod_yaml; create_k8s_pod_yaml()"
```

## Data Sources

The model is trained on the [PJM Energy Consumption Dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption), which provides hourly energy consumption data.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
