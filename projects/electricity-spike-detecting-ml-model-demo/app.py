import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model (cached for interactivity)
@st.cache_resource
def load_model():
    return joblib.load('xgboost_model.joblib')

model = load_model()

st.title('Real-Time Energy Spike Prediction with XGBoost')

periods = st.slider('Forecast Hours', min_value=1, max_value=48, value=24)
threshold_multiplier = st.slider('Spike Threshold Multiplier (based on mean)', min_value=1.0, max_value=3.0, value=1.5, step=0.1)

# Create future dataframe
last_date = pd.to_datetime('now')
future_index = pd.date_range(start=last_date, periods=periods, freq='H')
future_df = pd.DataFrame(index=future_index)
future_df = future_df.reset_index().rename(columns={'index': 'Datetime'})
future_df['Datetime'] = pd.to_datetime(future_df['Datetime'])
future_df = future_df.set_index('Datetime')

# Engineer features
future_df['hour'] = future_df.index.hour
future_df['dayofweek'] = future_df.index.dayofweek
future_df['quarter'] = future_df.index.quarter
future_df['month'] = future_df.index.month
future_df['year'] = future_df.index.year
future_df['dayofyear'] = future_df.index.dayofyear
future_df['dayofmonth'] = future_df.index.day
future_df['weekofyear'] = future_df.index.isocalendar().week
future_df['lag1'] = np.mean(future_df['hour'])  # Placeholder
future_df['lag24'] = future_df['lag1'].shift(24, fill_value=0)
future_df['rolling_mean7'] = future_df['lag1'].rolling(window=7).mean().fillna(0)

features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'lag1', 'lag24', 'rolling_mean7']
predictions = model.predict(future_df[features])
future_df['yhat'] = predictions
future_df['yhat_upper'] = predictions + np.std(predictions)  # Simple upper band

mean_demand = future_df['yhat'].mean()
threshold = mean_demand * threshold_multiplier

spikes = future_df[future_df['yhat_upper'] > threshold]

st.line_chart(future_df['yhat'])
st.write('Spike threshold:', threshold)

if not spikes.empty:
    st.warning(f'Spike predicted at {spikes.index[0]}')
else:
    st.success('No spikes predicted in the forecast horizon')