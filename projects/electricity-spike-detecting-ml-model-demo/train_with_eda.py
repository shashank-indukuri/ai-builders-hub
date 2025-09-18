import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
import os
import numpy as np

# Step 1: Load and explore the data (EDA)
def load_and_eda(path='data/PJME_hourly.csv'):
    if not os.path.exists(path):
        os.makedirs('data', exist_ok=True)
        url = "https://raw.githubusercontent.com/robikscube/hourly-energy-consumption/master/PJME_hourly.csv"
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
        print(f'Data downloaded to {path}')
    else:
        df = pd.read_csv(path)
    
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    df = df.sort_index()
    
    print('Initial data sample:')
    print(df.head())

    # Summary stats
    print('\nSummary statistics (PJME_MW):')
    print(df['PJME_MW'].describe())  # Mean, std, min/max—spot potential spikes (high max).

    # Check missing
    missing = df['PJME_MW'].isnull().sum()
    print(f'\nMissing values: {missing}')
    df['PJME_MW'] = df['PJME_MW'].fillna(method='ffill')

    # Visualize trend and seasonality
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df['PJME_MW'])
    plt.title('Energy Consumption Over Time (Trend/Spikes)')
    plt.xlabel('Datetime')
    plt.ylabel('PJME_MW')
    plt.show()  # "See daily/seasonal spikes—model will learn these."

    # Histogram
    plt.figure(figsize=(10,5))
    sns.histplot(df['PJME_MW'], bins=50)
    plt.title('Energy Distribution (Outliers/Spikes)')
    plt.show()  # "Skewed data indicates anomalies we'll predict."

    return df

# Step 2: Feature engineering and analysis
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    
    # Lag features (previous values)
    df['lag1'] = df['PJME_MW'].shift(1)
    df['lag24'] = df['PJME_MW'].shift(24)  # Previous day same hour
    
    # Rolling stats
    df['rolling_mean7'] = df['PJME_MW'].rolling(window=7).mean()
    
    df = df.dropna()  # Drop NaNs from shifts
    print('\nFeature Analysis: Engineered time-based (hour/month) and lag/rolling features to capture patterns/spikes.')
    return df

# Step 3: Train/test split (chronological)
def train_test_split(df, test_size=0.2):
    split_point = int(len(df) * (1 - test_size))
    train = df.iloc[:split_point]
    test = df.iloc[split_point:]
    print(f'\nTrain size: {len(train)}\nTest size: {len(test)}')
    return train, test

# Step 4: Train XGBoost and evaluate
def train_and_evaluate(train, test, features, target='PJME_MW'):
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    model = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, learning_rate=0.01, max_depth=3)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
    
    # Predict and evaluate
    test['prediction'] = model.predict(X_test)
    mae = mean_absolute_error(test[target], test['prediction'])
    rmse = np.sqrt(mean_squared_error(test[target], test['prediction']))
    print(f'\nMAE: {mae:.2f}, RMSE: {rmse:.2f}')
    
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(test.index, test[target], label='Actual')
    plt.plot(test.index, test['prediction'], label='Predicted')
    plt.legend()
    plt.title('Test Set: Actual vs Predicted')
    plt.show()
    
    # Feature importance
    fi = pd.DataFrame(data=model.feature_importances_, index=features, columns=['importance'])
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
    plt.show()
    
    return model

# Step 5: Save model
def save_model(model, path='xgboost_model.joblib'):
    joblib.dump(model, path)
    print(f'Model saved to {path}')

# ---- Main execution ----
print('Step 1: Load & explore data...')
df = load_and_eda()

print('Step 2: Feature engineering...')
df = create_features(df)
features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'lag1', 'lag24', 'rolling_mean7']

print('Step 3: Train/test split...')
train, test = train_test_split(df)

print('Step 4: Train & evaluate model...')
model = train_and_evaluate(train, test, features)

print('Step 5: Save trained model...')
save_model(model)