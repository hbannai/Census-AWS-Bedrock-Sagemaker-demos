"""
SageMaker Training Script - Population Forecasting Model
=========================================================
This script is executed BY SageMaker during the training job.
It reads training data, trains a model, and saves it as model artifacts.

Do NOT run this directly — it's invoked by forecast_training.py via SageMaker.
"""

import argparse
import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def train(args):
    """Train the population forecasting model."""

    # Load training data
    train_dir = args.train
    df = pd.read_csv(os.path.join(train_dir, 'training_data.csv'))
    print(f"Training data: {len(df)} counties")

    # Features for prediction
    feature_cols = ['Latest_Population', 'Avg_Growth_Rate', 'Trend_Slope',
                    'Pop_2012', 'Pop_2017', 'Num_Years_Data']

    # Target: predict population growth rate (which we'll use for forecasting)
    df['Target_Growth'] = df['Avg_Growth_Rate']

    # Remove invalid rows
    df = df.dropna(subset=feature_cols + ['Target_Growth'])
    df = df[np.isfinite(df['Target_Growth'])]

    X = df[feature_cols].values
    y = df['Target_Growth'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance:")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²:  {r2:.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save county data for lookup during inference
    county_data = df[['FIPS', 'County', 'State_FIPS', 'County_FIPS',
                      'Latest_Population', 'Avg_Growth_Rate', 'Trend_Slope',
                      'Pop_2012', 'Pop_2017', 'Num_Years_Data']].copy()
    county_path = os.path.join(args.model_dir, 'county_data.pkl')
    with open(county_path, 'wb') as f:
        pickle.dump(county_data, f)

    # Save feature columns for inference
    meta_path = os.path.join(args.model_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'feature_cols': feature_cols,
            'mae': mae,
            'r2': r2,
            'num_counties': len(df),
            'training_years': '2012-2022'
        }, f)

    print(f"Model saved to {model_path}")
    print(f"County data saved to {county_path}")


def model_fn(model_dir):
    """Load model for inference."""
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def input_fn(request_body, request_content_type):
    """Parse incoming inference request."""
    if request_content_type == 'application/json':
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions."""
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    # Load county data
    county_path = os.path.join(model_dir, 'county_data.pkl')
    with open(county_path, 'rb') as f:
        county_data = pickle.load(f)

    # Find the requested county
    county_fips = str(input_data.get('county_fips', '')).zfill(3)
    state_fips = str(input_data.get('state_fips', '')).zfill(2)
    fips = state_fips + county_fips

    county_row = county_data[county_data['FIPS'] == int(fips)]

    if county_row.empty:
        return {
            'error': f'County not found: FIPS {fips}',
            'available_counties': len(county_data)
        }

    row = county_row.iloc[0]

    # Prepare features
    feature_cols = ['Latest_Population', 'Avg_Growth_Rate', 'Trend_Slope',
                    'Pop_2012', 'Pop_2017', 'Num_Years_Data']
    features = row[feature_cols].values.reshape(1, -1)

    # Predict growth rate
    predicted_growth_rate = model.predict(features)[0]

    # Generate forecasts
    forecast_years = input_data.get('forecast_years', [2024, 2025, 2026, 2027, 2028])
    latest_pop = row['Latest_Population']
    predictions = []

    for i, year in enumerate(forecast_years):
        years_ahead = year - 2022
        predicted_pop = latest_pop * (1 + predicted_growth_rate) ** years_ahead
        predictions.append(round(predicted_pop))

    return {
        'county': row['County'],
        'fips': fips,
        'latest_population_2022': int(latest_pop),
        'predicted_growth_rate': round(predicted_growth_rate, 4),
        'avg_historical_growth_rate': round(row['Avg_Growth_Rate'], 4),
        'predictions': predictions,
        'forecast_years': forecast_years,
        'confidence': 'high' if row['Num_Years_Data'] >= 8 else 'medium',
        'model_metrics': {
            'based_on_years': int(row['Num_Years_Data']),
            'trend_slope': round(row['Trend_Slope'], 2)
        }
    }


def output_fn(prediction, accept):
    """Format output."""
    return json.dumps(prediction), 'application/json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forecast_years', type=int, default=5)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    args = parser.parse_args()

    train(args)
