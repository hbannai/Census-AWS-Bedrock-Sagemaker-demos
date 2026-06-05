"""
SageMaker Population Forecasting - Training & Deployment Script
================================================================
This script trains a time-series forecasting model on 10 years of
Census population data (2012-2022) and deploys it as a SageMaker endpoint.

Usage:
    1. Run this script from a SageMaker notebook or local machine with AWS credentials
    2. Ensure the census data has been pulled to S3 first (run Lambda: census-demo-data-pull)
    3. Update BUCKET and PREFIX below to match your CFN parameters

Prerequisites:
    pip install boto3 pandas scikit-learn sagemaker
"""

import boto3
import json
import pandas as pd
import numpy as np
from io import StringIO
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker import get_execution_role
import os

# ============================================================
# CONFIGURATION - Update these to match your CFN parameters
# ============================================================
BUCKET = os.environ.get('S3_BUCKET', 'YOUR-BUCKET-NAME')  # Replace or set env var
PREFIX = os.environ.get('S3_PREFIX', 'census-demo/')
REGION = os.environ.get('AWS_REGION', 'us-east-1')
ENDPOINT_NAME = 'census-forecast-endpoint'
INSTANCE_TYPE = os.environ.get('SAGEMAKER_INSTANCE', 'ml.m5.xlarge')


def get_ssm_params():
    """Read configuration from SSM Parameter Store (if deployed via CFN)."""
    try:
        ssm = boto3.client('ssm', region_name=REGION)
        bucket = ssm.get_parameter(Name='/census-demo/s3/bucket-name')['Parameter']['Value']
        prefix = ssm.get_parameter(Name='/census-demo/s3/prefix')['Parameter']['Value']
        return bucket, prefix
    except Exception:
        return BUCKET, PREFIX


def load_training_data():
    """Load historical population data from S3."""
    bucket, prefix = get_ssm_params()
    s3 = boto3.client('s3', region_name=REGION)

    obj = s3.get_object(Bucket=bucket, Key=f"{prefix}raw/historical_population.csv")
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

    print(f"Loaded {len(df):,} rows of historical population data")
    print(f"Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Counties: {df['County'].nunique():,}")
    return df


def prepare_features(df):
    """
    Create features for forecasting.
    For each county, calculate:
    - Population growth rate (year over year)
    - Average growth rate
    - Population trend (slope)
    """
    # Clean data
    df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
    df = df.dropna(subset=['Population'])
    df = df[df['Population'] > 0]

    # Create county identifier
    df['FIPS'] = df['State_FIPS'].astype(str).str.zfill(2) + df['County_FIPS'].astype(str).str.zfill(3)

    # Calculate features per county
    features = []
    for fips, group in df.groupby('FIPS'):
        group = group.sort_values('Year')
        if len(group) < 5:  # Need at least 5 years of data
            continue

        pops = group['Population'].values
        years = group['Year'].values

        # Calculate growth rates
        growth_rates = np.diff(pops) / pops[:-1]
        avg_growth_rate = np.mean(growth_rates)

        # Linear trend (slope)
        slope, _ = np.polyfit(years - years[0], pops, 1)

        # Latest population
        latest_pop = pops[-1]
        latest_year = years[-1]

        features.append({
            'FIPS': fips,
            'County': group['County'].iloc[-1],
            'State_FIPS': group['State_FIPS'].iloc[-1],
            'County_FIPS': group['County_FIPS'].iloc[-1],
            'Latest_Population': latest_pop,
            'Latest_Year': latest_year,
            'Avg_Growth_Rate': avg_growth_rate,
            'Trend_Slope': slope,
            'Pop_2012': pops[0] if len(pops) > 0 else 0,
            'Pop_2017': group[group['Year'] == 2017]['Population'].values[0] if 2017 in years else 0,
            'Pop_2022': pops[-1],
            'Num_Years_Data': len(group)
        })

    features_df = pd.DataFrame(features)
    print(f"Prepared features for {len(features_df):,} counties")
    return features_df


def upload_training_data(features_df):
    """Upload prepared training data to S3."""
    bucket, prefix = get_ssm_params()
    s3 = boto3.client('s3', region_name=REGION)

    # Save training CSV
    csv_buffer = StringIO()
    features_df.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}sagemaker/train/training_data.csv",
        Body=csv_buffer.getvalue().encode('utf-8')
    )
    print(f"Uploaded training data to s3://{bucket}/{prefix}sagemaker/train/training_data.csv")
    return f"s3://{bucket}/{prefix}sagemaker/train/"


def train_model(training_data_s3_path):
    """
    Train a simple forecasting model using SageMaker SKLearn.
    Uses linear regression + growth rate features to predict future population.
    """
    bucket, prefix = get_ssm_params()

    try:
        role = get_execution_role()
    except ValueError:
        # Not running in SageMaker - get role from SSM
        ssm = boto3.client('ssm', region_name=REGION)
        # Use the SageMaker role ARN from CFN output
        iam = boto3.client('iam', region_name=REGION)
        role = f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/census-demo-sagemaker-role"

    sklearn_estimator = SKLearn(
        entry_point='train_script.py',
        source_dir='.',
        role=role,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version='1.2-1',
        py_version='py3',
        output_path=f"s3://{bucket}/{prefix}sagemaker/model/",
        hyperparameters={
            'forecast_years': 5
        }
    )

    sklearn_estimator.fit({'train': training_data_s3_path})
    print("Training complete!")
    return sklearn_estimator


def deploy_endpoint(estimator):
    """Deploy the trained model as a SageMaker endpoint."""
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME
    )
    print(f"Endpoint deployed: {ENDPOINT_NAME}")
    return predictor


def test_prediction(county_fips='48201', state_fips='48'):
    """Test the deployed endpoint with Harris County, TX."""
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=REGION)

    payload = json.dumps({
        'county_fips': county_fips,
        'state_fips': state_fips,
        'forecast_years': [2024, 2025, 2026, 2027, 2028]
    })

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=payload
    )

    result = json.loads(response['Body'].read().decode())
    print(f"\n{'='*50}")
    print(f"PREDICTION - County FIPS: {county_fips}")
    print(f"{'='*50}")
    for year, pop in zip([2024, 2025, 2026, 2027, 2028], result.get('predictions', [])):
        print(f"  {year}: {pop:,.0f}")
    print(f"  Confidence: {result.get('confidence', 'N/A')}")
    print(f"  Growth Rate: {result.get('avg_growth_rate', 'N/A')}")
    return result


def cleanup_endpoint():
    """Delete the SageMaker endpoint to stop charges."""
    sagemaker_client = boto3.client('sagemaker', region_name=REGION)
    try:
        sagemaker_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Deleted endpoint: {ENDPOINT_NAME}")
    except Exception as e:
        print(f"Could not delete endpoint: {e}")


# ============================================================
# MAIN WORKFLOW
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("CENSUS POPULATION FORECASTING - SAGEMAKER DEMO")
    print("=" * 60)

    # Step 1: Load data
    print("\n[Step 1] Loading historical population data from S3...")
    df = load_training_data()

    # Step 2: Prepare features
    print("\n[Step 2] Preparing features...")
    features_df = prepare_features(df)

    # Step 3: Upload training data
    print("\n[Step 3] Uploading training data to S3...")
    s3_path = upload_training_data(features_df)

    # Step 4: Train model
    print("\n[Step 4] Training forecasting model on SageMaker...")
    estimator = train_model(s3_path)

    # Step 5: Deploy endpoint
    print("\n[Step 5] Deploying endpoint...")
    predictor = deploy_endpoint(estimator)

    # Step 6: Test prediction
    print("\n[Step 6] Testing prediction...")
    test_prediction('48201', '48')  # Harris County, TX

    print("\n✅ Demo B setup complete!")
    print(f"   Endpoint name: {ENDPOINT_NAME}")
    print(f"   To test: python forecast_training.py --test-only")
    print(f"   To cleanup: python forecast_training.py --cleanup")
