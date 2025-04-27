"""
Example of how to perform inference using a saved pipeline model.
This demonstrates how preprocessing is automatically applied during inference.
"""

import joblib
import pandas as pd
import numpy as np
import mlflow
import argparse

def load_model_from_local(model_path):
    """Load a model saved locally"""
    return joblib.load(model_path)

def load_model_from_mlflow(model_name, model_version=None, stage=None):
    """Load a model from MLflow model registry"""
    if stage:
        model_uri = f"models:/{model_name}/{stage}"
    elif model_version:
        model_uri = f"models:/{model_name}/{model_version}"
    else:
        # Get latest version
        model_uri = f"models:/{model_name}/latest"
    
    return mlflow.sklearn.load_model(model_uri)

def create_example_input():
    """Create a sample input for testing"""
    # Create a sample input matching the format of the raw Adult dataset
    sample = {
        'age': 38,
        'workclass': 'Private',
        'fnlwgt': 215646,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Divorced',
        'occupation': 'Handlers-cleaners',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'
    }
    
    # Create DataFrame from the sample
    df = pd.DataFrame([sample])
    return df

def main():
    parser = argparse.ArgumentParser(description="Example of inference with saved pipeline")
    parser.add_argument("--model_path", type=str, help="Path to saved model pipeline")
    parser.add_argument("--mlflow_model", type=str, help="Name of model in MLflow registry")
    parser.add_argument("--model_version", type=str, help="Version of model in MLflow registry")
    parser.add_argument("--model_stage", type=str, help="Stage of model in MLflow registry (e.g., Production, Staging)")
    parser.add_argument("--mlflow_tracking_uri", type=str, help="MLflow tracking URI")
    args = parser.parse_args()
    
    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    # Load model
    if args.model_path:
        print(f"Loading model from local path: {args.model_path}")
        model = load_model_from_local(args.model_path)
    elif args.mlflow_model:
        print(f"Loading model '{args.mlflow_model}' from MLflow registry")
        model = load_model_from_mlflow(args.mlflow_model, args.model_version, args.model_stage)
    else:
        raise ValueError("Either --model_path or --mlflow_model must be provided")
    
    # Create a sample input
    input_data = create_example_input()
    print("\nInput data:")
    print(input_data)
    
    # Make prediction
    # The pipeline automatically handles all preprocessing steps
    prediction_prob = model.predict_proba(input_data)
    prediction = model.predict(input_data)
    
    # Display results
    print("\nPrediction:")
    print(f"Class: {prediction[0]} ({'<=50K' if prediction[0] == 0 else '>50K'})")
    print(f"Probability: {prediction_prob[0][1]:.4f}")

if __name__ == "__main__":
    main()
