"""
Example of how to perform inference using a saved pipeline model.
This demonstrates how preprocessing is automatically applied during inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import argparse
from dotenv import load_dotenv

def load_model_from_local(model_path):
    """
    Load a saved model from a local file path.
    
    This function loads a previously saved scikit-learn model or pipeline
    from a local file using joblib.
    
    Parameters:
        model_path (str): Path to the saved model file
        
    Returns:
        object: The loaded model or pipeline
    """
    return joblib.load(model_path)

def load_model_from_mlflow(model_name=None, model_version=None, stage=None, run_id=None, artifact_path='model'):
    """
    Load a model from MLflow, either from the model registry or directly from a run.
    
    This function provides multiple ways to retrieve a model from MLflow:
    1. From the model registry using name, version, and/or stage
    2. Directly from a run using run_id and artifact_path
    
    Parameters:
        model_name (str, optional): Name of the registered model in MLflow model registry
        model_version (str, optional): Specific version of the model in the registry
        stage (str, optional): Stage of the model (e.g., 'Production', 'Staging', 'Archived')
        run_id (str, optional): MLflow run ID to load model directly from a run
        artifact_path (str, optional): Path to the model artifact within the run (default: 'model')
        
    Returns:
        object: The loaded model from MLflow
        
    Notes:
        - Registry loading: If model_name is provided, the function will load from the registry.
          If both model_version and stage are None, the latest version is loaded.
          If both are provided, stage takes precedence.
        - Run loading: If run_id is provided, model is loaded directly from that run's artifacts.
        - At least one of model_name or run_id must be provided.
    """
    # Validate input parameters
    if not model_name and not run_id:
        raise ValueError("Either model_name or run_id must be provided")
    
    # Case 1: Load from model registry
    if model_name:
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
            print(f"Loading model '{model_name}' with stage '{stage}' from registry")
        elif model_version:
            model_uri = f"models:/{model_name}/{model_version}"
            print(f"Loading model '{model_name}' version {model_version} from registry")
        else:
            # Get latest version
            model_uri = f"models:/{model_name}/latest"
            print(f"Loading latest version of model '{model_name}' from registry")
    
    # Case 2: Load directly from run
    elif run_id:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        print(f"Loading model directly from run {run_id}, artifact path: {artifact_path}")
    
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        # Try as a sklearn model if pyfunc loading fails
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception as nested_e:
            raise RuntimeError(f"Failed to load model from {model_uri}: {str(e)}. Also failed as sklearn model: {str(nested_e)}")

def create_example_input(dataset_type="adult"):
    """
    Create a sample input for testing inference.
    
    This function creates a sample dataframe with a single row that matches
    the structure of the specified dataset. This can be used for testing
    model inference on new data.
    
    Parameters:
        dataset_type (str): The type of dataset to create a sample for. 
                           Currently supports "adult" and "california_housing"
    
    Returns:
        pandas.DataFrame: A dataframe containing a single sample record
    """
    if dataset_type.lower() == "adult":
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
    elif dataset_type.lower() == "california_housing":
        # Create a sample input matching the format of the California Housing dataset
        sample = {
            'MedInc': 5.2,
            'HouseAge': 32.0,
            'AveRooms': 5.5,
            'AveBedrms': 1.2,
            'Population': 2100.0,
            'AveOccup': 3.2,
            'Latitude': 37.85,
            'Longitude': -122.25
        }
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Currently supports 'adult' and 'california_housing'")
    
    # Create DataFrame from the sample
    df = pd.DataFrame([sample])
    return df

def main():
    """
    Main function demonstrating model inference workflow.
    
    This function demonstrates a complete inference workflow:
    1. Parsing command line arguments
    2. Loading a model from either local storage or MLflow (registry or run)
    3. Creating sample input data
    4. Making predictions
    5. Displaying prediction results
    
    The function can be used as a template for creating inference scripts
    for deployed models.
    """
    parser = argparse.ArgumentParser(description="Example of inference with saved pipeline")
    parser.add_argument("--model_path", type=str, help="Path to saved model pipeline")
    parser.add_argument("--mlflow_model", type=str, help="Name of model in MLflow registry")
    parser.add_argument("--model_version", type=str, help="Version of model in MLflow registry")
    parser.add_argument("--model_stage", type=str, help="Stage of model in MLflow registry (e.g., Production, Staging)")
    parser.add_argument("--run_id", type=str, help="MLflow run ID to load model directly")
    parser.add_argument("--artifact_path", type=str, default="model", help="Path to model artifact within the run (default: 'model')")
    parser.add_argument("--mlflow_tracking_uri", type=str, help="MLflow tracking URI")
    parser.add_argument("--dataset_type", type=str, default="adult", help="Dataset type for sample input (adult or california_housing)")
    args = parser.parse_args()
    
    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    else:
        # Try to load from environment if not provided
        from dotenv import load_dotenv
        load_dotenv()
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
        if mlflow_uri:
            print(f"Using MLflow tracking URI from environment: {mlflow_uri}")
            mlflow.set_tracking_uri(mlflow_uri)
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Load model based on provided arguments
    try:
        if args.model_path:
            print(f"Loading model from local path: {args.model_path}")
            model = load_model_from_local(args.model_path)
        elif args.mlflow_model or args.run_id:
            model = load_model_from_mlflow(
                model_name=args.mlflow_model,
                model_version=args.model_version,
                stage=args.model_stage,
                run_id=args.run_id,
                artifact_path=args.artifact_path
            )
        else:
            raise ValueError("Either --model_path, --mlflow_model, or --run_id must be provided")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Create a sample input
    try:
        input_data = create_example_input(args.dataset_type)
        print("\nInput data:")
        print(input_data)
    except Exception as e:
        print(f"Error creating input data: {str(e)}")
        return
    
    # Make prediction
    try:
        # The pipeline automatically handles all preprocessing steps
        is_classifier = hasattr(model, 'predict_proba')
        prediction = model.predict(input_data)
        
        # Display results
        print("\nPrediction:")
        
        if args.dataset_type.lower() == "adult":
            if is_classifier:
                prediction_prob = model.predict_proba(input_data)
                print(f"Class: {prediction[0]} ({'<=50K' if prediction[0] == 0 else '>50K'})")
                print(f"Probability: {prediction_prob[0][1]:.4f}")
            else:
                print(f"Prediction: {prediction[0]}")
        elif args.dataset_type.lower() == "california_housing":
            # For regression problems
            print(f"Predicted median house value: ${prediction[0]:.2f}")
        else:
            print(f"Raw prediction: {prediction[0]}")
            if is_classifier and len(prediction_prob[0]) == 2:
                print(f"Probability: {prediction_prob[0][1]:.4f}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
