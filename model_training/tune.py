import os
import argparse
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import mlflow
from dotenv import load_dotenv

from data_processor import AdultDataProcessor
from model_tuner import ModelTuner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def load_hyperparameter_config(config_path):
    """
    Load hyperparameter search configuration from JSON or YAML file.
    
    Parameters:
        config_path (str): Path to the JSON or YAML configuration file
        
    Returns:
        dict: Hyperparameter search configuration
    """
    _, ext = os.path.splitext(config_path)
    
    with open(config_path, 'r') as f:
        if ext.lower() == '.json':
            return json.load(f)
        elif ext.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use .json, .yaml, or .yml")


def get_model_class(model_name):
    """
    Get the model class based on the model name.
    
    Parameters:
        model_name (str): Name of the model class
        
    Returns:
        class: The model class
    """
    model_classes = {
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'SVC': SVC,
        'KNeighborsClassifier': KNeighborsClassifier,
        'XGBClassifier': XGBClassifier
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {', '.join(model_classes.keys())}")
    
    return model_classes[model_name]


def main():
    """
    Main function for the Adult Income Classification hyperparameter tuning job.
    
    This function handles the complete hyperparameter tuning workflow including:
    1. Command-line argument parsing
    2. Configuration loading
    3. MLflow setup
    4. Data loading and preprocessing
    5. Model initialization with hyperparameter grid
    6. Hyperparameter tuning with cross-validation
    7. Evaluation and logging of all trials
    8. Selection and registration of the best model
    """
    parser = argparse.ArgumentParser(description="Adult Income Classification Hyperparameter Tuning Job")
    parser.add_argument("--data_path", type=str, default=None, 
                        help="Path to data directory (if None, UCI repo will be used)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to hyperparameter search configuration file (JSON or YAML)")
    parser.add_argument("--mlflow_experiment", type=str, default="Adult_Classification_Tuning",
                        help="MLflow experiment name")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None,
                        help="MLflow tracking URI")
    parser.add_argument("--search_type", type=str, default="grid", choices=["grid", "random"],
                        help="Type of hyperparameter search (grid or random)")
    parser.add_argument("--n_iter", type=int, default=10,
                        help="Number of iterations for random search (ignored for grid search)")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 to use all available cores)")
    parser.add_argument("--verbose", type=int, default=2,
                        help="Verbosity level (0=silent, 1=minimal, 2=detailed)")
    parser.add_argument("--no_real_time_logging", action="store_true",
                        help="Disable real-time MLflow logging (runs in parallel and logs at the end)")
    args = parser.parse_args()
    
    # Load hyperparameter search configuration
    config = load_hyperparameter_config(args.config)
    
    # Configure MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    else:
        # Try to load from environment if not provided
        load_dotenv()
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
    
    mlflow.set_experiment(args.mlflow_experiment)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process data
    print("🔄 Processing data...")
    processor = AdultDataProcessor()
    
    if args.data_path:
        print(f"Loading data from {args.data_path}")
        # Implement loading from local files if needed
        # This is a placeholder for loading from provided path
        raise NotImplementedError("Loading from local path not implemented yet")
    else:
        print("Loading data from UCI repository")
        processor.load_from_uci()
    
    # Create preprocessing pipeline
    print("🔄 Creating preprocessing pipeline...")
    column_info = processor.create_preprocessing_pipeline()
    
    # Get raw train/test split
    X_train, X_test, y_train, y_test = processor.train_test_split()
    
    # Skip transformation - we'll use the pipeline during tuning
    print("🔄 Using pipeline during tuning process...")
    
    # Save processed data and preprocessing pipeline for reference
    data_output_path = os.path.join(args.output_dir, 'train_test_data.pkl')
    joblib.dump((X_train, X_test, y_train, y_test), data_output_path)
    
    pipeline_output_path = os.path.join(args.output_dir, 'preprocessing_pipeline.pkl')
    joblib.dump(processor.preprocessing_pipeline, pipeline_output_path)
    
    # Save column information
    processor.save_metadata(args.output_dir)
    
    print(f"✅ Data processed and saved to {args.output_dir}")
    
    # Process each model configuration from the config file
    for model_config in config['models']:
        model_name = model_config['name']
        model_class_name = model_config['class']
        model_params = model_config.get('init_params', {})
        param_grid = model_config['param_grid']
        
        print(f"🔍 Tuning {model_name} with {args.search_type} search")
        
        # Get model class
        model_class = get_model_class(model_class_name)
        
        # Initialize model with base parameters
        model = model_class(**model_params)
        
        # Additional search parameters
        search_params = {}
        if args.search_type == 'random':
            search_params['n_iter'] = args.n_iter
        
        # Create tuner
        tuner = ModelTuner(
            model=model,
            param_grid=param_grid,
            model_name=model_name,
            experiment_name=args.mlflow_experiment,
            output_dir=args.output_dir,
            preprocessing_pipeline=processor.preprocessing_pipeline,
            search_type=args.search_type,
            search_params=search_params,
            cv=args.cv_folds,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
            real_time_logging=not args.no_real_time_logging
        )
        
        # Perform tuning - use the pipeline to handle preprocessing
        best_model = tuner.tune_and_log(X_train, X_test, y_train, y_test, use_pipeline=True)
        
        print(f"✅ Tuning completed for {model_name}")
    
    print("✅ All models tuned!")


if __name__ == "__main__":
    main()