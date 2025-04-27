import os
import argparse
import joblib
import numpy as np
import pandas as pd
import mlflow
from dotenv import load_dotenv

from data_processor import AdultDataProcessor
from model_trainer import ModelTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description="Adult Income Classification Training Job")
    parser.add_argument("--data_path", type=str, default=None, 
                        help="Path to data directory (if None, UCI repo will be used)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated list of models to train (all,logistic,rf,xgb,gb,knn)")
    parser.add_argument("--mlflow_experiment", type=str, default="Adult_Classification_Azure",
                        help="MLflow experiment name")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None,
                        help="MLflow tracking URI")
    parser.add_argument("--run_tensorflow", action="store_true", 
                        help="Whether to run TensorFlow DNN model")
    args = parser.parse_args()
    
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
    
    # Fit preprocessing pipeline on training data and transform both train and test
    print("🔄 Fitting preprocessing pipeline...")
    X_train_transformed, y_train = processor.fit_transform_train_data(X_train, y_train)
    X_test_transformed = processor.transform_test_data(X_test)
    
    # Save processed data and preprocessing pipeline
    data_output_path = os.path.join(args.output_dir, 'train_test_data.pkl')
    joblib.dump((X_train_transformed, X_test_transformed, y_train, y_test), data_output_path)
    
    pipeline_output_path = os.path.join(args.output_dir, 'preprocessing_pipeline.pkl')
    joblib.dump(processor.preprocessing_pipeline, pipeline_output_path)
    
    # Save column information
    processor.save_metadata(args.output_dir)
    
    print(f"✅ Data processed and saved to {args.output_dir}")
    
    # Parse models to train
    models_to_train = args.models.lower().split(',')
    
    # Define model configurations
    model_configs = []
    
    if 'all' in models_to_train or 'logistic' in models_to_train:
        model_configs.extend([
            ("LogisticRegression_C_0_1", LogisticRegression(max_iter=2000, solver="liblinear", C=0.1)),
            ("LogisticRegression_C_1_0", LogisticRegression(max_iter=2000, solver="liblinear", C=1.0)),
            ("LogisticRegression_C_10_0", LogisticRegression(max_iter=2000, solver="liblinear", C=10.0)),
        ])
    
    if 'all' in models_to_train or 'rf' in models_to_train:
        model_configs.extend([
            ("RandomForest_n200_md10", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
            ("RandomForest_n100_md5", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ("RandomForest_n300_md15", RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)),
        ])
    
    if 'all' in models_to_train or 'xgb' in models_to_train:
        model_configs.extend([
            ("XGBoost_default", XGBClassifier(eval_metric="logloss", random_state=42)),
            ("XGBoost_lr01_md3", XGBClassifier(learning_rate=0.1, max_depth=3, eval_metric="logloss", random_state=42)),
            ("XGBoost_lr005_md5", XGBClassifier(learning_rate=0.05, max_depth=5, eval_metric="logloss", random_state=42)),
        ])
    
    if 'all' in models_to_train or 'knn' in models_to_train:
        model_configs.extend([
            ("KNN_n5_minkowski", KNeighborsClassifier(n_neighbors=5, metric='minkowski')),
            ("KNN_n10_euclidean", KNeighborsClassifier(n_neighbors=10, metric='euclidean')),
            ("KNN_n3_manhattan", KNeighborsClassifier(n_neighbors=3, metric='manhattan')),
        ])
    
    if 'all' in models_to_train or 'gb' in models_to_train:
        model_configs.extend([
            ("GradientBoosting_n150_lr01_md5", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)),
            ("GradientBoosting_n100_lr005_md3", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)),
            ("GradientBoosting_n200_lr02_md7", GradientBoostingClassifier(n_estimators=200, learning_rate=0.2, max_depth=7, random_state=42)),
        ])
    
    # Train models
    print(f"🏃 Training {len(model_configs)} models...")
    for model_name, model in model_configs:
        print(f"🔍 Training {model_name}")
        trainer = ModelTrainer(
            model=model,
            model_name=model_name,
            experiment_name=args.mlflow_experiment,
            output_dir=args.output_dir,
            preprocessing_pipeline=processor.preprocessing_pipeline
        )
        # Train with the full pipeline (includes preprocessing)
        trainer.train_and_log(X_train, X_test, y_train, y_test, use_pipeline=True)
    
    # Train TensorFlow DNN model if requested
    if args.run_tensorflow:
        print("🧠 Training TensorFlow DNN model")
        # For TensorFlow, use the already transformed data
        train_tensorflow_model(X_train_transformed, X_test_transformed, y_train, y_test, args.output_dir)
    
    print("✅ Training completed successfully!")
    print(f"📊 Preprocessing pipeline saved to {pipeline_output_path}")

def train_tensorflow_model(X_train, X_test, y_train, y_test, output_dir):
    """Train a TensorFlow DNN model similar to the one in the notebook"""
    with mlflow.start_run(run_name="Deep Neural Network - TensorFlow"):
        # Make sure y is 1D array
        y_train = y_train.ravel() if hasattr(y_train, 'ravel') else y_train
        y_test = y_test.ravel() if hasattr(y_test, 'ravel') else y_test
        
        # Enable MLflow autologging for TensorFlow
        mlflow.tensorflow.autolog()
        
        # Build DNN
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Define early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model with early stopping
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Predictions for evaluation
        y_train_prob = model.predict(X_train).ravel()
        y_test_prob = model.predict(X_test).ravel()
        y_train_pred = (y_train_prob >= 0.5).astype(int)
        y_test_pred = (y_test_prob >= 0.5).astype(int)
        
        # Save model
        model_path = os.path.join(output_dir, 'tensorflow_model')
        model.save(model_path)
        print(f"Saved TensorFlow model to {model_path}")
        
        # Additional metrics and visualization handled by autologging

if __name__ == "__main__":
    main()
