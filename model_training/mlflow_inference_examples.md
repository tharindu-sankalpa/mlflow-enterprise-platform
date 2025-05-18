# MLflow Model Inference Examples

This document provides comprehensive examples for loading models from MLflow and using them for inference.

## Loading Models from MLflow

There are several ways to load models from MLflow:

1. From the Model Registry (by name, version, or stage)
2. Directly from an MLflow Run (by run ID)
3. From a local file (saved model artifact)

Our updated `inference_example.py` supports all these methods.

## Example Commands

### 1. Load a model from the Model Registry

```bash
# Load the latest version of a registered model
python model_training/inference_example.py --mlflow_model "RandomForestClassifier_tuned"

# Load a specific version of a registered model
python model_training/inference_example.py --mlflow_model "LogisticRegression_tuned" --model_version "1"

# Load a model at a specific stage (Production, Staging, or Archived)
python model_training/inference_example.py --mlflow_model "XGBClassifier_tuned" --model_stage "Production"
```

### 2. Load a model directly from an MLflow Run

```bash
# Load a model from a specific run ID
python model_training/inference_example.py --run_id "977841598692450c98260e4d76b23cdd"

# Specify a different artifact path within the run
python model_training/inference_example.py --run_id "977841598692450c98260e4d76b23cdd" --artifact_path "model"
```

### 3. Load a model from a local file (exported from MLflow)

```bash
# Load a model from a local file path
python model_training/inference_example.py --model_path "./outputs/RandomForest_best_model.joblib"
```

### 4. Using a custom MLflow tracking server

```bash
# Specify the MLflow tracking URI
python model_training/inference_example.py --mlflow_model "RandomForestClassifier_tuned" --mlflow_tracking_uri "http://localhost:5000"
```

### 5. Using different dataset types for inference

```bash
# For Adult Income Classification
python model_training/inference_example.py --run_id "977841598692450c98260e4d76b23cdd" --dataset_type "adult"

# For California Housing Regression
python model_training/inference_example.py --run_id "abc123def456" --dataset_type "california_housing"
```

## Programmatic Usage

You can also use the functions in `inference_example.py` directly in your Python code:

```python
from model_training.inference_example import load_model_from_mlflow, create_example_input

# Set the MLflow tracking URI if needed
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

# Load a model from MLflow
model = load_model_from_mlflow(
    model_name="RandomForestClassifier_tuned",
    stage="Production"
)

# Alternatively, load directly from a run
# model = load_model_from_mlflow(run_id="977841598692450c98260e4d76b23cdd")

# Create sample input data
input_data = create_example_input(dataset_type="adult")

# Make predictions
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)[0][1]

print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability:.4f}")
```

## Finding Run IDs and Model Names

### Finding Run IDs

1. Using the MLflow UI:
   - Navigate to the experiment
   - Find your run and click on it
   - The run ID will be in the URL and displayed in the run details

2. Using the MLflow API:
   ```python
   import mlflow
   
   # Set the tracking URI if using a remote server
   mlflow.set_tracking_uri("http://localhost:5000")
   
   # Get all runs from an experiment
   runs = mlflow.search_runs(experiment_names=["Adult_Classification_Local"])
   
   # Print run IDs and metrics
   for _, run in runs.iterrows():
       print(f"Run ID: {run.run_id}, Accuracy: {run['metrics.test_accuracy']}")
   ```

### Finding Registered Model Names

1. Using the MLflow UI:
   - Click on "Models" in the left navigation
   - Browse the list of registered models

2. Using the MLflow API:
   ```python
   import mlflow.tracking
   
   # Set the tracking URI if using a remote server
   client = mlflow.tracking.MlflowClient("http://localhost:5000")
   
   # List registered models
   for rm in client.search_registered_models():
       print(f"Name: {rm.name}")
       
       # Get latest versions
       for mv in client.search_model_versions(f"name='{rm.name}'"):
           print(f"  Version: {mv.version}, Stage: {mv.current_stage}")
   ```

## Troubleshooting

1. **Error: No module named 'mlflow'**
   - Ensure MLflow is installed: `pip install mlflow`

2. **MLFLOW_TRACKING_URI environment variable not set**
   - Set the environment variable or provide it via command line: `--mlflow_tracking_uri`

3. **Model version not found**
   - Check that the model version exists in the registry
   - Try using the latest version or a specific stage instead

4. **Run ID not found**
   - Verify the run ID exists in the experiment
   - Check if the artifact path is correct within the run

5. **Error loading model from MLflow**
   - Ensure you have all required dependencies for the model type
   - For sklearn models, try adding `--no-deps` flag to force scikit-learn loader