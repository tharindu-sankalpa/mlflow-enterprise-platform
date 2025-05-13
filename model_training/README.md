# Adult Income Classification Project

This project implements a machine learning pipeline for the UCI Adult Income dataset classification task. The pipeline includes data processing, model training with multiple algorithms, evaluation, and MLflow tracking integration. The implementation supports both local execution and remote training on Azure ML.

## Project Structure

```
model_training/
├── __init__.py             # Package initialization file
├── data_processor.py       # Handles data loading and preprocessing
├── model_trainer.py        # Model training and evaluation utilities
├── train.py                # Main training script
├── submit_job.py           # Azure ML job submission script
├── inference_example.py    # Example script for inference
├── requirements.txt        # Python dependencies
├── conda_dependencies.yml  # Conda environment for Azure ML
└── README.md               # This documentation file
```

## Modules and Classes

### 1. Data Processing (`data_processor.py`)

The `AdultDataProcessor` class handles loading, preprocessing, and splitting the Adult Income dataset. It uses scikit-learn's `Pipeline` and `ColumnTransformer` for reproducible preprocessing.

Key methods:
- `load_from_uci()`: Fetches data directly from the UCI repository
- `load_from_csv(file_path)`: Loads data from a local CSV file
- `create_preprocessing_pipeline()`: Creates a sklearn pipeline for data preprocessing
- `preprocess_data()`: Applies preprocessing steps to the data
- `train_test_split(test_size, random_state)`: Splits data into training and test sets
- `fit_transform_train_data(X_train, y_train)`: Fits preprocessing pipeline on training data
- `transform_test_data(X_test)`: Transforms test data using fitted pipeline
- `save_metadata(output_dir)`: Saves dataset metadata to file
- `get_feature_names()`: Returns feature names after transformation

### 2. Model Training (`model_trainer.py`)

The `ModelTrainer` class handles model training, evaluation, and MLflow tracking. It can combine preprocessing and model into a single pipeline.

Key methods:
- `train_and_log(X_train, X_test, y_train, y_test, use_pipeline)`: Trains model with preprocessing pipeline
- `_log_metrics(y_true, y_pred, y_prob, prefix)`: Logs classification metrics (accuracy, precision, etc.)
- `_log_confusion_matrix(cm, title, artifact_name)`: Generates and logs confusion matrix plots
- `_log_classification_report(y_true, y_pred, prefix)`: Logs detailed classification reports

### 3. Training Script (`train.py`)

The main script that orchestrates the training process. Supports various classification models:
- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- KNN
- TensorFlow DNN (optional)

Features:
- Automatic data loading from UCI repository
- Data preprocessing and feature engineering
- Model training with multiple algorithms
- Comprehensive evaluation metrics
- MLflow tracking integration
- Support for TensorFlow deep learning models
- Automatic artifact saving

### 4. Azure ML Job Submission (`submit_job.py`)

Utility script to submit training jobs to Azure ML. It handles:
- Authentication with Azure using DefaultAzureCredential
- Connection to Azure ML workspace
- Definition of compute environment
- Training command configuration
- Job submission and monitoring

### 5. Inference Example (`inference_example.py`)

Demonstrates how to use the trained pipeline for predictions on new data. The pipeline includes all preprocessing steps.

Features:
- Support for loading models from local files or MLflow registry
- Sample input generation for testing
- Prediction with automatic preprocessing
- Detailed result display

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mlflow-enterprise-platform
```

2. Install dependencies:
```bash
pip install -r model_training/requirements.txt
```

## Running Locally

You can run the training script locally with various configuration options:

```bash
python model_training/train.py --models all --mlflow_experiment "Adult_Classification_Local"
```

### Command Line Arguments

- `--data_path`: Path to data directory (if None, UCI repo will be used)
- `--output_dir`: Directory to save outputs (default: "./outputs")
- `--models`: Comma-separated list of models to train (options: all, logistic, rf, xgb, gb, knn)
- `--mlflow_experiment`: MLflow experiment name (default: "Adult_Classification_Azure")
- `--mlflow_tracking_uri`: MLflow tracking URI (if not provided, will use environment variable)
- `--run_tensorflow`: Flag to run TensorFlow DNN model

Examples:

```bash
# Train only logistic regression and random forest models
python model_training/train.py --models logistic,rf --output_dir ./my_outputs

# Train all models including TensorFlow
python model_training/train.py --models all --run_tensorflow

# Use a specific MLflow tracking server
python model_training/train.py --mlflow_tracking_uri http://localhost:5000

# Full example
python model_training/train.py --models all --run_tensorflow --mlflow_experiment "Adult_Classification_Local" --mlflow_tracking_uri http://127.0.0.1:5000
```

## Running on Azure ML

1. First, ensure you have set up an Azure ML workspace and compute target.

2. Use the provided conda dependencies file `conda_dependencies.yml` in the model_training directory:

```yaml
name: adult-classification-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pip=22.3.1
  - pip:
    - numpy>=1.20.0
    - pandas>=1.3.0
    - scikit-learn>=1.0.0
    - matplotlib>=3.4.0
    - mlflow>=2.0.0
    - tensorflow>=2.8.0
    - xgboost>=1.5.0
    - python-dotenv>=0.19.0
    - ucimlrepo>=0.0.3
    - joblib>=1.1.0
    - azureml-mlflow>=1.42.0
```

3. Submit a training job to Azure ML:

```bash
python model_training/submit_job.py \
  --subscription_id "<your-subscription-id>" \
  --resource_group "<your-resource-group>" \
  --workspace_name "<your-workspace-name>" \
  --compute_target "<your-compute-target>" \
  --experiment_name "Adult_Classification_Azure" \
  --models all \
  --run_tensorflow
```

### Azure ML Job Submission Arguments

- `--subscription_id`: Azure subscription ID (required)
- `--resource_group`: Azure resource group (required)
- `--workspace_name`: Azure ML workspace name (required)
- `--compute_target`: Azure ML compute target (required)
- `--experiment_name`: Experiment name (default: "Adult_Classification_Azure")
- `--models`: Models to train (default: "all")
- `--run_tensorflow`: Flag to run TensorFlow model
- `--environment_name`: Environment name (default: "adult-classification-env")

## Inference

The trained models include all preprocessing steps as part of a scikit-learn pipeline, making inference straightforward. You can use the saved pipeline directly with raw data.

### Using a trained model for inference:

```bash
# Load model from local file
python model_training/inference_example.py --model_path ./outputs/LogisticRegression_C_1_0_pipeline.joblib

# Or load from MLflow model registry
python model_training/inference_example.py --mlflow_model LogisticRegression_C_1_0 --model_stage Production
```

You can also use the model in your FastAPI service:

```python
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the pipeline once at startup
model_pipeline = joblib.load("./outputs/model_pipeline.joblib")

app = FastAPI()

class AdultData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.post("/predict/")
async def predict(data: AdultData):
    # Convert to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict - preprocessing happens automatically
    prediction = model_pipeline.predict(input_df)
    probability = model_pipeline.predict_proba(input_df)[0][1]
    
    return {
        "income_class": "<=50K" if prediction[0] == 0 else ">50K",
        "probability": float(probability)
    }
```

## MLflow Integration

This project uses MLflow to track experiments. For each model training run, the following information is logged:

- Model parameters
- Evaluation metrics (accuracy, precision, recall, F1, etc.)
- ROC AUC and PR AUC curves
- Confusion matrices
- Classification reports
- The complete pipeline (preprocessing + model) for easy deployment

To view the tracked information, start the MLflow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Dataset

The project uses the UCI Adult Income dataset, which contains demographic information about adults along with a binary classification of whether their income exceeds $50K per year. The dataset includes both categorical and numerical features.

Features include:
- Age
- Workclass
- Education level
- Marital status
- Occupation
- Relationship
- Race
- Sex
- Capital gain/loss
- Hours per week
- Native country

## Custom Transformers

The project includes a custom `IntToFloatTransformer` that converts integer columns to float64. This is important for handling missing values properly and preventing MLflow schema enforcement errors during inference.

## Results

After training, model outputs and evaluation metrics will be saved to the specified output directory and tracked in MLflow. You can compare models to select the best performing one for deployment.

The outputs directory will contain:
- Trained model files (.joblib)
- Full preprocessing pipelines
- Confusion matrix plots
- Classification reports
- Metadata about columns and preprocessing
