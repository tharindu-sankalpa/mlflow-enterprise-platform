# Adult Income Classification Project

This project implements a machine learning pipeline for the UCI Adult Income dataset classification task. The pipeline includes data processing, model training with multiple algorithms, evaluation, and MLflow tracking integration. The implementation supports both local execution and remote training on Azure ML.

## Project Structure

```
azure_training/
├── __init__.py             # Package initialization file
├── data_processor.py       # Handles data loading and preprocessing
├── model_trainer.py        # Model training and evaluation utilities
├── train.py                # Main training script
├── submit_job.py           # Azure ML job submission script
├── requirements.txt        # Python dependencies
└── conda_dependencies.yml  # Conda environment for Azure ML
```

## Modules and Classes

### 1. Data Processing (`data_processor.py`)

The `AdultDataProcessor` class handles loading, preprocessing, and splitting the Adult Income dataset.

Key methods:
- `load_from_uci()`: Fetches data directly from the UCI repository
- `load_from_csv(file_path)`: Loads data from a local CSV file
- `preprocess_data()`: Applies preprocessing steps (handling missing values, one-hot encoding)
- `train_test_split(test_size, random_state)`: Splits data into training and test sets
- `scale_features(X_train, X_test, columns)`: Scales numerical features using StandardScaler
- `save_metadata(output_dir)`: Saves dataset metadata to file

### 2. Model Training (`model_trainer.py`)

The `ModelTrainer` class handles model training, evaluation, and MLflow tracking.

Key methods:
- `train_and_log(X_train, X_test, y_train, y_test)`: Trains model and logs metrics to MLflow
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

### 4. Azure ML Job Submission (`submit_job.py`)

Utility script to submit training jobs to Azure ML.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mlflow_iris_example
```

2. Install dependencies:
```bash
pip install -r azure_training/requirements.txt
```

## Running Locally

You can run the training script locally with various configuration options:

```bash
python azure_training/train.py --models all --mlflow_experiment "Adult_Classification_Local"
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
python azure_training/train.py --models logistic,rf --output_dir ./my_outputs

# Train all models including TensorFlow
python azure_training/train.py --models all --run_tensorflow

# Use a specific MLflow tracking server
python azure_training/train.py --mlflow_tracking_uri http://localhost:5000
```

## Running on Azure ML

1. First, ensure you have set up an Azure ML workspace and compute target.

2. Create a conda dependencies file `conda_dependencies.yml` in the azure_training directory:

```yaml
name: adult-classification-env
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pip=21.0
  - pip:
    - azureml-core>=1.42.0
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
```

3. Submit a training job to Azure ML:

```bash
python azure_training/submit_job.py \
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

## MLflow Integration

This project uses MLflow to track experiments. For each model training run, the following information is logged:

- Model parameters
- Evaluation metrics (accuracy, precision, recall, F1, etc.)
- Confusion matrices
- Classification reports
- The model itself for easy deployment

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

## Results

After training, model outputs and evaluation metrics will be saved to the specified output directory and tracked in MLflow. You can compare models to select the best performing one for deployment.
