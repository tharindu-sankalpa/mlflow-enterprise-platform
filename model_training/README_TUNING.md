# Model Hyperparameter Tuning Module

This module extends the Adult Income Classification project with hyperparameter tuning capabilities. It leverages the existing data processing and model training architecture while adding robust hyperparameter optimization with comprehensive MLflow tracking.

## Features

- **Scikit-learn Model Support**: Works with any scikit-learn compatible classifier
- **Grid and Random Search**: Both exhaustive and randomized hyperparameter optimization
- **Flexible Configuration**: Define hyperparameter grids via YAML or JSON files
- **Cross-validation**: Validate model performance across multiple data folds
- **MLflow Integration**: Log every trial as a separate run for detailed comparison
- **Azure ML Support**: Optional submission to Azure ML compute
- **Comprehensive Metrics**: Track detailed classification metrics for each trial
- **Pipeline Integration**: Seamless integration with preprocessing pipelines

## Project Structure

```
model_training/
├── model_tuner.py           # ModelTuner class for hyperparameter optimization
├── tune.py                  # Main tuning script
├── submit_tune_job.py       # Azure ML job submission script
├── hyperparameter_configs/  # Configuration files for different models
│   ├── logistic_regression.json
│   ├── random_forest.yaml
│   ├── xgboost.yaml
│   └── multi_model.yaml
```

## Components

### 1. ModelTuner Class (`model_tuner.py`)

The `ModelTuner` class extends the functionality of the original `ModelTrainer` class with hyperparameter optimization capabilities:

- **Constructor**: Accepts a base model, parameter grid, and configuration options
- **tune_and_log()**: Main method that performs hyperparameter search and logs results to MLflow
- **Child Runs**: Each parameter combination is logged as a child run for detailed tracking
- **Best Model Selection**: Automatically identifies and logs the best performing model
- **Comprehensive Metrics**: Tracks detailed performance metrics for each trial

### 2. Tuning Script (`tune.py`)

The main tuning script that handles:

- Command-line arguments and configuration
- Data loading and preprocessing
- Model initialization with hyperparameter grids
- Running the tuning process
- Logging results to MLflow

### 3. Azure ML Support (`submit_tune_job.py`)

A script to submit tuning jobs to Azure ML, providing:

- Authentication with Azure
- Environment configuration
- Job submission and monitoring
- Remote execution on scalable compute

### 4. Hyperparameter Configuration Files

JSON or YAML files defining hyperparameter search spaces:

- `logistic_regression.json`: Configuration for logistic regression
- `random_forest.yaml`: Configuration for random forest classifiers
- `xgboost.yaml`: Configuration for XGBoost models
- `multi_model.yaml`: Configuration for multiple models in a single run

## Configuration Format

The hyperparameter configuration files follow this structure:

```yaml
models:
  - name: ModelName                # Name for the tuning run
    class: ModelClassName          # Class name of the model (e.g., "LogisticRegression")
    init_params:                   # Parameters for model initialization
      param1: value1
      param2: value2
    param_grid:                    # Hyperparameter search grid
      hyperparameter1: [val1, val2, val3]
      hyperparameter2: [val1, val2, val3]
```

For multi-model configurations, simply include multiple model entries:

```yaml
models:
  - name: FirstModel
    class: FirstModelClass
    # ... parameters ...
    
  - name: SecondModel
    class: SecondModelClass
    # ... parameters ...
```

## Running Locally

To run hyperparameter tuning locally:

```bash
python model_training/tune.py --config model_training/hyperparameter_configs/logistic_regression.json
```

### Command Line Arguments

- `--data_path`: Path to data directory (if None, UCI repo will be used)
- `--output_dir`: Directory to save outputs (default: "./outputs")
- `--config`: Path to hyperparameter search configuration file (required)
- `--mlflow_experiment`: MLflow experiment name (default: "Adult_Classification_Tuning")
- `--mlflow_tracking_uri`: MLflow tracking URI
- `--search_type`: Type of hyperparameter search ("grid" or "random", default: "grid")
- `--n_iter`: Number of iterations for random search (default: 10, ignored for grid search)
- `--cv_folds`: Number of cross-validation folds (default: 5)
- `--n_jobs`: Number of parallel jobs (-1 to use all cores, default: -1)
- `--verbose`: Verbosity level (0=silent, 1=minimal, 2=detailed, default: 2)
- `--no_real_time_logging`: Disable real-time MLflow logging (will run in parallel and log at the end)

### Examples

```bash
# Run grid search with logistic regression
python model_training/tune.py --config model_training/hyperparameter_configs/logistic_regression.json

# Run random search with XGBoost
python model_training/tune.py --config model_training/hyperparameter_configs/xgboost.yaml --search_type random --n_iter 20

# Run with a specific MLflow tracking server
python model_training/tune.py --config model_training/hyperparameter_configs/multi_model.yaml --mlflow_tracking_uri http://localhost:5000

# Run with maximum parallelism and no real-time logging (faster but logs at the end)
python model_training/tune.py --config model_training/hyperparameter_configs/random_forest.yaml --no_real_time_logging

# Run a quick example with real-time progress tracking
python model_training/tune.py --config model_training/hyperparameter_configs/quick_example.yaml
```

### Understanding Execution Modes

There are two main execution modes for tuning:

1. **Real-time Logging Mode (Default)**
   - Each trial is executed sequentially
   - MLflow logs are updated in real-time as each trial completes
   - Progress bar shows completion status
   - Great for monitoring ongoing progress in the MLflow UI
   - Use this when you want to see results as they happen

2. **Parallel Execution Mode (--no_real_time_logging)**
   - Trials are executed in parallel using all available cores
   - MLflow logs are created only after all trials complete
   - Much faster for large hyperparameter spaces
   - Use this when maximizing CPU utilization is more important than real-time monitoring
   - When running with `--no_real_time_logging`, you'll see the computation progress first, followed by the MLflow logging progress

### Model Signatures and Input Examples

All models saved to MLflow now include:

1. **Model Signatures**: Automatically inferred schema information about the expected inputs and outputs
2. **Input Examples**: Sample input data that can be used for model validation and serving

These features enhance the models' usability for deployment and inference, ensuring that:
- MLflow UI shows the expected input/output schemas
- Models can be loaded and served with proper type checking
- Inference code can validate inputs against the model's expected schema

## Running on Azure ML

To submit a hyperparameter tuning job to Azure ML:

```bash
python model_training/submit_tune_job.py \
  --subscription_id "<your-subscription-id>" \
  --resource_group "<your-resource-group>" \
  --workspace_name "<your-workspace-name>" \
  --compute_target "<your-compute-target>" \
  --config model_training/hyperparameter_configs/random_forest.yaml
```

### Azure ML Submission Arguments

- `--subscription_id`: Azure subscription ID (required)
- `--resource_group`: Azure resource group (required)
- `--workspace_name`: Azure ML workspace name (required)
- `--compute_target`: Azure ML compute target (required)
- `--experiment_name`: Experiment name (default: "Adult_Classification_Tuning")
- `--config`: Path to hyperparameter configuration file (required)
- `--search_type`: Type of hyperparameter search (default: "grid")
- `--n_iter`: Number of iterations for random search (default: 10)
- `--cv_folds`: Number of cross-validation folds (default: 5)
- `--environment_name`: Environment name (default: "adult-classification-env")

## Creating Custom Hyperparameter Configurations

You can create your own hyperparameter configuration files to test different parameter combinations:

1. Create a new JSON or YAML file in the `hyperparameter_configs` directory
2. Follow the format described in the "Configuration Format" section
3. Run the tuning script with your configuration file

Example for SVM:

```yaml
models:
  - name: SVM_RBF_Grid
    class: SVC
    init_params:
      probability: true
      random_state: 42
    param_grid:
      C: [0.1, 1.0, 10.0, 100.0]
      gamma: [0.001, 0.01, 0.1, 1.0]
      kernel: ["rbf"]
```

## MLflow Integration

The tuning module integrates deeply with MLflow to track all aspects of the hyperparameter optimization process:

- **Parent Run**: Each tuning job creates a parent run for the overall search
- **Child Runs**: Each parameter combination gets a dedicated child run
- **Parameters**: All hyperparameters are logged for each trial
- **Metrics**: Comprehensive metrics are tracked for both cross-validation and test performance
- **Artifacts**: Confusion matrices, classification reports, and model files are saved
- **Model Registry**: The best model is automatically registered in the MLflow registry

To view the tracked information, start the MLflow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Understanding the MLflow Run Structure

The tuning module creates a hierarchical structure in MLflow:

```
Parent Run: "LogisticRegression_Grid Tuning"
├── Child Run: "Trial 1"
│   ├── Parameters: {C: 0.001, penalty: l1, ...}
│   ├── Metrics: {mean_cv_score: 0.82, ...}
│   └── Artifacts: {...}
├── Child Run: "Trial 2"
│   ├── Parameters: {C: 0.01, penalty: l1, ...}
│   └── ...
└── ...
```

The best model will have additional metrics and artifacts, including:

- Training and testing metrics
- Confusion matrices
- Classification reports
- The serialized model

## Best Practices

1. **Start Small**: When creating new hyperparameter grids, start with a small set of values to ensure everything works correctly before expanding to a larger search space.

2. **Use Random Search for Large Spaces**: If you have many hyperparameters to tune, consider using random search with a reasonable number of iterations instead of grid search, which grows exponentially with the number of parameters.

3. **Monitor Resource Usage**: Grid search with large parameter spaces can be computationally expensive. Consider submitting these jobs to Azure ML for scalable compute resources.

4. **Visualize Results**: After tuning, use the MLflow UI to compare trials and understand the impact of different hyperparameters on model performance.

5. **Iterative Refinement**: Use the results of an initial broad search to identify promising regions of the hyperparameter space, then conduct a more focused search in those regions.

## Extending the Module

The module is designed to be easily extended:

- **Add New Models**: To support a new model type, add it to the `get_model_class` function in `tune.py`
- **Custom Scoring Metrics**: Customize the `scoring` parameter in the `ModelTuner` constructor
- **Additional Visualizations**: Extend the `ModelTuner` class to include additional visualization types
- **Bayesian Optimization**: Implement alternative search strategies like Bayesian optimization