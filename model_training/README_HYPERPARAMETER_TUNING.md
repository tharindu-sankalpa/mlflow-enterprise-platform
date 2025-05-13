# Hyperparameter Tuning with MLflow: Optimizing Model Performance

This guide explores the hyperparameter tuning functionality in the [MLflow Iris Example](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform) repository. All code examples and instructions reference this public repository.

When it comes to building machine learning models, hyperparameter tuning is a crucial step for achieving optimal performance. In this section, we'll explore how to efficiently tune hyperparameters while tracking all experiments using MLflow. We'll leverage the robust hyperparameter tuning module in our project to systematically search for the best parameter combinations across multiple models.

## Understanding Hyperparameter Tuning

Hyperparameters are configuration variables that determine how a model learns from data. Unlike regular parameters that the model learns during training (like weights in a neural network), hyperparameters must be set before training begins. Examples include:

- The learning rate in gradient-boosted trees
- The regularization strength (C) in logistic regression
- The maximum tree depth in random forests
- The number of hidden layers in a neural network

Finding the optimal hyperparameters often requires trying many combinations, which can be time-consuming and difficult to track manually. Our hyperparameter tuning module addresses these challenges by providing:

1. **Structured Search**: Systematic exploration of the hyperparameter space
2. **Comprehensive Tracking**: Complete logging of every trial in MLflow
3. **Performance Metrics**: Detailed evaluation of each parameter combination
4. **Visual Comparison**: Easy visualization of results in the MLflow UI

## The Hyperparameter Tuning Architecture

Our tuning module consists of three main components:

1. **ModelTuner Class**: The core component that handles hyperparameter search, cross-validation, and MLflow logging
2. **Configuration Files**: YAML or JSON files defining hyperparameter search spaces
3. **Tuning Script**: The main script that orchestrates the tuning process

Let's explore each component in detail.

### The ModelTuner Class

The [`ModelTuner` class](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/blob/main/model_training/model_tuner.py) is designed to handle the entire hyperparameter tuning workflow. Here's a simplified overview of its key methods:

```python
class ModelTuner:
    def __init__(
        self, 
        model,                     # Base model to tune
        param_grid,                # Hyperparameter grid to search
        model_name="Model",        # Name identifier
        experiment_name="Default", # MLflow experiment name
        output_dir="./outputs",    # Directory for local outputs
        preprocessing_pipeline=None, # Optional preprocessing pipeline
        search_type="grid",        # 'grid' or 'random' search
        search_params=None,        # Additional search parameters
        cv=5,                      # Number of cross-validation folds
        scoring="roc_auc",         # Metric to optimize
        n_jobs=-1,                 # Number of parallel jobs
        verbose=2,                 # Verbosity level
        real_time_logging=True     # Whether to log in real-time
    ):
        # Initialize attributes
        
    def tune_and_log(self, X_train, X_test, y_train, y_test, use_pipeline=True):
        """
        Tune hyperparameters, evaluate performance, and log results to MLflow.
        
        This method performs a complete hyperparameter tuning workflow.
        
        Returns:
            The best model from hyperparameter search
        """
        best_result = self.tune_with_custom_search(X_train, X_test, y_train, y_test, use_pipeline)
        return best_result['estimator']
```

The `tune_and_log` method is the main entry point that coordinates the entire tuning process. It handles:

1. Parameter combination generation
2. Cross-validation execution
3. Performance evaluation
4. MLflow logging
5. Best model selection

### Hyperparameter Configuration Files

One of the key features of our module is the ability to define hyperparameter search spaces in external configuration files (YAML or JSON) located in the [`hyperparameter_configs`](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/tree/main/model_training/hyperparameter_configs) directory. This approach offers several advantages:

- **Separation of Concerns**: Model configurations are separate from code
- **Version Control**: Configuration files can be versioned and shared
- **Flexibility**: Easy to add new parameter combinations without code changes
- **Multi-Model Support**: Configure multiple models in a single file

Here's an example configuration for logistic regression ([`logistic_regression.json`](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/blob/main/model_training/hyperparameter_configs/logistic_regression.json)):

```json
{
  "models": [
    {
      "name": "LogisticRegression_Grid",
      "class": "LogisticRegression",
      "init_params": {
        "max_iter": 2000,
        "random_state": 42
      },
      "param_grid": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"]
      }
    }
  ]
}
```

For more complex models like XGBoost, we can use YAML format ([`xgboost.yaml`](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/blob/main/model_training/hyperparameter_configs/xgboost.yaml)):

```yaml
models:
  - name: XGBoost_Grid
    class: XGBClassifier
    init_params:
      eval_metric: logloss
      random_state: 42
    param_grid:
      learning_rate: [0.01, 0.05, 0.1, 0.2]
      max_depth: [3, 5, 7, 9]
      min_child_weight: [1, 3, 5]
      gamma: [0, 0.1, 0.2]
      subsample: [0.6, 0.8, 1.0]
      colsample_bytree: [0.6, 0.8, 1.0]
      n_estimators: [100, 200, 300]
```

We can even define multiple models to tune in a single configuration file ([`multi_model.yaml`](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/blob/main/model_training/hyperparameter_configs/multi_model.yaml)):

```yaml
models:
  - name: LogisticRegression_Quick
    class: LogisticRegression
    init_params:
      max_iter: 2000
      random_state: 42
    param_grid:
      C: [0.1, 1.0, 10.0]
      penalty: ["l2"]
      solver: ["liblinear"]

  - name: RandomForest_Quick
    class: RandomForestClassifier
    init_params:
      random_state: 42
    param_grid:
      n_estimators: [100, 200]
      max_depth: [5, 10]
      min_samples_split: [2, 5]
```

### The Tuning Script

The [`tune.py` script](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/blob/main/model_training/tune.py) serves as the entry point for running hyperparameter tuning jobs. It handles:

1. Parsing command-line arguments
2. Loading configurations
3. Setting up MLflow tracking
4. Loading and preprocessing data
5. Creating and executing the tuning process

Here's a simplified overview of the `main()` function:

```python
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Adult Income Classification Hyperparameter Tuning Job")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to hyperparameter search configuration file (JSON or YAML)")
    parser.add_argument("--mlflow_experiment", type=str, default="Adult_Classification_Tuning",
                        help="MLflow experiment name")
    # ... other arguments ...
    args = parser.parse_args()
    
    # Load hyperparameter search configuration
    config = load_hyperparameter_config(args.config)
    
    # Configure MLflow
    mlflow.set_experiment(args.mlflow_experiment)
    
    # Process data
    processor = AdultDataProcessor()
    processor.load_from_uci()
    column_info = processor.create_preprocessing_pipeline()
    X_train, X_test, y_train, y_test = processor.train_test_split()
    
    # Process each model configuration from the config file
    for model_config in config['models']:
        model_name = model_config['name']
        model_class_name = model_config['class']
        model_params = model_config.get('init_params', {})
        param_grid = model_config['param_grid']
        
        # Get model class and initialize with base parameters
        model_class = get_model_class(model_class_name)
        model = model_class(**model_params)
        
        # Create tuner
        tuner = ModelTuner(
            model=model,
            param_grid=param_grid,
            model_name=model_name,
            experiment_name=args.mlflow_experiment,
            preprocessing_pipeline=processor.preprocessing_pipeline,
            search_type=args.search_type,
            # ... other parameters ...
        )
        
        # Perform tuning - use the pipeline to handle preprocessing
        best_model = tuner.tune_and_log(X_train, X_test, y_train, y_test, use_pipeline=True)
```

## Running Hyperparameter Tuning

Now let's walk through the process of running a hyperparameter tuning job and examining the results in MLflow.

### Running a Tuning Job

To run a hyperparameter tuning job, we use the `tune.py` script with a configuration file:

```bash
python model_training/tune.py --config model_training/hyperparameter_configs/logistic_regression.json
```

This command performs the following actions:

1. Loads the Adult Income dataset from the UCI repository
2. Creates a preprocessing pipeline for the data
3. Loads the hyperparameter search configuration
4. For each model in the configuration:
   - Creates a ModelTuner instance
   - Executes the hyperparameter search
   - Logs all trial results to MLflow
   - Identifies and registers the best model

You can customize the tuning process with additional arguments:

```bash
python model_training/tune.py \
    --config model_training/hyperparameter_configs/xgboost.yaml \
    --search_type random \
    --n_iter 20 \
    --mlflow_experiment "XGBoost_Random_Search" \
    --cv_folds 3
```

This example:
- Uses the XGBoost configuration file
- Performs random search instead of grid search
- Tries 20 random combinations
- Logs results to the "XGBoost_Random_Search" experiment
- Uses 3-fold cross-validation for faster execution

### Execution Modes

The tuning module supports two execution modes:

1. **Real-time Logging Mode (Default)**
   - Trials are executed sequentially
   - MLflow logs are updated in real-time
   - Progress bar shows completion status
   - Great for monitoring ongoing progress

2. **Parallel Execution Mode**
   - Enabled with the `--no_real_time_logging` flag
   - Trials are executed in parallel using all available cores
   - MLflow logs are created after all trials complete
   - Much faster for large hyperparameter spaces

```bash
# Run with maximum parallelism
python model_training/tune.py \
    --config model_training/hyperparameter_configs/random_forest.yaml \
    --no_real_time_logging
```

## Understanding MLflow Integration

The tuning module integrates deeply with MLflow to track all aspects of the hyperparameter optimization process. Let's explore how this integration works and what information is tracked.

### Hierarchical Run Structure

When running a hyperparameter tuning job, the module creates a hierarchical structure in MLflow:

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

### What's Tracked for Each Trial

For each parameter combination (trial), the following information is logged to MLflow:

#### Parameters
- All hyperparameters being tested (e.g., C, penalty, max_depth)
- Model initialization parameters (e.g., random_state)
- Search configuration (e.g., search_type, cv_folds)

#### Metrics
- Cross-validation metrics:
  - `mean_cv_score`: Average score across CV folds
  - `std_cv_score`: Standard deviation of scores
- Training metrics:
  - `train_accuracy`, `train_precision`, `train_recall`, `train_f1_score`
  - `train_log_loss`, `train_roc_auc`, `train_pr_auc`
- Testing metrics:
  - `test_accuracy`, `test_precision`, `test_recall`, `test_f1_score`
  - `test_log_loss`, `test_roc_auc`, `test_pr_auc`
- Confusion matrix elements:
  - `test_true_negative`, `test_false_positive`, `test_false_negative`, `test_true_positive`

#### Artifacts
- **Confusion Matrices**: Visualizations of model performance
- **Classification Reports**: Detailed text reports with precision, recall, and F1-score for each class
- **Model Files**: The trained model saved in MLflow format with:
  - Model signature (input/output schema)
  - Input example for validation
  - Full preprocessing pipeline + model

### Code Walk-through: Tracking a Trial

Let's look at how a single trial is logged to MLflow in [`model_tuner.py`](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/blob/main/model_training/model_tuner.py):

```python
def _log_one_trial(self, params, X_train, y_train, X_test, y_test, cv_folds, trial_idx):
    # Start a new run for this parameter combination
    run_name = f"Trial {trial_idx+1} - {self.model_name}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log run metadata
        mlflow.set_tag("trial_number", trial_idx + 1)
        mlflow.set_tag("model_type", self.model.__class__.__name__)
        mlflow.set_tag("search_type", self.search_type)
        
        # Log parameters
        mlflow.log_params(params)
        
        # Create and train estimator
        if self.full_pipeline is not None:
            # Create a clone of the pipeline with new parameters
            estimator = clone(self.full_pipeline)
            # Set model parameters in the pipeline
            model_params = {k.replace('model__', ''): v for k, v in params.items()}
            estimator.named_steps['model'].set_params(**model_params)
        else:
            # Create a clone of the model with new parameters
            estimator = clone(self.model)
            estimator.set_params(**params)
        
        # Perform cross-validation
        cv_scores = []
        for train_idx, val_idx in cv_folds:
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model
            estimator.fit(X_cv_train, y_cv_train)
            
            # Get validation score
            y_cv_val_prob = estimator.predict_proba(X_cv_val)[:, 1]
            score = roc_auc_score(y_cv_val, y_cv_val_prob)
            cv_scores.append(score)
        
        # Log cross-validation metrics
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        mlflow.log_metric("mean_cv_score", mean_cv_score)
        mlflow.log_metric("std_cv_score", std_cv_score)
        
        # Train on full training set
        estimator.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = estimator.predict(X_train)
        y_train_prob = estimator.predict_proba(X_train)[:, 1]
        y_test_pred = estimator.predict(X_test)
        y_test_prob = estimator.predict_proba(X_test)[:, 1]
        
        # Log detailed metrics
        cm_train = self._log_metrics(y_train, y_train_pred, y_train_prob, "train")
        cm_test = self._log_metrics(y_test, y_test_pred, y_test_prob, "test")
        
        # Log confusion matrices
        self._log_confusion_matrix(cm_train, f"Trial {trial_idx+1} - Train", f"{self.model_name}_train_{trial_idx+1}")
        self._log_confusion_matrix(cm_test, f"Trial {trial_idx+1} - Test", f"{self.model_name}_test_{trial_idx+1}")
        
        # Log classification reports
        self._log_classification_report(y_train, y_train_pred, f"{self.model_name}_train_{trial_idx+1}")
        self._log_classification_report(y_test, y_test_pred, f"{self.model_name}_test_{trial_idx+1}")
        
        # Log model with signature
        mlflow.sklearn.log_model(
            estimator, 
            "model",
            signature=mlflow.models.infer_signature(X_train, y_train),
            input_example=X_train.iloc[:5]
        )
```

This code shows how each trial is logged as a separate run in MLflow, with all parameters, metrics, and artifacts tracked. The `mlflow.sklearn.log_model` function is particularly important, as it saves the complete model pipeline (including preprocessing) in a format that can be easily loaded for inference.

## Exploring Results in the MLflow UI

After running hyperparameter tuning jobs, you can explore the results in the MLflow UI. Start the UI with:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser (or the centralized MLflow tracking server URL if you're using one).

Let's walk through some key views that help analyze tuning results:

### 1. Experiment View

The experiment view shows all runs within an experiment. For a tuning job, you'll see all trials listed with their metrics. You can:

- Sort runs by any metric (e.g., test_f1_score)
- Filter runs by parameter values
- Compare multiple runs

### 2. Run Comparison

Select multiple runs and click "Compare" to see a detailed comparison of metrics and parameters. This is extremely useful for understanding how different hyperparameters affect model performance.

### 3. Parameter Importance Analysis

The comparison view allows you to visualize the relationship between hyperparameters and metrics. For example, you might observe that:

- Increasing C values in logistic regression improves performance up to a point, then plateaus or declines
- Deeper trees in random forests improve training metrics but may lead to overfitting
- Certain learning rates consistently perform better regardless of other hyperparameters

### 4. Artifacts and Model Inspection

For each trial, you can explore:

- **Confusion Matrices**: Visual representation of correct and incorrect predictions
- **Classification Reports**: Detailed breakdown of precision, recall, and F1-score by class
- **Model Files**: Serialized models that can be directly loaded for inference

## Creating Custom Hyperparameter Configurations

One of the strengths of our tuning framework is its flexibility. You can easily create custom hyperparameter configurations to explore specific model types or parameter ranges.

### Example: Creating a Custom SVM Configuration

Here's an example of creating a configuration file for Support Vector Machines:

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

Save this as `model_training/hyperparameter_configs/svm.yaml` and run:

```bash
python model_training/tune.py --config model_training/hyperparameter_configs/svm.yaml
```

### Example: Complex Neural Network Tuning

For more complex models like neural networks, you might create a configuration like:

```yaml
models:
  - name: DeepNN_Grid
    class: MLPClassifier
    init_params:
      random_state: 42
      max_iter: 1000
    param_grid:
      hidden_layer_sizes: [[50], [100], [50, 30], [100, 50]]
      activation: ["relu", "tanh"]
      alpha: [0.0001, 0.001, 0.01]
      learning_rate_init: [0.001, 0.01, 0.1]
      solver: ["adam", "sgd"]
```

### Best Practices for Configuration Design

When creating hyperparameter configurations, consider these best practices:

1. **Start Small**: Begin with a small set of hyperparameters and values to ensure everything works correctly before expanding to a larger search space.

2. **Use Log Scales**: For parameters like learning rate or regularization strength, consider using a logarithmic scale (e.g., [0.001, 0.01, 0.1, 1.0] instead of [0.1, 0.2, 0.3, 0.4]).

3. **Include Baseline Models**: Always include simple models with default parameters as a baseline for comparison.

4. **Iterative Refinement**: Use the results of an initial broad search to identify promising regions of the hyperparameter space, then conduct a more focused search in those regions.

5. **Consider Compute Resources**: Grid search grows exponentially with the number of parameters and values. For large searches, consider using random search or reducing the parameter space.

## Integration with Azure ML

For large-scale hyperparameter tuning jobs that require significant compute resources, our module also supports submitting jobs to Azure ML. This allows you to leverage cloud computing power for extensive hyperparameter searches.

To submit a tuning job to Azure ML:

```bash
python model_training/submit_tune_job.py \
  --subscription_id "<your-subscription-id>" \
  --resource_group "<your-resource-group>" \
  --workspace_name "<your-workspace-name>" \
  --compute_target "<your-compute-target>" \
  --config model_training/hyperparameter_configs/random_forest.yaml
```

This command:
1. Authenticates with Azure
2. Configures the environment
3. Submits the tuning job to the specified compute target
4. Monitors the job progress

The results will be logged to the MLflow tracking server associated with your Azure ML workspace, providing the same rich visualization and comparison capabilities.

## Conclusion

Hyperparameter tuning is a critical step in developing high-performing machine learning models. Our tuning module, combined with MLflow tracking, provides a comprehensive solution for:

## Repository Code Files

The hyperparameter tuning functionality is implemented in these key files:

- [`model_tuner.py`](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/blob/main/model_training/model_tuner.py): The core class that handles hyperparameter search and MLflow logging
- [`tune.py`](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/blob/main/model_training/tune.py): The main script for running tuning jobs
- [`submit_tune_job.py`](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/blob/main/model_training/submit_tune_job.py): Script for submitting tuning jobs to Azure ML
- [`hyperparameter_configs/`](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/tree/main/model_training/hyperparameter_configs): Directory containing configuration files
- [`data_processor.py`](https://github.com/tharindu-sankalpa/mlflow-enterprise-platform/blob/main/model_training/data_processor.py): Handles data loading and preprocessing

This hyperparameter tuning module provides:

- Defining and executing hyperparameter searches
- Tracking all trials and metrics
- Visualizing and comparing results
- Identifying the optimal parameter combinations
- Saving and registering the best models

By leveraging this framework, you can systematically explore hyperparameter spaces, make data-driven decisions about model configuration, and maintain a complete record of all experiments. The integration with MLflow ensures that you can easily reproduce results, compare different approaches, and share findings with team members.

As you work with the module, remember that hyperparameter tuning is often an iterative process. Use the insights gained from each tuning run to refine your approach, focusing on the most promising areas of the hyperparameter space for subsequent iterations.