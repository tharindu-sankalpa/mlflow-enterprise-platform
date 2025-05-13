import mlflow
import mlflow.sklearn
import tempfile
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import itertools
from tqdm import tqdm
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid, ParameterSampler
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, average_precision_score
)


class ModelTuner:
    """
    Class for tuning classification models with hyperparameter search and MLflow integration.
    
    This class handles the hyperparameter tuning, evaluation, and logging of machine learning models
    for classification tasks. It provides capabilities for:
    
    1. Running grid search or randomized search for hyperparameter optimization
    2. Calculating and logging comprehensive metrics for each trial
    3. Generating visualizations for each parameter combination
    4. Producing detailed classification reports for each trial
    5. Saving models and artifacts to both local directories and MLflow
    6. Registering the best model in the MLflow registry
    
    Attributes:
        model: The base machine learning model to tune
        param_grid (dict): Dictionary of hyperparameters to search
        model_name (str): Identifier for the model
        experiment_name (str): Name of the MLflow experiment
        run_name (str): Name of the MLflow runs
        output_dir (str): Directory to save local outputs
        preprocessing_pipeline: Optional scikit-learn pipeline for preprocessing
        search_type (str): Type of search - 'grid' or 'random'
        search_params (dict): Additional parameters for the search
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric for hyperparameter optimization
        n_jobs (int): Number of parallel jobs for search
        verbose (int): Verbosity level
    """
    
    def __init__(
        self, 
        model, 
        param_grid, 
        model_name="Model", 
        experiment_name="Default", 
        output_dir="./outputs", 
        run_name=None, 
        preprocessing_pipeline=None,
        search_type="grid",
        search_params=None,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2,
        real_time_logging=True
    ):
        """
        Initialize the ModelTuner with model, parameters, and configuration.
        
        Parameters:
            model: Scikit-learn compatible model with fit, predict, and predict_proba methods
            param_grid (dict): Dictionary of hyperparameters to search
            model_name (str): Name identifier for the model
            experiment_name (str): Name of the MLflow experiment
            output_dir (str): Directory to save local output files
            run_name (str, optional): Base name for MLflow runs
            preprocessing_pipeline (optional): Scikit-learn pipeline for preprocessing
            search_type (str): Type of search - 'grid' or 'random'
            search_params (dict, optional): Additional parameters for the search like n_iter for random search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for hyperparameter optimization
            n_jobs (int): Number of parallel jobs (-1 to use all processors)
            verbose (int): Verbosity level (0: none, 1: minimal, 2: detailed)
            real_time_logging (bool): Whether to log trials to MLflow in real time
        """
        self.model = model
        self.param_grid = param_grid
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.run_name = run_name or f"{model_name}"
        self.output_dir = output_dir
        self.preprocessing_pipeline = preprocessing_pipeline
        self.search_type = search_type.lower()
        self.search_params = search_params or {}
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.real_time_logging = real_time_logging
        
        # Calculate number of available CPUs if using -1
        if self.n_jobs == -1:
            self.n_jobs = cpu_count()
        
        # Create full pipeline if preprocessing is provided
        if self.preprocessing_pipeline is not None:
            self.full_pipeline = Pipeline([
                ('preprocessor', self.preprocessing_pipeline),
                ('model', self.model)
            ])
        else:
            self.full_pipeline = None
        
        mlflow.set_experiment(experiment_name)
    
    def _log_metrics(self, y_true, y_pred, y_prob, prefix):
        """
        Log classification metrics to MLflow.
        
        Calculates and logs comprehensive classification metrics to MLflow,
        including accuracy, precision, recall, F1-score, log loss, ROC AUC,
        PR AUC, and confusion matrix elements.
        
        Parameters:
            y_true (array-like): Ground truth labels
            y_pred (array-like): Predicted labels
            y_prob (array-like): Predicted probabilities for the positive class
            prefix (str): Prefix for metric names (e.g., 'train' or 'test')
            
        Returns:
            numpy.ndarray: Confusion matrix
        """
        mlflow.log_metric(f"{prefix}_accuracy", accuracy_score(y_true, y_pred))
        mlflow.log_metric(f"{prefix}_precision", precision_score(y_true, y_pred))
        mlflow.log_metric(f"{prefix}_recall", recall_score(y_true, y_pred))
        mlflow.log_metric(f"{prefix}_f1_score", f1_score(y_true, y_pred))
        mlflow.log_metric(f"{prefix}_log_loss", log_loss(y_true, y_prob))
        mlflow.log_metric(f"{prefix}_roc_auc", roc_auc_score(y_true, y_prob))
        mlflow.log_metric(f"{prefix}_pr_auc", average_precision_score(y_true, y_prob))

        # Confusion matrix raw values
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        mlflow.log_metric(f"{prefix}_true_negative", tn)
        mlflow.log_metric(f"{prefix}_false_positive", fp)
        mlflow.log_metric(f"{prefix}_false_negative", fn)
        mlflow.log_metric(f"{prefix}_true_positive", tp)

        return cm

    def _log_confusion_matrix(self, cm, title, artifact_name):
        """
        Generate and log confusion matrix visualization.
        
        Creates a visual representation of the confusion matrix and saves it
        both as an MLflow artifact and to the local output directory.
        
        Parameters:
            cm (numpy.ndarray): Confusion matrix array
            title (str): Title for the confusion matrix plot
            artifact_name (str): Base name for the saved artifact
        """
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            mlflow.log_artifact(tmpfile.name, artifact_path="confusion_matrices")
        plt.close()

        # Also save to output directory
        out_path = os.path.join(self.output_dir, f"{artifact_name}_confusion_matrix.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        plt.savefig(out_path)
        plt.close()

    def _log_classification_report(self, y_true, y_pred, prefix):
        """
        Generate and log detailed classification report.
        
        Creates a text-based classification report including precision, recall,
        F1-score, and support for each class. Saves the report as both an MLflow
        artifact and to the local output directory.
        
        Parameters:
            y_true (array-like): Ground truth labels
            y_pred (array-like): Predicted labels
            prefix (str): Prefix for the report filename
        """
        report = classification_report(y_true, y_pred)
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tmpfile:
            tmpfile.write(report)
            tmpfile.flush()
            mlflow.log_artifact(tmpfile.name, artifact_path="classification_reports")

        # Also save to output directory
        out_path = os.path.join(self.output_dir, f"{prefix}_classification_report.txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
    
    def _log_cv_results(self, search_results):
        """
        Log cross-validation results from hyperparameter search.
        
        Saves the full cross-validation results as a CSV file and
        logs it as an MLflow artifact.
        
        Parameters:
            search_results (dict): CV results with trials
        """
        # Convert results to DataFrame
        cv_results_df = pd.DataFrame(search_results)
        
        # Save to local file
        results_path = os.path.join(self.output_dir, f"{self.model_name}_cv_results.csv")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        cv_results_df.to_csv(results_path, index=False)
        
        # Log to MLflow
        mlflow.log_artifact(results_path, artifact_path="cv_results")
    
    def _prepare_param_grid_for_pipeline(self, param_grid):
        """
        Prepare parameter grid for use with a pipeline.
        
        When using a pipeline, parameter names need to be prefixed with 'model__'
        to correctly target the model component of the pipeline.
        
        Parameters:
            param_grid (dict): Original parameter grid
            
        Returns:
            dict: Parameter grid with adjusted key names for pipeline use
        """
        if self.full_pipeline is not None:
            return {f'model__{key}': value for key, value in param_grid.items()}
        return param_grid
    
    def _generate_cv_folds(self, X, y):
        """
        Generate cross-validation folds.
        
        Parameters:
            X (array-like): Training feature data
            y (array-like): Training target data
            
        Returns:
            list: List of (train_idx, val_idx) tuples for cross-validation
        """
        if isinstance(self.cv, int):
            cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
            return list(cv.split(X, y))
        return self.cv
    
    def _calculate_total_trials(self, param_grid):
        """
        Calculate the total number of parameter combinations for grid search.
        
        Parameters:
            param_grid (dict): Parameter grid
            
        Returns:
            int: Total number of parameter combinations
        """
        if self.search_type == 'grid':
            return len(list(ParameterGrid(param_grid)))
        else:  # random search
            n_iter = self.search_params.get('n_iter', 10)
            return n_iter
    
    def _log_one_trial(self, params, X_train, y_train, X_test, y_test, cv_folds, trial_idx, estimator=None, extra_results=None):
        """
        Log a single hyperparameter trial to MLflow.
        
        Parameters:
            params (dict): Parameter combination to evaluate
            X_train (array-like): Training feature data
            y_train (array-like): Training target data
            X_test (array-like): Testing feature data
            y_test (array-like): Testing target data
            cv_folds (list): Cross-validation folds
            trial_idx (int): Trial index
            estimator (object, optional): Fitted estimator (if already trained)
            extra_results (dict, optional): Extra results to include
            
        Returns:
            dict: Evaluation results
        """
        # Start a new run for this parameter combination
        run_name = f"Trial {trial_idx+1} - {self.model_name}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log run metadata
            mlflow.set_tag("trial_number", trial_idx + 1)
            mlflow.set_tag("model_type", self.model.__class__.__name__)
            mlflow.set_tag("search_type", self.search_type)
            
            # Log parameters
            mlflow.log_params(params)
            
            if estimator is None:
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
                    if hasattr(estimator, 'predict_proba'):
                        y_cv_val_prob = estimator.predict_proba(X_cv_val)[:, 1]
                        score = roc_auc_score(y_cv_val, y_cv_val_prob)
                    else:
                        y_cv_val_pred = estimator.predict(X_cv_val)
                        score = accuracy_score(y_cv_val, y_cv_val_pred)
                    
                    cv_scores.append(score)
                
                # Log cross-validation metrics
                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)
                
                # Train on full training set
                estimator.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = estimator.predict(X_train)
                y_train_prob = estimator.predict_proba(X_train)[:, 1]
                y_test_pred = estimator.predict(X_test)
                y_test_prob = estimator.predict_proba(X_test)[:, 1]
                
                # Calculate scores
                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test, y_test_pred)
                test_roc_auc = roc_auc_score(y_test, y_test_prob)
                
            else:
                # Use provided estimator and scores
                mean_cv_score = extra_results['mean_cv_score']
                std_cv_score = extra_results['std_cv_score']
                train_score = extra_results['train_score']
                test_score = extra_results['test_score']
                test_roc_auc = extra_results['test_roc_auc']
                y_train_pred = extra_results['y_train_pred']
                y_train_prob = extra_results['y_train_prob']
                y_test_pred = extra_results['y_test_pred']
                y_test_prob = extra_results['y_test_prob']
            
            # Log metrics
            mlflow.log_metric("mean_cv_score", mean_cv_score)
            mlflow.log_metric("std_cv_score", std_cv_score)
            mlflow.log_metric("train_score", train_score)
            mlflow.log_metric("test_score", test_score)
            mlflow.log_metric("test_roc_auc", test_roc_auc)
            
            # Log detailed metrics
            cm_train = self._log_metrics(y_train, y_train_pred, y_train_prob, "train")
            cm_test = self._log_metrics(y_test, y_test_pred, y_test_prob, "test")
            
            # Log confusion matrices
            self._log_confusion_matrix(cm_train, f"Trial {trial_idx+1} - Train", f"{self.model_name}_train_{trial_idx+1}")
            self._log_confusion_matrix(cm_test, f"Trial {trial_idx+1} - Test", f"{self.model_name}_test_{trial_idx+1}")
            
            # Log classification reports
            self._log_classification_report(y_train, y_train_pred, f"{self.model_name}_train_{trial_idx+1}")
            self._log_classification_report(y_test, y_test_pred, f"{self.model_name}_test_{trial_idx+1}")
            
            # Create model signature
            input_example = X_train.iloc[:5]
            
            # Log model with signature
            mlflow.sklearn.log_model(
                estimator, 
                "model",
                signature=mlflow.models.infer_signature(X_train, y_train),
                input_example=input_example
            )
            
            # Print run URL
            run_id = run.info.run_id
            run_url = mlflow.get_registry_uri() or "http://127.0.0.1:5000"
            if run_url and "http" in run_url:
                print(f"🏃 View run {run_name} at: {run_url}/#/experiments/{run.info.experiment_id}/runs/{run_id}")
                print(f"🧪 View experiment at: {run_url}/#/experiments/{run.info.experiment_id}")
            
            return {
                'params': params,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': std_cv_score,
                'train_score': train_score,
                'test_score': test_score,
                'test_roc_auc': test_roc_auc,
                'estimator': estimator,
                'run_id': run_id,
                'y_train_pred': y_train_pred,
                'y_train_prob': y_train_prob,
                'y_test_pred': y_test_pred,
                'y_test_prob': y_test_prob
            }
    
    def _evaluate_params_parallel(self, params, X_train, y_train, X_test, y_test, cv_folds, trial_idx):
        """
        Evaluate a single parameter combination without logging to MLflow.
        Used for parallel processing.
        
        Parameters:
            params (dict): Parameter combination to evaluate
            X_train (array-like): Training feature data
            y_train (array-like): Training target data
            X_test (array-like): Testing feature data
            y_test (array-like): Testing target data
            cv_folds (list): Cross-validation folds
            trial_idx (int): Trial index
            
        Returns:
            dict: Evaluation results
        """
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
            if hasattr(estimator, 'predict_proba'):
                y_cv_val_prob = estimator.predict_proba(X_cv_val)[:, 1]
                score = roc_auc_score(y_cv_val, y_cv_val_prob)
            else:
                y_cv_val_pred = estimator.predict(X_cv_val)
                score = accuracy_score(y_cv_val, y_cv_val_pred)
            
            cv_scores.append(score)
        
        # Train on full training set
        estimator.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = estimator.predict(X_train)
        y_train_prob = estimator.predict_proba(X_train)[:, 1]
        y_test_pred = estimator.predict(X_test)
        y_test_prob = estimator.predict_proba(X_test)[:, 1]
        
        return {
            'params': params,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'train_score': accuracy_score(y_train, y_train_pred),
            'test_score': accuracy_score(y_test, y_test_pred),
            'test_roc_auc': roc_auc_score(y_test, y_test_prob),
            'estimator': estimator,
            'y_train_pred': y_train_pred,
            'y_train_prob': y_train_prob,
            'y_test_pred': y_test_pred,
            'y_test_prob': y_test_prob,
            'trial_idx': trial_idx
        }
    
    def _log_summary_results(self, all_results, best_result):
        """
        Log a summary of all trials in a CSV file.
        
        Parameters:
            all_results (list): List of evaluation results for all trials
            best_result (dict): Result for the best trial
        """
        # Prepare CSV file with results
        summary_data = []
        for i, result in enumerate(all_results):
            summary_data.append({
                'trial': i + 1,
                'run_id': result.get('run_id', ''),
                'params': str(result['params']),
                'mean_cv_score': result['mean_cv_score'],
                'std_cv_score': result['std_cv_score'],
                'train_score': result['train_score'],
                'test_score': result['test_score'],
                'test_roc_auc': result['test_roc_auc'],
                'is_best': (result == best_result)
            })
        
        # Convert to DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, f"{self.model_name}_tuning_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Also save as HTML for easier viewing
        html_path = os.path.join(self.output_dir, f"{self.model_name}_tuning_summary.html")
        summary_df.to_html(html_path, index=False)
        
        print(f"📊 Tuning summary saved to {summary_path}")
    
    def tune_with_custom_search(self, X_train, X_test, y_train, y_test, use_pipeline=True):
        """
        Tune hyperparameters with custom search implementation that logs each trial in real-time.
        
        Parameters:
            X_train (array-like): Training feature data
            X_test (array-like): Testing feature data
            y_train (array-like): Training target data
            y_test (array-like): Testing target data
            use_pipeline (bool): Whether to use the full preprocessing pipeline
                               
        Returns:
            dict: The best model from hyperparameter search with its results
        """
        # Ensure data is in DataFrame format for easier indexing in CV
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        if not isinstance(y_train, (pd.DataFrame, pd.Series)):
            y_train = pd.Series(y_train)
        if not isinstance(y_test, (pd.DataFrame, pd.Series)):
            y_test = pd.Series(y_test)
        
        # Determine which model/pipeline to use
        if use_pipeline and self.full_pipeline is not None:
            grid = self._prepare_param_grid_for_pipeline(self.param_grid)
        else:
            grid = self.param_grid
        
        # Generate parameter combinations
        if self.search_type == 'grid':
            param_list = list(ParameterGrid(grid))
        else:  # random search
            n_iter = self.search_params.get('n_iter', 10)
            param_list = list(ParameterSampler(grid, n_iter=n_iter, random_state=42))
        
        # Generate CV folds once
        cv_folds = self._generate_cv_folds(X_train, y_train)
        
        # Print summary of the search
        total_trials = len(param_list)
        cv_folds_count = len(cv_folds)
        total_fits = total_trials * cv_folds_count
        
        print(f"\n{'='*40}")
        print(f"Hyperparameter Search Summary:")
        print(f"{'='*40}")
        print(f"Model: {self.model_name}")
        print(f"Search type: {self.search_type}")
        print(f"Total parameter combinations: {total_trials}")
        print(f"Cross-validation folds: {cv_folds_count}")
        print(f"Total model fits: {total_fits}")
        print(f"Parallel jobs: {self.n_jobs}")
        print(f"Real-time MLflow logging: {'Disabled' if not self.real_time_logging else 'Enabled'}")
        print(f"{'='*40}\n")
        
        # Execute trials
        all_results = []
        
        if self.real_time_logging:
            # Execute trials one by one with real-time logging
            for i, params in enumerate(tqdm(param_list, desc="Evaluating hyperparameters")):
                result = self._log_one_trial(params, X_train, y_train, X_test, y_test, cv_folds, i)
                all_results.append(result)
        else:
            # Execute trials in parallel
            print(f"Running {total_trials} trials in parallel with {self.n_jobs} workers...")
            parallel_results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._evaluate_params_parallel)(params, X_train, y_train, X_test, y_test, cv_folds, i) 
                for i, params in enumerate(param_list)
            )
            
            # Log results to MLflow after all parallel execution is complete
            print(f"Parallel execution complete. Logging results to MLflow...")
            for result in tqdm(parallel_results, desc="Logging results to MLflow"):
                trial_idx = result['trial_idx']
                # Log to MLflow
                mlflow_result = self._log_one_trial(
                    result['params'], 
                    X_train, y_train, X_test, y_test,
                    cv_folds, trial_idx,
                    estimator=result['estimator'],
                    extra_results=result
                )
                all_results.append(mlflow_result)
        
        # Find the best parameter set based on mean_cv_score
        best_idx = np.argmax([r['mean_cv_score'] for r in all_results])
        best_result = all_results[best_idx]
        
        # Save best model to output directory
        import joblib
        best_model_path = os.path.join(self.output_dir, f"{self.model_name}_best_model.joblib")
        joblib.dump(best_result['estimator'], best_model_path)
        
        # Register the best model in MLflow
        model_uri = f"runs:/{best_result['run_id']}/model"
        try:
            registered_model = mlflow.register_model(model_uri, f"{self.model_name}_tuned")
            print(f"\nBest model registered as: {self.model_name}_tuned (version {registered_model.version})")
        except Exception as e:
            print(f"\nWarning: Could not register model: {e}")
        
        # Prepare and save CV results
        cv_results = {
            'params': [str(r['params']) for r in all_results],
            'mean_cv_score': [r['mean_cv_score'] for r in all_results],
            'std_cv_score': [r['std_cv_score'] for r in all_results],
            'train_score': [r['train_score'] for r in all_results],
            'test_score': [r['test_score'] for r in all_results],
            'test_roc_auc': [r['test_roc_auc'] for r in all_results],
            'rank': [sorted(range(len(all_results)), 
                           key=lambda i: all_results[i]['mean_cv_score'], 
                           reverse=True).index(i) + 1 
                    for i in range(len(all_results))]
        }
        
        # Log summary results
        self._log_summary_results(all_results, best_result)
        
        print(f"\nTuning completed. Best parameters:")
        for k, v in best_result['params'].items():
            print(f"  {k}: {v}")
        print(f"Best CV score: {best_result['mean_cv_score']:.4f}")
        print(f"Best test score: {best_result['test_score']:.4f}")
        print(f"Best test ROC AUC: {best_result['test_roc_auc']:.4f}")
        print(f"Best model saved to: {best_model_path}")
        
        return best_result
    
    def tune_and_log(self, X_train, X_test, y_train, y_test, use_pipeline=True):
        """
        Tune hyperparameters, evaluate performance, and log results to MLflow.
        
        This method performs a complete hyperparameter tuning workflow by calling
        the appropriate search strategy.
        
        Parameters:
            X_train (array-like): Training feature data
            X_test (array-like): Testing feature data
            y_train (array-like): Training target data
            y_test (array-like): Testing target data
            use_pipeline (bool): Whether to use the full preprocessing pipeline
                               (if False, assumes data is already preprocessed)
                                
        Returns:
            The best model from hyperparameter search
        """
        # Use our custom search implementation for real-time logging
        best_result = self.tune_with_custom_search(X_train, X_test, y_train, y_test, use_pipeline)
        return best_result['estimator']