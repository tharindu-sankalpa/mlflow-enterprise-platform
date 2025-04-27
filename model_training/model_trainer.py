import mlflow
import mlflow.sklearn
import tempfile
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, average_precision_score
)


class ModelTrainer:
    """
    Class for training and evaluating classification models with MLflow integration.
    
    This class handles the training, evaluation, and logging of machine learning models
    for classification tasks. It provides capabilities for:
    
    1. Training models with or without preprocessing pipelines
    2. Calculating and logging comprehensive metrics
    3. Generating visualizations (confusion matrices)
    4. Producing detailed classification reports
    5. Saving models and artifacts to both local directories and MLflow
    6. Registering models in the MLflow registry
    
    Attributes:
        model: The machine learning model to train
        model_name (str): Identifier for the model
        experiment_name (str): Name of the MLflow experiment
        run_name (str): Name of the MLflow run
        output_dir (str): Directory to save local outputs
        preprocessing_pipeline: Optional scikit-learn pipeline for preprocessing
        full_pipeline: Combined preprocessing + model pipeline
    """
    
    def __init__(self, model, model_name="Model", experiment_name="Default", output_dir="./outputs", run_name=None, preprocessing_pipeline=None):
        """
        Initialize the ModelTrainer with model and configuration parameters.
        
        Parameters:
            model: Scikit-learn compatible model with fit, predict, and predict_proba methods
            model_name (str): Name identifier for the model
            experiment_name (str): Name of the MLflow experiment
            output_dir (str): Directory to save local output files
            run_name (str, optional): Custom name for the MLflow run
            preprocessing_pipeline (optional): Scikit-learn pipeline for preprocessing
        """
        self.model = model
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.run_name = run_name or f"{model_name} Training"
        self.output_dir = output_dir
        self.preprocessing_pipeline = preprocessing_pipeline
        
        # Create full pipeline if preprocessing is provided
        if self.preprocessing_pipeline is not None:
            self.full_pipeline = Pipeline([
                ('preprocessor', self.preprocessing_pipeline),
                ('model', self.model)
            ])
        else:
            self.full_pipeline = None
        
        mlflow.set_experiment(experiment_name)
        mlflow.sklearn.autolog()

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

    def train_and_log(self, X_train, X_test, y_train, y_test, use_pipeline=True):
        """
        Train model, evaluate performance, and log results to MLflow.
        
        This method performs a complete workflow:
        1. Trains the model on the training data
        2. Makes predictions on both training and test sets
        3. Calculates and logs comprehensive metrics
        4. Generates and logs visualizations
        5. Saves the model locally and registers it in MLflow
        
        Parameters:
            X_train (array-like): Training feature data
            X_test (array-like): Testing feature data
            y_train (array-like): Training target data
            y_test (array-like): Testing target data
            use_pipeline (bool): Whether to use the full preprocessing pipeline
                                (if False, assumes data is already preprocessed)
                                
        Returns:
            The trained model or pipeline
        """
        with mlflow.start_run(run_name=self.run_name):
            if use_pipeline and self.full_pipeline is not None:
                # Train the full pipeline
                self.full_pipeline.fit(X_train, y_train)
                model_to_use = self.full_pipeline
                
                # Store model reference for easier access
                self.model = self.full_pipeline.named_steps['model']
            else:
                # If no pipeline or use_pipeline is False, just train the model
                self.model.fit(X_train, y_train)
                model_to_use = self.model
            
            # Train predictions
            y_train_pred = model_to_use.predict(X_train)
            y_train_prob = model_to_use.predict_proba(X_train)[:, 1]

            # Test predictions
            y_test_pred = model_to_use.predict(X_test)
            y_test_prob = model_to_use.predict_proba(X_test)[:, 1]

            # Log train/test metrics
            cm_train = self._log_metrics(y_train, y_train_pred, y_train_prob, "train")
            cm_test = self._log_metrics(y_test, y_test_pred, y_test_prob, "test")

            # Log confusion matrices
            self._log_confusion_matrix(cm_train, "Confusion Matrix - Train", f"{self.model_name}_train")
            self._log_confusion_matrix(cm_test, "Confusion Matrix - Test", f"{self.model_name}_test")

            # Log classification reports
            self._log_classification_report(y_train, y_train_pred, f"{self.model_name}_train")
            self._log_classification_report(y_test, y_test_pred, f"{self.model_name}_test")
            
            # Save model to output directory
            model_path = os.path.join(self.output_dir, f"{self.model_name}.joblib")
            import joblib
            joblib.dump(model_to_use, model_path)
            
            # Log pipeline separately for better visibility
            if use_pipeline and self.full_pipeline is not None:
                pipeline_path = os.path.join(self.output_dir, f"{self.model_name}_pipeline.joblib")
                joblib.dump(self.full_pipeline, pipeline_path)
                mlflow.log_artifact(pipeline_path, "pipeline")
            
            # Register the model in MLflow for easy deployment
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            try:
                mlflow.register_model(model_uri, self.model_name)
            except Exception as e:
                print(f"Warning: Could not register model: {e}")
            
            # Return the trained model
            return model_to_use
