import mlflow
import mlflow.sklearn
import tempfile
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, average_precision_score
)


class ModelTrainer:
    """Class for training and logging classification models"""
    
    def __init__(self, model, model_name="Model", experiment_name="Default", output_dir="./outputs", run_name=None):
        self.model = model
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.run_name = run_name or f"{model_name} Training"
        self.output_dir = output_dir
        
        mlflow.set_experiment(experiment_name)
        mlflow.sklearn.autolog()

    def _log_metrics(self, y_true, y_pred, y_prob, prefix):
        """Log classification metrics to MLflow"""
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
        """Generate and log confusion matrix plot"""
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
        """Generate and log classification report"""
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

    def train_and_log(self, X_train, X_test, y_train, y_test):
        """Train model and log metrics and artifacts"""
        with mlflow.start_run(run_name=self.run_name):
            self.model.fit(X_train, y_train)

            # Train predictions
            y_train_pred = self.model.predict(X_train)
            y_train_prob = self.model.predict_proba(X_train)[:, 1]

            # Test predictions
            y_test_pred = self.model.predict(X_test)
            y_test_prob = self.model.predict_proba(X_test)[:, 1]

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
            joblib.dump(self.model, model_path)
            
            # Register the model in MLflow for easy deployment
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            try:
                mlflow.register_model(model_uri, self.model_name)
            except Exception as e:
                print(f"Warning: Could not register model: {e}")
