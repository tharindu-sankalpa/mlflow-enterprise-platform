# train.py
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.datasets import load_iris
from pathlib import Path
import mlflow
import mlflow.sklearn

# Ignore some warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---

# **** ADD THIS LINE ****
# Choose ONE of the URIs below based on how you are accessing MLflow:
# Option A: If accessing via the LoadBalancer External IP
MLFLOW_TRACKING_URI = "http://135.235.186.123" # Port 80 is default for http

# Option B: If accessing via 'kubectl port-forward' mapping to local port 5000
# MLFLOW_TRACKING_URI = "http://localhost:5000"

# Option C: If running this script INSIDE the SAME Kubernetes cluster
# MLFLOW_TRACKING_URI = "http://mlflow.mlflow.svc.cluster.local:80" # Use internal service DNS

# Set the tracking URI for MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# ***********************

print(f"MLflow Version: {mlflow.__version__}")
print(f"Tracking URI: {mlflow.tracking.get_tracking_uri()}") # Verify the correct URI is set


# Parameters for the run
TEST_SPLIT_RATIO = 0.3
RANDOM_STATE = 42
LOGISTIC_REGRESSION_C = 0.1  # Inverse of regularization strength
EDA_ARTIFACT_DIR = "eda_plots" # Subdirectory for EDA artifacts within MLflow

# Create local directory for temporary EDA plots before logging
Path(EDA_ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)

# --- MLflow Run ---
try:
    # Start an MLflow run. MLflow entities will be recorded under this run.
    # Using 'with' ensures the run is properly closed even if errors occur.
    with mlflow.start_run(run_name="Iris Logistic Regression Run") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        print("MLflow Run Started...")

        # --- 1. Load Data ---
        print("Loading Iris dataset...")
        iris = load_iris()
        iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                               columns=iris['feature_names'] + ['target'])
        target_names = iris.target_names
        print(f"Dataset shape: {iris_df.shape}")
        # Log dataset characteristics
        mlflow.log_param("dataset_name", "Iris")
        mlflow.log_param("dataset_rows", iris_df.shape[0])
        mlflow.log_param("dataset_cols", iris_df.shape[1])

        # --- 2. Exploratory Data Analysis (EDA) ---
        print("Performing EDA and logging artifacts...")
        # Basic stats
        stats_path = "eda_summary_stats.csv"
        iris_df.describe().to_csv(stats_path)
        mlflow.log_artifact(stats_path, artifact_path=EDA_ARTIFACT_DIR)
        print(f"- Logged: {stats_path} to {EDA_ARTIFACT_DIR}/")

        # Pairplot
        pairplot_path = os.path.join(EDA_ARTIFACT_DIR, "eda_pairplot.png")
        sns.pairplot(iris_df, hue='target')
        plt.savefig(pairplot_path)
        plt.close() # Close plot to free memory
        mlflow.log_artifact(pairplot_path)
        print(f"- Logged: {pairplot_path}")

        # Feature Distributions
        for feature in iris.feature_names:
            plt.figure(figsize=(8, 4))
            sns.histplot(data=iris_df, x=feature, hue='target', kde=True)
            plot_path = os.path.join(EDA_ARTIFACT_DIR, f"eda_hist_{feature.replace(' ', '_')}.png")
            plt.title(f"Distribution of {feature}")
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)
            print(f"- Logged: {plot_path}")

        # --- 3. Data Splitting ---
        print("Splitting data...")
        X = iris_df[iris.feature_names]
        y = iris_df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE, stratify=y
        )
        mlflow.log_param("test_split_ratio", TEST_SPLIT_RATIO)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("train_set_shape", X_train.shape)
        mlflow.log_param("test_set_shape", X_test.shape)
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # --- 4. Model Training ---
        print("Training Logistic Regression model...")
        model = LogisticRegression(C=LOGISTIC_REGRESSION_C, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        print("Model training complete.")

        # Log model parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("logistic_regression_C", LOGISTIC_REGRESSION_C)

        # --- 5. Model Evaluation ---
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) # Get probabilities for potential future logging

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Log metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)

        # Log classification report as artifact (CSV and text)
        report_csv_path = "evaluation_classification_report.csv"
        report_df.to_csv(report_csv_path)
        mlflow.log_artifact(report_csv_path)
        print(f"- Logged: {report_csv_path}")

        report_txt_path = "evaluation_classification_report.txt"
        with open(report_txt_path, "w") as f:
             f.write(classification_report(y_test, y_pred, target_names=target_names))
        mlflow.log_artifact(report_txt_path)
        print(f"- Logged: {report_txt_path}")

        # Log confusion matrix as artifact (plot)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        cm_path = os.path.join(EDA_ARTIFACT_DIR, "evaluation_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        print(f"- Logged: {cm_path}")

        # --- 6. Log Model ---
        print("Logging the trained model...")
        # mlflow.sklearn.log_model logs the model in multiple formats (python_function, pickle)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris-logistic-regression-model", # Name in artifacts
            registered_model_name="iris-logistic-regression" # Optional: Register model
        )
        print("- Logged: Sklearn model 'iris-logistic-regression-model'")


        # --- 7. Log Environment & Code ---
        print("Logging environment and script...")
        # Log the lock file to reproduce environment
        if os.path.exists("poetry.lock"):
             mlflow.log_artifact("poetry.lock")
             print("- Logged: poetry.lock")
        elif os.path.exists("requirements.txt"): # Fallback if not using poetry
             mlflow.log_artifact("requirements.txt")
             print("- Logged: requirements.txt")

        # Log the training script itself
        mlflow.log_artifact(__file__)
        print(f"- Logged: {__file__}")

        print("MLflow Run Finished Successfully.")


except Exception as e:
    print(f"An error occurred during the MLflow run: {e}")
    # Optionally, log the error to MLflow if the run started
    # mlflow.log_param("run_error", str(e)) # Be careful logging potentially large error strings
    raise # Re-raise the exception after logging attempt

finally:
    # Clean up temporary local files if desired (optional)
    if os.path.exists(stats_path) : os.remove(stats_path)
    if os.path.exists(report_csv_path) : os.remove(report_csv_path)
    if os.path.exists(report_txt_path) : os.remove(report_txt_path)
    # Plots are saved directly to artifact paths or cleaned up by plt.close()
    print("Cleanup complete.")