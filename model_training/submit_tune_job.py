from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
import os
import argparse

def main():
    """
    Main function for submitting an Azure ML hyperparameter tuning job.
    
    This function handles the entire workflow for submitting a hyperparameter tuning job to 
    Azure Machine Learning:
    1. Parsing command line arguments
    2. Authenticating with Azure using DefaultAzureCredential
    3. Connecting to the Azure ML workspace
    4. Configuring the training environment
    5. Defining the tuning command
    6. Submitting the job to Azure ML
    7. Providing status information and monitoring URLs
    
    The job runs the tune.py script on the specified Azure ML compute target, using
    the conda environment defined in conda_dependencies.yml.
    """
    parser = argparse.ArgumentParser(description="Submit Adult Classification hyperparameter tuning job to Azure ML")
    parser.add_argument("--subscription_id", type=str, required=True, help="Azure subscription ID")
    parser.add_argument("--resource_group", type=str, required=True, help="Azure resource group")
    parser.add_argument("--workspace_name", type=str, required=True, help="Azure ML workspace name")
    parser.add_argument("--compute_target", type=str, required=True, help="Azure ML compute target")
    parser.add_argument("--experiment_name", type=str, default="Adult_Classification_Tuning", help="Experiment name")
    parser.add_argument("--config", type=str, required=True, help="Path to hyperparameter configuration file")
    parser.add_argument("--search_type", type=str, default="grid", choices=["grid", "random"], help="Type of hyperparameter search")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of iterations for random search")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--environment_name", type=str, default="adult-classification-env", help="Environment name")
    args = parser.parse_args()
    
    # Get credential
    credential = DefaultAzureCredential()
    
    # Connect to workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name
    )
    
    # Define environment
    custom_env = Environment(
        name=args.environment_name,
        description="Environment for Adult Classification hyperparameter tuning",
        conda_file="./conda_dependencies.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    
    # Get the base filename from the config path
    config_filename = os.path.basename(args.config)
    
    # Create the command string with appropriate arguments
    command_str = (
        f"python tune.py "
        f"--config hyperparameter_configs/{config_filename} "
        f"--mlflow_experiment {args.experiment_name} "
        f"--search_type {args.search_type} "
        f"--n_iter {args.n_iter} "
        f"--cv_folds {args.cv_folds}"
    )
    
    # Create the job
    tuning_job = command(
        code="./",  # Local path where the code is stored
        command=command_str,
        environment=args.environment_name,
        compute=args.compute_target,
        display_name=f"Adult-Classification-Tuning-{args.search_type}",
        experiment_name=args.experiment_name,
        description=f"Adult Income Classification Hyperparameter Tuning Job - {args.search_type} search"
    )
    
    # Submit the job
    returned_job = ml_client.jobs.create_or_update(tuning_job)
    print(f"Submitted job: {returned_job.name}")
    
    # Get a URL for the job
    job_studio_url = returned_job.studio_url
    print(f"Job URL: {job_studio_url}")

if __name__ == "__main__":
    main()