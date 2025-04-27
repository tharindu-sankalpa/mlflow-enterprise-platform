from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Submit Adult Classification training job to Azure ML")
    parser.add_argument("--subscription_id", type=str, required=True, help="Azure subscription ID")
    parser.add_argument("--resource_group", type=str, required=True, help="Azure resource group")
    parser.add_argument("--workspace_name", type=str, required=True, help="Azure ML workspace name")
    parser.add_argument("--compute_target", type=str, required=True, help="Azure ML compute target")
    parser.add_argument("--experiment_name", type=str, default="Adult_Classification_Azure", help="Experiment name")
    parser.add_argument("--models", type=str, default="all", help="Models to train (all,logistic,rf,xgb,gb,knn)")
    parser.add_argument("--run_tensorflow", action="store_true", help="Run TensorFlow model")
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
        description="Environment for Adult Classification training",
        conda_file="./conda_dependencies.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    
    # Create the command
    # Define command
    training_job = command(
        code="./",  # Local path where the code is stored
        command=f"python train.py --models {args.models} --mlflow_experiment {args.experiment_name} {'--run_tensorflow' if args.run_tensorflow else ''}",
        environment=args.environment_name,
        compute=args.compute_target,
        display_name="Adult-Classification-Training",
        experiment_name=args.experiment_name,
        description="Adult Income Classification Training Job"
    )
    
    # Submit the job
    returned_job = ml_client.jobs.create_or_update(training_job)
    print(f"Submitted job: {returned_job.name}")
    
    # Get a URL for the job
    job_studio_url = returned_job.studio_url
    print(f"Job URL: {job_studio_url}")

if __name__ == "__main__":
    main()
