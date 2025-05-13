# MLflow Enterprise Platform: Complete Experimentation and Deployment Solution

This repository provides a production-ready, enterprise-grade platform for machine learning experimentation, tracking, and deployment using MLflow. It offers a comprehensive solution that covers the full ML lifecycle from development to production deployment, with robust tooling for hyperparameter optimization, model tracking, and cloud integration.

## Enterprise-Grade Features

- **Scalable Architecture**: Deployment on Kubernetes with persistent storage
- **Production Workflows**: Standardized pipelines from training to deployment
- **Security Integration**: Authentication and role-based access control
- **Cloud-Native Infrastructure**: Seamless integration with Azure services
- **Comprehensive Observability**: Advanced tracking and monitoring of all experiments
- **Reproducible Research**: Environment versioning and experiment records
- **Collaborative Development**: Team-oriented ML experiment management

## Repository Structure

- `data/`: Contains datasets and metadata files
  - `california_housing.csv`: Housing dataset for regression examples
  - `metadata/`: JSON files with dataset schema information
  - `train_test_data.pkl`: Preprocessed train/test data

- `kubernetes/`: Production deployment on Kubernetes
  - `mlflow-values.yaml`: Helm chart values for MLflow server
  - `azure-resource-provisioning.sh`: Scripts for setting up Azure resources

- `model_training/`: Core ML training and tuning code
  - `data_processor.py`: Data loading and preprocessing pipeline
  - `model_trainer.py`: Model training with MLflow tracking
  - `model_tuner.py`: Hyperparameter tuning infrastructure
  - `train.py`: Main training script
  - `tune.py`: Main hyperparameter tuning script
  - `hyperparameter_configs/`: YAML/JSON configurations for tuning different models

- `notebook/`: Jupyter notebooks for interactive examples
  - `adult-classification.ipynb`: Classification example with Adult Income dataset
  - `california_housing-regression.ipynb`: Regression example with housing data

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tharindu-sankalpa/mlflow-enterprise-platform.git
   cd mlflow-enterprise-platform
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```
   
   Or with pip:
   ```bash
   pip install -r model_training/requirements.txt
   ```

## Getting Started

### Local MLflow Tracking

1. Start a local MLflow server:
   ```bash
   mlflow ui
   ```

2. Run a basic training job:
   ```bash
   python model_training/train.py
   ```

3. Try hyperparameter tuning:
   ```bash
   python model_training/tune.py --config model_training/hyperparameter_configs/logistic_regression.json
   ```

### Enterprise Deployment on Azure Kubernetes Service

For a production-grade MLflow server deployment, follow the instructions in the `kubernetes/README.md` file. This will guide you through:

1. Setting up Azure resources with `azure-resource-provisioning.sh`
2. Deploying MLflow on AKS with Helm
3. Configuring persistent storage with Azure Blob Storage
4. Setting up PostgreSQL database for metadata
5. Implementing authentication and access control

## Documentation

For detailed documentation on specific components:

- [Hyperparameter Tuning Framework](./model_training/README_HYPERPARAMETER_TUNING.md)
- [Model Training System](./model_training/README.md)
- [Enterprise Deployment on Kubernetes](./kubernetes/README.md)

## Key Platform Capabilities

This enterprise platform provides:

1. **Advanced Experiment Tracking**: Organized logging of parameters, metrics, and artifacts with hierarchical organization
2. **Modular Preprocessing Pipelines**: Scikit-learn pipelines integrated with MLflow for reproducible data transformations
3. **Comprehensive Hyperparameter Optimization**: Scalable grid and random search with parallel execution
4. **Enterprise Cloud Integration**: Production deployment on Azure with Kubernetes and managed services
5. **MLOps Best Practices**: CI/CD integration, model registry, and automated workflows
6. **Collaborative Model Development**: Team-oriented experimentation and knowledge sharing

## Example Production Workflows

### Classification Model Development Workflow

The repository includes a complete enterprise workflow for model development:

```bash
# Train a simple logistic regression model with tracking
python model_training/train.py

# Perform distributed hyperparameter tuning
python model_training/tune.py --config model_training/hyperparameter_configs/multi_model.yaml --no_real_time_logging

# Submit job to Azure ML compute cluster
python model_training/submit_tune_job.py \
  --subscription_id "<your-subscription-id>" \
  --resource_group "<your-resource-group>" \
  --workspace_name "<your-workspace-name>" \
  --compute_target "<your-compute-target>" \
  --config model_training/hyperparameter_configs/random_forest.yaml
```

### Interactive Development Environment

For team exploration and development:

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open notebooks in the `notebook/` directory

## Enterprise Integration Points

- **CI/CD Systems**: Hooks for integrating with Azure DevOps, GitHub Actions, etc.
- **Container Platforms**: Kubernetes deployment for scalable, cloud-agnostic operation
- **Data Storage Systems**: Configurable backends for artifacts and metadata
- **Authentication Services**: Integration with Azure AD and other identity providers
- **Monitoring Tools**: Exportable metrics for integration with enterprise monitoring systems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The UCI Adult Income Dataset
- The MLflow team for their excellent tool
- All contributors to the Python data science ecosystem