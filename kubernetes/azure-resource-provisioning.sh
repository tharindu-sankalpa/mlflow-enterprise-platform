

export RESOURCE_GROUP="tharindu-mlflow-rg"
export LOCATION="centralindia"
export AKS_CLUSTER_NAME="tharindu-mlflow-aks"
export STORAGE_ACCOUNT_NAME="tharindumlflow$(openssl rand -hex 4)" # Needs to be globally unique
export CONTAINER_NAME="artifactroot"
export K8S_NAMESPACE="mlflow"

az login --use-device-code

az group create --name $RESOURCE_GROUP --location $LOCATION

az provider register --namespace Microsoft.ContainerService
az provider show --namespace Microsoft.ContainerService --query "registrationState"

az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_CLUSTER_NAME \
  --node-count 1 \
  --enable-managed-identity \
  --generate-ssh-keys \
  --location $LOCATION

az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER_NAME --overwrite-existing

az storage account create \
  --name $STORAGE_ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS \
  --kind StorageV2

az storage container create \
  --name $CONTAINER_NAME \
  --account-name $STORAGE_ACCOUNT_NAME


export AZURE_STORAGE_ACCESS_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT_NAME --query "[0].value" -o tsv)


helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo update


echo "Your Container Name: $CONTAINER_NAME"
echo "Your Storage Account Name: $STORAGE_ACCOUNT_NAME"
echo "Your Azure Storage Access Key: $AZURE_STORAGE_ACCESS_KEY"


helm install mlflow community-charts/mlflow \
  --namespace $K8S_NAMESPACE \
  --create-namespace \
  -f mlflow-values.yaml


kubectl get pods -n $K8S_NAMESPACE
kubectl get svc -n $K8S_NAMESPACE
kubectl get pvc  -n $K8S_NAMESPACE
kubectl get secret  -n $K8S_NAMESPACE
kubectl describe deployment mlflow -n $K8S_NAMESPACE

export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="<YOUR_MLFLOW_UI_ADMIN_PASSWORD>"

# helm upgrade mlflow community-charts/mlflow
# helm upgrade mlflow community-charts/mlflow --namespace $K8S_NAMESPACE -f mlflow-values.yaml

# Ubunut 20.04 LTS
mlflow server \
    --backend-store-uri file:///home/tharindu/git/mlflow_iris_example/mlflow/mlruns \
    --default-artifact-root file:///home/tharindu/git/mlflow_iris_example/mlflow/mlruns \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 2

# WSL 2 Ubuntu 22.04 LTS
mlflow server \
    --backend-store-uri file:///home/tharindu/repos/mlflow_iris_example/mlflow/mlruns \
    --default-artifact-root file:///home/tharindu/repos/mlflow_iris_example/mlflow/mlruns \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 2