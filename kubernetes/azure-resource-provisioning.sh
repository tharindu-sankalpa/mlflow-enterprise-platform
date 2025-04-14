

export RESOURCE_GROUP="tharindu-mlflow-rg"
export LOCATION="centralindia"
export AKS_CLUSTER_NAME="tharindu-mlflow-aks"
export STORAGE_ACCOUNT_NAME="tharindumlflow$(openssl rand -hex 4)" # Needs to be globally unique
export CONTAINER_NAME="artifactroot"
export K8S_NAMESPACE="mlflow"


helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo update

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
  --account-name $STORAGE_ACCOUNT_NAME \
  --auth-mode login # Assumes your logged-in az user has rights, or use --account-key



export AZURE_STORAGE_ACCESS_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT_NAME --query "[0].value" -o tsv)


sed -i.bak \
    -e "s|<CONTAINER_NAME>|$CONTAINER_NAME|g" \
    -e "s|<STORAGE_ACCOUNT_NAME>|$STORAGE_ACCOUNT_NAME|g" \
    -e "s|<YOUR_AZURE_STORAGE_ACCESS_KEY>|$AZURE_STORAGE_ACCESS_KEY|g" \
    mlflow-values.yaml


helm install mlflow community-charts/mlflow \
  --namespace $K8S_NAMESPACE \
  --create-namespace \
  -f mlflow-values.yaml


kubectl get pods --namespace mlflow
kubectl get svc --namespace mlflow
kubectl get pvc --namespace mlflow
kubectl get secret --namespace mlflow


export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="S3cr3+"

# helm upgrade mlflow community-charts/mlflow
# helm upgrade mlflow community-charts/mlflow --namespace $K8S_NAMESPACE -f mlflow-values.yaml

