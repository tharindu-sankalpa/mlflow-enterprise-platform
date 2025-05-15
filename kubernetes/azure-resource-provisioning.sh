

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

################
# update mlflow-values.yaml
################
helm install mlflow community-charts/mlflow \
  --namespace $K8S_NAMESPACE \
  --create-namespace \
  -f mlflow-values.yaml


kubectl get pods -n $K8S_NAMESPACE
kubectl get svc -n $K8S_NAMESPACE
kubectl get pvc  -n $K8S_NAMESPACE
kubectl get secret  -n $K8S_NAMESPACE
kubectl describe deployment mlflow -n $K8S_NAMESPACE

export SUBSCRIPTION_ID=$(az account show --query id -o tsv)

# It's crucial to use the *actual* name of your storage account created earlier.
# If you don't remember it, you can list storage accounts in your resource group:
# az storage account list --resource-group $RESOURCE_GROUP --query "[].name" -o tsv
# Then set it:
# export STORAGE_ACCOUNT_NAME="your-actual-storage-account-name"
echo "Using Resource Group: $RESOURCE_GROUP"
echo "Using Storage Account: $STORAGE_ACCOUNT_NAME"
echo "Using Subscription ID: $SUBSCRIPTION_ID"

# Choose a Name for your Service Principal
export SP_NAME="mlflow-artifact-uploader-sp-$(openssl rand -hex 3)" # Creates a unique name
echo "Will create Service Principal named: $SP_NAME"


APP_CREATE_OUTPUT=$(az ad app create --display-name "$SP_NAME")
APP_ID=$(echo $APP_CREATE_OUTPUT | jq -r '.appId')
echo $APP_ID

SP_CREATE_OUTPUT=$(az ad sp create --id "$APP_ID")
SP_OBJECT_ID=$(echo $SP_CREATE_OUTPUT | jq -r '.id') 
echo $SP_OBJECT_ID


CLIENT_SECRET_OUTPUT=$(az ad app credential reset --id "$APP_ID" --append --display-name "mlflow_client_secret")
CLIENT_SECRET=$(echo $CLIENT_SECRET_OUTPUT | jq -r '.password')
echo $CLIENT_SECRET


AZURE_TENANT_ID=$(az account show --query tenantId -o tsv)
echo "Azure Tenant ID: $AZURE_TENANT_ID"

echo "Constructing scope for role assignment..."
STORAGE_ACCOUNT_RESOURCE_ID="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$STORAGE_ACCOUNT_NAME"
echo "Scope: $STORAGE_ACCOUNT_RESOURCE_ID"

az role assignment create \
    --assignee-object-id "$SP_OBJECT_ID" \
    --assignee-principal-type ServicePrincipal \
    --role "Storage Blob Data Contributor" \
    --scope "$STORAGE_ACCOUNT_RESOURCE_ID"

echo $APP_ID
echo $CLIENT_SECRET
echo $AZURE_TENANT_ID


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

mlflow models serve -m runs:/977841598692450c98260e4d76b23cdd/model -p 1234 --enable-mlserver
mlflow models serve -m runs:/977841598692450c98260e4d76b23cdd/model -p 1234 --enable-mlserver


mlflow models serve \
  -m runs:/35afb161a5524d5598efe2267b6f3035/model \
  -p 1234 \
  --enable-mlserver

mlflow models serve -m runs:/35afb161a5524d5598efe2267b6f3035/model -p 1234 --enable-mlserver

mlflow models build-docker -m "runs:/977841598692450c98260e4d76b23cdd/model" -n classifier-service