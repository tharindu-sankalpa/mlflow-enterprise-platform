

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

########################################################################################################
# --- Install KServe and dependencies ---
echo "Installing KServe and dependencies..."

# Create namespaces
export KSERVE_NAMESPACE="kserve"
kubectl create namespace $KSERVE_NAMESPACE

# Install Knative Serving CRDs
echo "Installing Knative Serving CRDs..."
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.18.0/serving-crds.yaml

# Install Knative Serving core components
echo "Installing Knative Serving core components..."
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.18.0/serving-core.yaml

# Download and install Istio
echo "Downloading and installing Istio..."
ISTIO_VERSION="1.26.0"
ISTIO_DIR="istio-${ISTIO_VERSION}"

if [ ! -d "$ISTIO_DIR" ]; then
  curl -L https://istio.io/downloadIstio | ISTIO_VERSION=$ISTIO_VERSION sh -
fi

# Add Istio binaries to PATH temporarily
export PATH="$PWD/$ISTIO_DIR/bin:$PATH"
echo "Istio downloaded. Using istioctl version: $(istioctl version --remote=false)"

# Create Istio system namespace
kubectl create namespace istio-system --dry-run=client -o yaml | kubectl apply -f -

# Create Istio configuration file
cat > istio-config.yaml << EOF
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
spec:
  profile: default
  components:
    egressGateways:
    - name: istio-egressgateway
      enabled: true
    ingressGateways:
    - name: istio-ingressgateway
      enabled: true
EOF

# Install Istio with the configuration
echo "Installing Istio with custom configuration..."
istioctl install -f istio-config.yaml --verify -y

# Install Istio monitoring and visualization addons
echo "Installing Istio addons (Prometheus, Grafana, Jaeger, Kiali)..."
kubectl apply -f $ISTIO_DIR/samples/addons/prometheus.yaml
kubectl apply -f $ISTIO_DIR/samples/addons/grafana.yaml
kubectl apply -f $ISTIO_DIR/samples/addons/jaeger.yaml
kubectl apply -f $ISTIO_DIR/samples/addons/kiali.yaml

# Wait for Istio components to be ready
echo "Waiting for Istio components to be ready..."
kubectl wait --for=condition=ready pod --all -n istio-system --timeout=300s

# Install Knative Istio controller
echo "Installing Knative Istio controller..."
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.18.0/net-istio.yaml

# Install Certificate Manager
echo "Installing Certificate Manager..."
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml

# Wait for Cert Manager to be ready before installing KServe
echo "Waiting for Cert Manager to be ready..."
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=300s
kubectl wait --for=condition=ready pod -l app=webhook -n cert-manager --timeout=300s

# Install KServe main components
echo "Installing KServe core components..."
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.15.0/kserve.yaml

# Wait for KServe controller and webhook to be ready
echo "Waiting for KServe controller to be ready..."
kubectl wait --for=condition=ready pod -l control-plane=kserve-controller-manager -n kserve --timeout=300s

# Apply cluster resources with retry logic for webhook availability
echo "Installing KServe cluster resources with retry logic..."
MAX_RETRIES=5
for i in $(seq 1 $MAX_RETRIES); do
  echo "Attempting to install KServe cluster resources (attempt $i of $MAX_RETRIES)..."
  kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.15.0/kserve-cluster-resources.yaml && break
  echo "Retrying in 10 seconds..."
  sleep 10
done

# Create a secret for Azure Blob storage (for KServe to access MLflow artifacts)
echo "Creating Azure Blob storage secret for KServe..."
kubectl create secret generic azure-secret -n kserve \
  --from-literal=AZURE_STORAGE_ACCESS_KEY="$AZURE_STORAGE_ACCESS_KEY" \
  --save-config=false

# 3. Add annotation separately
kubectl annotate secret azure-secret -n kserve \
  serving.kserve.io/azure-storage-account-name="$STORAGE_ACCOUNT_NAME"

# Create a ServiceAccount for KServe to access Azure Blob
echo "Creating ServiceAccount for KServe with Azure access..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kserve-sa
  namespace: $KSERVE_NAMESPACE
secrets:
- name: azure-secret
EOF

echo "KServe installation complete!"
echo "You can now deploy ML models from MLflow to KServe using:"
echo "wasbs://$CONTAINER_NAME@$STORAGE_ACCOUNT_NAME.blob.core.windows.net/<experiment-id>/<run-id>/artifacts/model"



export ACR_NAME="tharindumlflowacr$(openssl rand -hex 3)"
export RESOURCE_GROUP="tharindu-mlflow-rg"
export LOCATION="centralindia"

# Create ACR
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --location $LOCATION

# Enable admin user (for pulling images)
az acr update --name $ACR_NAME --admin-enabled true

export ACR_NAME=tharindumlflowacr2f76fa

# Get ACR login server
export ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
echo "ACR Login Server: $ACR_LOGIN_SERVER"


# Get ACR credentials
export ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
export ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query 'passwords[0].value' --output tsv)

# Verify credentials
echo "ACR Username: $ACR_USERNAME"
echo "ACR Password: $ACR_PASSWORD"

# If AKS is AMD64, build for AMD64:
docker build --platform linux/amd64 -t $ACR_LOGIN_SERVER/mlflow-xgboost-serving:v1.0.0 -f model_serving/Dockerfile.simple .












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