#!/usr/bin/env bash
# =============================================================================
# InvestIQ — IBM Cloud Code Engine deployment script
# Deploys the FastAPI backend as a Code Engine application.
# Usage: bash deploy/code_engine_deploy.sh
# Requires: ibmcloud CLI with Code Engine plugin (ibmcloud plugin install code-engine)
# =============================================================================

set -euo pipefail

# ── Config (edit these) ───────────────────────────────────────────────────────
REGION="us-south"
RESOURCE_GROUP="default"
CE_PROJECT="investiq-prod"
CE_APP_API="investiq-api"
CE_APP_DASH="investiq-dashboard"
REGISTRY="us.icr.io"
NAMESPACE="investiq"
IMAGE_API="${REGISTRY}/${NAMESPACE}/investiq-api:latest"
IMAGE_DASH="${REGISTRY}/${NAMESPACE}/investiq-dashboard:latest"

# Secrets — set these in your environment before running
: "${WATSONX_API_KEY:?Need WATSONX_API_KEY}"
: "${WATSONX_PROJECT_ID:?Need WATSONX_PROJECT_ID}"
: "${COS_API_KEY:?Need COS_API_KEY}"
: "${COS_INSTANCE_CRN:?Need COS_INSTANCE_CRN}"

echo "=== InvestIQ: Deploying to IBM Cloud Code Engine ==="

# 1. Login
ibmcloud login -r "${REGION}" -g "${RESOURCE_GROUP}"
ibmcloud ce project select --name "${CE_PROJECT}" 2>/dev/null || \
    ibmcloud ce project create --name "${CE_PROJECT}"

# 2. Create registry secret for pulling images
ibmcloud ce registry create \
    --name investiq-registry \
    --server "${REGISTRY}" \
    --username iamapikey \
    --password "${WATSONX_API_KEY}" 2>/dev/null || true

# 3. Create secret with all env vars
ibmcloud ce secret create \
    --name investiq-secrets \
    --from-literal WATSONX_API_KEY="${WATSONX_API_KEY}" \
    --from-literal WATSONX_PROJECT_ID="${WATSONX_PROJECT_ID}" \
    --from-literal COS_API_KEY="${COS_API_KEY}" \
    --from-literal COS_INSTANCE_CRN="${COS_INSTANCE_CRN}" \
    --from-literal DEMO_MODE="false" \
    --from-literal LOG_LEVEL="INFO" \
    2>/dev/null || \
ibmcloud ce secret update \
    --name investiq-secrets \
    --from-literal WATSONX_API_KEY="${WATSONX_API_KEY}" \
    --from-literal WATSONX_PROJECT_ID="${WATSONX_PROJECT_ID}" \
    --from-literal COS_API_KEY="${COS_API_KEY}" \
    --from-literal COS_INSTANCE_CRN="${COS_INSTANCE_CRN}"

# 4. Deploy FastAPI backend
echo "--- Deploying FastAPI API ---"
ibmcloud ce application create \
    --name "${CE_APP_API}" \
    --image "${IMAGE_API}" \
    --registry-secret investiq-registry \
    --env-from-secret investiq-secrets \
    --port 8000 \
    --min-scale 1 \
    --max-scale 5 \
    --cpu 1 \
    --memory 4G \
    --command "uvicorn" \
    --argument "api.main:app" \
    --argument "--host" --argument "0.0.0.0" \
    --argument "--port" --argument "8000" \
    2>/dev/null || \
ibmcloud ce application update \
    --name "${CE_APP_API}" \
    --image "${IMAGE_API}" \
    --env-from-secret investiq-secrets

# Get API URL
API_URL=$(ibmcloud ce application get --name "${CE_APP_API}" --output json | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',{}).get('url',''))")
echo "API URL: ${API_URL}"

# 5. Deploy Streamlit Dashboard
echo "--- Deploying Streamlit Dashboard ---"
ibmcloud ce application create \
    --name "${CE_APP_DASH}" \
    --image "${IMAGE_DASH}" \
    --registry-secret investiq-registry \
    --env-from-secret investiq-secrets \
    --env "API_BASE_URL=${API_URL}" \
    --port 8501 \
    --min-scale 1 \
    --max-scale 3 \
    --cpu 1 \
    --memory 2G \
    --command "streamlit" \
    --argument "run" --argument "dashboard/app.py" \
    --argument "--server.port" --argument "8501" \
    --argument "--server.address" --argument "0.0.0.0" \
    2>/dev/null || \
ibmcloud ce application update \
    --name "${CE_APP_DASH}" \
    --image "${IMAGE_DASH}" \
    --env-from-secret investiq-secrets

DASH_URL=$(ibmcloud ce application get --name "${CE_APP_DASH}" --output json | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',{}).get('url',''))")

echo ""
echo "======================================================================"
echo "✅ InvestIQ deployed successfully!"
echo "   API:       ${API_URL}"
echo "   Dashboard: ${DASH_URL}"
echo "   API Docs:  ${API_URL}/docs"
echo "======================================================================"
