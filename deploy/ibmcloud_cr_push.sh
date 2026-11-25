#!/usr/bin/env bash
# =============================================================================
# InvestIQ — IBM Cloud Container Registry push script
# Usage: bash deploy/ibmcloud_cr_push.sh
# Requires: ibmcloud CLI, docker
# =============================================================================

set -euo pipefail

# ── Config (edit these) ───────────────────────────────────────────────────────
REGION="us-south"
REGISTRY="us.icr.io"
NAMESPACE="investiq"
IMAGE_NAME="investiq-api"
TAG="latest"
RESOURCE_GROUP="default"

# ── Full image path ───────────────────────────────────────────────────────────
FULL_IMAGE="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${TAG}"

echo "=== InvestIQ: Building & pushing to IBM Cloud Container Registry ==="
echo "Image: ${FULL_IMAGE}"

# 1. Login
ibmcloud login -r "${REGION}" -g "${RESOURCE_GROUP}" --no-region
ibmcloud cr login

# 2. Create namespace (idempotent)
ibmcloud cr namespace-add "${NAMESPACE}" || true

# 3. Build image from investiq/ directory
cd "$(dirname "$0")/.."
docker build -t "${FULL_IMAGE}" .

# 4. Push
docker push "${FULL_IMAGE}"

echo "=== Push complete: ${FULL_IMAGE} ==="
