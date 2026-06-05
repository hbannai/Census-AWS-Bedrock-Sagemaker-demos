#!/bin/bash
# ============================================================
# Census Demo - Cleanup Script
# ============================================================
# Deletes all resources to stop charges.
# Usage: ./cleanup.sh [REGION]
# ============================================================

set -e

REGION=${1:-"us-east-1"}
STACK_NAME="census-demo-workshop"
ENDPOINT_NAME="census-forecast-endpoint"

echo "============================================================"
echo "  CENSUS DEMO WORKSHOP - CLEANUP"
echo "============================================================"
echo ""

# Step 1: Delete SageMaker endpoint
echo "[Step 1/4] Deleting SageMaker endpoint..."
aws sagemaker delete-endpoint \
  --endpoint-name ${ENDPOINT_NAME} \
  --region ${REGION} 2>/dev/null && echo "  ✅ Endpoint deleted" || echo "  ⚠️  Endpoint not found (already deleted or not deployed)"

# Delete endpoint config
aws sagemaker delete-endpoint-config \
  --endpoint-config-name ${ENDPOINT_NAME} \
  --region ${REGION} 2>/dev/null && echo "  ✅ Endpoint config deleted" || echo "  ⚠️  Endpoint config not found"

echo ""

# Step 2: Delete Bedrock resources (manual reminder)
echo "[Step 2/4] Bedrock resources (delete manually in console):"
echo "  - Delete Bedrock Agent (if created)"
echo "  - Delete Bedrock Knowledge Base (if created)"
echo "  - Delete Bedrock Guardrail (if created)"
echo ""

# Step 3: Delete CloudFormation stack
echo "[Step 3/4] Deleting CloudFormation stack..."
aws cloudformation delete-stack \
  --stack-name ${STACK_NAME} \
  --region ${REGION}

echo "  Waiting for stack deletion..."
aws cloudformation wait stack-delete-complete \
  --stack-name ${STACK_NAME} \
  --region ${REGION} 2>/dev/null && echo "  ✅ Stack deleted" || echo "  ⚠️  Stack deletion may still be in progress"

echo ""

# Step 4: Clean up S3 data (optional)
echo "[Step 4/4] S3 data cleanup:"
echo "  The census data in your S3 bucket was NOT deleted."
echo "  To remove it manually:"
echo "    aws s3 rm s3://YOUR-BUCKET/census-demo/ --recursive"
echo ""

echo "============================================================"
echo "  ✅ CLEANUP COMPLETE"
echo "============================================================"
echo ""
echo "  Remaining manual steps:"
echo "  1. Delete Bedrock Knowledge Base in console"
echo "  2. Delete Bedrock Agent in console"
echo "  3. Delete Bedrock Guardrail in console"
echo "  4. (Optional) Delete census data from S3"
echo "============================================================"
