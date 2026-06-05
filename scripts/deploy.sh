#!/bin/bash
# ============================================================
# Census Demo - One-Command Deploy Script
# ============================================================
# Usage: ./deploy.sh <BUCKET_NAME> <ACCOUNT_ID> [REGION] [MODEL_ID] [INSTANCE_TYPE]
#
# Example:
#   ./deploy.sh my-demo-bucket 123456789012
#   ./deploy.sh my-demo-bucket 123456789012 us-gov-west-1
# ============================================================

set -e

# Parameters
BUCKET_NAME=${1:?"ERROR: Provide S3 bucket name as first argument"}
ACCOUNT_ID=${2:?"ERROR: Provide AWS Account ID as second argument"}
REGION=${3:-"us-east-1"}
MODEL_ID=${4:-"anthropic.claude-sonnet-4-20250514-v1:0"}
INSTANCE_TYPE=${5:-"ml.m5.xlarge"}
S3_PREFIX="census-demo/"
STACK_NAME="census-demo-workshop"

echo "============================================================"
echo "  CENSUS DEMO WORKSHOP - DEPLOYMENT"
echo "============================================================"
echo "  Bucket:        ${BUCKET_NAME}"
echo "  Account:       ${ACCOUNT_ID}"
echo "  Region:        ${REGION}"
echo "  Model:         ${MODEL_ID}"
echo "  Instance Type: ${INSTANCE_TYPE}"
echo "  Stack Name:    ${STACK_NAME}"
echo "============================================================"
echo ""

# Step 1: Deploy CloudFormation
echo "[Step 1/3] Deploying CloudFormation stack..."
aws cloudformation deploy \
  --template-file ../cfn/template.yaml \
  --stack-name ${STACK_NAME} \
  --capabilities CAPABILITY_NAMED_IAM \
  --region ${REGION} \
  --parameter-overrides \
    AWSAccountId=${ACCOUNT_ID} \
    AWSRegion=${REGION} \
    BedrockModelId=${MODEL_ID} \
    ExistingS3BucketName=${BUCKET_NAME} \
    S3Prefix=${S3_PREFIX} \
    SageMakerInstanceType=${INSTANCE_TYPE}

echo "  ✅ CloudFormation deployed!"
echo ""

# Step 2: Pull Census Data
echo "[Step 2/3] Pulling Census data via Lambda..."
aws lambda invoke \
  --function-name census-demo-data-pull \
  --region ${REGION} \
  --payload '{}' \
  /tmp/census-data-pull-response.json

echo "  Lambda response:"
cat /tmp/census-data-pull-response.json
echo ""
echo "  ✅ Census data pulled to S3!"
echo ""

# Step 3: Upload batch inference input
echo "[Step 3/3] Uploading batch inference input file..."
aws s3 cp ../bedrock/batch_inference_input.jsonl \
  s3://${BUCKET_NAME}/${S3_PREFIX}batch/input.jsonl \
  --region ${REGION}

echo "  ✅ Batch inference input uploaded!"
echo ""

# Done
echo "============================================================"
echo "  ✅ DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "  1. Demo A: Create Bedrock Knowledge Base (see docs/workshop.md)"
echo "  2. Demo B: Run sagemaker/forecast_training.py"
echo "  3. Demo C: Create Bedrock Agent (see docs/workshop.md)"
echo ""
echo "  Data location: s3://${BUCKET_NAME}/${S3_PREFIX}"
echo "  CFN Stack:     ${STACK_NAME}"
echo "============================================================"
