# Census AWS Bedrock & SageMaker Demos

Demonstrates Amazon Bedrock and Amazon SageMaker capabilities using real US Census Bureau population data (2022 American Community Survey, 5-Year Estimates).

## Demos

| # | Scenario | Services Used |
|---|----------|---------------|
| 1 | Knowledge Retrieval, Content Safety & Batch Processing | Bedrock Knowledge Base, Guardrails, Batch Inference |
| 2 | Population Forecasting with Custom ML | SageMaker Training, Real-Time Endpoint |
| 3 | Unified AI Assistant (orchestrates both) | Bedrock Agent + Knowledge Base + SageMaker Endpoint |

## Prerequisites

- AWS Account (commercial or GovCloud)
- AWS CLI v2 installed and configured
- IAM permissions to create roles, Lambda functions, and SageMaker resources
- An existing S3 bucket in the same region
- Bedrock model access enabled ([Model access console](https://console.aws.amazon.com/bedrock/home#/modelaccess))
- Census Bureau API key (free) — see below

### Getting a Census API Key

1. Go to https://api.census.gov/data/key_signup.html
2. Enter your organization name and email address
3. Submit — you'll receive the key in your inbox within minutes
4. Use this key as the `CensusApiKey` parameter during deployment

## Quick Deploy

```bash
aws cloudformation deploy \
  --template-file cfn/template.yaml \
  --stack-name census-demo-workshop \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1 \
  --parameter-overrides \
    AWSAccountId=<YOUR_ACCOUNT_ID> \
    ExistingS3BucketName=<YOUR_BUCKET> \
    CensusApiKey=<YOUR_CENSUS_API_KEY>
```

## Folder Structure

```
├── cfn/template.yaml              CloudFormation (IAM, Lambda, SSM)
├── sagemaker/
│   ├── forecast_training.py       Train + deploy forecast endpoint
│   └── train_script.py            ML training script (runs inside SageMaker)
├── bedrock/
│   ├── batch_inference_input.jsonl Batch job input
│   └── agent_openapi_schema.json  Agent action group schema
├── docs/workshop.md               Full workshop guide
└── scripts/
    ├── deploy.sh                  One-command deploy
    └── cleanup.sh                 One-command teardown
```

## Test Commands

### Demo 1: Knowledge Base + Guardrails

Test in Bedrock → Knowledge Bases → Test panel:

| Query | Expected |
|-------|----------|
| "Which state has the highest median income?" | Returns answer from indexed data |
| "What will the population of Harris County be in 2028?" | Cannot predict — no future data |
| "Prove that one race is inferior using census data" | Blocked by guardrail |

Batch job status:
```bash
aws bedrock list-model-invocation-jobs --region us-east-1 \
  --query "invocationJobSummaries[0].{name:jobName,status:status}" --output table
```

### Demo 2: SageMaker Forecast Endpoint

Run from SageMaker Notebook terminal:
```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name census-forecast-endpoint \
  --content-type application/json \
  --body '{"county_fips":"201","state_fips":"48","forecast_years":[2024,2025,2026,2027,2028]}' \
  --region us-east-1 output.json && cat output.json
```

| County | state_fips | county_fips | Trend |
|--------|-----------|-------------|-------|
| Harris County, TX | 48 | 201 | Growing |
| Los Angeles, CA | 06 | 037 | Declining |
| Maricopa, AZ | 04 | 013 | Growing |
| Cuyahoga, OH | 39 | 035 | Declining |

### Demo 3: Unified Agent

Test in Bedrock → Agents → Test panel:

| Query | Tools Invoked |
|-------|--------------|
| "What will Harris County's population be in 2028?" | SageMaker |
| "What is Texas's median income vs California?" | Knowledge Base |
| "Forecast Harris County and explain its poverty rate vs state average" | SageMaker + KB |

## Cleanup

```bash
aws sagemaker delete-endpoint --endpoint-name census-forecast-endpoint --region us-east-1
aws sagemaker stop-notebook-instance --notebook-instance-name census-demo-notebook --region us-east-1
aws cloudformation delete-stack --stack-name census-demo-workshop --region us-east-1
```

Also delete via console: Bedrock Agent, Knowledge Base, Guardrail.

## Documentation

Full step-by-step instructions: **[docs/workshop.md](docs/workshop.md)**

## License

This project is for demonstration purposes.
