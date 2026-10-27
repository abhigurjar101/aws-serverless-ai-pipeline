# Serverless document-processing prototype

An AWS SAM architecture exercise with S3-triggered ingestion, a DynamoDB metadata audit, and an HTTP query handler. The offline demonstration chunks sample text, computes deterministic hash vectors, ranks supplied passages by token overlap, and returns excerpts with source names.

## Run without AWS

```bash
git clone https://github.com/abhigurjar101/aws-serverless-ai-pipeline.git
cd aws-serverless-ai-pipeline
python3 client/test_pipeline.py
python3 -m unittest discover -s tests -v
```

Local mode is the default, even when boto3 is installed. It constructs no AWS clients. The demonstration runner prints examples; the separate test suite makes assertions. Hash vectors are test fixtures, not semantic embeddings. Recorded local elapsed time does not measure deployed AWS or model inference latency.

## Architecture and implementation boundary

```text
S3 object -> ingest Lambda -> text chunks + optional Titan call
                         -> DynamoDB metadata audit
HTTP POST /query -> caller-supplied documents -> lexical ranking
                                             -> excerpts or optional Bedrock generation
```

The audit table stores metadata, not a searchable vector index. Query does not retrieve previously uploaded S3 documents. The handlers expect UTF-8 text, not PDFs. There is no end-to-end persistent RAG index, authentication, scale test, or deployment evidence in this repository.

## Optional AWS experiment

AWS mode requires boto3, configured credentials, region-specific Bedrock model access and `USE_REAL_AWS_BEDROCK=true` in the Lambda environment. Build with AWS SAM and inspect the generated CloudFormation before deploying. `template.yaml` defines the bucket, metadata table, two functions, and HTTP API. `requirements.txt` is at the repository root; package boto3 with the Lambda code or use the runtime SDK intentionally. The template does not enable live Bedrock calls by default.

Review IAM scope, API authentication, storage, model permissions and retention before using real documents. AWS storage, requests, logs, and model calls can incur charges; this project does not claim zero idle cost. Remote failures propagate rather than silently reporting a successful synthetic result.

## Request example

```json
{"question":"When are invoices due?","documents":[{"source":"sample.txt","text":"Invoices are due after seven days."}]}
```

Malformed JSON and invalid request shapes return 400. No-overlap queries return no evidence. Local answers quote supplied passages; they are not generated legal conclusions. Remaining work includes persistent retrieval, ingestion idempotency, authorization, prompt-injection evaluation, deployment validation and benchmarked retrieval quality.

## Structure

- `lambdas/`: ingest and query handlers
- `template.yaml`: prototype infrastructure
- `samples/`: synthetic contract text
- `client/test_pipeline.py`: offline demonstration
- `tests/`: chunk bounds, network-free local mode, evidence grounding and validation

[Abhi Kumar Gurjar](https://www.linkedin.com/in/abhigurjar)
