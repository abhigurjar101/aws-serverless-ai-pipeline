"""
AWS Lambda: Ingest Processor
Triggered by Amazon S3 ObjectCreated events.
Extracts document text, performs semantic chunking, generates embeddings
via Amazon Bedrock, and indexes document records into Amazon DynamoDB.
"""

import json
import os
import urllib.parse
import hashlib
import time
from typing import List, Dict, Any

# Optional AWS SDK imports (gracefully fall back for local test harness)
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


DYNAMODB_TABLE = os.environ.get("DOCUMENT_TABLE_NAME", "EnterpriseDocumentAudit")
EMBEDDING_MODEL_ID = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v1")


def generate_local_embedding(text: str, dimensions: int = 256) -> List[float]:
    """
    Deterministic pseudo-embedding for local testing when AWS Bedrock credentials
    are not present, preserving semantic vector length.
    """
    hasher = hashlib.sha256(text.encode("utf-8")).hexdigest()
    raw_vals = [int(hasher[i:i+2], 16) / 255.0 for i in range(0, min(len(hasher), dimensions * 2), 2)]
    # Repeat or pad to reach dimensions
    vector = (raw_vals * (dimensions // len(raw_vals) + 1))[:dimensions]
    # L2 normalize
    norm = sum(x ** 2 for x in vector) ** 0.5 or 1.0
    return [round(x / norm, 4) for x in vector]


def get_bedrock_embedding(bedrock_client, text: str) -> List[float]:
    """Generates embedding vector via AWS Bedrock Titan Embeddings with fast local fallback."""
    if bedrock_client and os.environ.get("USE_REAL_AWS_BEDROCK", "false").lower() == "true":
        try:
            body = json.dumps({"inputText": text})
            response = bedrock_client.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body
            )
            response_body = json.loads(response["body"].read())
            return response_body["embedding"]
        except Exception as e:
            raise RuntimeError("Bedrock embedding failed") from e
    
    return generate_local_embedding(text)


def semantic_chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Breaks raw document text into readable, context-preserving paragraph chunks
    with sliding window overlap so thoughts aren't cut in half.
    """
    if (isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0
            or isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < chunk_size):
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    words = text.split()
    chunks = []
    for start in range(0, len(words), chunk_size - overlap):
        part = words[start:start + chunk_size]
        combined = " ".join(part)
        chunks.append({"chunk_id": len(chunks), "text": combined,
                       "word_count": len(part), "char_length": len(combined)})
        if start + chunk_size >= len(words):
            break
    return chunks



def lambda_handler(event, context):
    """
    Main AWS Lambda entrypoint for S3 PutObject events.
    """
    print(f"Received event: {json.dumps(event)}")
    
    # Initialize AWS clients if available
    use_aws = os.environ.get("USE_REAL_AWS_BEDROCK", "false").lower() == "true"
    if use_aws and not HAS_BOTO3:
        raise RuntimeError("Install boto3 before enabling AWS mode")
    s3_client = boto3.client("s3") if use_aws else None
    bedrock_client = boto3.client("bedrock-runtime") if use_aws else None
    dynamodb = boto3.resource("dynamodb") if use_aws else None

    results = []

    for record in event.get("Records", []):
        bucket = record.get("s3", {}).get("bucket", {}).get("name", "local-demo-bucket")
        key = urllib.parse.unquote_plus(
            record.get("s3", {}).get("object", {}).get("key", "sample.txt")
        )
        file_size = record.get("s3", {}).get("object", {}).get("size", 0)

        print(f"📥 Processing document: s3://{bucket}/{key} ({file_size} bytes)")

        # Read object contents
        if s3_client:
            try:
                response = s3_client.get_object(Bucket=bucket, Key=key)
                content = response["Body"].read().decode("utf-8", errors="ignore")
            except Exception as e:
                raise RuntimeError(f"Unable to read S3 object: {key}") from e
        else:
            content = record.get("mock_content", f"Enterprise operational document content for {key}.")

        # Chunk document
        chunks = semantic_chunk_text(content)
        print(f"✂️ Created {len(chunks)} semantic chunks for {key}")

        processed_chunks = []
        for c in chunks:
            vector = get_bedrock_embedding(bedrock_client, c["text"])
            c["vector_dimension"] = len(vector)
            c["sample_embedding"] = vector[:5]  # show first 5 dims for debug
            processed_chunks.append(c)

        doc_record = {
            "document_id": hashlib.md5(f"{bucket}/{key}".encode()).hexdigest(),
            "s3_uri": f"s3://{bucket}/{key}",
            "filename": os.path.basename(key),
            "processed_at": int(time.time()),
            "chunk_count": len(chunks),
            "status": "AUDIT_RECORDED" if use_aws else "LOCAL_DEMO",
            "chunks": processed_chunks
        }

        # Write metadata audit record to DynamoDB if client exists
        if dynamodb:
            try:
                table = dynamodb.Table(DYNAMODB_TABLE)
                table.put_item(Item={
                    "doc_id": doc_record["document_id"],
                    "s3_uri": doc_record["s3_uri"],
                    "filename": doc_record["filename"],
                    "processed_at": doc_record["processed_at"],
                    "chunk_count": doc_record["chunk_count"],
                    "status": "AUDIT_RECORDED"
                })
            except Exception as e:
                raise RuntimeError("DynamoDB audit write failed") from e

        results.append(doc_record)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": f"Successfully processed {len(results)} document(s)",
            "documents": results
        })
    }
