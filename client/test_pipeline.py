#!/usr/bin/env python3
"""
Local Pipeline Test Runner & AWS Architecture Simulator
Simulates:
  1. S3 PutObject event triggering lambdas.ingest_handler
  2. Document text extraction & semantic chunking
  3. API Gateway POST /query invoking lambdas.query_handler
"""

import sys
import os
import json

# Add root directory to sys.path to easily import lambdas
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from lambdas import ingest_handler, query_handler


def run_pipeline_simulation():
    print("=" * 80)
    print("🚀 AWS SERVERLESS AI PIPELINE: LOCAL VERIFICATION RUNNER")
    print("=" * 80)

    sample_doc_path = os.path.join(PROJECT_ROOT, "samples", "sample_contract.txt")
    with open(sample_doc_path, "r", encoding="utf-8") as f:
        contract_text = f.read()

    print(f"\n📂 Step 1: Simulating S3 ObjectCreated Event...")
    mock_s3_event = {
        "Records": [
            {
                "eventVersion": "2.1",
                "eventSource": "aws:s3",
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": "enterprise-rag-incoming-bucket"},
                    "object": {"key": "incoming/Master_Services_Agreement_2026.txt", "size": len(contract_text)}
                },
                "mock_content": contract_text
            }
        ]
    }

    ingest_res = ingest_handler.lambda_handler(mock_s3_event, context=None)
    ingest_body = json.loads(ingest_res["body"])
    doc_info = ingest_body["documents"][0]

    print(f"✅ Ingestion Successful!")
    print(f"   • Document ID: {doc_info['document_id']}")
    print(f"   • Chunks Created: {doc_info['chunk_count']}")
    print(f"   • Status: {doc_info['status']}")

    test_queries = [
        "What is the required notice period to terminate this contract?",
        "What encryption and security protocols are guaranteed?",
        "What is the late fee penalty if invoices are unpaid past 45 days?"
    ]

    print("\n" + "=" * 80)
    print("🤖 Step 2: Simulating API Gateway POST /query with AWS Bedrock RAG")
    print("=" * 80)

    # Convert chunks to retrieval format
    retrieval_corpus = [
        {"id": f"chunk_{c['chunk_id']}", "source": doc_info["filename"], "text": c["text"]}
        for c in doc_info["chunks"]
    ]

    for idx, q in enumerate(test_queries, 1):
        print(f"\n[Query #{idx}]: \"{q}\"")
        api_event = {
            "httpMethod": "POST",
            "path": "/query",
            "body": json.dumps({
                "question": q,
                "documents": retrieval_corpus
            })
        }
        query_res = query_handler.lambda_handler(api_event, context=None)
        res_data = json.loads(query_res["body"])
        
        print(f"  ⚡ Latency: {res_data['latency_ms']} ms")
        print(f"  📚 Citations: {', '.join(res_data['retrieved_sources'])}")
        print(f"  💬 Answer: {res_data['answer']}")

    print("\n" + "=" * 80)
    print("🎉 All pipeline stages executed successfully with zero cloud bill!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline_simulation()
