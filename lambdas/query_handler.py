"""
AWS Lambda: Query Processor
Triggered by Amazon API Gateway HTTP POST /query.
Retrieves semantic context for a user question and invokes
Amazon Bedrock Foundation Models (Claude 3.5 Sonnet / Amazon Titan)
to return a grounded, citation-backed response.
"""

import json
import os
import time
from typing import Dict, Any, List

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_CHAT_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0")


def calculate_jaccard_similarity(query_tokens: set, doc_tokens: set) -> float:
    """Computes token overlap similarity for ranking."""
    if not query_tokens or not doc_tokens:
        return 0.0
    intersection = len(query_tokens.intersection(doc_tokens))
    union = len(query_tokens.union(doc_tokens))
    return round(intersection / union, 4) if union > 0 else 0.0


def retrieve_relevant_context(question: str, knowledge_base: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Ranks available document chunks against the user question.
    """
    q_words = set(w.lower().strip(".,?!;:") for w in question.split() if len(w) > 2)
    scored = []
    
    for item in knowledge_base:
        doc_words = set(w.lower().strip(".,?!;:") for w in item.get("text", "").split())
        score = calculate_jaccard_similarity(q_words, doc_words)
        scored.append((score, item))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k] if score > 0]


def invoke_bedrock_llm(bedrock_client, prompt: str, system_prompt: str, question: str = "") -> str:
    """Invokes Amazon Bedrock LLM with proper inference parameters, with smart local fallback."""
    if bedrock_client and os.environ.get("USE_REAL_AWS_BEDROCK", "false").lower() == "true":
        try:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }

            response = bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload)
            )
            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"]
        except Exception as e:
            raise RuntimeError("Bedrock generation failed") from e

    # The offline demonstration returns supplied evidence, never a canned factual answer.
    return "Local extractive demonstration; relevant source excerpts follow:\n" + prompt


def lambda_handler(event, context):
    """
    API Gateway Lambda Proxy Integration handler.
    """
    start_time = time.time()
    
    def bad_request(message):
        return {"statusCode": 400, "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": message})}

    body = event.get("body", {})
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (ValueError, TypeError):
            return bad_request("body must be valid JSON")
    if not isinstance(body, dict):
        return bad_request("body must be a JSON object")
    question = body.get("question")
    provided_context = body.get("documents")
    if not isinstance(question, str) or not question.strip():
        return bad_request("question must be non-empty text")
    if not isinstance(provided_context, list) or not all(
        isinstance(item, dict) and isinstance(item.get("text"), str)
        and isinstance(item.get("source"), str) for item in provided_context
    ):
        return bad_request("documents must be a list of objects with text and source strings")

    top_chunks = retrieve_relevant_context(question, provided_context, top_k=2)
    context_str = "\n\n".join([f"Source [{c.get('source')}]:\n{c.get('text')}" for c in top_chunks])

    system_instruction = (
        "You are an enterprise AI legal and document intelligence assistant. "
        "Answer the user's question accurately using ONLY the provided verified context. "
        "Cite the document source for every statement. If the answer is not in the text, clearly state that."
    )
    
    user_prompt = f"DOCUMENT CONTEXT:\n{context_str}\n\nUSER QUESTION:\n{question}\n\nANSWER:"

    use_aws = os.environ.get("USE_REAL_AWS_BEDROCK", "false").lower() == "true"
    if use_aws and not HAS_BOTO3:
        raise RuntimeError("Install boto3 before enabling AWS mode")
    bedrock_client = boto3.client("bedrock-runtime") if use_aws and top_chunks else None
    if not top_chunks:
        llm_answer = "No matching evidence was found in the supplied documents."
    elif not use_aws:
        llm_answer = "Local extractive demonstration:\n" + context_str
    else:
        llm_answer = invoke_bedrock_llm(bedrock_client, user_prompt, system_instruction, question=question)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    response_payload = {
        "question": question,
        "answer": llm_answer,
        "retrieved_sources": [c.get("source") for c in top_chunks],
        "latency_ms": latency_ms,
        "status": "SUCCESS" if use_aws else "LOCAL_DEMO"
    }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(response_payload)
    }
