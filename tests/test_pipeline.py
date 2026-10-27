import json
import os
import unittest
from unittest.mock import patch
from lambdas import ingest_handler as ingest, query_handler as query

class PipelineTests(unittest.TestCase):
    def test_chunks_obey_word_bound_and_overlap(self):
        result = ingest.semantic_chunk_text("one two three four five six seven", 4, 2)
        self.assertEqual([c["text"] for c in result], ["one two three four", "three four five six", "five six seven"])
        self.assertEqual(ingest.semantic_chunk_text(""), [])
        with self.assertRaises(ValueError):
            ingest.semantic_chunk_text("text", 4, 4)

    @patch.dict(os.environ, {"USE_REAL_AWS_BEDROCK": "false"})
    def test_local_mode_never_constructs_sdk_clients(self):
        with patch.object(ingest, "HAS_BOTO3", True), patch.object(ingest, "boto3", create=True) as sdk:
            result = ingest.lambda_handler({"Records": [{"mock_content": "Orchids need humidity."}]}, None)
            self.assertEqual(json.loads(result["body"])["documents"][0]["status"], "LOCAL_DEMO")
            sdk.client.assert_not_called()
            sdk.resource.assert_not_called()

    @patch.dict(os.environ, {"USE_REAL_AWS_BEDROCK": "false"})
    def test_answers_come_from_supplied_evidence(self):
        documents = [{"source": "example.txt", "text": "Payment invoices are due after seven days."}]
        response = query.lambda_handler({"body": json.dumps({"question": "payment invoices", "documents": documents})}, None)
        body = json.loads(response["body"])
        self.assertIn("seven days", body["answer"])
        self.assertNotIn("1.5%", body["answer"])
        self.assertEqual(body["retrieved_sources"], ["example.txt"])
        body = json.loads(query.lambda_handler({"body": {"question": "volcanoes", "documents": documents}}, None)["body"])
        self.assertEqual(body["retrieved_sources"], [])

    def test_malformed_requests(self):
        for body in ["{", "[]", {"question": " ", "documents": []}, {"question": "hello", "documents": [1]}]:
            self.assertEqual(query.lambda_handler({"body": body}, None)["statusCode"], 400)
