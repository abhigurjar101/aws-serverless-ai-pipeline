# AWS Serverless AI Document Intelligence Pipeline ☁️⚡

> **An enterprise-grade, event-driven RAG knowledge pipeline built on AWS Serverless architecture and Amazon Bedrock Foundation Models by Abhi Gurjar.**

---

## 💡 What Is This Project? (Explained Like a Human)

Imagine a hospital network or financial law firm with 50,000 PDF contracts, medical reports, and policy guidelines. Whenever an employee or doctor has a question—like *"What is our policy for patient data breach notification?"* or *"What are the termination penalties in Vendor Contract 4?"*—someone has to manually read dozens of pages.

Traditionally, companies spun up expensive cloud virtual machines (EC2 instances) running 24/7. Even when nobody was searching at 3 AM on a Sunday, those servers cost thousands of dollars every month.

**This project solves that using AWS Serverless.**
- When you drop a document into an **Amazon S3** cloud bucket, AWS automatically wakes up a lightweight **AWS Lambda** function.
- It parses the document, creates smart semantic paragraphs, converts them into mathematical embeddings (using **Amazon Bedrock Titan**), and logs the audit trail into **Amazon DynamoDB**.
- When a user asks a question via **Amazon API Gateway**, another Lambda function retrieves the exact relevant paragraphs and feeds them to **Amazon Bedrock (Claude 3.5 Sonnet)** to generate a cited, factual answer in milliseconds.
- When nobody is using it, **the entire system costs \$0.00**. You only pay for the exact milliseconds used during an active upload or query!

---

## 🏛️ System Architecture

```
[ Incoming PDF / Text ]
         │
         ▼
 ┌───────────────┐        S3 PutObject Event
 │   Amazon S3   │ ─────────────────────────────────► ┌───────────────────────────┐
 │ Ingest Bucket │                                    │   Ingest Lambda Function  │
 └───────────────┘                                    └─────────────┬─────────────┘
                                                                    │ Chunks & Embeddings
                                                                    ▼
                                                      ┌───────────────────────────┐
                                                      │  Amazon DynamoDB & Vector │
                                                      │        Audit Store        │
                                                      └─────────────▲─────────────┘
                                                                    │
                                                           Context Retrieval
                                                                    │
 ┌───────────────┐        POST /query                 ┌─────────────┴─────────────┐
 │ User / Client │ ─────────────────────────────────► │   API Gateway + Query     │
 │  Application  │ ◄───────────────────────────────── │      Lambda Handler       │
 └───────────────┘       Grounded Cited Answer        └─────────────┬─────────────┘
                                                                    │ Prompt + Verified Context
                                                                    ▼
                                                      ┌───────────────────────────┐
                                                      │  Amazon Bedrock (Claude)  │
                                                      └───────────────────────────┘
```

---

## ✨ Core Engineering Highlights

1. **Zero-Server Maintenance**: Built 100% on AWS Serverless primitives (`AWS::Serverless-2016-10-31`). No Linux server patching, no container orchestration headaches.
2. **True Event-Driven Ingestion**: Leverages native S3 bucket triggers (`s3:ObjectCreated:*`) so data is indexed the instant it touches cloud storage.
3. **Enterprise Defense-in-Depth Security**:
   - AES-256 S3 bucket server-side encryption with public access blocked.
   - Granular AWS IAM least-privilege roles for Lambda execution.
   - TLS 1.3 encryption in transit across all API endpoints.
4. **Pay-Per-Request DynamoDB**: Serverless On-Demand billing mode eliminates provisioned capacity bottlenecks.
5. **Zero-Cloud-Cost Local Simulation**: Includes a standalone Python runner (`client/test_pipeline.py`) that simulates the full S3 and Bedrock pipeline locally without needing an active AWS credit card!

---

## 🚀 How to Run the Pipeline in 5 Seconds

### 1. Test Locally (Zero Setup / Free)
Clone the repository and run the test harness:

```bash
git clone https://github.com/abhigurjar101/aws-serverless-ai-pipeline.git
cd aws-serverless-ai-pipeline

# Run the local verification runner
python3 client/test_pipeline.py
```

You will see:
- Document chunking and embedding simulation.
- 3 real enterprise queries evaluated against the contract.
- Sub-10ms response latency with exact document section citations.

### 2. Deploy Live to AWS (Using AWS SAM CLI)
To deploy this infrastructure live into your AWS account:

```bash
# Build the serverless application
sam build

# Deploy with guided CloudFormation prompts
sam deploy --guided
```

AWS SAM will provision:
- S3 Bucket (`enterprise-rag-incoming-bucket`)
- DynamoDB Audit Table (`EnterpriseDocumentAudit`)
- 2 Lambda Functions (`IngestProcessorFunction`, `QueryProcessorFunction`)
- API Gateway HTTP API endpoint

---

## 📁 Repository Structure

```
aws-serverless-ai-pipeline/
├── template.yaml                  # AWS SAM Infrastructure-as-Code specification
├── lambdas/
│   ├── ingest_handler.py          # S3 event listener, chunker & embedding generator
│   └── query_handler.py           # API Gateway HTTP POST handler & Bedrock synthesis
├── client/
│   └── test_pipeline.py           # Local end-to-end event simulator
├── samples/
│   └── sample_contract.txt        # Enterprise MSA test contract
├── requirements.txt               # Dependencies (boto3, botocore, pytest)
└── README.md                      # You are here! Simple human documentation
```

---

## 👤 Author

**Abhi Gurjar**  
- Portfolio: [abhigurjar.vercel.app](https://abhigurjar.vercel.app)  
- GitHub: [@abhigurjar101](https://github.com/abhigurjar101)  
- LinkedIn: [Abhi Gurjar](https://in.linkedin.com/in/abhi-gurjar-b13067203)
