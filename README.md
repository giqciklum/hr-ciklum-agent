# HR Spain Ciklum Bot — AI-Agentic HR Assistant

> **Ciklum AI Academy — Level 3 Engineering Capstone Project**
> **Author:** Ghazanfar Iqbal Begum (giq@ciklum.com)

## Project Overview

HR Spain Ciklum Bot is an AI-powered HR assistant deployed on Google Chat that answers employee questions about Ciklum Spain's HR policies, benefits, and procedures. It processes internal HR documents (PDFs, PPTX, DOCX), builds a semantic search index, and uses retrieval-augmented generation (RAG) to provide accurate, context-grounded responses in both Spanish and English.

**Live on Google Chat** as "HR SPAIN CIKLUM BOT" — available to all Ciklum Spain employees.

---

## Agentic AI Components

This project covers all five core components of an AI-Agentic system:

### 1. Data Preparation & Contextualization

- **Multi-format document processing:** Supports PDF, PPTX, and DOCX files
- **Vision OCR for complex layouts:** Uses Gemini 2.0 Flash vision capabilities to extract text from scanned documents, tables, and image-heavy pages that standard text extraction misses
- **Structural enrichment:** Automatically injects semantic headings into document text based on keyword detection, improving retrieval precision for topic-specific queries
- **Table extraction:** Converts document tables to Markdown format for better semantic understanding
- **Automated pipeline:** Google Drive → Cloud Function (sync) → GCS bucket → Cloud Build (indexing) → Vector DB

### 2. RAG Pipeline Design

- **Embeddings:** Local HuggingFace `all-MiniLM-L6-v2` model (CPU-only, no API dependency)
- **Vector store:** ChromaDB with 445 semantic chunks across 11 HR documents
- **Text splitting:** `RecursiveCharacterTextSplitter` (chunk_size=1500, overlap=200)
- **Multi-Query Retriever:** Generates multiple reformulations of the user's question to improve recall
- **History-Aware Retriever:** Reformulates follow-up questions using conversation history
- **Top-K retrieval:** Returns 10 most relevant chunks per query

### 3. Reasoning & Reflection

- **Chain-of-thought prompting:** The system prompt instructs the LLM to reason step-by-step internally
- **Uncertainty management:** The agent self-reflects on context quality — provides partial answers when context is poor, acknowledges limitations when empty
- **Discriminative reasoning:** Focuses only on the specific process the user asked about
- **Bilingual detection:** Automatically detects Spanish or English and responds in the same language

### 4. Tool-Calling Mechanisms

- **Vision OCR Tool:** Automatically invokes Gemini 2.0 Flash vision for complex document pages (tables, images, scans)
- **Layout detection heuristic:** `has_complex_layout()` decides between standard extraction and vision OCR
- **Google Chat integration:** Receives messages via HTTP webhook, responds with formatted cards
- **Conversation memory:** Per-user chat history (last 5 exchanges) for contextual follow-ups

### 5. Evaluation

- **Test retrieval script** (`test_retrieval.py`): Validates vector store returns relevant chunks
- **Manual QA testing:** Tested with real HR questions in Spanish and English
- **Response quality criteria:** Accuracy, completeness, formatting, language consistency

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Google Gemini 2.0 Flash | Chat responses + Vision OCR |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 | Local semantic embeddings (CPU) |
| **Vector DB** | ChromaDB | Semantic search index |
| **Framework** | LangChain | RAG pipeline orchestration |
| **Web Server** | Flask + Gunicorn | HTTP endpoint |
| **Chat Platform** | Google Chat API | User interface |
| **Cloud Runtime** | Google Cloud Run | Serverless container hosting |
| **CI/CD** | Google Cloud Build | Automated build & deploy |
| **Storage** | Google Cloud Storage | Document & DB storage |
| **Secrets** | Google Secret Manager | API key management |

---

## Architecture

See `architecture.mmd` for the full Mermaid diagram.

```
HR Documents (Google Drive)
    ↓ Cloud Function (auto-sync)
GCS Bucket (hr-bot-docs)
    ↓ Cloud Build trigger (on push to master)
build_index.py → Vision OCR + Enrichment + ChromaDB (445 chunks)
    ↓ Upload to GCS (hr-bot-db)
Cloud Run (downloads DB on startup via entrypoint.sh)
    ↓
Google Chat ←→ Flask /chat ←→ RAG Chain ←→ Gemini 2.0 Flash
```

---

## Project Structure

```
hr-ciklum-agent/
├── app.py                 # Flask app with RAG chain (Gemini + HuggingFace)
├── build_index.py         # Document processor & vector index builder (v24)
├── entrypoint.sh          # Cloud Run startup script (downloads DB from GCS)
├── Dockerfile             # Container image with pre-baked HuggingFace model
├── cloudbuild.yaml        # CI/CD pipeline (6 steps)
├── requirements.txt       # Python dependencies (pinned versions)
├── test_retrieval.py      # Retrieval evaluation tests
├── sync_function/         # Cloud Function for Google Drive → GCS sync
├── architecture.mmd       # Architecture diagram (Mermaid)
└── README.md              # This file
```

---

## How to Run

### Prerequisites

- Python 3.11+
- Google Cloud project with: Cloud Run, Cloud Build, Cloud Storage, Secret Manager, Google Chat API
- Gemini API key stored in Secret Manager as `hr-bot-gemini-key`

### Local Development

```bash
git clone https://github.com/giqciklum/hr-ciklum-agent.git
cd hr-ciklum-agent

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

export GOOGLE_API_KEY="your-gemini-api-key"
export PERSIST_DIRECTORY="chroma_db_v2"

# Place HR documents in docs/ folder
python build_index.py

python app.py
# Server runs at http://localhost:8080
```

### Cloud Deployment

Every push to `master` triggers the full CI/CD pipeline automatically:

```bash
git push origin master
```

The pipeline runs 6 steps:
1. **pull-docs** — Downloads HR documents from GCS
2. **build-vector-index** — Builds ChromaDB index with all documents
3. **upload-db-to-gcs** — Syncs vector database to GCS
4. **docker-build** — Builds container image (with HuggingFace model pre-baked)
5. **docker-push** — Pushes to Google Container Registry
6. **run-deploy** — Deploys to Cloud Run (europe-west1, 2Gi memory, CPU boost)

---

## Key Design Decisions

1. **Local embeddings over API embeddings:** Switched from OpenAI/Gemini API embeddings to local HuggingFace `all-MiniLM-L6-v2` to eliminate rate limits and remove external API dependency
2. **Pre-baked model in Docker image:** HuggingFace model downloaded during Docker build, not at runtime — prevents cold-start failures from HuggingFace rate limits
3. **CPU-only PyTorch:** Uses `torch+cpu` (~200MB) instead of full CUDA torch (~4GB), reducing container image size
4. **Structural enrichment:** Topic-aware heading injection before chunking improves retrieval accuracy
5. **Vision OCR fallback:** Complex document pages automatically processed with Gemini vision

---

## Migration History

Originally built with OpenAI (GPT-4o + text-embedding-3-large via Azure Gateway). Migrated to Google stack as part of the AI Academy capstone:

| Component | Before | After |
|-----------|--------|-------|
| **LLM** | OpenAI GPT-4o | Google Gemini 2.0 Flash |
| **Embeddings** | OpenAI text-embedding-3-large | HuggingFace all-MiniLM-L6-v2 (local) |
| **API Gateway** | Azure OpenAI Gateway | Google AI Studio (direct) |
| **Secret** | OPENAI_API_KEY | GOOGLE_API_KEY |

---

## Author

**Ghazanfar Iqbal Begum** — DevOps Engineer at Ciklum
Built as part of the Ciklum AI Academy Level 3 Engineering Capstone Project (2025)
