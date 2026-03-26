FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies (no Cloud SDK needed - ChromaDB is baked into image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace embedding model into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code + chroma_db_v2 (baked into image by Cloud Build)
COPY . .

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
