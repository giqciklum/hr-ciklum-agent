# Dockerfile Final (v6 - Gemini + HuggingFace Embeddings)
FROM python:3.11-slim-bookworm

# Install system dependencies + Cloud SDK
RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg poppler-utils python3-crcmod; \
  install -d -m 0755 /etc/apt/keyrings; \
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /etc/apt/keyrings/cloud.google.gpg; \
  echo "deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list; \
  apt-get update; \
  apt-get install -y --no-install-recommends google-cloud-cli; \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch FIRST (avoids downloading 4GB GPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Then install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + entrypoint
COPY . .
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
