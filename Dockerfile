# Dockerfile Final (v5 - keyrings + base estable)
FROM python:3.11-slim-bookworm

# Instalar dependencias del sistema (sin apt-key) + Cloud SDK
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

# Directorio de trabajo
WORKDIR /app

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código + entrypoint
COPY . .
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
