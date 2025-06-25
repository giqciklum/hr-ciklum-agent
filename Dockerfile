# Dockerfile Final (v4 - Con Herramientas de Google Cloud)
FROM python:3.11-slim

# Instalar dependencias del sistema, incluyendo las herramientas para GCS
RUN apt-get update && apt-get install -y curl gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y \
    google-cloud-cli \
    poppler-utils \
    python3-crcmod \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el fichero de requisitos
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y el entrypoint
COPY . .

# Hacer ejecutable el script de inicio
RUN chmod +x /app/entrypoint.sh

# Usar el script de inicio para lanzar la aplicación
ENTRYPOINT ["/app/entrypoint.sh"]