# Dockerfile Final (Stateless & Production Ready)
FROM python:3.11-slim

# Instalar dependencias del sistema y gsutil
RUN apt-get update && apt-get install -y poppler-utils python3-crcmod --no-install-recommends && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hacer ejecutable el script de inicio
RUN chmod +x /app/entrypoint.sh

# Usar el script de inicio para lanzar la aplicación
ENTRYPOINT ["/app/entrypoint.sh"]