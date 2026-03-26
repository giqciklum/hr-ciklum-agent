#!/bin/sh
# entrypoint.sh (v2 - ChromaDB baked into Docker image)

set -e

echo ">> Entrypoint: Iniciando la aplicación Gunicorn..."
echo ">> Entrypoint: ChromaDB pre-cargada en /app/${PERSIST_DIRECTORY:-chroma_db_v2}"
exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 120 app:app
