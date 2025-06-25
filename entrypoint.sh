#!/bin/sh
# entrypoint.sh

# Salir inmediatamente si un comando falla para una depuración más fácil.
set -e

# Leer variables de entorno configuradas por Cloud Build.
GCS_BUCKET=$GCS_BUCKET_NAME
DB_FOLDER=$PERSIST_DIRECTORY
DB_PATH_IN_GCS="gs://${GCS_BUCKET}/${DB_FOLDER}"
LOCAL_DB_PATH="/app/${DB_FOLDER}"

echo ">> Entrypoint: Iniciando script de arranque..."

# Validar que las variables de entorno necesarias están presentes.
if [ -z "$GCS_BUCKET" ] || [ -z "$DB_FOLDER" ]; then
    echo ">> Entrypoint ERROR: Las variables de entorno GCS_BUCKET_NAME o PERSIST_DIRECTORY no están configuradas."
    exit 1
fi

# Descargar la base de datos desde GCS al sistema de ficheros del contenedor.
echo ">> Entrypoint: Descargando la base de datos desde ${DB_PATH_IN_GCS} a ${LOCAL_DB_PATH}..."
gsutil -m cp -r "${DB_PATH_IN_GCS}" /app/
echo ">> Entrypoint: Descarga de la base de datos completada."

# Ejecutar el comando principal de la aplicación.
# 'exec' reemplaza el proceso del shell con el de gunicorn, lo cual es una buena práctica.
echo ">> Entrypoint: Iniciando la aplicación Gunicorn..."
exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 120 app:app