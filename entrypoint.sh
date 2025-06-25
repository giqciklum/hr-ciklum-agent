#!/bin/sh
# entrypoint.sh (Corrected Version for Separate Buckets)

# Salir inmediatamente si un comando falla.
set -e

# MODIFICATION: Read the new environment variable for the database bucket.
GCS_DB_BUCKET=$GCS_DB_BUCKET_NAME
DB_FOLDER=$PERSIST_DIRECTORY

# MODIFICATION: Construct the GCS path using the dedicated database bucket.
DB_PATH_IN_GCS="gs://${GCS_DB_BUCKET}/${DB_FOLDER}"
LOCAL_DB_PATH="/app/${DB_FOLDER}"

echo ">> Entrypoint: Iniciando script de arranque..."

# MODIFICATION: Update the validation check for the new variable.
if [ -z "$GCS_DB_BUCKET" ] || [ -z "$DB_FOLDER" ]; then
    echo ">> Entrypoint ERROR: Las variables de entorno GCS_DB_BUCKET_NAME o PERSIST_DIRECTORY no están configuradas."
    exit 1
fi

# Descargar la base de datos desde el bucket de base de datos a GCS.
echo ">> Entrypoint: Descargando la base de datos desde ${DB_PATH_IN_GCS} a ${LOCAL_DB_PATH}..."
gsutil -m cp -r "${DB_PATH_IN_GCS}" /app/
echo ">> Entrypoint: Descarga de la base de datos completada."

# Ejecutar el comando principal de la aplicación.
echo ">> Entrypoint: Iniciando la aplicación Gunicorn..."
exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 120 app:app