# === Fase 1: El Constructor ===
# En esta fase instalamos todo lo necesario para CONSTRUIR el índice.
# Usamos una imagen completa de Python.
FROM python:3.11-slim as builder

# Establecemos el directorio de trabajo
WORKDIR /app

# Actualizamos e instalamos Tesseract para el OCR de imágenes (si es necesario)
# Si no usas OCR en imágenes, puedes comentar estas líneas para acelerar la construcción.
# RUN apt-get update && apt-get install -y tesseract-ocr

# Copiamos primero el fichero de requisitos para aprovechar el cache de Docker
COPY requirements.txt .

# Instalamos todas las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el código fuente de la aplicación
COPY . .

# --- ¡EL PASO CLAVE! ---
# Ejecutamos el script que construye la base de datos vectorial.
# Esta creará la carpeta 'chroma_db_v2' dentro de esta fase 'builder'.
RUN python build_index.py


# === Fase 2: El Final ===
# Esta es la imagen final, limpia y ligera, que se desplegará en Cloud Run.
FROM python:3.11-slim

WORKDIR /app

# Copiamos solo las dependencias de producción desde la fase 'builder'
# Esto hace la imagen final mucho más pequeña.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copiamos el código fuente de la aplicación desde la fase 'builder'
COPY --from=builder /app/app.py .
COPY --from=builder /app/.env .

# --- ¡LA SOLUCIÓN! ---
# Copiamos la base de datos vectorial YA CREADA desde la fase 'builder' a la imagen final.
# Ahora el "cerebro" está en la caja.
COPY --from=builder /app/chroma_db_v2 ./chroma_db_v2

# Exponemos el puerto que usará Gunicorn
EXPOSE 8080

# El comando que ejecutará Cloud Run para iniciar la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
