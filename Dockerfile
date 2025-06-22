# --- Etapa 1: El Constructor (Builder) ---
# Aquí instalamos todo lo necesario para CONSTRUIR el índice.
FROM python:3.11-slim AS builder

# Instalar dependencias del sistema y de Python
WORKDIR /app
RUN apt-get update && apt-get install -y poppler-utils --no-install-recommends && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar los documentos y el script para construir el índice
COPY docs ./docs
COPY build_index.py .

# ¡Ejecutar la construcción del índice! Esto crea /app/chroma_db
RUN python build_index.py

# --- Etapa 2: La Imagen Final (Final Image) ---
# Esta será la imagen que se despliegue. Es mucho más ligera.
FROM python:3.11-slim

WORKDIR /app

# Instalar SOLO las dependencias de Python necesarias para EJECUTAR la app.
# Las copiamos desde la etapa 'builder' para no tener que volver a descargarlas.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copiar el código de la aplicación y LA BASE DE DATOS YA GENERADA desde la etapa 'builder'.
COPY --from=builder /app/app.py .
COPY --from=builder /app/chroma_db ./chroma_db

# PASO DE VERIFICACIÓN (¡EXCELENTE IDEA DE LA OTRA IA!)
# La build fallará aquí si 'chroma_db' no se copió correctamente.
RUN test -d /app/chroma_db && echo "✅ La base de datos 'chroma_db' se ha incluido correctamente en la imagen final." || \
    (echo "❌ ¡ERROR CRÍTICO! 'chroma_db' no se encuentra en /app/. La build fallará." && exit 1)

# Configurar el entorno y el comando de inicio de la aplicación.
ENV PYTHONUNBUFFERED=1
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "app:app"]