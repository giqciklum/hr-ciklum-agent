# Dockerfile Corregido y Optimizado

# 1. Usar una imagen base de Python oficial
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo
WORKDIR /app

# 3. Copiar PRIMERO el fichero de requisitos
COPY requirements.txt .

# 4. Instalar las dependencias. Esta capa se guardará en caché.
RUN pip install --no-cache-dir -r requirements.txt

# 5. AHORA, copiar el resto de la aplicación, incluyendo el `chroma_db` pre-construido
COPY . .

# 6. Configurar el entorno y el comando de inicio
ENV PYTHONUNBUFFERED=1
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "app:app"]