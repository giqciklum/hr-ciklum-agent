# Dockerfile Definitivo y Robusto

# 1. Usar una imagen base de Python oficial
FROM python:3.11-slim

# 2. Instalar dependencias del sistema operativo (para pdf2image)
RUN apt-get update && apt-get install -y poppler-utils --no-install-recommends

# 3. Establecer el directorio de trabajo
WORKDIR /app

# 4. Copiar los ficheros de requisitos y del código fuente
COPY requirements.txt .
COPY . .

# 5. Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. AHORA, EJECUTAR EL SCRIPT PARA CREAR LA BASE DE DATOS DENTRO DEL CONTENEDOR
RUN python build_index.py

# 7. Configurar el entorno y el comando de inicio
ENV PYTHONUNBUFFERED=1
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "app:app"]