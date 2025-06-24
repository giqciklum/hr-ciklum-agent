# Dockerfile Definitivo y Robusto

# 1. Usar una imagen base de Python oficial.
FROM python:3.11-slim

# 2. Instalar dependencias del sistema operativo (necesarias para pdf2image).
# Es más eficiente instalarlo en una sola capa RUN.
RUN apt-get update && apt-get install -y poppler-utils --no-install-recommends && rm -rf /var/lib/apt/lists/*

# 3. Establecer el directorio de trabajo.
WORKDIR /app

# 4. Copiar PRIMERO el fichero de requisitos.
# Esto aprovecha el caché de Docker. Si requirements.txt no cambia, no se volverán a instalar.
COPY requirements.txt .

# 5. Instalar las dependencias de Python.
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar TODO el resto del código fuente y los documentos.
# Esto incluye app.py, build_index.py y la carpeta 'docs' descargada en Cloud Build.
COPY . .

# 7. AHORA, EJECUTAR EL SCRIPT PARA CREAR LA BASE DE DATOS DENTRO DEL CONTENEDOR.
# La carpeta 'chroma_db' se creará aquí, dentro de la imagen.
RUN python build_index.py

# 8. Configurar el entorno y el comando de inicio de la aplicación.
ENV PYTHONUNBUFFERED=1
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "app:app"]