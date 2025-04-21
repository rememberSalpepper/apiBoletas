FROM python:3.9-slim

# Instala dependencias sistema + Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-spa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependencias Python
COPY requirements.txt .
# Usa --no-cache-dir para reducir tamaño de imagen
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia el código de la aplicación
COPY . .

# Variable de entorno para el puerto (Cloud Run la usa)
ENV PORT 8080

# Expone el puerto
EXPOSE 8080

# Comando para iniciar la app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]