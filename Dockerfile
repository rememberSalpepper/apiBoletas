# Usa una imagen base de Python ligera
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requerimientos y lo instala
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia el resto de la aplicación (incluyendo main.py)
COPY . .

# Expone el puerto 8080 (Cloud Run espera la aplicación en este puerto)
ENV PORT 8080

# Comando para iniciar Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
