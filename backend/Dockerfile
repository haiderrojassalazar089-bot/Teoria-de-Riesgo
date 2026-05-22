# ══════════════════════════════════════════════════════════════
# Dockerfile para RiskLab API
# ══════════════════════════════════════════════════════════════

FROM python:3.11.9-slim-bookworm

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        && \
    rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copiar código
COPY app ./app

# Exponer puerto
EXPOSE 8000

# Comando de inicio usando python -m
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
