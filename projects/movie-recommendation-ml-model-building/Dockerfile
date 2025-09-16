# Dockerfile for the production API service
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Optional: system libs that can help with numerical packages on some platforms
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY src ./src
COPY app ./app
COPY artifacts ./artifacts
COPY data ./data

EXPOSE 8000

# Run API with Uvicorn (single worker; scale with replicas at orchestrator)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
