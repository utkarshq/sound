FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY docker/model-hub/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/config /app/scripts

# Copy scripts
COPY docker/model-hub/scripts/ /app/scripts/

CMD ["python", "/app/scripts/model_manager.py"]
