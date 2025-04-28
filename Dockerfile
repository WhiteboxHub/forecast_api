FROM python:3.9-slim as base

# Install system dependencies required for LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
FROM base as model_training
COPY model_training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model_training/ .
COPY data/ /data/

# Build the API
FROM base as api
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ .

# Create non-root user
RUN useradd -m appuser
USER appuser

# Command to run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

