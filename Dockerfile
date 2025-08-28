FROM python:3.9-slim

WORKDIR /app

# Install system dependencies first (required for TensorFlow)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories
RUN mkdir -p models

EXPOSE 8000

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]