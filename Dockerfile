# Use Python 3.12.4 slim
FROM python:3.12.4-slim

# Set working directory
WORKDIR /app

# Environment variables for Python & pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    postgresql-client \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements.txt ./

# Install python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the entire project
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose internal service port (mapped to 3004 outside)
EXPOSE 8000

# Run the AI service (adjust main:app if needed)
CMD ["python", "test_chat_service.py"]
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]