FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies safely
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libatlas3-base \
    libatlas-base-dev \
    liblapack-dev \
    curl \
    git \
    apt-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 5000

# Run the app
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 app:app"]
