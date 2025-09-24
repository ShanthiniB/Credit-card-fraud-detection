# ==============================
# Dockerfile for Fraud Detection App
# ==============================

# Use official Python 3.13 image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (for pandas, scikit-learn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project into the container
COPY . .

# Expose port (Render expects 10000, but we'll map Flask/Gunicorn to $PORT)
EXPOSE 5000

# Start the app with Gunicorn
# On Render, Render injects PORT as an env var, so we use it
CMD exec gunicorn --bind 0.0.0.0:$PORT app:app
