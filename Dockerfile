# Use official Python runtime as base image
FROM python:3.9-slim

# Set maintainer information
LABEL maintainer="ML Docker Lab"
LABEL description="Machine Learning Environment with TensorFlow"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV JUPYTER_ENABLE_LAB=yes

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directories for ML workflow
RUN mkdir -p /app/notebooks /app/models /app/data /app/src

# Copy application files
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Expose ports for Jupyter and Flask
EXPOSE 8888 5000

# Default command to start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
