# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies with longer timeout
RUN pip install --no-cache-dir --default-timeout=120 -r requirements.txt

# Copy project files
COPY CNN_pytorch.py .
COPY CNN_tensorflow.py .
COPY menu.sh .

# Make menu.sh executable
RUN chmod +x menu.sh

# Default command
CMD ["/bin/bash"]
