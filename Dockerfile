FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip

# Install PyTorch first
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyG extensions with compatible wheels
RUN pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.1+cpu.html

# Install remaining requirements
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for data and outputs
RUN mkdir -p dataset predictions weights

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "main.py"] 