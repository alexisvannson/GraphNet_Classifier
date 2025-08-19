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

# Install PyTorch 2.1.0 (more stable with PyG)
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install PyG extensions with compatible wheels for PyTorch 2.1.0
RUN pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Install torch-geometric
RUN pip install torch-geometric

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