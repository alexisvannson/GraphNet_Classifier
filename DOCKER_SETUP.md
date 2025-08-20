# Docker Setup Guide

## Installing Docker on macOS

### Option 1: Docker Desktop (Recommended)
1. Visit https://www.docker.com/products/docker-desktop
2. Download Docker Desktop for Mac
3. Install and follow the setup wizard
4. Start Docker Desktop from Applications

### Option 2: Homebrew
```bash
brew install --cask docker
```

### Option 3: Command Line Tools
```bash
# Install Docker CLI
brew install docker

# Install Docker Compose
brew install docker-compose
```

## Verifying Installation
```bash
docker --version
docker-compose --version
```

## Using Docker with GraphNet Classifier

### Build the Image
```bash
docker build -t graphnet-classifier .
```

> Apple Silicon (M1/M2) note: The Dockerfile uses CPU-only PyTorch and prebuilt PyG wheels compatible with amd64 and arm64 via pip. No extra flags are required.

### Run with Docker Compose (Recommended)
```bash
# Start the container and run training (default command)
docker-compose up --build

# View logs if needed
docker-compose logs -f graphnet-classifier
```

### Run tests instead of training
Edit `docker-compose.yml` and set the command to run tests:
```yaml
command: python test_setup.py
```
Then:
```bash
docker-compose up --build
```

### Run Directly with Docker
```bash
docker run -it --rm \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/predictions:/app/predictions \
  -v $(pwd)/weights:/app/weights \
  graphnet-classifier
```

## Troubleshooting Docker Issues

### PyG Compatibility Issues
If you encounter PyG import errors in Docker:
1. The Dockerfile uses PyTorch 2.1.0 for better compatibility
2. PyG extensions are installed from official wheel index
3. The app requirements avoid reinstalling torch/PyG, preventing version conflicts

### Permission Issues
```bash
# If you get permission errors
sudo chown -R $USER:$USER dataset predictions weights
```

### Clean Build
```bash
# Force rebuild without cache
docker build --no-cache -t graphnet-classifier .
```

## Development Workflow

### Interactive Development
```bash
# Start container in background, dropping into a shell
docker-compose run --rm graphnet-classifier bash

# Inside the container, you can run
python test_setup.py
python main.py
```

### Production Deployment
```bash
# Build optimized image
docker build -t graphnet-classifier:prod .

# Run with specific command
docker run -d \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/predictions:/app/predictions \
  -v $(pwd)/weights:/app/weights \
  graphnet-classifier:prod python main.py
```

## Environment Variables

The Docker container uses these environment variables:
- `PYTHONPATH=/app`: Sets Python path
- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered

## Volume Mounts

The Docker setup mounts these directories:
- `./dataset` → `/app/dataset`: Training data
- `./predictions` → `/app/predictions`: Output predictions
- `./weights` → `/app/weights`: Saved model weights

## Benefits of Docker

1. **Reproducible Environment**: Same setup across different machines
2. **Isolation**: No conflicts with system Python packages
3. **Easy Deployment**: Package everything in one container
4. **Version Control**: Pin exact versions of all dependencies
5. **Clean Testing**: Fresh environment for each test run 