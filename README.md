# GraphNet Classifier

A Graph Neural Network (GNN) and MLP image classifier (chihuahua vs muffin). This README focuses on the simplest, most reliable way to run and develop the project: with Docker.

### Why Docker?
- Guarantees a working PyTorch + PyG stack (no torch_scatter issues)
- No conflicts with your host Python
- Same environment for everyone and in CI

## Prerequisites
- Docker Desktop installed and running
- Git, terminal

## One-time setup
```bash
# Clone
git clone <repository-url>
cd GraphNet_Classifier

# Build the image (pins Torch 2.1.0 and compatible PyG wheels)
docker compose build
```

## Quick start
```bash
# Run with docker compose (default command runs main.py)
docker compose up
# Stop
docker compose down
```

What happens:
- The container runs `python main.py`
- Volumes are mounted so your data and results persist on your host:
  - `./dataset -> /app/dataset`
  - `./weights -> /app/weights`
  - `./predictions -> /app/predictions`

## Interactive development (recommended)
```bash
# Open a shell inside the container with project mounted
docker compose run --rm graphnet-classifier bash

# Inside the container you can run:
python -c "import models.GNN; print('GNN OK')"
python main.py
python utils/inference.py
```
Tips inside the container:
- The working directory is `/app` (your repo root)
- Python path is set so `import models...` works
- Your code edits on the host are immediately visible in the container

## Docker Development vs Production Configurations

This project includes two Docker configurations for different use cases:

### Development Configuration (`docker-compose.dev.yml`)
**Best for**: Active development, debugging, and experimentation

**Key Features:**
- ✅ **Live code changes**: Edit files on your host, see changes immediately in container
- ✅ **No rebuilds needed**: Code changes don't require rebuilding the Docker image
- ✅ **Faster iteration**: Perfect for debugging and testing different approaches
- ✅ **Simpler setup**: Single volume mount for the entire project

**Usage:**
```bash
# Build and run development version
docker compose -f docker-compose.dev.yml up --build

# Interactive shell with development setup
docker compose -f docker-compose.dev.yml run --rm graphnet-classifier bash
```

**How it works:**
- Uses `Dockerfile.dev` (no `COPY . .` command)
- Mounts entire project directory: `- .:/app`
- Relies entirely on volume mounting for source code

### Production Configuration (`docker-compose.yml`)
**Best for**: Deployment, final testing, and reproducible builds

**Key Features:**
- ✅ **Reproducible**: Exact code state is preserved in the image
- ✅ **Optimized**: Faster startup (no volume mounting overhead)
- ✅ **Controlled**: Explicit control over what gets mounted
- ✅ **Deployment-ready**: Self-contained image with all code

**Usage:**
```bash
# Build and run production version
docker compose up --build

# Interactive shell with production setup
docker compose run --rm graphnet-classifier bash
```

**How it works:**
- Uses `Dockerfile` (includes `COPY . .` command)
- Mounts specific directories: `./dataset`, `./predictions`, `./weights`
- Code is copied into image during build

### When to Use Which?

| Use Case | Configuration | Why |
|----------|---------------|-----|
| **Active development** | Development (`docker-compose.dev.yml`) | Live code changes, no rebuilds |
| **Debugging** | Development (`docker-compose.dev.yml`) | Immediate feedback on code changes |
| **Final testing** | Production (`docker-compose.yml`) | Reproducible environment |
| **Deployment** | Production (`docker-compose.yml`) | Self-contained, optimized |
| **CI/CD** | Production (`docker-compose.yml`) | Consistent builds |

### Real-World Example

**Scenario**: You're debugging your `main.py` file

**With Development Config:**
```bash
# 1. Start development container
docker compose -f docker-compose.dev.yml run --rm graphnet-classifier bash

# 2. Edit main.py on your host (in another terminal/editor)
# 3. Run in container - sees changes immediately!
python main.py
```

**With Production Config:**
```bash
# 1. Edit main.py on your host
# 2. Rebuild image (required for code changes)
docker compose build --no-cache
# 3. Run container - now sees changes
docker compose run --rm graphnet-classifier bash
python main.py
```

### Switching Between Configurations

**From Development to Production:**
```bash
# Stop development container
docker compose -f docker-compose.dev.yml down

# Start production container
docker compose up --build
```

**From Production to Development:**
```bash
# Stop production container
docker compose down

# Start development container
docker compose -f docker-compose.dev.yml up --build
```

### Troubleshooting Docker Configurations

**"Old version" of code running:**
- **Development config**: Make sure you're using `docker-compose.dev.yml`
- **Production config**: Rebuild after code changes: `docker compose build --no-cache`

**Import errors in container:**
- Check that your code changes are visible in the container
- For development: restart container to pick up new files
- For production: rebuild image after code changes

**Volume mounting issues:**
- Development: Uses single mount `- .:/app`
- Production: Uses specific mounts for data directories

## Common tasks

### Train GNN
- Edit `main.py` to adjust training parameters (epochs, resize_value, max_samples, etc.)
- Then run inside the container:
```bash
python main.py
```
Weights are saved under `weights/GNN` (mounted volume, persisted on host).

### Run MLP inference
```bash
python utils/inference.py
```
By default, it uses the MLP weights path configured in `utils/inference.py`.

### Rebuild after dependency changes
If you change `requirements.txt`, `Dockerfile`, or want a clean build:
```bash
docker compose build --no-cache
```

## Local (no Docker) – not recommended
You can use a local venv, but binary wheels for PyG often mismatch on macOS and cause segfaults. If you must:
```bash
python3.10 -m venv Graphnet
source Graphnet/bin/activate
pip install --upgrade pip
# Pin exact CPU wheels known to work together
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install -r requirements.txt
python -c "import models.GNN; print('GNN OK')"
```
If imports fail or segfault, prefer Docker.

## Troubleshooting

- "torch_scatter" import errors or segfaults
  - Run inside Docker. The image pins compatible versions.
  - Keep GNN imports lazy (already done in `main.py`).

- Changes not reflected in the container
  - If you changed dependencies, rebuild: `docker compose build`
  - For code-only changes, a restart is enough: `docker compose down && docker compose up`

- Permission issues with mounted folders
```bash
sudo chown -R "$USER":"$USER" dataset predictions weights
```

## Project structure (key parts)
```
GraphNet_Classifier/
├── models/
│   ├── GNN.py              # GraphNet + CombinedModel (lazy import safe)
│   └── MLP.py              # Baseline MLP
├── utils/
│   ├── image_to_graph/...  # Image → graph utilities
│   ├── train_model.py      # Generic train loop
│   └── inference.py        # Inference helpers
├── dataset/                # Mounted data (host <-> container)
├── predictions/            # Mounted outputs
├── weights/                # Mounted model weights
├── Dockerfile              # Reproducible runtime
├── docker-compose.yml      # One-command run
└── README.md               # This guide
```

## FAQ
- Do I need a virtualenv if I use Docker?
  - No. The container isolates Python for you.
- Where do my outputs go?
  - In `predictions/` and `weights/` on your host (they are mounted volumes).
- Can I run a different command?
  - Yes: `docker compose run --rm graphnet-classifier bash` and run any Python command from the shell.

---
If you’re new to Docker, the only commands you really need are:
```bash
docker compose build
docker compose up  # Ctrl-C to stop
docker compose down
```
And for an interactive shell inside the environment:
```bash
docker compose run --rm graphnet-classifier bash
```
