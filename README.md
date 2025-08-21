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
