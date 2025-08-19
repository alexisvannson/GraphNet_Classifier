# GraphNet Classifier

A Graph Neural Network (GNN) based image classifier that converts images to pixel graphs and uses message passing for classification.

## Features

- Converts images to pixel graphs with 4-neighborhood connectivity
- Graph Neural Network with message passing layers
- Support for both MLP and GNN-based classification
- Docker setup for reproducible environment

## Quick Start with Docker

### 1. Build and run with Docker Compose (Recommended)

```bash
# Build the container
docker-compose up -d

# Access the container for development
docker-compose exec graphnet-classifier bash

# Inside the container, run training
python main.py
```

### 2. Build and run with Docker directly

```bash
# Build the image
docker build -t graphnet-classifier .

# Run the container
docker run -it --rm \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/predictions:/app/predictions \
  -v $(pwd)/weights:/app/weights \
  graphnet-classifier
```

## Dataset Structure

Place your images in the following structure:
```
dataset/
├── chihuahua/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── muffin/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Usage

### Training

```python
# Train GNN model (default)
python main.py

# Train MLP model
python -c "from main import train_MLP; train_MLP(epochs=10)"
```

### Model Architecture

- **GraphNet**: Converts images to pixel graphs and processes them with message passing
- **MLP**: Traditional multi-layer perceptron for comparison
- **CombinedModel**: Wraps GraphNet with a classifier head

## Environment Setup (Local)

If you prefer to run locally instead of Docker:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyG extensions (CPU version)
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.1+cpu.html
pip install torch-geometric
```

## Project Structure

```
GraphNet_Classifier/
├── models/
│   ├── GNN.py          # Graph Neural Network implementation
│   └── MLP.py          # Multi-layer perceptron
├── utils/
│   ├── dataset.py      # Dataset loading utilities
│   ├── image_to_graph.py      # Image to graph conversion
│   └── image_to_graph_knn.py  # KNN-based graph construction
├── dataset/            # Training data
├── predictions/        # Output predictions
├── weights/           # Saved model weights
├── main.py           # Main training script
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker configuration
└── docker-compose.yml # Docker Compose setup
```

## Troubleshooting

### PyG Import Issues
If you encounter PyG import errors, ensure you're using compatible versions:
- PyTorch 2.2.1
- Python 3.10
- PyG extensions from the official wheel index

### Docker Issues
- Ensure Docker and Docker Compose are installed
- Check that ports are not already in use
- Verify dataset directory structure

## License

This project is for educational and research purposes.
