# GraphNet Classifier

A Graph Neural Network (GNN) based image classifier that converts images to pixel graphs and uses message passing for classification. This project compares traditional MLP approaches with Graph Neural Networks for image classification tasks.

## 🚀 Features

- **Graph Neural Network**: Converts images to pixel graphs with 4-neighborhood connectivity
- **MLP Comparison**: Traditional multi-layer perceptron for baseline comparison
- **Docker Setup**: Reproducible environment with all dependencies
- **Inference Pipeline**: Complete inference and comparison tools
- **Optimized Training**: Configurable dataset size and model complexity for fast experimentation

## 📊 Dataset

The project is configured for binary classification (chihuahua vs muffin) but can be extended to other datasets.

### Dataset Structure
```
dataset/
├── chihuahua/
│   ├── img_0_1071.jpg
│   ├── img_0_1074.jpg
│   └── ... (641 images)
└── muffin/
    ├── img_0_1072.jpg
    ├── img_0_1075.jpg
    └── ... (542 images)
```

**Total**: 1,183 images across 2 classes

## 🐳 Quick Start with Docker

### Prerequisites
- Docker Desktop installed and running
- Git

### 1. Clone and Setup
```bash
git clone <repository-url>
cd GraphNet_Classifier
```

### 2. Build and Run Training
```bash
# Build and start training (default: 50 samples, 20 epochs)
docker compose up --build

# View logs
docker compose logs -f graphnet-classifier

# Stop training
docker compose down
```

### 3. Run Tests First (Optional)
Edit `docker-compose.yml` and change the command to:
```yaml
command: python test_setup.py
```
Then run:
```bash
docker compose up --build
```

## 🎯 Training Configuration

### Current Optimized Settings (Fast Testing)
```python
train_GNN(epochs=20, hidden_layers=2, max_samples=50)
```
- **50 samples** (instead of 1,183)
- **64×64 resolution** (instead of 128×128)
- **3 GraphNet blocks** (instead of 10)
- **Expected time**: ~2-5 minutes

### Production Settings (Full Dataset)
```python
train_GNN(epochs=30, max_samples=None, resize_value=128, n_blocks=10)
```
- **Full dataset** (1,183 images)
- **128×128 resolution**
- **10 GraphNet blocks**
- **Expected time**: ~2-5 days

### Custom Configuration
Edit `main.py` to adjust:
- `epochs`: Number of training epochs
- `max_samples`: Limit dataset size for faster testing
- `resize_value`: Image resolution (32, 64, 128)
- `n_blocks`: Number of GraphNet blocks (2-10)

## 🔍 Inference and Comparison

### After Training (when weights are saved)

```python
# Single image inference
test_image = 'dataset/chihuahua/img_0_1071.jpg'

# GNN inference
inference_GNN(test_image)

# MLP inference
inference_MLP(test_image)

# Compare both models
compare_models(test_image)
```

### Output Files
- `predictions/gnn_prediction_[image].png` - GNN results
- `predictions/mlp_prediction_[image].png` - MLP results
- `predictions/comparison_[image].png` - Model comparison

## 🏗️ Model Architecture

### GraphNet
- **Node Features**: RGB pixel values (3 channels)
- **Edge Features**: Relative position + distance
- **Message Passing**: Multiple GraphNet blocks
- **Classification**: Flattened node features → MLP classifier

### MLP (Comparison)
- **Input**: Flattened image pixels
- **Architecture**: Multi-layer perceptron
- **Output**: Class probabilities

### CombinedModel
- Wraps GraphNet with classification head
- Handles graph → tensor conversion
- Outputs class predictions

## 📁 Project Structure

```
GraphNet_Classifier/
├── models/
│   ├── GNN.py              # Graph Neural Network implementation
│   └── MLP.py              # Multi-layer perceptron
├── utils/
│   ├── image_to_graph.py   # Image to graph conversion
│   ├── image_to_graph_knn.py # KNN-based graph construction
│   ├── dataset.py          # Dataset utilities
│   └── train_model.py      # Training utilities
├── dataset/                # Training data (mounted volume)
├── predictions/            # Output predictions (mounted volume)
├── weights/               # Saved model weights (mounted volume)
├── main.py               # Main training and inference script
├── test_setup.py         # Environment testing script
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
└── DOCKER_SETUP.md      # Detailed Docker instructions
```

## 🔧 Development Workflow

### Interactive Development
```bash
# Start container with shell access
docker compose run --rm graphnet-classifier bash

# Inside container
python test_setup.py    # Test environment
python main.py          # Run training
```

### Environment Testing
```bash
# Test all components
docker compose run --rm graphnet-classifier python test_setup.py
```

### Custom Training
```bash
# Edit main.py to change parameters, then rebuild
docker compose up --build
```

## 🐛 Troubleshooting

### Common Issues

**Docker not found**
```bash
# Install Docker Desktop
brew install --cask docker
open -a Docker
```

**PyG Import Errors**
- The Dockerfile uses PyTorch 2.1.0 with compatible PyG wheels
- All dependencies are pre-installed in the container

**Permission Issues**
```bash
sudo chown -R $USER:$USER dataset predictions weights
```

**Slow Training**
- Reduce `max_samples` in `main.py`
- Lower `resize_value` (32, 64 instead of 128)
- Reduce `n_blocks` (2-3 instead of 10)

### Performance Tips

**For Fast Testing:**
```python
train_GNN(epochs=1, max_samples=10, resize_value=32, n_blocks=2)
# Time: ~5-10 seconds
```

**For Medium Scale:**
```python
train_GNN(epochs=10, max_samples=200, resize_value=64, n_blocks=5)
# Time: ~10-15 minutes
```

**For Production:**
```python
train_GNN(epochs=30, max_samples=None, resize_value=128, n_blocks=10)
# Time: ~2-5 days
```

## 📈 Expected Results

### Training Performance
- **Fast testing**: 30 seconds - 2 minutes
- **Medium scale**: 10-15 minutes
- **Full dataset**: 2-5 days

### Model Comparison
- **GNN**: Better at capturing spatial relationships
- **MLP**: Faster training, simpler architecture
- **Comparison**: Side-by-side evaluation with confidence scores

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- PyTorch Geometric for GNN implementation
- Docker for containerization
- The chihuahua/muffin dataset for testing
