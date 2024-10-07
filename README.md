# GNN-based Image Classification

This project implements a custom GNN model for image classification by transforming images into graphs and applying message-passing algorithms. The Theory is explained in GNN.pdf(french) or https://alexisvannson.hashnode.dev/applying-graph-neural-networks-for-better-image-classification (english)

## Overview

The code includes:
- Custom dataset for image loading (`CustomDataset`).
- Image-to-graph conversion (`image_to_graph`).
- GNN-based model (`GraphNet` and processors).
- Combined model for classification.

## Dataset

A `CustomDataset` class is used to load images from a directory using `torchvision.datasets.ImageFolder`.

## Image-to-Graph Conversion

The function `image_to_graph` divides an image into patches, extracts patch embeddings, and connects patches based on k-nearest neighbors.

- **Patch Size:** 16x16.
- **Embedding Dimension:** 768.
- **Graph Construction:** cKDTree for k-NN (k=9).

## Model Architecture

### GraphNet

`GraphNet` applies node and edge processing layers using multi-layer perceptrons (MLP) and a message-passing structure through several blocks.

- **Node Processing:** Uses an `MLP` to update node features.
- **Edge Processing:** Uses another `MLP` for edge feature update.
- **Graph Processing:** Message-passing over nodes and edges with residual connections.

### Combined Model

The `CombinedModel` takes the GNN output and applies a linear classifier for final binary classification.

## Training Setup

- **Optimizer:** Adam.
- **Loss Function:** CrossEntropy for binary classification.
- **Batch Size:** 32.
- **Learning Rate:** 0.001.
- **Early Stopping:** Monitors validation loss with patience of 5 epochs.

## Example Usage

```python
# Load dataset
dataset = CustomDataset(image_folder='/path/to/dataset')

# Split into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Initialize model
gnn = GraphNet(...)
model = CombinedModel(gnn)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        for img, label in zip(images, labels):
            x, pos, edge_index = image_to_graph(img)
            output = model(x, pos, edge_index)
            loss = criterion(output.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
        optimizer.step()
