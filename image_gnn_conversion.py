import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

def convert_image_to_graph(image_path, patch_size=16, embed_dim=768, k=9):
    # Load the image
    image = Image.open(image_path)

    # Preprocess the image
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    # Function to divide the image into patches
    def image_to_patches(image, patch_size):
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(image.size(0), -1, 3, patch_size, patch_size)
        return patches

    # Divide the image into patches
    patches = image_to_patches(image, patch_size)

    # Class to extract patch feature vectors
    class PatchEmbedding(nn.Module):
        def __init__(self, patch_size, in_channels=3, embed_dim=768):
            super().__init__()
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, x):
            x = self.proj(x).flatten(2).transpose(1, 2)
            return x

    # Extract patch feature vectors
    patch_embedding = PatchEmbedding(patch_size, embed_dim=embed_dim)
    node_features = patch_embedding(image)
    node_features = node_features.squeeze(0).detach().numpy()

    # Generate node positions
    num_patches_per_row = int(image.size(2) / patch_size)
    pos = np.array([(i // num_patches_per_row, i % num_patches_per_row) for i in range(num_patches_per_row * num_patches_per_row)])

    # Function to build the adjacency matrix
    def build_graph(node_features, k=9):
        tree = cKDTree(node_features)
        adj_matrix = np.zeros((node_features.shape[0], node_features.shape[0]))
        for i in range(node_features.shape[0]):
            dists, indices = tree.query(node_features[i], k=k+1)  # k+1 because the closest neighbor is the node itself
            adj_matrix[i, indices[1:]] = 1  # skip the first index because it's the node itself
        return adj_matrix

    # Build the graph
    adj_matrix = build_graph(node_features, k)
    graph = nx.from_numpy_array(adj_matrix)

    # Convert adjacency matrix to edge_index
    edge_index = np.array(list(graph.edges)).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Convert node features to tensor
    x = torch.tensor(node_features, dtype=torch.float)

    # Convert positions to tensor
    pos = torch.tensor(pos, dtype=torch.float)

    return x, pos, edge_index
