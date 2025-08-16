import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def image_to_patches(image, patch_size):
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(image.size(0), -1, 3, patch_size, patch_size)
        return patches

# Classe pour extraire les vecteurs de caractéristiques des patches
    class PatchEmbedding(nn.Module):
        def __init__(self, patch_size, in_channels=3, embed_dim=768):
            super().__init__()
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, x):
            x = self.proj(x).flatten(2).transpose(1, 2)
            return x
 
# Fonction pour construire la matrice d'adjacence
def build_graph(node_features, k=9):
    tree = cKDTree(node_features)
    adj_matrix = np.zeros((node_features.shape[0], node_features.shape[0]))
    for i in range(node_features.shape[0]):
        dists, indices = tree.query(node_features[i], k=k+1)  # k+1 car le voisin le plus proche est le nœud lui-même
        adj_matrix[i, indices[1:]] = 1  # sauter le premier index car c'est le nœud lui-même
    return adj_matrix


def image_to_graph(image_path, patch_size=16, embed_dim=768, k=9):
    """
    Convertit une image en  graphe où chaque nœud représente un patch de l'image.
    Chaque noeud contient un vecteur de caractéristiques dérivé du patch, et des arêtes sont formées
    en fonction de la similarité de ces vecteurs de caractéristiques avec l'algorithme knn

    Paramètres:
    - image_path (str): Chemin vers l'image d'entrée.
    - patch_size (int): Taille des patches pour diviser l'image. La valeur par défaut est 16.
    - embed_dim (int): Dimension du vecteur d'embedding pour chaque patch. La valeur par défaut est 768.
    - k (int): Nombre de voisins les plus proches pour connecter chaque nœud dans le graphe. La valeur par défaut est 9.

    Retourne:
    - x (torch.Tensor): Matrice des caractéristiques des nœuds de taille (num_patches, embed_dim).
    Chaque ligne de cette matrice représente un patch de l'image. Les caractéristiques de chaque patch
    sont extraites à partir des valeurs RGB des pixels dans ce patch.
    - node_coords (torch.Tensor): Positions des nœuds dans la grille de l'image originale.
    - edge2nodes (torch.Tensor): Liste des arêtes au format COO.
    """
    # Charger l'image
    image = Image.open(image_path)

    # Prétraiter l'image
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    patches = image_to_patches(image, patch_size)

    # Extraire les vecteurs de caractéristiques des patches
    patch_embedding = PatchEmbedding(patch_size, embed_dim=embed_dim)
    node_features = patch_embedding(image)
    node_features = node_features.squeeze(0).detach().numpy()

    # Générer les positions des nœuds
    num_patches_per_row = int(image.size(2) / patch_size)
    node_coords = np.array([(i // num_patches_per_row, i % num_patches_per_row) for i in range(num_patches_per_row * num_patches_per_row)])

    # Construire le graphe
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
