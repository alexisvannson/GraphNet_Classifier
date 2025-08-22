from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torch


def image_to_graph_pixel(image_or_path, resize_value=128, diagonals=False):
    """
    Convert an image into a pixel graph representation.

    Args:
        image_or_path (PIL.Image.Image | str): PIL image or path to image.
        resize_value (int, optional): Output resolution (resize_value x resize_value).

    Returns:
        x (torch.FloatTensor): Node features of shape (num_nodes, num_features).
        pos (torch.FloatTensor): Node positions of shape (num_nodes, 2).
        edge_index (torch.LongTensor): Graph connectivity of shape (2, num_edges).
    """
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert('RGB')
    else:
        image = image_or_path.convert('RGB')

    resized_image = image.resize((resize_value, resize_value))
    tab = np.array(resized_image)  # H, W, C
    H, W, C = tab.shape

    # Node features: flatten pixel values => (H*W, C)
    x = tab.reshape(H * W, C)

    # Node positions: (row, col)
    pos = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1).reshape(-1, 2)

    # Edge index 4-neighborhood
    edges = []
    def duplicate(values, a, b):
        return (a, b) in values or (b, a) in values
    for row in range(H):
        for col in range(W):
            idx = row * W + col
            if row > 0 and not duplicate(edges, idx, (row - 1) * W + col):
                edges.append([idx, (row - 1) * W + col])
            if row < H - 1 and not duplicate(edges, idx, (row + 1) * W + col):
                edges.append([idx, (row + 1) * W + col])
            if col > 0 and not duplicate(edges, idx, row * W + (col - 1)):
                edges.append([idx, row * W + (col - 1)])
            if col < W - 1 and not duplicate(edges, idx, row * W + (col + 1)):
                edges.append([idx, row * W + (col + 1)])
            if diagonals:
                if row > 0 and col > 0 and not duplicate(edges, idx, (row - 1) * W + (col - 1)):
                    edges.append([idx, (row - 1) * W + (col - 1)])
                if row > 0 and col < W - 1 and not duplicate(edges, idx, (row - 1) * W + (col + 1)):
                    edges.append([idx, (row - 1) * W + (col + 1)])
                if row < H - 1 and col > 0 and not duplicate(edges, idx, (row + 1) * W + (col - 1)):
                    edges.append([idx, (row + 1) * W + (col - 1)])
                if row < H - 1 and col < W - 1 and not duplicate(edges, idx, (row + 1) * W + (col + 1)):
                    edges.append([idx, (row + 1) * W + (col + 1)])

    edge_index = np.transpose(np.array(edges, dtype=np.int64))  # (2, E)

    return x, pos, edge_index


class DatasetLoader(Dataset):
    def __init__(self, dataset_path='dataset', resize_value=128, diagonals=False):
        self.dataset_path = dataset_path
        self.dataset = datasets.ImageFolder(self.dataset_path)
        self.resize_value = resize_value
        self.diagonals = diagonals
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        x, pos, edge_index = image_to_graph_pixel(image, self.resize_value, self.diagonals)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(pos, dtype=torch.float32), torch.tensor(edge_index, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        