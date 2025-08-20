from torch.utils.data import Dataset
import torchvision.datasets as datasets
import time
import torch

from utils.image_to_graph.image_to_graph_optimized import image_to_graph_pixel_optimized
from utils.image_to_graph.image_to_graph_superpixel import image_to_graph_superpixel
from utils.image_to_graph.image_to_graph_patch import image_to_graph_patch

class OptimizedDatasetLoader(Dataset):
    def __init__(self, dataset_path='dataset', resize_value=128, diagonals=False, 
                 method='pixel', n_segments=100, patch_size=8, use_cache=True):
        self.dataset_path = dataset_path
        self.dataset = datasets.ImageFolder(self.dataset_path)
        self.resize_value = resize_value
        self.diagonals = diagonals
        self.method = method
        self.n_segments = n_segments
        self.patch_size = patch_size
        self.use_cache = use_cache
        
        print(f"Using {method} method with resize_value={resize_value}")
        if method == 'pixel':
            print(f"Graph size: {resize_value*resize_value} nodes")
        elif method == 'superpixel':
            print(f"Target superpixels: {n_segments}")
        elif method == 'patch':
            print(f"Patch size: {patch_size}, patches: {(resize_value//patch_size)**2}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.method == 'pixel':
            x, pos, edge_index = image_to_graph_pixel_optimized(
                image, self.resize_value, self.diagonals, self.use_cache)
        elif self.method == 'superpixel':
            x, pos, edge_index = image_to_graph_superpixel(
                image, self.resize_value, self.n_segments)
        elif self.method == 'patch':
            x, pos, edge_index = image_to_graph_patch(
                image, self.resize_value, self.patch_size)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return (x, pos, edge_index), torch.tensor(label, dtype=torch.long)
