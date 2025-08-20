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


# Performance comparison function
def benchmark_graph_methods(image_path, resize_values=[32, 64, 128]):
    """Benchmark different graph construction methods."""
    print("Graph Construction Performance Benchmark")
    print("=" * 50)
    
    methods = ['pixel', 'superpixel', 'patch']
    
    for resize_value in resize_values:
        print(f"\nResize value: {resize_value}x{resize_value}")
        print("-" * 30)
        
        for method in methods:
            try:
                start_time = time.time()
                
                if method == 'pixel':
                    x, pos, edge_index = image_to_graph_pixel_optimized(
                        image_path, resize_value, diagonals=False, use_cache=True)
                elif method == 'superpixel':
                    x, pos, edge_index = image_to_graph_superpixel(
                        image_path, resize_value, n_segments=resize_value//2)
                elif method == 'patch':
                    patch_size = max(4, resize_value // 8)
                    x, pos, edge_index = image_to_graph_patch(
                        image_path, resize_value, patch_size)
                
                end_time = time.time()
                
                print(f"{method:12s}: {end_time-start_time:.3f}s | "
                      f"Nodes: {x.shape[0]:4d} | Edges: {edge_index.shape[1]:6d}")
                
            except Exception as e:
                print(f"{method:12s}: Error - {e}")
    
    print("\n" + "=" * 50) 