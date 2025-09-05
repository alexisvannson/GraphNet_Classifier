from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np
import torch
from convert import image_to_graph_pixel_optimized
from torchvision import transforms

class GraphDatasetLoader(Dataset):
    def __init__(self, dataset_path='dataset', resize_value=28, npoints_of_scheme=4, 
                 n_segments=100, patch_size=8, use_cache=True, grayscale=False):
        self.dataset_path = dataset_path
        # Apply resizing once via torchvision transform to standardize inputs
        self.transform = transforms.Compose([
            transforms.Resize((resize_value, resize_value)),
        ])
        self.dataset = datasets.ImageFolder(self.dataset_path, transform=self.transform)
        self.resize_value = resize_value
        self.npoints_of_scheme = npoints_of_scheme
        self.grayscale = grayscale
        self.use_cache = use_cache
        
        if grayscale:
            print("Processing images as grayscale (optimized for MNIST)")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        x, pos, edge_index = image_to_graph_pixel_optimized(
            image,
            resize_value=self.resize_value,
            npoints_of_scheme=self.npoints_of_scheme,
            use_cache=self.use_cache,
            grayscale=self.grayscale,
        )
        
        # Convert numpy arrays to PyTorch tensors
        x = torch.tensor(x, dtype=torch.float32)
        pos = torch.tensor(pos, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        return (x, pos, edge_index), torch.tensor(label, dtype=torch.long)


class DatasetLoader(Dataset):
    def __init__(self, dataset_path='dataset', resize_value=28, grayscale=False):
        self.dataset_path = dataset_path
        self.transform = transforms.Compose([
            transforms.Resize((resize_value, resize_value)),
        ])
        self.dataset = datasets.ImageFolder(self.dataset_path, transform=self.transform)
        self.resize_value = resize_value
        self.grayscale = grayscale