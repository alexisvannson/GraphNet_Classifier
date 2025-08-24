import torch
import numpy as np
from PIL import Image
from .image_to_graph_optimized import create_grid_edges_optimized

def image_to_graph_patch(image_or_path, resize_value=128, patch_size=8, grayscale=False):
    """
    Alternative approach: Use patches to reduce graph size. 
    
    Args:
        image_or_path: PIL image or path
        resize_value: Output resolution
        patch_size: Size of each patch
        grayscale (bool): Whether to convert to grayscale (for MNIST optimization)
    
    Returns:
        x, pos, edge_index: Graph representation with patch-based nodes
    """
    if isinstance(image_or_path, str):
        if grayscale:
            image = Image.open(image_or_path).convert('L')
        else:
            image = Image.open(image_or_path).convert('RGB')
    else:
        if grayscale:
            image = image_or_path.convert('L')
        else:
            image = image_or_path.convert('RGB')

    resized_image = image.resize((resize_value, resize_value))
    img_array = np.array(resized_image)
    
    if grayscale:
        H, W = img_array.shape
        # Add channel dimension for grayscale
        img_array = img_array.reshape(H, W, 1)
    else:
        H, W, C = img_array.shape
    
    patch_h, patch_w = patch_size, patch_size
    
    # Calculate number of patches
    n_patches_h = H // patch_h
    n_patches_w = W // patch_w
    
    features = []
    positions = []
    
    # Extract patch features
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            # Feature: mean RGB of patch (or grayscale)
            if grayscale:
                mean_value = np.mean(patch)
                features.append([mean_value])  # Single channel
            else:
                mean_rgb = np.mean(patch, axis=(0, 1))
                features.append(mean_rgb)
            
            # Position: center of patch
            center_y = i * patch_h + patch_h // 2
            center_x = j * patch_w + patch_w // 2
            positions.append([center_y, center_x])
    
    x, pos = features, positions
    
    # Create grid edges between patches
    edge_index = create_grid_edges_optimized(n_patches_h, n_patches_w, diagonals=False)
    
    return x, pos, edge_index
