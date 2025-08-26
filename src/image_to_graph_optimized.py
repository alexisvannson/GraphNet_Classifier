"""
Optimized Image-to-Graph Conversion Module

This module provides efficient functions to convert images to graph representations
for use with Graph Neural Networks (GNNs). It supports multiple conversion methods:
- Pixel-based: Each pixel becomes a node with 4/8 connectivity
- Superpixel-based: Uses SLIC algorithm for segmentation
- Patch-based: Divides image into patches
- CSR-based: Compressed Sparse Row format for sparse representations

Features:
- Vectorized operations for better performance
- Memory-efficient implementations
- Support for both grayscale and RGB images
- Caching for repeated operations
- Multiple connectivity patterns
"""

import numpy as np
from PIL import Image
import torch
from typing import Tuple, Optional, Union, Dict, Any
import warnings
from functools import lru_cache
import os
from scipy.sparse import csr_matrix
from scipy import ndimage
from sklearn.feature_extraction import image as skimage
from sklearn.cluster import MiniBatchKMeans
import cv2


class ImageToGraphConverter:
    """Main class for converting images to graph representations."""
    
    def __init__(self, cache_size: int = 128):
        """
        Initialize the converter with caching.
        
        Args:
            cache_size: Number of recent conversions to cache
        """
        self.cache_size = cache_size
        self._edge_cache: Dict[str, np.ndarray] = {}
    
    def _get_cache_key(self, size: int, connectivity: str, diagonals: bool) -> str:
        """Generate cache key for edge indices."""
        return f"{size}_{connectivity}_{diagonals}"
    
    def _generate_edge_indices(self, height: int, width: int, connectivity: str = "8", 
                              diagonals: bool = True) -> np.ndarray:
        """
        Generate edge indices for a grid of given dimensions.
        
        Args:
            height: Image height
            width: Image width
            connectivity: "4" or "8" connectivity
            diagonals: Whether to include diagonal edges (for 8-connectivity)
            
        Returns:
            edge_index: (2, num_edges) array of edge indices
        """
        cache_key = self._get_cache_key(height * width, connectivity, diagonals)
        
        if cache_key in self._edge_cache:
            return self._edge_cache[cache_key]
        
        # Generate all possible node indices
        nodes = np.arange(height * width).reshape(height, width)
        
        # Define neighbor offsets
        if connectivity == "4":
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif connectivity == "8":
            if diagonals:
                offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            else:
                offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            raise ValueError(f"Unsupported connectivity: {connectivity}")
        
        # Generate edges using vectorized operations
        edges = []
        for di, dj in offsets:
            # Shift the grid
            shifted = np.roll(np.roll(nodes, di, axis=0), dj, axis=1)
            
            # Create edges from original to shifted positions
            valid_mask = np.ones_like(nodes, dtype=bool)
            if di < 0:
                valid_mask[-di:, :] = False
            elif di > 0:
                valid_mask[:-di, :] = False
            if dj < 0:
                valid_mask[:, -dj:] = False
            elif dj > 0:
                valid_mask[:, :-dj] = False
            
            # Get valid edges
            src = nodes[valid_mask].flatten()
            dst = shifted[valid_mask].flatten()
            
            # Add to edges list
            edges.extend(list(zip(src, dst)))
        
        # Convert to numpy array and remove duplicates
        edge_index = np.array(edges, dtype=np.int64).T
        edge_index = np.unique(edge_index, axis=1)
        
        # Cache the result
        if len(self._edge_cache) < self.cache_size:
            self._edge_cache[cache_key] = edge_index
        
        return edge_index


def image_to_graph_pixel_optimized(
    image: Union[str, Image.Image, np.ndarray, torch.Tensor],
    resize_value: int = 28,
    diagonals: bool = False,
    use_cache: bool = True,
    grayscale: bool = True,
    connectivity: str = "4"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert image to graph using pixel-based approach with optimizations.
    
    Args:
        image: Input image (path, PIL Image, numpy array, or torch tensor)
        resize_value: Size to resize image to (resize_value x resize_value)
        diagonals: Whether to include diagonal edges
        use_cache: Whether to use edge caching
        grayscale: Whether to convert to grayscale
        connectivity: "4" or "8" connectivity
        
    Returns:
        x: (num_nodes, num_features) node features
        pos: (num_nodes, 2) node positions
        edge_index: (2, num_edges) edge indices
    """
    # Load and preprocess image
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        img = Image.fromarray(image.numpy())
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Resize image
    img = img.resize((resize_value, resize_value), Image.Resampling.LANCZOS)
    
    # Convert to grayscale if requested
    if grayscale:
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        x = arr.flatten()[:, None]  # (num_nodes, 1)
    else:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        x = arr.reshape(-1, 3)  # (num_nodes, 3)
    
    # Generate positions (grid coordinates)
    pos = np.array([[i, j] for i in range(resize_value) for j in range(resize_value)], 
                   dtype=np.float32)
    
    # Generate edge indices
    if use_cache:
        converter = ImageToGraphConverter()
        edge_index = converter._generate_edge_indices(resize_value, resize_value, 
                                                     connectivity, diagonals)
    else:
        # Generate without caching
        nodes = np.arange(resize_value * resize_value).reshape(resize_value, resize_value)
        
        if connectivity == "4":
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif connectivity == "8":
            if diagonals:
                offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            else:
                offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        edges = []
        for di, dj in offsets:
            shifted = np.roll(np.roll(nodes, di, axis=0), dj, axis=1)
            valid_mask = np.ones_like(nodes, dtype=bool)
            if di < 0:
                valid_mask[-di:, :] = False
            elif di > 0:
                valid_mask[:-di, :] = False
            if dj < 0:
                valid_mask[:, -dj:] = False
            elif dj > 0:
                valid_mask[:, :-dj] = False
            
            src = nodes[valid_mask].flatten()
            dst = shifted[valid_mask].flatten()
            edges.extend(list(zip(src, dst)))
        
        edge_index = np.array(edges, dtype=np.int64).T
        edge_index = np.unique(edge_index, axis=1)
    
    return x, pos, edge_index


def image_to_graph_superpixel(
    image: Union[str, Image.Image, np.ndarray],
    resize_value: int = 128,
    n_segments: int = 100,
    compactness: float = 10.0,
    grayscale: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert image to graph using superpixel segmentation.
    
    Args:
        image: Input image
        resize_value: Size to resize image to
        n_segments: Number of superpixels
        compactness: Superpixel compactness
        grayscale: Whether to convert to grayscale
        
    Returns:
        x: (num_nodes, num_features) node features
        pos: (num_nodes, 2) node positions
        edge_index: (2, num_edges) edge indices
    """
    try:
        from skimage.segmentation import slic
        from skimage.util import img_as_float
    except ImportError:
        raise ImportError("scikit-image is required for superpixel segmentation")
    
    # Load and preprocess image
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Resize image
    img = img.resize((resize_value, resize_value), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    if grayscale:
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.stack([arr] * 3, axis=-1)  # Convert to 3-channel for SLIC
    else:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
    
    # Apply SLIC segmentation
    segments = slic(arr, n_segments=n_segments, compactness=compactness, 
                   start_label=0, channel_axis=2)
    
    # Get unique segment IDs
    unique_segments = np.unique(segments)
    num_segments = len(unique_segments)
    
    # Calculate features for each superpixel
    x = []
    pos = []
    
    for segment_id in unique_segments:
        mask = segments == segment_id
        segment_pixels = arr[mask]
        
        # Calculate mean color as features
        if grayscale:
            features = [np.mean(segment_pixels[:, 0])]  # Use only first channel
        else:
            features = [np.mean(segment_pixels[:, i]) for i in range(3)]
        
        # Calculate centroid position
        y_coords, x_coords = np.where(mask)
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        
        x.append(features)
        pos.append([centroid_y, centroid_x])
    
    x = np.array(x, dtype=np.float32)
    pos = np.array(pos, dtype=np.float32)
    
    # Create edges between adjacent superpixels
    edges = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            # Check if segments are adjacent
            mask_i = segments == unique_segments[i]
            mask_j = segments == unique_segments[j]
            
            # Dilate one mask and check intersection
            from scipy.ndimage import binary_dilation
            dilated_i = binary_dilation(mask_i)
            if np.any(dilated_i & mask_j):
                edges.append((i, j))
                edges.append((j, i))  # Undirected graph
    
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    
    return x, pos, edge_index


def image_to_graph_patch(
    image: Union[str, Image.Image, np.ndarray],
    resize_value: int = 128,
    patch_size: int = 8,
    grayscale: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert image to graph using patch-based approach.
    
    Args:
        image: Input image
        resize_value: Size to resize image to
        patch_size: Size of each patch
        grayscale: Whether to convert to grayscale
        
    Returns:
        x: (num_nodes, num_features) node features
        pos: (num_nodes, 2) node positions
        edge_index: (2, num_edges) edge indices
    """
    # Load and preprocess image
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Resize image
    img = img.resize((resize_value, resize_value), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    if grayscale:
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr[:, :, None]  # Add channel dimension
    else:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
    
    # Calculate number of patches
    num_patches_h = resize_value // patch_size
    num_patches_w = resize_value // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Extract patches and calculate features
    x = []
    pos = []
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Extract patch
            patch = arr[i*patch_size:(i+1)*patch_size, 
                       j*patch_size:(j+1)*patch_size]
            
            # Calculate features (mean of each channel)
            if grayscale:
                features = [np.mean(patch[:, :, 0])]
            else:
                features = [np.mean(patch[:, :, c]) for c in range(3)]
            
            # Calculate patch center position
            center_y = (i + 0.5) * patch_size
            center_x = (j + 0.5) * patch_size
            
            x.append(features)
            pos.append([center_y, center_x])
    
    x = np.array(x, dtype=np.float32)
    pos = np.array(pos, dtype=np.float32)
    
    # Create edges between adjacent patches
    edges = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            current_idx = i * num_patches_w + j
            
            # Check 4-connectivity with adjacent patches
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < num_patches_h and 0 <= nj < num_patches_w:
                    neighbor_idx = ni * num_patches_w + nj
                    edges.append((current_idx, neighbor_idx))
    
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    
    return x, pos, edge_index


def map_image_to_csr(
    image: Union[str, Image.Image, np.ndarray],
    resize_value: int = 28,
    patch_size: int = 1,
    grayscale: bool = True,
    threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert image to CSR format for sparse graph representation.
    
    Args:
        image: Input image
        resize_value: Size to resize image to
        patch_size: Size of patches (1 for pixel-level)
        grayscale: Whether to convert to grayscale
        threshold: Threshold for considering a pixel as foreground
        
    Returns:
        x: (num_nodes, num_features) node features
        pos: (num_nodes, 2) node positions  
        edge_index: (2, num_edges) edge indices
    """
    # Load and preprocess image
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Resize image
    img = img.resize((resize_value, resize_value), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    if grayscale:
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.mean(arr, axis=2)  # Convert to grayscale for CSR
    
    # Apply threshold to get foreground pixels
    foreground_mask = arr > threshold
    
    # Get foreground pixel indices
    foreground_indices = np.where(foreground_mask)
    num_foreground = len(foreground_indices[0])
    
    if num_foreground == 0:
        # No foreground pixels, return empty graph
        return (np.zeros((0, 1), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((2, 0), dtype=np.int64))
    
    # Create node features (pixel values)
    x = arr[foreground_mask][:, None]  # (num_foreground, 1)
    
    # Create node positions
    pos = np.column_stack([foreground_indices[0], foreground_indices[1]]).astype(np.float32)
    
    # Create edges between adjacent foreground pixels
    edges = []
    for i in range(num_foreground):
        y, x = foreground_indices[0][i], foreground_indices[1][i]
        
        # Check 4-connectivity
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < resize_value and 0 <= nx < resize_value and 
                foreground_mask[ny, nx]):
                # Find index of neighbor
                neighbor_idx = np.where((foreground_indices[0] == ny) & 
                                      (foreground_indices[1] == nx))[0][0]
                edges.append((i, neighbor_idx))
    
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    
    return x, pos, edge_index


# Convenience functions for backward compatibility
def image_to_graph(image, resize_value=28, diagonals=True, grayscale=True):
    """Backward compatibility wrapper for image_to_graph_pixel_optimized."""
    return image_to_graph_pixel_optimized(image, resize_value, diagonals, True, grayscale)


def map_csr_to_graph(csr):
    """Convert CSR matrix to edge indices."""
    row, col = csr.nonzero()
    edge_index = np.vstack([row, col])
    return edge_index


# Performance monitoring
class PerformanceMonitor:
    """Monitor performance of graph conversion operations."""
    
    def __init__(self):
        self.timings = {}
        self.counts = {}
    
    def time_operation(self, operation_name: str, func, *args, **kwargs):
        """Time a function execution."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        if operation_name not in self.timings:
            self.timings[operation_name] = []
            self.counts[operation_name] = 0
        
        self.timings[operation_name].append(end_time - start_time)
        self.counts[operation_name] += 1
        
        return result
    
    def get_stats(self):
        """Get performance statistics."""
        stats = {}
        for op_name in self.timings:
            times = self.timings[op_name]
            stats[op_name] = {
                'count': self.counts[op_name],
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        return stats


# Global performance monitor
_perf_monitor = PerformanceMonitor()


def get_performance_stats():
    """Get performance statistics for all operations."""
    return _perf_monitor.get_stats() 