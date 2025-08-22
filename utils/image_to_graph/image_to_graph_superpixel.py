import torch
import numpy as np
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float


def image_to_graph_superpixel(image_or_path, resize_value=128, n_segments=100, compactness=10):
    """
    Alternative approach: Use superpixels to reduce graph size.
    
    Args:
        image_or_path: PIL image or path
        resize_value: Output resolution
        n_segments: Number of superpixels
        compactness: Superpixel compactness
    
    Returns:
        x, pos, edge_index: Graph representation with fewer nodes
    """
    
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert('RGB')
    else:
        image = image_or_path.convert('RGB')

    resized_image = image.resize((resize_value, resize_value))
    img_array = img_as_float(np.array(resized_image))
    
    # Create superpixels
    segments = slic(img_array, n_segments=n_segments, compactness=compactness, start_label=0)
    
    # Calculate superpixel features (mean RGB values)
    unique_segments = np.unique(segments)
    n_segments_actual = len(unique_segments)
    
    # Features: mean RGB for each superpixel
    features = []
    positions = []
    
    for segment_id in unique_segments:
        mask = segments == segment_id
        mean_rgb = np.mean(img_array[mask], axis=0)
        features.append(mean_rgb)
        
        # Position: centroid of superpixel
        y_coords, x_coords = np.where(mask)
        centroid_y, centroid_x = np.mean(y_coords), np.mean(x_coords)
        positions.append([centroid_y, centroid_x])
    
    x, pos = features, positions
    
    # Create edges between adjacent superpixels
    edges = []
    for i in range(n_segments_actual):
        for j in range(i + 1, n_segments_actual):
            # Check if superpixels are adjacent
            mask_i = segments == unique_segments[i]
            mask_j = segments == unique_segments[j]
            
            # Dilate one mask and check intersection
            from scipy.ndimage import binary_dilation
            dilated_i = binary_dilation(mask_i)
            if np.any(dilated_i & mask_j):
                edges.append([i, j])
                edges.append([j, i])  # Undirected graph
    
    if edges:
        edge_index = np.array(edges).T
    else:
        edge_index = np.empty((2, 0))
    
    return x, pos, edge_index
