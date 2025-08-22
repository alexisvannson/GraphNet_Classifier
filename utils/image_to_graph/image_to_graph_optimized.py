from PIL import Image
import numpy as np
import torch
from functools import lru_cache


def create_grid_edges_optimized(H, W, diagonals=False):
    """
    Create grid edges using vectorized operations - much faster than the original approach.
    
    Args:
        H, W: Height and width of the grid
        diagonals: Whether to include diagonal connections
    
    Returns:
        edge_index: (2, num_edges) tensor of edge indices
    """
    # Create node indices for the grid
    nodes = np.arange(H * W).reshape(H, W)
    
    # Horizontal edges (left to right)
    h_edges = np.stack([nodes[:, :-1].flatten(), nodes[:, 1:].flatten()], axis=1)
    
    # Vertical edges (top to bottom)
    v_edges = np.stack([nodes[:-1, :].flatten(), nodes[1:, :].flatten()], axis=1)
    
    edges = [h_edges, v_edges]
    
    if diagonals:
        # Diagonal edges (top-left to bottom-right)
        d1_edges = np.stack([nodes[:-1, :-1].flatten(), nodes[1:, 1:].flatten()], axis=1)
        # Diagonal edges (top-right to bottom-left)
        d2_edges = np.stack([nodes[:-1, 1:].flatten(), nodes[1:, :-1].flatten()], axis=1)
        edges.extend([d1_edges, d2_edges])
    
    # Combine all edges
    edge_index = np.concatenate(edges, axis=0).T  # (2, num_edges)
    
    return torch.tensor(edge_index, dtype=torch.long)


@lru_cache(maxsize=128)
def get_cached_edge_index(resize_value, diagonals):
    """
    Cache edge indices for repeated use - same graph topology for same image size.
    """
    return create_grid_edges_optimized(resize_value, resize_value, diagonals)


def image_to_graph_pixel_optimized(image_or_path, resize_value=128, diagonals=False, use_cache=True):
    """
    Optimized version of image to graph conversion.
    
    Args:
        image_or_path (PIL.Image.Image | str): PIL image or path to image.
        resize_value (int, optional): Output resolution (resize_value x resize_value).
        diagonals (bool): Whether to include diagonal connections
        use_cache (bool): Whether to use cached edge indices
    
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
    x = torch.tensor(tab.reshape(H * W, C), dtype=torch.float32)

    # Node positions: (row, col) - vectorized
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pos = torch.tensor(np.stack([rows.flatten(), cols.flatten()], axis=1), dtype=torch.float32)

    # Get edge indices (cached for same image size)
    if use_cache:
        edge_index = get_cached_edge_index(resize_value, diagonals)
    else:
        edge_index = create_grid_edges_optimized(H, W, diagonals)
    
    return x, pos, edge_index
