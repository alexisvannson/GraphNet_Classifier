import numpy as np
from PIL import Image


def image_to_graph(image, resize_value=28, diagonals=True, grayscale=True):
    """
    Convert a PIL image to a graph edge_index (COO format) for GNNs.
    Args:
        image: PIL Image
        resize_value: int, resize image to (resize_value, resize_value)
        diagonals: bool, whether to include diagonal edges
        grayscale: bool, convert to grayscale
    Returns:
        x: (num_nodes, num_features) node features (flattened image pixels)
        pos: (num_nodes, 2) node positions (row, col)
        edge_index: (2, num_edges) edge indices in COO format
    """
    # Convert image to numpy array
    img = image.resize((resize_value, resize_value))
    if grayscale:
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0  # shape: (H, W)
        x = arr.flatten()[:, None]  # (num_nodes, 1)
    else:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # shape: (H, W, 3)
        x = arr.reshape(-1, 3)  # (num_nodes, 3)

    H, W = resize_value, resize_value
    pos = np.array([[i, j] for i in range(H) for j in range(W)], dtype=np.float32)  # (num_nodes, 2)

    # Build edge_index
    edge_list = []
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            # 4-connectivity
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    nidx = ni * W + nj
                    edge_list.append((idx, nidx))
            if diagonals:
                for di, dj in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        nidx = ni * W + nj
                        edge_list.append((idx, nidx))
    # Convert to numpy array
    edge_index = np.array(edge_list, dtype=np.int64).T  # shape: (2, num_edges)
    return x, pos, edge_index



def map_vtk_to_csr(vtk_mesh):
    """
    Map a VTK mesh (e.g., from pyvista or vtk) to CSR-like arrays.
    Returns:
        node_coords: (num_nodes, 3) array of node (x, y, z) positions
        elem2nodes: (num_elems, nodes_per_elem) array mapping each element to its node indices
        p_elem2nodes: (num_elems+1,) array, CSR pointer for elem2nodes
    """
    # Extract node coordinates
    points = vtk_mesh.points  # shape: (num_nodes, 3)
    node_coords = np.array(points, dtype=np.float32)

    # Extract cell connectivity
    # For pyvista/vtk, cells are stored as a flat array: [n0, id0_0, id0_1, ..., n1, id1_0, ...]
    # We'll convert to a 2D array (num_elems, nodes_per_elem)
    cells = vtk_mesh.cells
    # Parse the flat cell array
    elem2nodes = []
    i = 0
    while i < len(cells):
        n = cells[i]
        elem_nodes = cells[i+1:i+1+n]
        elem2nodes.append(elem_nodes)
        i += n + 1
    elem2nodes = np.array(elem2nodes, dtype=np.int32)

    # CSR pointer
    nodes_per_elem = elem2nodes.shape[1] if elem2nodes.ndim == 2 else None
    if nodes_per_elem is not None:
        p_elem2nodes = np.arange(0, len(elem2nodes) * nodes_per_elem + 1, nodes_per_elem, dtype=np.int32)
    else:
        # For variable-size elements
        lengths = [len(e) for e in elem2nodes]
        p_elem2nodes = np.zeros(len(elem2nodes) + 1, dtype=np.int32)
        p_elem2nodes[1:] = np.cumsum(lengths)

        # Flatten elem2nodes for variable-size elements
        elem2nodes = np.concatenate(elem2nodes)

    return node_coords, elem2nodes, p_elem2nodes



def map_image_to_csr(image_path, resize_value=10):
    """
    assume grayscale image, from image path then resize and return csr matrix
    assumes that most frequent value is background (true for MNIST) and only non-background values are considered
    """
    from scipy.sparse import csr_matrix
    image = Image.open(image_path)
    image = image.resize((resize_value, resize_value))
    
    image = np.array(image)
    counts = np.bincount(image.flatten())
    background = np.argmax(counts)
    image[image == background] = 0
    image = image.astype(np.float32) / 255.0
    return csr_matrix(image)


def map_csr_to_graph(csr):
    """
    Given a scipy.sparse.csr_matrix, return edge_index (PyTorch geometric style) for the nonzero entries.
    Each nonzero entry (i, j) is an edge from node i to node j.
    Returns:
        edge_index: np.ndarray of shape (2, num_edges)
    """
    # Get the row and column indices of nonzero entries
    row, col = csr.nonzero()
    # Stack as edge_index (shape: 2, num_edges)
    edge_index = np.vstack([row, col])
    return edge_index
