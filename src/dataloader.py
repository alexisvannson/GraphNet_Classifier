from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np
import torch

#from utils.image_to_graph.image_to_graph_optimized import image_to_graph_pixel_optimized
#from utils.image_to_graph.image_to_graph_superpixel import image_to_graph_superpixel
#from utils.image_to_graph.image_to_graph_patch import image_to_graph_patch

class OptimizedDatasetLoader(Dataset):
    def __init__(self, dataset_path='dataset', resize_value=128, diagonals=False, 
                 method='pixel', n_segments=100, patch_size=8, use_cache=True, grayscale=False):
        self.dataset_path = dataset_path
        self.dataset = datasets.ImageFolder(self.dataset_path)
        self.resize_value = resize_value
        self.diagonals = diagonals
        self.method = method
        self.n_segments = n_segments
        self.patch_size = patch_size
        self.use_cache = use_cache
        self.grayscale = grayscale
        
        print(f"Using {method} method with resize_value={resize_value}")
        if grayscale:
            print("Processing images as grayscale (optimized for MNIST)")
        if method == 'pixel':
            print(f"Graph size: {resize_value*resize_value} nodes")
        elif method == 'superpixel':
            print(f"Target superpixels: {n_segments}")
        elif method == 'patch':
            print(f"Patch size: {patch_size}, patches: {(resize_value//patch_size)**2}")
        elif method == 'csr':
            print(f"CSR size: {resize_value*resize_value} nodes")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.method == 'pixel':
            x, pos, edge_index = image_to_graph_pixel_optimized(
                image, self.resize_value, self.diagonals, self.use_cache, self.grayscale)
        elif self.method == 'superpixel':
            x, pos, edge_index = image_to_graph_superpixel(
                image, self.resize_value, self.n_segments, grayscale=self.grayscale)
        elif self.method == 'patch':
            x, pos, edge_index = image_to_graph_patch(
                image, self.resize_value, self.patch_size, grayscale=self.grayscale)
        elif self.method == 'csr':
            x, pos, edge_index = map_image_to_csr(image, self.resize_value, self.patch_size, grayscale=self.grayscale)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Convert numpy arrays to PyTorch tensors
        x = torch.tensor(x, dtype=torch.float32)
        pos = torch.tensor(pos, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        return (x, pos, edge_index), torch.tensor(label, dtype=torch.long)



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




from PIL import Image
"""
resize_value = 10
image = Image.open('data/mnist/test/0/mnist_test_00028.png')
image = image.resize((resize_value, resize_value))
image = image.convert('L')
image = np.array(image)
counts = np.bincount(image.flatten())
background = np.argmax(counts)
print(background)
image[image == background] = 0
values = image.flatten()[image.flatten() > 0]
row_indices = np.where(image.flatten() > 0)[0]



#col_indices = np.where(image.flatten() > 0)[1]
print("-"*60)
#print(col_indices)

print(image)
print(values)
print("row_indices", row_indices)
row_indices = np.vectorize(lambda x: x % resize_value)(row_indices)
print("row_indices", row_indices)
"""

def image_to_sparse_matrix(image_path, resize_value=10):
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


def build_graph_from_csr(csr_matrix):
    "assume csr matrix, return graph"
    # get shape of csr matrix
    rows, cols = csr_matrix.shape
    # get non-zero values
    values = csr_matrix.data
    # get row indices
    row_indices = csr_matrix.indices
    # get column indices
    print(row_indices)
    print(values)
    pass


image_path = 'data/mnist/test/0/mnist_test_00028.png'
print(image_to_sparse_matrix(image_path))