import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

def image_to_patches(image, patch_size):
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(image.size(0), -1, 3, patch_size, patch_size)
        return patches

# Classe pour extraire les vecteurs de caractéristiques des patches
    class PatchEmbedding(nn.Module):
        def __init__(self, patch_size, in_channels=3, embed_dim=768):
            super().__init__()
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, x):
            x = self.proj(x).flatten(2).transpose(1, 2)
            return x
 
# Fonction pour construire la matrice d'adjacence
def build_graph(node_features, k=9):
    tree = cKDTree(node_features)
    adj_matrix = np.zeros((node_features.shape[0], node_features.shape[0]))
    for i in range(node_features.shape[0]):
        dists, indices = tree.query(node_features[i], k=k+1)  # k+1 car le voisin le plus proche est le nœud lui-même
        adj_matrix[i, indices[1:]] = 1  # sauter le premier index car c'est le nœud lui-même
    return adj_matrix

def image_to_graph_pixel(image_path):
        image = Image.open(image_path)
        transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor() ])
        image = transform(image).unsqueeze(0)
        _write_numpy2d_to_file_png(tab, filename)

def _transform_numpy2d_to_hsv(tab):
    """Non applicable.
    """
    imgval = numpy.zeros((tab.shape[0], tab.shape[1], 3), dtype=numpy.float64)
    for i in range(0, tab.shape[0]):
        for j in range(0, tab.shape[1]):
            if tab[i, j] == 1:
                imgval[i, j, 0] = 1  # h = elem2subdom/nsubdoms
                imgval[i, j, 1] = 1  # s
                imgval[i, j, 2] = 1  # v
            else:
                imgval[i, j, 0] = 1  # h
                imgval[i, j, 1] = 1  # s
                imgval[i, j, 2] = 0  # v
 
    return imgval
 
def _map_ab_to_cd(a, b, c, d, x):
    """Non applicable.
    """
    # Map $x \in [a,b]$ to $x \in [c,d]$.
    alpha = (c - d) / (a - b)
    beta = c - alpha * a
 
    return alpha * x + beta
 
 
def _transform_numpy2d_to_mrg(tab, selection):
    """Non applicable.
    """
    ny = tab.shape[0]
    nx = tab.shape[1]
 
    out_mesh = mesh4u._mesh.Mesh()
    out_mesh.spacedim = 3
    nnodes = (nx + 1) * (ny + 1)
    nodes_per_elem = 4
    if selection == None:
        nelems = nx * ny
    elif selection == 0:
        nelems = numpy.count_nonzero(tab==0)
    else:
        nelems = numpy.count_nonzero(tab)
    out_mesh.nelems = nelems
    out_mesh.p_elem2nodes = numpy.empty(out_mesh.nelems + 1, dtype=numpy.int64)
    out_mesh.p_elem2nodes[0] = 0
    for i in range(0, out_mesh.nelems):
        out_mesh.p_elem2nodes[i + 1] = out_mesh.p_elem2nodes[i] + nodes_per_elem
    out_mesh.elem2nodes = numpy.empty(out_mesh.nelems * nodes_per_elem, dtype=numpy.int64)
 
    ## nx*ny quad elements contains nodes: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
    k = 0
    for j in range(0, ny):
        jj = (ny - 1) - j  ## (0,0) of numpy tab in top-left, (0,0) of mesh in bottom-left
        for i in range(0, nx):
            if selection == None or tab[jj, i] == selection:
                out_mesh.elem2nodes[k + 0] = j * (nx + 1) + i
                out_mesh.elem2nodes[k + 1] = j * (nx + 1) + i + 1
                out_mesh.elem2nodes[k + 2] = (j + 1) * (nx + 1) + i + 1
                out_mesh.elem2nodes[k + 3] = (j + 1) * (nx + 1) + i
                k += nodes_per_elem
    out_mesh.elem_type = numpy.empty(out_mesh.nelems, dtype=numpy.int64)
    out_mesh.elem_type[:] = mesh4u._mesh.VTK_QUAD
 
    # coordinates of (nx+1)*(ny+1) nodes of cartesian grid
    node_coords = numpy.empty((nnodes, out_mesh.spacedim), dtype=numpy.float64)
    xmin = 0
    xmax = nx
    ymin = 0
    ymax = ny
    xymax = max([xmax, ymax])
    xmax = _map_ab_to_cd(xmin, xmax, xmin, 1.*xmax/xymax, xmax)
    ymax = _map_ab_to_cd(ymin, ymax, ymin, 1.*ymax/xymax, ymax)
 
    k = 0
    for j in range(0, ny + 1):
        yy = ymin + (j * (ymax - ymin) / ny)
        for i in range(0, nx + 1):
            xx = xmin + (i * (xmax - xmin) / nx)
            node_coords[k, :] = xx, yy, 0.
            k += 1
 
    if selection == None:
        out_mesh.nnodes = (nx + 1) * (ny + 1)
        out_mesh.node_coords = numpy.empty((out_mesh.nnodes, out_mesh.spacedim), dtype=numpy.float64)
        for i in range(0, out_mesh.nnodes):
            out_mesh.node_coords[i, :] = node_coords[i, :]
        # local to global numbering
        out_mesh_l2g = numpy.arange(0, out_mesh.nnodes, 1, dtype=numpy.int64)
        out_mesh.node_l2g = out_mesh_l2g
 
    else:
        node_mask = numpy.zeros(nnodes, dtype=numpy.int64)
        node_mask[out_mesh.elem2nodes] = 1
        node_l2g = numpy.nonzero(node_mask)[0]
 
        # local to global numbering
        out_mesh.node_l2g = node_l2g
 
        out_mesh.nnodes = numpy.count_nonzero(node_mask)
        node_g2l = numpy.zeros(nnodes, dtype=numpy.int64)
        for i in range(0, out_mesh.nnodes):
            node_g2l[node_l2g[i]] = i
 
        for i in range(0, out_mesh.nelems):
            for ii in range(out_mesh.p_elem2nodes[i], out_mesh.p_elem2nodes[i+1]):
                out_mesh.elem2nodes[ii] = node_g2l[out_mesh.elem2nodes[ii]]
 
        out_mesh.node_coords = numpy.empty((out_mesh.nnodes, out_mesh.spacedim), dtype=numpy.float64)
        #out_mesh.node_coords = node_coords[node_g2l,:]
        for i in range(0, out_mesh.nnodes):
            out_mesh.node_coords[i, :] = node_coords[node_l2g[i], :]
 
    return out_mesh, out_mesh.node_l2g
        
def _write_numpy2d_to_file_png(tab, filename):
    """ convert numpy2d to png
    """
    import matplotlib.pyplot
 
    imghsv = _transform_numpy2d_to_hsv(tab)
 
    fig, ax = matplotlib.pyplot.subplots()
    imgrgb = matplotlib.colors.hsv_to_rgb(imghsv)
    img = ax.imshow(imgrgb)
    ax.autoscale()
 
    (root, ext) = os.path.splitext(filename)
    if ext == '.bmp' or ext == '.jpg' or ext == '.png' or ext == 'gif' or ext == 'tiff':
        output_file = root + '_imsavegrey' + ext
        matplotlib.pyplot.imsave(output_file, tab, cmap='Greys')
 
        output_file = root + '_imsaverainbow' + ext
        matplotlib.pyplot.imsave(output_file, tab, cmap='rainbow')
        matplotlib.pyplot.close(fig)
 
    else:
        matplotlib.pyplot.show()
        matplotlib.pyplot.close(fig)
 
    return

def image_to_graph_knn(image_path, patch_size=16, embed_dim=768, k=9):
    """
    Convertit une image en  graphe où chaque nœud représente un patch de l'image.
    Chaque noeud contient un vecteur de caractéristiques dérivé du patch, et des arêtes sont formées
    en fonction de la similarité de ces vecteurs de caractéristiques avec l'algorithme knn

    Paramètres:
    - image_path (str): Chemin vers l'image d'entrée.
    - patch_size (int): Taille des patches pour diviser l'image. La valeur par défaut est 16.
    - embed_dim (int): Dimension du vecteur d'embedding pour chaque patch. La valeur par défaut est 768.
    - k (int): Nombre de voisins les plus proches pour connecter chaque nœud dans le graphe. La valeur par défaut est 9.

    Retourne:
    - x (torch.Tensor): Matrice des caractéristiques des nœuds de taille (num_patches, embed_dim).
    Chaque ligne de cette matrice représente un patch de l'image. Les caractéristiques de chaque patch
    sont extraites à partir des valeurs RGB des pixels dans ce patch.
    - pos (torch.Tensor): Positions des nœuds dans la grille de l'image originale.
    - edge_index (torch.Tensor): Liste des arêtes au format COO.
    """
    # Charger l'image
    image = Image.open(image_path)

    # Prétraiter l'image
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    patches = image_to_patches(image, patch_size)

    # Extraire les vecteurs de caractéristiques des patches
    patch_embedding = PatchEmbedding(patch_size, embed_dim=embed_dim)
    node_features = patch_embedding(image)
    node_features = node_features.squeeze(0).detach().numpy()

    # Générer les positions des nœuds
    num_patches_per_row = int(image.size(2) / patch_size)
    node_coords = np.array([(i // num_patches_per_row, i % num_patches_per_row) for i in range(num_patches_per_row * num_patches_per_row)])

    # Construire le graphe
    adj_matrix = build_graph(node_features, k)
    graph = nx.from_numpy_array(adj_matrix)

    # Convert adjacency matrix to edge_index
    edge_index = np.array(list(graph.edges)).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Convert node features to tensor
    x = torch.tensor(node_features, dtype=torch.float)

    # Convert positions to tensor
    pos = torch.tensor(node_coords, dtype=torch.float)

    return x, pos, edge_index
