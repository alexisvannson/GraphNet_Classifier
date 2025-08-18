from PIL import Image
import numpy as np



def image_to_graph_pixel(image_path, resize_value=128, diagonals=False):
    """
    Convert an image into a pixel graph representation.

    Args:
        image_path (str): Path to the input image file.
        resize_value (int, optional): Image will be resized to (resize_value x resize_value). Default = 128.

    Returns:
        x (torch.FloatTensor): 
            Node features of shape (num_nodes, num_features).  
            - num_nodes = H * W (pixels)  
            - num_features = C (channels, e.g. 3 for RGB)  
            - Each row = pixel values (R,G,B).

        pos (torch.FloatTensor): 
            Node positions of shape (num_nodes, 2).  
            - Each row = (row_index, col_index) of the pixel in the grid.

        edge_index (torch.LongTensor): 
            Graph connectivity of shape (2, num_edges).  
            - Each column = (source_node, target_node)  
            - Pixels are connected to their 4-neighbors (up, down, left, right).
    """
    image = Image.open(image_path).convert('RGB')
    resized_image = image.resize((resize_value, resize_value))

    tab = np.array(resized_image)
    
    C, H, W = tab.shape # (3, 128, 128) par défaut
    
    # Node features: flatten pixel values => shape: (num_nodes, num_features)
    x = np.transpose(tab.reshape(C, H*W))
    
    # Node positions: (row, col) coordinates
    pos = [[i // W, i % W] for i in range(H*W)]
    
    # Edge index: connect each pixel to its 4 neighbors (up, down, left, right)
    edges = []
    duplicate = lambda values, x, y: True if (x, y) in values or (y, x) in values else  False
    for row in range(H):
        for col in range(W):
            idx = row * W + col
            if (row > 0) and not duplicate(edges, idx,(row-1)*W + col):     # up
                edges.append([idx, (row-1)*W + col])
            if row < H-1 and not duplicate(edges, idx,(row+1)*W + col):     # down
                edges.append([idx, (row+1)*W + col])
            if col > 0 and not duplicate(edges, idx,row*W + (col-1)):       # left
                edges.append([idx, row*W + (col-1)])
            if col < W-1 and not duplicate(edges, idx,row*W + (col+1)):     # right
                edges.append([idx, row*W + (col+1)])
                
             # we can also add the diagonals if diagonals is set to True
            if diagonals:
                if row > 0 and col > 0 and not duplicate(edges, idx,(row-1)*W + (col-1)):              # up-left
                    edges.append([idx, (row-1)*W + (col-1)])
                if row > 0 and col < W-1 and not duplicate(edges, idx,(row-1)*W + (col+1)):            # up-right
                    edges.append([idx, (row-1)*W + (col+1)])
                if row < H-1 and col > 0 and not duplicate(edges, idx,(row+1)*W + (col-1)):             # down-left
                    edges.append([idx, (row+1)*W + (col-1)])
                if row < H-1 and col < W-1 and not duplicate(edges, idx,(row+1)*W + (col+1)):           # down-right
                    edges.append([idx, (row+1)*W + (col+1)])
    
    edge_index = np.transpose(edges)  # shape: (2, num_edges)
    
    return x, pos, edge_index

