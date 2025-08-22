import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

# from image_to_graph.image_to_graph_optimized import image_to_graph_pixel_optimized
from models.MLP import MLP
from models.GNN import GraphNet, CombinedModel
from utils.image_to_graph.image_to_graph_optimized import image_to_graph_pixel_optimized


def mlp_inference(image_path, weights='weights/MLP/final_model_.pth', resize_value=128):
    in_dim = resize_value * resize_value * 3  # RGB image
    out_dim = 2  # Assuming binary classification (chihuahua vs muffin)
    model = MLP(in_dim=in_dim, out_dim=out_dim)
    model.load_state_dict(torch.load(weights))
    model.eval()
    
    with torch.no_grad():
        img = Image.open(image_path).resize((resize_value, resize_value)).convert('RGB')
        img_array = np.array(img)
        input_tensor = torch.tensor(img_array.flatten(), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        output = model(input_tensor)
    return output


def gnn_inference(image_path: str, weights_path: str = 'weights/GNN/best_model_epoch2.pth', resize_value: int = 64):
	"""Run a single-image GNN inference with lazy imports to avoid segfaults in misconfigured hosts."""
	from utils.image_to_graph.image_to_graph_optimized import image_to_graph_pixel_optimized
	try:
		from models.GNN import CombinedModel, GraphNet
	except Exception as e:
		print('Failed to import GNN modules. Run inside Docker (see README). Error:', e)
		return
	graph_net = GraphNet(num_local_features=3, space_dim=2, out_channels=1, n_blocks=3)
	model = CombinedModel(graph_net=graph_net, num_nodes=resize_value * resize_value, classes=2)
	# Load weights
	state = torch.load(weights_path, map_location='cpu')
	model.load_state_dict(state)
	model.eval()
	with torch.no_grad():
		graph_tuple = image_to_graph_pixel_optimized(image_path, resize_value=resize_value)
		logits = model(graph_tuple)
		print('GNN logits:', logits)
