import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
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
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
    return logits, probabilities


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
		x, pos, edge_index = image_to_graph_pixel_optimized(image_path, resize_value=resize_value)
		# Convert numpy arrays to PyTorch tensors
		x = torch.tensor(x, dtype=torch.float32)
		pos = torch.tensor(pos, dtype=torch.float32)
		edge_index = torch.tensor(edge_index, dtype=torch.long)
		
		# Debug: print input shapes and values
		print(f'Input x shape: {x.shape}, x range: [{x.min():.3f}, {x.max():.3f}]')
		print(f'Input pos shape: {pos.shape}, pos range: [{pos.min():.3f}, {pos.max():.3f}]')
		print(f'Input edge_index shape: {edge_index.shape}, edge_index range: [{edge_index.min()}, {edge_index.max()}]')
		
		graph_tuple = (x, pos, edge_index)
		logits = model(graph_tuple)
		
		# Debug: print logits shape
		print(f'Logits shape: {logits.shape}')
		
		# Ensure logits has the right shape for softmax
		if logits.dim() == 1:
			logits = logits.unsqueeze(0)  # Add batch dimension if missing
		
		probabilities = F.softmax(logits, dim=-1)  # Use dim=-1 instead of dim=1
		print('GNN logits:', logits)
		print('GNN probabilities:', probabilities)
		return logits, probabilities

print(gnn_inference('dataset/chihuahua/img_0_8.jpg', weights_path='weights/GNN/pixel_dim64_3block/final_model.pth', resize_value=64))
print(gnn_inference('dataset/muffin/img_0_31.jpg.jpg', weights_path='weights/GNN/pixel_dim64_3block/final_model.pth', resize_value=64))
print(gnn_inference('dataset/chihuahua/img_0_137.jpg', weights_path='weights/GNN/pixel_dim64_3block/final_model.pth', resize_value=64))
print(gnn_inference('dataset/muffin/img_0_431.jpg', weights_path='weights/GNN/pixel_dim64_3block/final_model.pth', resize_value=64))