from models.MLP import MLP
from torch.utils.data import DataLoader
import torch
import torchvision.datasets as datasets
from torchvision import transforms

# Deferred GNN import to avoid import-time segfaults
from utils.train_model import train

from utils.inference import inference_MLP


def load_data(dataset_path, resize_value=128, batch_size=8):
	transform = transforms.Compose([transforms.Resize((resize_value, resize_value)), transforms.ToTensor()])

	dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return dataloader


def train_MLP(epochs=30, channels=3, resize_value=128, batch_size=8, hidden_layers=2,output_path='weights/MLP'):
	input_dim = channels * resize_value * resize_value 

	dataset = load_data('dataset', resize_value, batch_size)

	num_classes = len(dataset.dataset.classes)
	model = MLP(in_dim=input_dim, out_dim=num_classes, hidden_layers=hidden_layers)
	
	train(model, dataset, epochs, output_path)


def train_GNN(epochs=30, channels=3, resize_value=64, batch_size=8, hidden_layers=2, max_samples=None, method='pixel', use_cache=True,output_path='weights/GNN'):
	# Local import to avoid import-time segfault from torch_scatter
	from models.GNN import CombinedModel, GraphNet
	# Graph dataset produces tuples (x, pos, edge_index), label
	from utils.dataloader import OptimizedDatasetLoader
	
	# Use optimized dataset loader with caching
	original_dataset = OptimizedDatasetLoader(
		dataset_path='dataset', 
		resize_value=resize_value,
		method=method,  # 'pixel', 'superpixel', or 'patch'
		use_cache=use_cache
	)
	
	# Get number of classes before potentially creating subset
	num_classes = len(original_dataset.dataset.classes)
	
	# Limit dataset size for faster testing
	if max_samples and max_samples < len(original_dataset):
		from torch.utils.data import Subset
		import random
		random.seed(42)  # For reproducibility
		indices = random.sample(range(len(original_dataset)), max_samples)
		dataset = Subset(original_dataset, indices)
		print(f"Using subset of {max_samples} samples for faster training")
	else:
		dataset = original_dataset
	
	dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])
	
	# Adjust num_nodes based on method
	if method == 'pixel':
		num_nodes = resize_value * resize_value
	elif method == 'superpixel':
		num_nodes = resize_value // 2  # Approximate number of superpixels
	elif method == 'patch':
		num_nodes = (resize_value // 8) ** 2  # Approximate number of patches
	else:
		num_nodes = resize_value * resize_value
	
	graph_net = GraphNet(num_local_features=3, space_dim=2, out_channels=1, n_blocks=3)
	model = CombinedModel(graph_net=graph_net, num_nodes=num_nodes, classes=num_classes)
	
	print(f"Training GNN with {method} method, {num_nodes} nodes")
	train(model, dataloader, epochs, method, output_path)
	

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


if __name__ == '__main__':
	print('start')
	# Example GNN inference (ensure weights exist under weights/GNN/)
	gnn_image = 'dataset/chihuahua/img_0_8.jpg'
	gnn_weights = 'weights/final_model.pth'
	gnn_inference(gnn_image, gnn_weights, resize_value=64)
	#train_GNN(epochs=100)