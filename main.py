from models.MLP import MLP
from torch.utils.data import DataLoader
import torch
import torchvision.datasets as datasets
from torchvision import transforms

# Deferred GNN import to avoid import-time segfaults
from utils.train_model import train

#from utils.inference import mlp_inference, gnn_inference


def load_data(dataset_path, resize_value=128, batch_size=8):
	transform = transforms.Compose([transforms.Resize((resize_value, resize_value)), transforms.ToTensor()])

	dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return dataloader


def train_MLP(epochs=30, channels=3, resize_value=128, batch_size=8, hidden_layers=2, output_path='weights/MLP'):
	input_dim = channels * resize_value * resize_value 

	dataset = load_data('dataset', resize_value, batch_size)

	num_classes = len(dataset.dataset.classes)
	model = MLP(in_dim=input_dim, out_dim=num_classes, hidden_layers=hidden_layers)
	
	train(model, dataset, epochs, patience=5, output_path=output_path)


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
	train(model, dataloader, epochs, patience=5, output_path=output_path)

if __name__ == '__main__':
	print('start')
	# Example GNN inference (ensure weights exist under weights/GNN/)
	#train_MLP(epochs=100, resize_value=128,hidden_layers=5, output_path='weights/MLP/dim128_5hidden_dim')
	train_GNN(epochs=100, resize_value=128, output_path='weights/GNN/dim128_3block')
