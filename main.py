from models.MLP import MLP
from torch.utils.data import DataLoader
import torch
import torchvision.datasets as datasets
from torchvision import transforms

from models.GNN import CombinedModel, GraphNet
from utils.train_model import train


def load_data(dataset_path, resize_value=128, batch_size=8):
	transform = transforms.Compose([transforms.Resize((resize_value, resize_value)), transforms.ToTensor()])

	dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return dataloader


def train_MLP(epochs=30, channels=3, resize_value=128, batch_size=8, hidden_layers=2):
	input_dim = channels * resize_value * resize_value 

	dataset = load_data('dataset', resize_value, batch_size)

	num_classes = len(dataset.dataset.classes)
	model = MLP(in_dim=input_dim, out_dim=num_classes, hidden_layers=hidden_layers)
	
	train(model, dataset, epochs)


def train_GNN(epochs=30, channels=3, resize_value=64, batch_size=8, hidden_layers=2, max_samples=None, method='pixel', use_cache=True):
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
	train(model, dataloader, epochs)
	


if __name__ == '__main__':
	# Train GNN model with optimized preprocessing
	# Choose one of these methods:
	
	# Method 1: Optimized pixel-based (fastest)
	#train_GNN(epochs=20, hidden_layers=2, method='pixel', use_cache=True)
	
	# Method 2: Superpixel-based (reduces graph size significantly)
	train_GNN(epochs=20, hidden_layers=2, max_samples=None, method='pixel')
	
	# Method 3: Patch-based (good balance)
	# train_GNN(epochs=20, hidden_layers=2, max_samples=50, method='patch')
	
	# Example inference (uncomment after training completes and weights are saved)
	#test_image = 'dataset/chihuahua/img_0_1071.jpg'  # Use any image from your dataset
	#inference_GNN(test_image,weights='weights/GNN/best_model_epoch2.pth')
	#inference_MLP(test_image)
	#compare_models(test_image)