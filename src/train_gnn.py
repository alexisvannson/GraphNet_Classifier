from utils.train_model import train
from models.GNN import CombinedModel, GraphNet
from torch.utils.data import DataLoader


def train_GNN(epochs=30, channels=3, resize_value=64, batch_size=8, hidden_layers=2, max_samples=None, method='pixel', use_cache=True,output_path='weights/GNN',dataset_path='dataset',patience=5, grayscale=False):
	# Local import to avoid import-time segfault from torch_scatter
	from models.GNN import CombinedModel, GraphNet
	# Graph dataset produces tuples (x, pos, edge_index), label
	from utils.dataloader import OptimizedDatasetLoader
	
	# Use optimized dataset loader with caching
	original_dataset = OptimizedDatasetLoader(
		dataset_path=dataset_path, 
		resize_value=resize_value,
		method=method,  # 'pixel', 'superpixel', or 'patch'
		use_cache=use_cache,
		grayscale=grayscale
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
	
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: batch[0])
	
	# Adjust num_nodes based on method
	if method == 'pixel':
		num_nodes = resize_value * resize_value
	elif method == 'superpixel':
		num_nodes = resize_value // 2  # Approximate number of superpixels
	elif method == 'patch':
		num_nodes = (resize_value // 8) ** 2  # Approximate number of patches
	else:
		num_nodes = resize_value * resize_value
	
	# Set num_local_features based on grayscale parameter
	num_local_features = 1 if grayscale else 3
	
	graph_net = GraphNet(num_local_features=num_local_features, space_dim=2, out_channels=1, n_blocks=3)
	model = CombinedModel(graph_net=graph_net, num_nodes=num_nodes, classes=num_classes)
	
	print(f"Training GNN with {method} method, {num_nodes} nodes, {num_local_features} local features")
	train(model, dataloader, epochs, patience=patience, output_path=output_path)

