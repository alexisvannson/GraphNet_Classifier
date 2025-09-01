from ai_mlp import MLP
from train_model import train, train_with_val_test
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms


def load_data(dataset_path, resize_value=128, batch_size=8):
    transform = transforms.Compose([transforms.Resize((resize_value, resize_value)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_MLP(epochs=30, channels=3, resize_value=128, batch_size=8, hidden_layers=2, output_path='weights/MLP', dataset_path='dataset', show=False, to_save=True):
	input_dim = channels * resize_value * resize_value 

	# Load the dataset directly, not as a DataLoader
	transform = transforms.Compose([transforms.Resize((resize_value, resize_value)), transforms.ToTensor()])
	dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

	num_classes = len(dataset.classes)
	model = MLP(in_dim=input_dim, out_dim=num_classes, hidden_layers=hidden_layers)
	
	train_with_val_test(model, dataset, epochs, patience=5, output_path=output_path, batch_size=batch_size, show=show, to_save=to_save)



def train_GNN(epochs=30,resize_value=64, batch_size=8, n_blocks=2, max_samples=None, output_path='weights/GNN',dataset_path='data/mnist/test', patience=5, grayscale=False):
	# Local import to avoid import-time segfault from torch_scatter
	from ai_gnn import CombinedModel, GraphNet
	# Graph dataset produces tuples (x, pos, edge_index), label
	from datasets import OptimizedDatasetLoader
	
	# Use optimized dataset loader with caching 
	original_dataset = OptimizedDatasetLoader(
		dataset_path=dataset_path, 
		resize_value=resize_value,
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
	
	num_nodes = resize_value * resize_value
	
	# Set num_local_features based on grayscale parameter
	num_local_features = 1 if grayscale else 3
	
	graph_net = GraphNet(num_local_features=num_local_features, space_dim=2, out_channels=1, n_blocks=n_blocks)
	model = CombinedModel(graph_net=graph_net, num_nodes=num_nodes, classes=num_classes)
	
	train(model, dataloader, epochs, patience=patience, output_path=output_path)


if __name__ == '__main__':
	train_MLP(epochs=100, resize_value=28, hidden_layers=5,dataset_path='data/mnist/test', output_path='weights/MLP/test3_mtx', show=True, to_save=True)
	#train_GNN(epochs=100, resize_value=28, n_blocks=10, dataset_path='data/mnist/test', output_path='weights/GNN/dim28_10hidden_dim', grayscale=True)