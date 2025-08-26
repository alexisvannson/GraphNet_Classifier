from models.MLP import MLP
from torch.utils.data import DataLoader
import torch
import torchvision.datasets as datasets
from torchvision import transforms

from utils.train_model import train


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



if __name__ == '__main__':
	print('start')
	# Example GNN inference (ensure weights exist under weights/GNN/)
	#train_MLP(epochs=100, resize_value=128,hidden_layers=5, output_path='weights/MLP/dim128_5hidden_dim')
	#train_MLP(epochs=100, resize_value=28,hidden_layers=5, output_path='weights/MLP/MNIST/dim28_5hidden_dim')
	
