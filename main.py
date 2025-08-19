from models.MLP import MLP
from torch.utils.data import DataLoader
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchinfo import summary
import os
import matplotlib.pyplot as plt
from PIL import Image

from models.GNN import CombinedModel, GraphNet
from utils.image_to_graph import DatasetLoader
from utils.train_model import train


def load_data(dataset_path, resize_value=128):
	transform = transforms.Compose([transforms.Resize((resize_value, resize_value)),
									transforms.ToTensor()])

	dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
	dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
	return dataloader


def train_MLP(epochs=30, channels=3, resize_value=128, batch_size=8, hidden_layers=2):
	
	input_dim = channels * resize_value * resize_value 

	dataset = load_data('dataset', resize_value)

	num_classes = len(dataset.dataset.classes)
	model = MLP(in_dim=input_dim, out_dim=num_classes,hidden_layers=hidden_layers)
	
	train(model, dataset, epochs)


def train_GNN(epochs=30, channels=3, resize_value=64, batch_size=8, hidden_layers=2, max_samples=None):  # Reduced from 128 to 64
	# Graph dataset produces tuples (x, pos, edge_index), label
	original_dataset = DatasetLoader(resize_value=resize_value)
	
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
	graph_net = GraphNet(num_local_features=3, space_dim=2, out_channels=1, n_blocks=3)  # Reduced from 10 to 3 blocks
	model = CombinedModel(graph_net=graph_net, num_nodes=resize_value*resize_value, classes=num_classes)
	# print(summary(model))  # Optional, can be large
	train(model, dataloader, epochs)
	

def inference_MLP(image_path, input_dim=128*128*3, num_classes=2, weights='weights/final_model.pth', output_folder='predictions', resize_value=128):
	"""
	Run MLP inference on a single image.
	
	Args:
		image_path (str): Path to the input image
		input_dim (int): Input dimension for the model
		num_classes (int): Number of classes
		weights (str): Path to the trained model weights
		output_folder (str): Folder to save predictions
		resize_value (int): Image resize dimension
	"""
	os.makedirs(output_folder, exist_ok=True)

	model = MLP(in_dim=input_dim, out_dim=num_classes)
	model.load_state_dict(torch.load(weights, map_location='cpu'))
	model.eval()

	with torch.no_grad():
		# Load and preprocess image
		image = Image.open(image_path).convert('RGB')
		resized_image = image.resize((resize_value, resize_value))
		
		# Convert to tensor and normalize
		transform = transforms.Compose([
			transforms.ToTensor(),
		])
		input_tensor = transform(resized_image).unsqueeze(0)  # Add batch dimension
		
		# Run inference
		output = model(input_tensor)
		probabilities = torch.softmax(output, dim=1)
		predicted_class = torch.argmax(probabilities, dim=1).item()
		confidence = probabilities[0][predicted_class].item()
		
		# Get class names
		dataset = datasets.ImageFolder(root='dataset')
		class_names = dataset.classes
		
		print(f"MLP Prediction: {class_names[predicted_class]} (confidence: {confidence:.3f})")
		
		# Save prediction visualization
		plt.figure(figsize=(10, 5))
		
		plt.subplot(1, 2, 1)
		plt.imshow(resized_image)
		plt.title(f"Input Image")
		plt.axis('off')
		
		plt.subplot(1, 2, 2)
		plt.bar(range(len(class_names)), probabilities[0].numpy())
		plt.title(f"MLP Predictions")
		plt.xlabel("Classes")
		plt.ylabel("Probability")
		plt.xticks(range(len(class_names)), class_names, rotation=45)
		
		# Save the plot
		image_name = os.path.basename(image_path).split('.')[0]
		output_path = os.path.join(output_folder, f"mlp_prediction_{image_name}.png")
		plt.tight_layout()
		plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
		plt.close()
		
		print(f"MLP prediction saved to: {output_path}")
		return predicted_class, confidence


def inference_GNN(image_path, resize_value=64, num_classes=2, weights='weights/final_model.pth', output_folder='predictions'):
	"""
	Run GNN inference on a single image.
	
	Args:
		image_path (str): Path to the input image
		resize_value (int): Image resize dimension (should match training)
		num_classes (int): Number of classes
		weights (str): Path to the trained model weights
		output_folder (str): Folder to save predictions
	"""
	os.makedirs(output_folder, exist_ok=True)
	
	# Load and preprocess image to graph
	image = Image.open(image_path).convert('RGB')
	resized_image = image.resize((resize_value, resize_value))
	
	# Convert to graph using the same function as training
	from utils.image_to_graph import image_to_graph_pixel
	x, pos, edge_index = image_to_graph_pixel(resized_image, resize_value=resize_value)
	
	# Add batch dimension
	x = x.unsqueeze(0)  # (1, num_nodes, features)
	pos = pos.unsqueeze(0)  # (1, num_nodes, 2)
	edge_index = edge_index.unsqueeze(0)  # (1, 2, num_edges)
	
	# Create model (same architecture as training)
	graph_net = GraphNet(num_local_features=3, space_dim=2, out_channels=1, n_blocks=3)
	model = CombinedModel(graph_net=graph_net, num_nodes=resize_value*resize_value, classes=num_classes)
	
	# Load trained weights
	model.load_state_dict(torch.load(weights, map_location='cpu'))
	model.eval()
	
	with torch.no_grad():
		# Run inference
		output = model((x, pos, edge_index))
		probabilities = torch.softmax(output, dim=1)
		predicted_class = torch.argmax(probabilities, dim=1).item()
		confidence = probabilities[0][predicted_class].item()
		
		# Get class names
		dataset = datasets.ImageFolder(root='dataset')
		class_names = dataset.classes
		
		print(f"GNN Prediction: {class_names[predicted_class]} (confidence: {confidence:.3f})")
		
		# Save prediction visualization
		plt.figure(figsize=(15, 5))
		
		plt.subplot(1, 3, 1)
		plt.imshow(resized_image)
		plt.title(f"Input Image ({resize_value}x{resize_value})")
		plt.axis('off')
		
		plt.subplot(1, 3, 2)
		# Visualize graph structure (simplified)
		plt.scatter(pos[0, :, 1].numpy(), pos[0, :, 0].numpy(), c=x[0, :, 0].numpy(), s=1, alpha=0.7)
		plt.title("Graph Nodes (R channel)")
		plt.axis('equal')
		plt.colorbar()
		
		plt.subplot(1, 3, 3)
		plt.bar(range(len(class_names)), probabilities[0].numpy())
		plt.title(f"GNN Predictions")
		plt.xlabel("Classes")
		plt.ylabel("Probability")
		plt.xticks(range(len(class_names)), class_names, rotation=45)
		
		# Save the plot
		image_name = os.path.basename(image_path).split('.')[0]
		output_path = os.path.join(output_folder, f"gnn_prediction_{image_name}.png")
		plt.tight_layout()
		plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
		plt.close()
		
		print(f"GNN prediction saved to: {output_path}")
		return predicted_class, confidence


def compare_models(image_path, resize_value=64, num_classes=2, mlp_weights='weights/final_model.pth', gnn_weights='weights/final_model.pth', output_folder='predictions'):
	"""
	Compare MLP and GNN predictions on the same image.
	
	Args:
		image_path (str): Path to the input image
		resize_value (int): Image resize dimension
		num_classes (int): Number of classes
		mlp_weights (str): Path to MLP model weights
		gnn_weights (str): Path to GNN model weights
		output_folder (str): Folder to save comparison
	"""
	os.makedirs(output_folder, exist_ok=True)
	
	print(f"Comparing MLP vs GNN on: {image_path}")
	print("-" * 50)
	
	# Run both models
	mlp_class, mlp_conf = inference_MLP(image_path, input_dim=resize_value*resize_value*3, num_classes=num_classes, 
									   weights=mlp_weights, output_folder=output_folder, resize_value=resize_value)
	
	gnn_class, gnn_conf = inference_GNN(image_path, resize_value=resize_value, num_classes=num_classes, 
									   weights=gnn_weights, output_folder=output_folder)
	
	# Get class names
	dataset = datasets.ImageFolder(root='dataset')
	class_names = dataset.classes
	
	print("-" * 50)
	print("COMPARISON RESULTS:")
	print(f"MLP:  {class_names[mlp_class]} (confidence: {mlp_conf:.3f})")
	print(f"GNN:  {class_names[gnn_class]} (confidence: {gnn_conf:.3f})")
	
	if mlp_class == gnn_class:
		print("✅ Both models agree!")
	else:
		print("❌ Models disagree!")
	
	# Create comparison plot
	plt.figure(figsize=(12, 8))
	
	plt.subplot(2, 3, 1)
	image = Image.open(image_path).convert('RGB')
	resized_image = image.resize((resize_value, resize_value))
	plt.imshow(resized_image)
	plt.title("Input Image")
	plt.axis('off')
	
	plt.subplot(2, 3, 2)
	plt.bar(['MLP', 'GNN'], [mlp_conf, gnn_conf], color=['blue', 'orange'])
	plt.title("Confidence Comparison")
	plt.ylabel("Confidence")
	plt.ylim(0, 1)
	
	plt.subplot(2, 3, 3)
	plt.bar(['MLP', 'GNN'], [mlp_class, gnn_class], color=['blue', 'orange'])
	plt.title("Predicted Class")
	plt.ylabel("Class Index")
	plt.yticks(range(len(class_names)), class_names)
	
	# Add text summary
	plt.subplot(2, 3, 4)
	plt.text(0.1, 0.8, f"MLP: {class_names[mlp_class]}\nConfidence: {mlp_conf:.3f}", fontsize=12, transform=plt.gca().transAxes)
	plt.text(0.1, 0.4, f"GNN: {class_names[gnn_class]}\nConfidence: {gnn_conf:.3f}", fontsize=12, transform=plt.gca().transAxes)
	plt.axis('off')
	
	# Save comparison
	image_name = os.path.basename(image_path).split('.')[0]
	comparison_path = os.path.join(output_folder, f"comparison_{image_name}.png")
	plt.tight_layout()
	plt.savefig(comparison_path, bbox_inches='tight', pad_inches=0)
	plt.close()
	
	print(f"Comparison saved to: {comparison_path}")
	return mlp_class, mlp_conf, gnn_class, gnn_conf


if __name__ == '__main__':
	# Train GNN model
	train_GNN(epochs=20, hidden_layers=2, max_samples=50)  
	
	# Example inference (uncomment after training completes and weights are saved)
	# test_image = 'dataset/chihuahua/img_0_1071.jpg'  # Use any image from your dataset
	# inference_GNN(test_image)
	# inference_MLP(test_image)
	# compare_models(test_image)