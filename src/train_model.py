import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from datetime import datetime
from tqdm import tqdm

def train(model, dataset, epochs, patience=5, output_path='weights', start_weights=None):
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()
	best_loss = float('inf')
	patience_counter = 0
	
	if start_weights:
		model.load_state_dict(torch.load(start_weights))
	
	# Create the full output path directory structure
	os.makedirs(output_path, exist_ok=True)
	print(f"Training model in {output_path}")
	
	# Create a timestamped log file for this training run
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_path = os.path.join(output_path, f'training_logs_{timestamp}.txt')
	
	# Write training start info
	with open(log_path, "w") as the_file:
		the_file.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
		the_file.write(f"Epochs: {epochs}, Patience: {patience}\n")
		the_file.write(f"Output path: {output_path}\n")
		the_file.write("-" * 50 + "\n")
	for epoch in range(epochs):
		checkpoint1 = time.time()
		epoch_loss = 0
		num_batches = 0
		for sample, label in tqdm(dataset):
			# tensor for MLP, (x, pos, edge_index) for GNN
			logits = model(sample)
			loss = criterion(logits, label)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()
			num_batches += 1
		
		avg_loss = epoch_loss / max(1, num_batches)
		print(f"Epoch {epoch+1}/{epochs}, avg_loss={avg_loss:.4f}")
		checkpoint2 = time.time()
		print(f"epoch: {epoch + 1} needed {checkpoint2 - checkpoint1} time")
		# Save training logs in the same directory as the weights
		with open(log_path, "a") as the_file:
			the_file.write(f"Epoch {epoch+1}/{epochs}, avg_loss={avg_loss:.4f}\n")
			the_file.write(f"Epoch {epoch+1}/{epochs}, needed {(checkpoint2 - checkpoint1) / 60:.2f} minutes\n")
		
		# Early stopping
		if avg_loss < best_loss:
			best_loss = avg_loss
			patience_counter = 0
			# Save best model in the same directory as final model
			best_model_path = os.path.join(output_path, f'best_model_epoch{epoch+1}.pth')
			torch.save(model.state_dict(), best_model_path)
			print(f"Saved best model: {best_model_path}")
		else:
			patience_counter += 1
			
		if patience_counter >= patience:
			print(f"Early stopping at epoch {epoch+1}")
			break
		
	# Save final model in the same directory as best models
	final_model_path = os.path.join(output_path, f'final_model.pth')
	torch.save(model.state_dict(), final_model_path)
	print(f"Saved final model: {final_model_path}")
	
	# Write training completion info to log
	with open(log_path, "a") as the_file:
		the_file.write("-" * 50 + "\n")
		the_file.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
		the_file.write(f"Best loss achieved: {best_loss:.4f}\n")
		the_file.write(f"Final model saved: {final_model_path}\n")


def train_with_val_test(model, dataset, epochs=30, patience=5, output_path='weights', log_path='train_log.txt', val_ratio=0.1, test_ratio=0.1, criterion=None, optimizer=None, random_seed=42):
	"""
	Train a model with train/validation/test split (80/10/10), save losses per epoch, and plot/save loss curves.
	"""
	import torch
	import os
	import numpy as np
	from datetime import datetime
	import random
	import matplotlib.pyplot as plt
	from torch.utils.data import Subset, DataLoader

	# Set random seed for reproducibility
	random.seed(random_seed)
	np.random.seed(random_seed)
	torch.manual_seed(random_seed)

	# Split dataset indices
	n_total = len(dataset)
	indices = list(range(n_total))
	random.shuffle(indices)
	n_test = int(test_ratio * n_total)
	n_val = int(val_ratio * n_total)
	n_train = n_total - n_val - n_test

	train_indices = indices[:n_train]
	val_indices = indices[n_train:n_train+n_val]
	test_indices = indices[n_train+n_val:]

	train_set = Subset(dataset, train_indices)
	val_set = Subset(dataset, val_indices)
	test_set = Subset(dataset, test_indices)

	# Use same batch size as original dataset if possible
	batch_size = getattr(dataset, 'batch_size', 8)
	# If dataset is not a DataLoader, default to 8
	if not isinstance(dataset, DataLoader):
		batch_size = 8

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

	# Prepare for training
	if criterion is None:
		from lossfunction import get_loss_function
		criterion = get_loss_function("cross_entropy")
	if optimizer is None:
		import torch.optim as optim
		optimizer = optim.Adam(model.parameters(), lr=1e-3)

	best_val_loss = float('inf')
	patience_counter = 0
	train_losses = []
	val_losses = []

	if not os.path.exists(output_path):
		os.makedirs(output_path)
	log_path = os.path.join(output_path, "train_val_log.txt")

	for epoch in range(epochs):
		model.train()
		epoch_train_loss = 0.0
		num_train_batches = 0
		for sample, label in train_loader:
			logits = model(sample)
			loss = criterion(logits, label)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			epoch_train_loss += loss.item()
			num_train_batches += 1
		avg_train_loss = epoch_train_loss / max(1, num_train_batches)
		train_losses.append(avg_train_loss)

		# Validation
		model.eval()
		epoch_val_loss = 0.0
		num_val_batches = 0
		with torch.no_grad():
			for sample, label in val_loader:
				logits = model(sample)
				loss = criterion(logits, label)
				epoch_val_loss += loss.item()
				num_val_batches += 1
		avg_val_loss = epoch_val_loss / max(1, num_val_batches)
		val_losses.append(avg_val_loss)

		# Logging
		with open(log_path, "a") as the_file:
			the_file.write(f"Epoch {epoch+1}/{epochs}, train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}\n")

		print(f"Epoch {epoch+1}/{epochs}, train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

		# Early stopping on validation loss
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			patience_counter = 0
			best_model_path = os.path.join(output_path, f'best_model_val_epoch{epoch+1}.pth')
			torch.save(model.state_dict(), best_model_path)
			print(f"Saved best model (val): {best_model_path}")
		else:
			patience_counter += 1

		if patience_counter >= patience:
			print(f"Early stopping at epoch {epoch+1} (val loss)")
			break

	# Save final model
	final_model_path = os.path.join(output_path, f'final_model_val.pth')
	torch.save(model.state_dict(), final_model_path)
	print(f"Saved final model: {final_model_path}")

	# Save losses as .mtx files (Matrix Market format)
	train_loss_arr = np.array(train_losses).reshape(-1, 1)
	val_loss_arr = np.array(val_losses).reshape(-1, 1)
	np.savetxt(os.path.join(output_path, "train_loss.mtx"), train_loss_arr, fmt="%.6f")
	np.savetxt(os.path.join(output_path, "val_loss.mtx"), val_loss_arr, fmt="%.6f")

	# Plot losses and save as .png and .mtx
	plt.figure()
	plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
	plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Training and Validation Loss")
	plt.legend()
	plt.grid(True)
	plot_path = os.path.join(output_path, "loss_curve.png")
	plt.savefig(plot_path)
	plt.close()

	# Save loss values for plotting in .mtx (2 columns: train, val)
	loss_matrix = np.column_stack([train_loss_arr, val_loss_arr])
	np.savetxt(os.path.join(output_path, "loss_curve.mtx"), loss_matrix, fmt="%.6f")

	# Test set evaluation
	model.eval()
	test_loss = 0.0
	num_test_batches = 0
	with torch.no_grad():
		for sample, label in test_loader:
			logits = model(sample)
			loss = criterion(logits, label)
			test_loss += loss.item()
			num_test_batches += 1
	avg_test_loss = test_loss / max(1, num_test_batches)
	with open(log_path, "a") as the_file:
		the_file.write(f"Test loss: {avg_test_loss:.4f}\n")
	print(f"Test loss: {avg_test_loss:.4f}")
