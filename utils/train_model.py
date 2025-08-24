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
