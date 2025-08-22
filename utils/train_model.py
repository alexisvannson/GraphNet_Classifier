import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

def train(model, dataset, epochs, method="", patience=5, output_path='weights', start_weights=None):
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()
	best_loss = float('inf')
	patience_counter = 0
	
	if start_weights:
		model.load_state_dict(torch.load(start_weights))
	
	os.makedirs(output_path, exist_ok=True)

	for epoch in range(epochs):
		checkpoint1 = time.time()
		epoch_loss = 0
		num_batches = 0
		for sample, label in dataset:
			# Support both MLP and GNN inputs
			if isinstance(sample, tuple):
				# Graph data: (x, pos, edge_index)
				logits = model(sample)
			else:
				# Image tensor for MLP
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
		print(f"epoch: {epoch} needed {checkpoint2 - checkpoint1} time")
		with open(os.path.join('output_pth', 'trainning_logs.txt'), "a") as the_file:
			the_file.write(f"Epoch {epoch+1}/{epochs}, avg_loss={avg_loss:.4f}\n")
			the_file.write(f"Epoch {epoch+1}/{epochs}, needed {(checkpoint2 - checkpoint1) / 60:.2f} minutes\n")
		
		# Early stopping
		if avg_loss < best_loss:
			best_loss = avg_loss
			patience_counter = 0
			torch.save(model.state_dict(), os.path.join(output_path, f'best_model_{method}_epoch{epoch+1}.pth'))
		else:
			patience_counter += 1
			
		if patience_counter >= patience:
			print(f"Early stopping at epoch {epoch+1}")
			break
		
	torch.save(model.state_dict(), os.path.join(output_path, f'final_model_{method}.pth'))
