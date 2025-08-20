import torch
import torch.nn as nn
import torch.optim as optim
import os

def train(model, dataset, epochs, method="", patience=5, output_path='weights'):
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()
	best_loss = float('inf')
	patience_counter = 0
	
	for epoch in range(epochs):
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
		
		# Early stopping
		if avg_loss < best_loss:
			best_loss = avg_loss
			patience_counter = 0
			os.makedirs(output_path, exist_ok=True)
			torch.save(model.state_dict(), os.path.join(output_path, f'best_model_{method}_epoch{epoch+1}.pth'))
		else:
			patience_counter += 1
			
		if patience_counter >= patience:
			print(f"Early stopping at epoch {epoch+1}")
			break
	os.makedirs(output_path, exist_ok=True)
	torch.save(model.state_dict(), os.path.join(output_path, f'final_model_{method}.pth'))
