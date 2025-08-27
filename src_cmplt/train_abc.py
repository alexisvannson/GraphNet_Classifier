import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gen_excel import ImageRegressionDataset
from src.ai_mlp import MLP
import numpy as np
from gen_excel import generate_excel, generate_all_images_from_excel


def main_training(training_data='images'):
    dataset = ImageRegressionDataset('params.xlsx',training_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    patience = 5
    best = 1e5
    for epoch in range(100):
        for inputs, targets in dataloader:
            targets = targets.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if loss > best:
                patience -= 1
        else:
            patience = 5
            best = loss
        if patience == 0:
            torch.save(model.state_dict(), f'mlp_weights_epoch{epoch+1}.pth')
            break

        print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), 'mlp_weights.pth')

    if __name__ == "__main__":
        a_vals = np.linspace(1, 3, 10)
        b_vals = np.linspace(10, 30, 10)
        c_vals = np.linspace(1, 3, 10)
        param_list = [(a, b, c) for a in a_vals for b in b_vals for c in c_vals]
        generate_excel(param_list)
        generate_all_images_from_excel()
        main_training()