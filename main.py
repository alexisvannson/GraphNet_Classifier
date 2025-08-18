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
from models.GNN import CombinedModel

def load_data(dataset_path, resize_value=128):
    transform = transforms.Compose([transforms.Resize((resize_value, resize_value)),
                                transforms.ToTensor()])

    dataset = datasets.ImageFolder(root='dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    return dataloader

def train(model, dataset, epochs, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for image, label in dataset:
            pred = model(image)
            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, avg_loss={avg_loss:.4f}")
        
        # Correct early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_epoch{epoch+1}.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    torch.save(model.state_dict(), 'final_model.pth')

def train_MLP(epochs=30, channels=3, resize_value=128, batch_size=8):
    
    input_dim = channels * resize_value * resize_value 

    dataset = load_data('dataset', resize_value)

    num_classes = len(dataset.dataset.classes)
    print('num classes', num_classes)
    model = MLP(in_dim=input_dim, out_dim=num_classes)
    
    train(model, dataset, epochs)

def inference_MLP(input_dim=128*128*3, num_classes=2, weights='final_model.pth',output_folder='predictions',resize_value=128):# à tester
    os.makedirs(output_folder, exist_ok=True)

    model = MLP(in_dim=input_dim, out_dim=num_classes)
    model.load_state_dict(torch.load(weights))
    model.eval()

    with torch.no_grad():
        
        image = Image.open(image_path).convert('RGB')
        resized_image = image.resize((resize_value, resize_value))

        input_tensor = torch.tensor(resized_image, dtype=torch.float32)
        output = model(input_tensor).view(128, 128).numpy()
        output[output < 0.8] *= 0.2 
        plt.imshow(output) 
        plt.axis('off')
        image_path = os.path.join(output_folder, f"prediction_mlp_{image_path}.png")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    print("Image sauvegardée dans le fichier predictions")
if __name__ == '__main__':
    train_MLP()

#à faire, puis comparer
"""def train_GNN(epochs=30, channels=3, resize_value=128, batch_size=8):
    
    input_dim = channels * resize_value * resize_value 

    dataset = load_data('dataset', resize_value)

    num_classes = len(dataset.dataset.classes)
    print('num classes', num_classes)
    model = MLP(in_dim=input_dim, out_dim=num_classes)
    
    train(model, dataset, epochs)"""