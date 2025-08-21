import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from image_to_graph.image_to_graph_optimized import image_to_graph_pixel_optimized
from models.MLP import MLP
from models.GNN import GraphNet, CombinedModel


def inference_MLP(image_path, weights='weights/MLP/final_model.pth', resize_value=128):
    in_dim = resize_value * resize_value * 3  # RGB image
    out_dim = 2  # Assuming binary classification (chihuahua vs muffin)
    model = MLP(in_dim=in_dim, out_dim=out_dim)
    model.load_state_dict(torch.load(weights))
    model.eval()
    
    with torch.no_grad():
        input_tensor = torch.tensor(Image.open(image_path).resize((resize_value, resize_value)).convert('RGB').flatten(),dtype=torch.float32)
        output = model(input_tensor)
    return output

def inference_GNN(image_path, weights='weights/MLP/final_model.pth', resize_value=64):
    # Create the same model architecture as used in training
    num_nodes = resize_value * resize_value
    graph_net = GraphNet(num_local_features=3, space_dim=2, out_channels=1, n_blocks=3)
    model = CombinedModel(graph_net=graph_net, num_nodes=num_nodes, classes=2)
    
    model.load_state_dict(torch.load(weights))
    model.eval()
    
    with torch.no_grad():
        input_tensor = image_to_graph_pixel_optimized(image_or_path=image_path, resize_value=resize_value)
        output = model(input_tensor)
    return output

image = '/Users/philippevannson/Desktop/ongoing_stuff/GraphNet_Classifier/dataset/chihuahua/img_0_8.jpg'

print("MLP inference:")
print(inference_MLP(image))
print("\nGNN inference:")
print(inference_GNN(image))