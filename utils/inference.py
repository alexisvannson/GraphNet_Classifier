import torch
from PIL import Image

from image_to_graph.image_to_graph_optimized import image_to_graph_pixel_optimized
from models.MLP import MLP
from models.GNN import GraphNet


def inference_MLP(image_path, weights='../weights/MLP/final_model.pth', resize_value=128):
    model = MLP()
    model.load_state_dict(torch.load(weights))
    model.eval()
    
    with torch.no_grad():
        input_tensor = torch.tensor(Image.open(image_path).resize(resize_value, resize_value).convert('RGB').flatten(),dtype=torch.float32)
        output = model(input_tensor)
    return output

def inference_GNN(image_path, weights='../weights/MLP/final_model.pth'):
    model = GraphNet()
    model.load_state_dict(torch.load(weights))
    model.eval()
    
    with torch.no_grad():
        input_tensor = image_to_graph_pixel_optimized(image_or_path=image_path)
        output = model(input_tensor)
    return output

image = '/Users/philippevannson/Desktop/ongoing_stuff/GraphNet_Classifier/dataset/chihuahua/img_0_8.jpg'

print(inference_MLP(image))