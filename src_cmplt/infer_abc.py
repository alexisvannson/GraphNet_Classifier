import torch
import os
import matplotlib.pyplot as plt
from src.ai_mlp import MLP

def main_inference(a, b, c, weights='mlp_weights.pth',output_folder='predictions'):
    os.makedirs(output_folder, exist_ok=True)

    model = MLP()
    model.load_state_dict(torch.load(weights))
    model.eval()

    with torch.no_grad():
        input_tensor = torch.tensor([[a, b, c]], dtype=torch.float32)
        output = model(input_tensor).view(128, 128).numpy()
        output[output < 0.8] *= 0.2 
        plt.imshow(output, cmap='gray') # a modifier
        plt.axis('off')
        image_path = os.path.join(output_folder, f"prediction_a{a}_b{b}_c{c}.png")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    print("Image sauvegardée dans le fichier predictions")


if __name__ == "__main__":
    pass
    #main_inference(1, 1, 1)
    #main_inference(8,2,3, weights='mlp_weights_epoch20.pth')