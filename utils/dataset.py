import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageRegressionDataset(Dataset):
    def __init__(self, excel_path, image_folder):
        self.df = pd.read_excel(excel_path)
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        params = self.df.iloc[idx][['a', 'b', 'c']].values.astype(np.float32)
        img_path = os.path.join(self.image_folder, f"img_{idx:03}.png")
        image = Image.open(img_path) #.convert('L')  
        image = self.transform(image)
        return torch.tensor(params), image