import os
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

def download_mnist_and_save_images():
    """
    Download MNIST dataset using torchvision and save as individual image files
    """
    print("Downloading MNIST dataset using torchvision...")
    
    # Download MNIST dataset without transforms to get raw images
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True, 
        transform=None,  # No transform to get raw images
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=None,  # No transform to get raw images
        download=True
    )
    
    print(f"Downloaded {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    
    # Save training images
    save_images_as_files(train_dataset, "./dataset", "train")
    
    # Save test images
    save_images_as_files(test_dataset, "./dataset", "test")
    
    return len(train_dataset), len(test_dataset)

def save_images_as_files(dataset, output_dir, split_name):
    """
    Save MNIST images as individual files organized by class
    """
    output_path = Path(output_dir) / "mnist" / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    for i in range(10):
        (output_path / str(i)).mkdir(exist_ok=True)
    
    # Save images
    for i, (image, label) in enumerate(dataset):
        # Convert PIL image to numpy array if needed
        if hasattr(image, 'numpy'):
            # If it's a tensor, convert to numpy
            img_array = image.numpy()
            if len(img_array.shape) == 3 and img_array.shape[0] == 1:
                # Remove channel dimension for grayscale
                img_array = img_array.squeeze(0)
            img = Image.fromarray(img_array, mode='L')
        else:
            # If it's already a PIL image
            img = image
        
        # Save to appropriate class directory
        class_dir = output_path / str(label)
        img_path = class_dir / f"{split_name}_{i:05d}.png"
        img.save(img_path)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Saved {i + 1} {split_name} images...")
    
    print(f"Saved {len(dataset)} {split_name} images to {output_path}")

def main():
    """
    Main function to download MNIST and save as images
    """
    print("Starting MNIST download and image extraction...")
    
    # Download MNIST and save as images
    train_count, test_count = download_mnist_and_save_images()
    
    print("MNIST dataset download and image extraction complete!")
    print(f"Training samples: {train_count}")
    print(f"Test samples: {test_count}")
    print(f"Images saved to: ./dataset/mnist/")

if __name__ == "__main__":
    main()