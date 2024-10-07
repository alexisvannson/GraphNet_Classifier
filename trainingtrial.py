#!/usr/bin/env python
# -*- coding: utf-8 -*-

__authors__ = 'KLB, MC, FM'
__copyright__ = 'Copyright (c) 1996 Magoules Research Group. All rights reserved.'
__date__ = '0000-00-00'
__version__ = '0.9.0'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from GNN import convert_image_to_graph, GraphNet, CombinedModel


class model_MLP1(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(model_MLP1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Vérifiez et aplatissez les images si nécessaire
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        latent = out
        return out, latent


class model_EfficientUNet(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(model_EfficientUNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder
        input_channel = input_channels
        output_channel = input_channel *2

        self.down1 = conv_block(input_channel, output_channel)
        self.pool1 = nn.MaxPool2d(2)
        
        input_channel = output_channel
        output_channel *= 2

        self.down2 = conv_block(input_channel, output_channel)
        self.pool2 = nn.MaxPool2d(2)
        
        input_channel = output_channel
        output_channel *= 2

        self.down3 = conv_block(input_channel, output_channel)
        self.pool3 = nn.MaxPool2d(2)
        
        input_channel = output_channel
        output_channel *= 2

        self.down4 = conv_block(input_channel, output_channel)
        self.pool4 = nn.MaxPool2d(2)
        
        input_channel = output_channel
        output_channel *= 2

        self.down5 = conv_block(input_channel, output_channel)

        # Decoder
        input_channel = output_channel
        output_channel //= 2

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = conv_block(input_channel, output_channel)

        input_channel = output_channel
        output_channel //= 2

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = conv_block(input_channel, output_channel)

        input_channel = output_channel
        output_channel //= 2

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = conv_block(input_channel, output_channel)

        input_channel = output_channel
        output_channel //= 2

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = conv_block(input_channel, output_channel)

        input_channel = output_channel
        output_channel //= 2

        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = conv_block(input_channel, output_channel)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.pool1(x1)
        
        x3 = self.down2(x2)
        x4 = self.pool2(x3)
        
        x5 = self.down3(x4)
        x6 = self.pool3(x5)
        
        x7 = self.down4(x6)
        x8 = self.pool4(x7)
        
        x9 = self.down5(x8)
        
        # Decoder
        x = self.up1(x9)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = self.conv4(x)
        
        x = self.up5(x)
        x = self.conv5(x)
        
        latent = x
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x, latent


class model_CNN(nn.Module):
    """
    CNN avec paramètres modulables
    model = CNNClassifier(input_channels=3, num_classes=2, num_conv_layers=3, conv_channels=[32, 64, 128])
    """
    def __init__(self, input_channels=3, num_classes=2, num_conv_layers=2, conv_channels=[32, 64], adaptive_pool_size=(4, 4)):
        super(model_CNN, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.convs = nn.ModuleList()
        
        # Créer dynamiquement les couches de convolution
        for i in range(num_conv_layers):
            in_channels = input_channels if i == 0 else conv_channels[i-1]
            out_channels = conv_channels[i]
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Utiliser AdaptiveAvgPool2d pour obtenir une taille de sortie fixe
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_pool_size)
        
        # Calculez le nombre d'entrées pour la première couche entièrement connectée
        flattened_size = conv_channels[-1] * adaptive_pool_size[0] * adaptive_pool_size[1]
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        for i in range(self.num_conv_layers):
            x = self.pool(F.relu(self.convs[i](x)))
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        if self.fc2.out_features == 1:
            x = torch.sigmoid(self.fc2(x))  # Utilisez sigmoid pour la classification binaire
        else:
            x = F.softmax(self.fc2(x), dim=1)  # Utilisez softmax pour la classification multiclasse
        return x


class model_GNN(nn.Module):
    """
    GNN avce couches linéaires de classification
    """
    def __init__(self):
        super(model_GNN, self).__init__()
        from GNN import MLP, EdgeProcessor, NodeProcessor, GraphProcessor, GraphNet, LinearClassifier, CombinedModel, convert_image_to_graph
        self.convert_image_to_graph =  convert_image_to_graph()
        self.gnn = GraphNet(
            num_global_features=0,
            num_local_features=768,
            space_dim=2,
            out_channels=128,
            n_blocks=10,
            out_dim_node=128,
            out_dim_edge=128,
            hidden_dim_node=128,
            hidden_dim_edge=128,
            hidden_dim_decoder=128,
            hidden_dim_processor_node=128,
            hidden_dim_processor_edge=128,
            hidden_layers_node=2,
            hidden_layers_edge=2,
            hidden_layers_decoder=2,
            hidden_layers_processor_node=2,
            hidden_layers_processor_edge=2,
            norm_type="LayerNorm",
            activation="ReLU",
            initializer=None,
        )
        
        self.model = CombinedModel(self.gnn)
    def forward(self, image_path):
        x, node_coords , edge2nodes = self.convert_image_to_graph(image_path)
        output = self.model(x, node_coords , edge2nodes)
        return output


    
    
    
    
    
    
    
    
    
    
    
    
    

    

    pass


class customize_dataset_for_jpg_training(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialise une instance du dataset personnalisé.
        
        Args:
            root_dir (str): Le répertoire racine contenant les données.
            transform (callable, optional): Transformation à appliquer aux échantillons.
        """
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        """
        Retourne la taille du dataset.
        
        Returns:
            int: Nombre total d'échantillons dans le dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Obtient un échantillon spécifique du dataset.
        
        Args:
            idx (int): L'indice de l'échantillon à obtenir.
        
        Returns:
            tuple: Tuple contenant l'image et son étiquette correspondante.
        """
        image, label = self.dataset[idx]
        return image, label


class customize_dataset_for_jpg_trial(torch.utils.data.Dataset):
        def __init__(self, image_paths, transform=None):
            self.image_paths = image_paths
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            return image


def plot_latent_space_3d_PCA_training(model, data_loader, **kwargs):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in data_loader:
            _, latent = model(images)
            embeddings.append(latent.numpy())
            labels.append(batch_labels.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    pca = PCA(n_components=3, random_state=123)
    embeddings_3d = pca.fit_transform(embeddings.reshape(embeddings.shape[0], -1))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(np.unique(labels))):
        indices = labels == i
        ax.scatter(embeddings_3d[indices, 0], embeddings_3d[indices, 1], embeddings_3d[indices, 2], label=f'Class {i}')

    ax.set_title('PCA Visualization of Latent Space (3D)')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.legend()

    if 'filename' in kwargs and kwargs['filename']:
        output_file = kwargs['filename']
        (root, ext) = os.path.splitext(output_file)
        plt.savefig(root + '_pca' + ext, format=ext[1:])
        plt.close()
    else:
        plt.show()
        plt.close()

    return


def plot_latent_space_3d_PCA_trial(model, data_loader, **kwargs):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images in data_loader:
            _, latent = model(images)
            embeddings.append(latent.numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    pca = PCA(n_components=3, random_state=123)
    embeddings_3d = pca.fit_transform(embeddings.reshape(embeddings.shape[0], -1))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2])

    ax.set_title('PCA Visualization of Latent Space (3D)')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')

    if 'filename' in kwargs and kwargs['filename']:
        output_file = kwargs['filename']
        (root, ext) = os.path.splitext(output_file)
        plt.savefig(root + '_pca' + ext, format=ext[1:])
        plt.close()
    else:
        plt.show()
        plt.close()

    return


def plot_latent_space_3d_TSNE_training(model, data_loader, **kwargs):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in data_loader:
            _, latent = model(images)
            embeddings.append(latent.numpy())
            labels.append(batch_labels.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    tsne = TSNE(n_components=3, random_state=123)
    embeddings_3d = tsne.fit_transform(embeddings.reshape(embeddings.shape[0], -1))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(np.unique(labels))):
        indices = labels == i
        ax.scatter(embeddings_3d[indices, 0], embeddings_3d[indices, 1], embeddings_3d[indices, 2], label=f'Class {i}')

    ax.set_title('t-SNE Visualization of Latent Space (3D)')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.legend()

    if 'filename' in kwargs and kwargs['filename']:
        output_file = kwargs['filename']
        (root, ext) = os.path.splitext(output_file)
        plt.savefig(root + '_tsne' + ext, format=ext[1:])
        plt.close()
    else:
        plt.show()
        plt.close()
    
    return


def plot_latent_space_3d_TSNE_trail(model, data_loader, **kwargs):
    """
    Visualise l'espace latent en 3D en utilisant t-SNE.
    
    Args:
        model (torch.nn.Module): Le modèle utilisé pour extraire l'espace latent.
        data_loader (torch.utils.data.DataLoader): Le DataLoader contenant les données.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images in data_loader:  
            _, latent = model(images)
            embeddings.append(latent.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    
    # Réduction de la dimensionnalité des embeddings à 3 avec t-SNE
    tsne = TSNE(n_components=3, perplexity=1, random_state=123)
    embeddings_3d = tsne.fit_transform(embeddings.reshape(embeddings.shape[0], -1))  
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')  
    
    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2])
    
    ax.set_title('t-SNE Visualization of Latent Space (3D)')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')

    if 'filename' in kwargs and kwargs['filename']:
        output_file = kwargs['filename']
        (root, ext) = os.path.splitext(output_file)
        plt.savefig(root + '_tsne' + ext, format=ext[1:])
        plt.close()
    else:
        plt.show()
        plt.close()
    
    return


def plot_error_metrics(num_epochs, train_errors, val_errors, **kwargs):

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_errors, label='Training Error')
    plt.plot(range(1, num_epochs + 1), val_errors, label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training and Validation Errors')
    plt.legend()

    if 'filename' in kwargs and kwargs['filename']:
        output_file = kwargs['filename']
        (root, ext) = os.path.splitext(output_file)
        plt.savefig(root + '_error' + ext, format=ext[1:])
        plt.close()
    else:
        plt.show()
        plt.close()
    
    return


def plot_accuracy_metrics(num_epochs,  train_accuracies, val_accuracies, **kwargs):
    
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    if 'filename' in kwargs and kwargs['filename']:
        output_file = kwargs['filename']
        (root, ext) = os.path.splitext(output_file)
        plt.savefig(root + '_accuracy' + ext, format=ext[1:])
        plt.close()
    else:
        plt.show()
        plt.close()
    
    return


def run_jpg_training_mlp(input_filename, save_filename, output_filename):
    
    input_size, hidden_size, output_size = 64*64*3, 64*64*3, 4
    # Model initialization
    model = model_MLP1(input_size, hidden_size, output_size)

    # The loss function
    criterion = nn.CrossEntropyLoss()

    # Our optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001) # lr = learning rate

    # Define transformation to be applied to images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor()          # Convert images to PyTorch tensors
    ])
    # Create custom dataset instances for training and validation
    train_dataset = customize_dataset_for_jpg_training(root_dir=input_filename, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    # Create DataLoader instances for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

    train_errors = []
    val_errors = []
    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0.0
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            images = images.view(images.size(0), -1)

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted_train = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted_train.eq(labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_errors.append(avg_train_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc='Validation', unit='batch'):
                images = images.view(images.size(0), -1)
                outputs, _ = model(images)

                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted_val = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted_val.eq(labels).sum().item()

        avg_val_loss = total_val_loss / len(valid_loader)
        val_errors.append(avg_val_loss)
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), output_filename)
            print(f"Best model saved at epoch {epoch + 1} with validation accuracy: {val_accuracy:.2f}%")

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

    _ = plot_error_metrics(num_epochs, train_errors, val_errors, filename=save_filename)
    _ = plot_accuracy_metrics(num_epochs, train_accuracies, val_accuracies, filename=save_filename)
    _ = plot_latent_space_3d_PCA_training(model, train_loader, filename=save_filename)
    #_ = plot_latent_space_3d_TSNE_training(model, train_loader, filename=save_filename)

    return


def run_jpg_classification_mlp(input_filename):
    """
    """
    input_size, hidden_size, output_size = 64*64*3, 64*64*3, 4
    # Load the saved model
    model = model_MLP1(input_size, hidden_size, output_size) # model initialization
    model.load_state_dict(torch.load(input_filename)) # take already prepared weights
    model.eval()  # Set the model to evaluation mode

    # Define image transformations
    image_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize the image to match model input size
        transforms.ToTensor()  # Convert PIL image to PyTorch tensor
    ])

    # Charger et prétraiter l'image
    #image_path = 'shuttle_2.jpeg'
    #image_path = 'montgolfière_2.jpeg'
    image_path = 'avion_de_ligne_2.jpeg'
    #image_path = 'avion_de_chasse_2.jpeg'

    #image_path = 'chihuahua.jpg'

    image = Image.open(image_path)
    image = image_transform(image)
    image = image.view(1, -1)  # Add batch dimension

    # Passage avant à travers le modèle pour obtenir les prédictions
    with torch.no_grad():
        output, _ = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Définir les étiquettes de classe
    class_labels = ['avion_de_chasse', 'avion_de_ligne', 'mongolfière', 'navette_spatiale']
    #class_labels = ["chihuahua", "muffin"]

    # Afficher la prédiction
    print(f'The predicted class is: {class_labels[predicted_class]}')
    print(probabilities)

    return


def run_jpg_training_unet(input_filename, save_filename, output_filename):

    input_shape = (3, 128, 128)
    num_classes = 4

    model = model_EfficientUNet(num_classes, input_channels=input_shape[0])
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(model.parameters(), lr=0.001)

    # Define transformation to be applied to images
    # Transformation pipeline for data augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(30),      # Rotate randomly up to 30 degrees
        transforms.RandomResizedCrop(input_shape[1]),  # Crop randomly and resize to 128x128
        transforms.RandomHorizontalFlip(),  # Flip horizontally with a probability of 0.5
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue randomly
        transforms.ToTensor()  # Convert images to PyTorch tensors
    ])

    # Create custom dataset instances for training and validation
    train_dataset = customize_dataset_for_jpg_training(root_dir=input_filename, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    # Create a random number generator for reproducibility
    random_generator = torch.Generator().manual_seed(42)

    # Split the dataset into training and validation sets while maintaining consistent batch sizes
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, valid_size],
        generator=random_generator  # Ensure consistent random sampling
    )

    # Create DataLoader instances for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)


    train_errors = []
    val_errors = []
    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0.0
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            images = images.view(images.size(0), input_shape[0], input_shape[1], input_shape[2])

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted_train = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted_train.eq(labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_errors.append(avg_train_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc='Validation', unit='batch'):
                images = images.view(images.size(0), input_shape[0], input_shape[1], input_shape[2])
                outputs, _ = model(images)

                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted_val = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted_val.eq(labels).sum().item()

        avg_val_loss = total_val_loss / len(valid_loader)
        val_errors.append(avg_val_loss)
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), output_filename)
            print(f"Best model saved at epoch {epoch + 1} with validation accuracy: {val_accuracy:.2f}%")

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

    _ = plot_error_metrics(num_epochs, train_errors, val_errors, filename=save_filename)
    _ = plot_accuracy_metrics(num_epochs, train_accuracies, val_accuracies, filename=save_filename)
    _ = plot_latent_space_3d_PCA_training(model, train_loader, filename=save_filename)
    #_ = plot_latent_space_3d_TSNE_training(model, train_loader, filename=save_filename)   

    return


def run_jpg_classification_unet(input_filename):

    input_shape = (3, 128, 128)  # Supposant que les images d'entrée sont en RGB avec des dimensions 128x128
    num_classes = 4
    model = model_EfficientUNet(num_classes, input_channels=input_shape[0])
    model.load_state_dict(torch.load(input_filename)) # Charger les poids déjà préparés
    model.eval()  # Mettre le modèle en mode évaluation

    # Définir les transformations d'image
    image_transform = transforms.Compose([
        transforms.Resize((input_shape[1], input_shape[2])),  # Redimensionner l'image pour correspondre à la taille d'entrée du modèle
        transforms.ToTensor()  # Convertir l'image PIL en tenseur PyTorch
    ])

    # Charger et prétraiter l'image
    #image_path = 'shuttle_2.jpeg'
    #image_path = 'montgolfière_2.jpeg'
    image_path = 'avion_de_ligne_2.jpeg'
    #image_path = 'avion_de_chasse_2.jpeg'

    #image_path = 'chihuahua.jpg'

    image = Image.open(image_path)
    image = image_transform(image)
    image = image.view(1, input_shape[0], input_shape[1], input_shape[2])

    # Passage avant à travers le modèle pour obtenir les prédictions
    with torch.no_grad():
        output, _ = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Définir les étiquettes de classe
    class_labels = ['avion_de_chasse', 'avion_de_ligne', 'mongolfière', 'navette_spatiale']
    #class_labels = ["chihuahua", "muffin"]

    # Afficher la prédiction
    print(f'The predicted class is: {class_labels[predicted_class]}')
    print(probabilities)

    # Chemins des images
    image_paths = ['avion_de_chasse_2.jpeg', 'avion_de_ligne_2.jpeg', 'montgolfière_2.jpeg', 'shuttle_2.jpeg']
    #image_paths = ['chihuahua.jpg', 'muffin.jpg']

    # Transformer les images
    transform = transforms.Compose([
        transforms.Resize((input_shape[1], input_shape[2])),
        transforms.ToTensor(),
    ])

    # Créer le dataset
    custom_dataset = customize_dataset_for_jpg_trial(image_paths, transform=transform)

    # Créer le DataLoader
    test_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

    _ = plot_latent_space_3d_PCA_trial(model, test_loader, filename='latent_trial.jpg')
    #_ = plot_latent_space_3d_TSNE_trial(model, test_loader, filename='latent_trial.jpg')

    return


def run_jpg_training_cnn(input_filename):
     # Hyperparameters
    
    input_channels = 3
    num_classes = 2
    num_conv_layers = 3 
    conv_channels = [32, 64, 128]
    num_epochs = 25
    batch_size = 32
    learning_rate = 0.001
    patience = 5

    model = model_CNN(input_channels=input_channels, num_classes=num_classes, num_conv_layers=num_conv_layers, conv_channels=conv_channels)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor()          # Convert images to PyTorch tensors
    ])
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = customize_dataset_for_jpg_training(root_dir=input_filename, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    # Create DataLoader instances for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_saved = False

    # Training loop with early stopping
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_loss = 0

            for img, label in zip(images, labels):
                x, pos, edge_index = convert_image_to_graph(img)
                output = model(x, pos, edge_index)
                loss = criterion(output.unsqueeze(0), label.unsqueeze(0))  # Assurez-vous que les dimensions correspondent
                batch_loss += loss.item()
                loss.backward()
            print(i)

            optimizer.step()
            train_loss += batch_loss / len(images)

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {batch_loss / len(images):.4f}')

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                batch_val_loss = 0
                for img, label in zip(images, labels):
                    x, pos, edge_index = convert_image_to_graph(img)
                    output = model(x, pos, edge_index)
                    loss = criterion(output.unsqueeze(0), label.unsqueeze(0))
                    batch_val_loss += loss.item()
                val_loss += batch_val_loss / len(images)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss or model_saved == False:
            model_saved = True
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.ckpt')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

        

def run_jpg_training_gnn(input_filename):    
    # Hyperparameters

    num_classes = 2  # chihuahua vs muffin
    num_epochs = 25
    batch_size = 32
    learning_rate = 0.001
    patience = 5


    # Initialize GNN model
    gnn = GraphNet(
        num_global_features=0,
        num_local_features=768,
        space_dim=2,
        out_channels=128,
        n_blocks=10,
        out_dim_node=128,
        out_dim_edge=128,
        hidden_dim_node=128,
        hidden_dim_edge=128,
        hidden_dim_decoder=128,
        hidden_dim_processor_node=128,
        hidden_dim_processor_edge=128,
        hidden_layers_node=2,
        hidden_layers_edge=2,
        hidden_layers_decoder=2,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        norm_type="LayerNorm",
        activation="ReLU",
        initializer=None,
    )

    model = CombinedModel(gnn)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor()          # Convert images to PyTorch tensors
    ])

    train_dataset = customize_dataset_for_jpg_training(root_dir=input_filename, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    # Create DataLoader instances for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gnn.parameters(), lr=learning_rate)


    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop with early stopping
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_loss = 0

            for img, label in zip(images, labels):
                x, pos, edge_index = convert_image_to_graph(img)
                output = model(x, pos, edge_index)
                loss = criterion(output.unsqueeze(0), label.unsqueeze(0))  # Assurez-vous que les dimensions correspondent
                batch_loss += loss.item()
                loss.backward()
            print(i)

            optimizer.step()
            train_loss += batch_loss / len(images)

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {batch_loss / len(images):.4f}')

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                batch_val_loss = 0
                for img, label in zip(images, labels):
                    x, pos, edge_index = convert_image_to_graph(img)
                    output = model(x, pos, edge_index)
                    loss = criterion(output.unsqueeze(0), label.unsqueeze(0))
                    batch_val_loss += loss.item()
                val_loss += batch_val_loss / len(images)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.ckpt')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break


    
def run_jpg_classification_cnn(input_filename):
    """
    AV: copy paste
    """
    input_size, hidden_size, output_size = 64*64*3, 64*64*3, 4
    # Load the saved model
    model = model_CNN(input_size, hidden_size, output_size) # model initialization
    model.eval()  # Set the model to evaluation mode

    # Define image transformations
    image_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize the image to match model input size
        transforms.ToTensor()  # Convert PIL image to PyTorch tensor
    ])

    #image_path = 'avion_de_ligne_2.jpeg'
    #image_path = 'avion_de_chasse_2.jpeg'

    image_path = 'chihuahua.jpg'

    image = Image.open(image_path)
    image = image_transform(image)
    image = image.view(1, -1)  # Add batch dimension

    # Passage avant à travers le modèle pour obtenir les prédictions
    with torch.no_grad():
        output, _ = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Définir les étiquettes de classe
    #class_labels = ['avion_de_chasse', 'avion_de_ligne', 'mongolfière', 'navette_spatiale']
    class_labels = ["chihuahua", "muffin"]

    # Afficher la prédiction
    print(f'The predicted class is: {class_labels[predicted_class]}')
    print(probabilities)

    

def run_jpg_classification_gnn(input_filename):    
    """
    AV: copy paste
    """
    # Load the saved model
    model = model_GNN() # model initialization
    model.eval()  # Set the model to evaluation mode

    # Charger et prétraiter l'image
    image_path = input_filename
    x, node_coords, edge2nodes = convert_image_to_graph(image_path)
    # Passage avant à travers le modèle pour obtenir les prédictions
    with torch.no_grad():
        output, _ = model(x, node_coords, edge2nodes)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Définir les étiquettes de classe
    #class_labels = ['avion_de_chasse', 'avion_de_ligne', 'mongolfière', 'navette_spatiale']
    class_labels = ["chihuahua", "muffin"]

    # Afficher la prédiction
    print(f'The predicted class is: {class_labels[predicted_class]}')
    print(probabilities)



if __name__ == "__main__":

    input_filename = './dataset_avion'
    save_filename = 'model_unet.jpg'
    output_filename = 'model_unet.pth'
    _ = run_jpg_training_unet(input_filename=input_filename, save_filename=save_filename, output_filename=output_filename)

    filename = 'model_unet.pth'
    _ = run_jpg_classification_unet(filename)

# pour le training on a les fichiers: 
# model_unet.pth
# model_unet_pca.jpg, model_unet_tsne.jpg
# model_unet_accuracy.jpg, model_unet_error.jpg
# model_unet_accuracy.txt, model_unet_error.txt // 1 ligne par ligne, sur chaque ligne la valeur de l'err ou de l 'accuracy avec 

# idem pour la classification

    input_filename = './dataset_avion'
    save_filename = 'model_unet.jpg'
    output_filename = 'model_mlp.pth'
    _ = run_jpg_training_mlp(input_filename=input_filename, save_filename=save_filename, output_filename=output_filename)

    filename = 'model_mlp.pth'
    _ = run_jpg_classification_mlp(filename)



    _ = run_jpg_training_cnn(input_filename=input_filename, save_filename=save_filename, output_filename=output_filename)

    filename = 'model_mlp.pth'
    _ = run_jpg_classification_mlp(filename)
    # ..todo: AV idem CNN

    # ..todo: AV idem GNN
    

    print("End.")
