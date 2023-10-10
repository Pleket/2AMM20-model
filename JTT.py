from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import matplotlib.pyplot as plt

import os
import tarfile
from PIL import Image

def train(resized_directory):
    num_classes = 200
    input_shape = (224, 224, 3)  # RGB images

    (train_loader, test_loader) = preprocess()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    
    model = resnet50(weights=ResNet50_Weights.DEFAULT, input_shape=input_shape)

    # Freeze the layers except the last 4 layers
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, num_classes),
                                    nn.LogSoftmax(dim=1))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)

def preprocess():
    #resize to 224x224 pixels
    target_directory = 'C:/Users/Gebruiker/OneDrive - TU Eindhoven/TUe/Master/2AMM20/CUB_200_2011/images'
    #target_directory = 'your target directory'
    resized_directory = 'C:/Users/Gebruiker/OneDrive - TU Eindhoven/TUe/Master/2AMM20/CUB_200_2011_resized'
    #resized_directory = 'your resized directory'
    # target_size = (224, 224)
    # # Iterate through all images in the dataset and resize them
    # # Iterate through the subdirectories (one for each class)
    # for class_dir in os.listdir(target_directory):
    #     class_path = os.path.join(target_directory, class_dir)
    #     if os.path.isdir(class_path):
    #         for filename in os.listdir(class_path):
    #             if filename.endswith('.jpg'):  # Assuming images are in JPEG format
    #                 image_path = os.path.join(class_path, filename)
    #                 img = Image.open(image_path)
    #                 img = img.resize(target_size, Image.ANTIALIAS)
                    
    #                 # Save the resized image to the destination directory
    #                 destination_path = os.path.join(resized_directory, class_dir, filename)
    #                 os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    #                 img.save(destination_path)

    transforms = transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ])
    
    batch_size = 32

    full_data = torchvision.datasets.ImageFolder(root=resized_directory, transform=transforms)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_data))
    val_size = len(full_data) - train_size
    train_dataset, val_dataset = random_split(full_data, [train_size, val_size])

    return (torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True),
        torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False))