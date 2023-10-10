import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import os
import tarfile
from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess():
    #resize to 224x224 pixels
    target_directory = 'C:/Users/Gebruiker/OneDrive - TU Eindhoven/TUe/Master/2AMM20/CUB_200_2011/images'
    #target_directory = 'your target directory'
    resized_directory = 'C:/Users/Gebruiker/OneDrive - TU Eindhoven/TUe/Master/2AMM20/CUB_200_2011_resized'
    #resized_directory = 'your resized directory'
    target_size = (224, 224)
    # Iterate through all images in the dataset and resize them
    # Iterate through the subdirectories (one for each class)
    for class_dir in os.listdir(target_directory):
        class_path = os.path.join(target_directory, class_dir)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith('.jpg'):  # Assuming images are in JPEG format
                    image_path = os.path.join(class_path, filename)
                    img = Image.open(image_path)
                    img = img.resize(target_size, Image.ANTIALIAS)
                    
                    # Save the resized image to the destination directory
                    destination_path = os.path.join(resized_directory, class_dir, filename)
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    img.save(destination_path)

    # transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(256),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225] )
    # ])

    # Clean up: Remove the extracted dataset if no longer needed
    #os.rmdir(target_directory)

def train(resized_directory):
    batch_size = 32
    num_classes = 200
    input_shape = (224, 224, 3)  # RGB images

    full_data = torchvision.datasets.ImageFolder(root=resized_directory)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_data))
    val_size = len(full_data) - train_size
    train_dataset, val_dataset = random_split(full_data, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False)
    
