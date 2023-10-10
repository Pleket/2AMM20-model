from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
import random

import os
import tarfile
from PIL import Image

def train_resnet50(directory):
    num_classes = 200
    input_shape = (224, 224, 3)  # RGB images

    (train_loader, test_loader) = preprocess(directory)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    
    #model = resnet50(weights=ResNet50_Weights.DEFAULT, input_shape=input_shape)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

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

    epochs = 2
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(train_loader))
                test_losses.append(test_loss/len(test_loader))                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(test_loader):.3f}.. "
                    f"Test accuracy: {accuracy/len(test_loader):.3f}")
                running_loss = 0
                model.train()
    torch.save(model, 'aerialmodel.pth')

    # Plot the training and validation losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()
    
def extract(tgz_file,target_directory):
    with tarfile.open(tgz_file, 'r:gz') as tar:
        tar.extractall(path=target_directory)

def preprocess(directory, select_percentage = 100):
    transform = transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ])
    
    batch_size = 32
    
    data = torchvision.datasets.ImageFolder(root=directory, transform=transform)
    
    if select_percentage != 100:
        # Create a dictionary to store indices of samples from each class
        class_indices = {}
        for idx, (directory, class_label) in enumerate(data.imgs):
            if class_label not in class_indices:
                class_indices[class_label] = []
            class_indices[class_label].append(idx)

        selected_indices = []
        
        # Randomly select a percentage of data from each class
        for class_label, indices in class_indices.items():
            num_samples = len(indices)
            num_samples_to_select = int(num_samples * (select_percentage / 100.0))
            selected_indices.extend(random.sample(indices, num_samples_to_select))

        # Create a Subset of the dataset using the selected indices
        data = torch.utils.data.Subset(data, selected_indices)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    return (torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True),
        torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False))
    
#extract('C:/Users/Gebruiker/OneDrive - TU Eindhoven/TUe/Master/2AMM20/waterbird_complete95_forest2water2.tar.gz','C:/Users/Gebruiker/OneDrive - TU Eindhoven/TUe/Master/2AMM20')
#train_loader, val_loader = preprocess('C:/Users/Gebruiker/OneDrive - TU Eindhoven/TUe/Master/2AMM20/waterbird_complete95_forest2water2',select_percentage=5)
train_resnet50('C:/Users/Gebruiker/OneDrive - TU Eindhoven/TUe/Master/2AMM20/waterbird_complete95_forest2water2')


