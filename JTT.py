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

def train_resnet50(resized_directory):
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

    epochs = 1
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

    full_data = torchvision.datasets.ImageFolder(root=target_directory, transform=transforms)

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