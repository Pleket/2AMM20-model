from torch.utils.data import Dataset
from math import floor, ceil
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random
import pandas as pd 
import shutil
import os

import tarfile
from PIL import Image

#You can use this code when uploading the data into Jupyterhub. Upload
#as a zipfile and then this code will extract it.
# import zipfile as zf
# files = zf.ZipFile("waterbird_complete95_2class.zip", 'r')
# files.extractall('data')
# files.close()

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        class_folders = os.listdir(data_dir)
        for class_folder in class_folders:
            class_path = os.path.join(data_dir, class_folder)
            if os.path.isdir(class_path):
                class_name = class_folder
                if class_name == "landbirds":
                    class_label = 0
                elif class_name == "waterbirds":
                    class_label = 1
                else:
                    continue

                for image_filename in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_filename)
                    self.image_paths.append(image_path)
                    self.labels.append(class_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        filename = os.path.basename(image_path)
        label = self.labels[idx]

        return image, filename, label
    
def preprocess(directory, batch_size, select_percentage = 100):
    transform = transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ])
    
    data = CustomDataset(data_dir=directory, transform=transform)

    if select_percentage != 100:
        data = select_fraction(data, select_percentage)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    return (DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True),
        DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False),
        )

def train_resnet50(train_loader, test_loader, model_path, epochs, learning_rate):
    num_classes = 2
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
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = ceil(len(train_loader))
    train_losses, test_losses = [], []
    print("We're training with ", len(train_loader), " batches and ", len(train_loader) * 32, " images.")
    for epoch in range(epochs):
        for inputs, filenames, labels in train_loader:
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
                    for inputs, filenames, labels in test_loader:
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
    torch.save(model, model_path)

    # Plot the training and validation losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()
    
def select_fraction(data, select_percentage):
    print("The initial size of the data is ", len(data))
    indices = list(range(len(data)))
    random.shuffle(indices)
    subset_indices = indices[:int(len(data) * (select_percentage / 100.0))]
    return torch.utils.data.Subset(data, subset_indices)

def get_misclassified_weights(model, train_loader, lambda_up):
    weights = []
    with torch.no_grad():
        for images, filenames, labels in train_loader:
            device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
            images, labels = images.to(device), labels.to(device) # This line is required because otherwise, inputs and labels are on the CPU while the model is on the GPU
            outputs = model(images)
            predicted_labels = torch.argmax(outputs, dim=1)
            misclassified_mask = predicted_labels != labels
            for classification in misclassified_mask:
                if classification:
                    weights.append(lambda_up)
                else:
                    weights.append(1)   
    return weights

def filename_to_group_dict(csv_file_path):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_file_path)

    # Initialize an empty dictionary
    filename_to_group = {}

    # Iterate through the DataFrame and map conditions to values
    for index, row in data.iterrows():
        filename = row['img_filename']
        last_slash_index = filename.rfind('/')
        if last_slash_index != -1:
            filename = filename[last_slash_index + 1:]
        y = row['y']
        place = row['place']

        if y == 0 and place == 0:
            filename_to_group[filename] = 0 #0 is landbird with land background
        elif y == 0 and place == 1:
            filename_to_group[filename] = 1 #1 is landbird with water background
        elif y == 1 and place == 0:
            filename_to_group[filename] = 2 #2 is waterbird with land background
        elif y == 1 and place == 1:
            filename_to_group[filename] = 3 #3 is waterbird with water background

    return filename_to_group

def compute_accuracy_per_group(model, test_loader, filename_to_group_dict):
    group_correct = {0:0,1:0,2:0,3:0}
    group_total = {0:0,1:0,2:0,3:0}
    group_accuracies = {}
    
    for batch in test_loader:
        images, filenames, labels = batch
        images, labels = images.to(device), labels.to(device)

        for image, filename, label in zip(images, filenames, labels):
            image = image.unsqueeze(0)
            output = model(image)
            group = filename_to_group_dict.get(filename)
            if torch.argmax(output) == label.item():
                group_correct[group] += 1
            group_total[group] += 1
    
    print(group_correct)
            
    group_accuracies[0] = group_correct[0]/group_total[0]
    group_accuracies[1] = group_correct[1]/group_total[1]
    group_accuracies[2] = group_correct[2]/group_total[2]
    group_accuracies[3] = group_correct[3]/group_total[3]
    
    return group_accuracies

path_to_data = 'data/waterbird_complete95_2class'

train_loader, test_loader = preprocess(path_to_data ,batch_size = 32, select_percentage=100)
filename_to_group_dict = filename_to_group_dict('metadata.csv')

train_resnet50(train_loader, test_loader, 'JTT_one', epochs = 10, learning_rate = 0.002)

model = torch.load('JTT_one')

accuracies_per_group_dict_one = compute_accuracy_per_group(model, test_loader, filename_to_group_dict)
print(accuracies_per_group_dict_one)

weights = get_misclassified_weights(model, train_loader, 2.5)
sampler = WeightedRandomSampler(weights, ceil(sum(weights)), replacement=True)

train_loader_new = DataLoader(
            train_loader.dataset, 
            batch_size=32, 
            sampler=sampler,
            shuffle=False)
print("Now, length of train_loader is ", len(train_loader_new))

train_resnet50(train_loader_new, test_loader, 'JTT_two', epochs = 10, learning_rate = 0.002)
model_twice = torch.load('JTT_two')

accuracies_per_group_dict_two = compute_accuracy_per_group(model_twice, test_loader, filename_to_group_dict)

print(accuracies_per_group_dict_two)
