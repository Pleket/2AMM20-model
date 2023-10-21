import torch
import torch.nn as nn
from losses import Loss
import torch.optim as optim

# Example input data
# Features (X) and labels (y)
X = torch.tensor([[0.2, 0.3], [0.5, 0.7], [0.8, 0.9], [0.4, 0.2], [0.7, 0.5]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0], [1]], dtype=torch.float32)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        #need a sigmoid since our model should output probabilities
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Create the neural network
model = SimpleNN()

# Define your custom CVaR loss class here
# ...

# Create an instance of the custom CVaR loss with desired parameters
cvar_loss = Loss(alpha=0.001, reg=0.01, tol=1e-4, maxiter=50)

# Define an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

num_epochs = 100
# Training the model with the custom CVaR loss
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    
    # Calculate the binary cross-entropy loss for each input
    individual_losses = nn.BCELoss(reduction='none')(outputs, y)
    
    # Calculate the custom CVaR loss over the individual losses
    loss = cvar_loss(individual_losses)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    
    loss.backward()
    optimizer.step()

# Test the model on new data
new_data = torch.tensor([[0.3, 0.4], [0.6, 0.6], [0.9, 0.2]], dtype=torch.float32)
predictions = model(new_data)

print("Predictions:")
for i in range(len(new_data)):
    print(f"Input: {new_data[i].numpy()}, Prediction: {predictions[i].item()}")