import torch
import torch.nn as nn
from test.losses import Loss


n_samples = 100
n_features = 2
X = torch.randn(n_samples, n_features)
y = torch.randn(n_samples)

# Create a simple linear model
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

# Instantiate the loss function and the model
cvar_loss = Loss(alpha=0.1, reg=0.01, tol=1e-4, maxiter=50)
model = SimpleModel(n_features)

# Define an optimizer (e.g., stochastic gradient descent)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop (you can customize the number of epochs)
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    # Calculate the loss using the defined CVaR loss function
    loss_value = cvar_loss(outputs, None)  # Pass None as p for simplicity

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_value.item()}')