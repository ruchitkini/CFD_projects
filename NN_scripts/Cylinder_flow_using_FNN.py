""" Using FNN, predicting the solution of flow over a cylinder in 2D using 500 data points """

import torch
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import time

# Set the directory where CSV files are stored
csv_dir = 'Cylinder_flow_data'

# List all CSV files in the directory
csv_files = [os.path.join(csv_dir, file) for file in os.listdir(csv_dir) if file.endswith('.csv')]

input_data = []
output_data = []

# Exact the data from each file
for file in csv_files:
    df = pd.read_csv(file, skiprows=5)

    features = df[['X [ m ]', ' Y [ m ]']].values    # Extracting x and y coordinates
    labels = df[[' Pressure [ Pa ]', ' Velocity u [ m s^-1 ]', ' Velocity v [ m s^-1 ]']].values

    input_data.append(features)
    output_data.append(labels)

# Convert list of arrays to a single numpy array
input_data = np.array(input_data).reshape(-1, 2)
output_data = np.array(output_data).reshape(-1, 3)

# Now convert numpy arrays to PyTorch tensors
input_data = torch.tensor(input_data).float()
output_data = torch.tensor(output_data).float()

# Split into training and testing sets
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# Create PyTorch DataLoader for batch processing
batch_size = 32

# Convert train and test data to TensorDataset
train_dataset = TensorDataset(train_inputs, train_outputs)
test_dataset = TensorDataset(test_inputs, test_outputs)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Activation function
        self.sigmoid = nn.Sigmoid()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)  # Linear transform (input)
        out = self.sigmoid(out)  # Non-linearity
        out = self.fc2(out)  # Linear transform (output)

        return out


model = FNN(input_dim=2, hidden_dim=30, output_dim=3)
loss_function = nn.MSELoss()
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr)


iter = 0
num_epochs = 500
print("Training the model ...")
start_time = time.time()
for epoch in range(num_epochs):
    for inputs, labels in train_dataset:
        outputs = model(inputs)                 # Forward pass: Compute predicted y by passing inputs to the model
        loss = loss_function(outputs, labels)   # Compute loss
        optimizer.zero_grad()                   # Zero gradients before backward pass
        loss.backward()                         # Backward pass: Compute gradients
        optimizer.step()                        # Update model parameters

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

end_time = time.time()
total_time = end_time - start_time
print(f'Training completed, required {total_time} s')

# Evaluate the model on test data
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for inputs, labels in test_dataset:
        outputs = model(inputs)
        test_loss += loss_function(outputs, labels).item()

    avg_test_loss = test_loss / len(test_dataset)
    print(f"Average Test Loss: {avg_test_loss:.4f}")

# Example: Use the model for inference on a new data point (replace with actual test data)
test_input = torch.tensor([0.5, 0.3])  # Example input (x, y)
predicted_output = model(test_input)
print(f"Predicted output for input {test_input}: {predicted_output}")


# Create a grid of points where you want to predict (x, y)
x_range = np.linspace(-5, 5, 100)  # X-coordinates range
y_range = np.linspace(-5, 5, 100)  # Y-coordinates range
X, Y = np.meshgrid(x_range, y_range)

# Flatten the grid for model input (since the model expects a list of points)
grid_points = np.vstack([X.ravel(), Y.ravel()]).T  # Shape (10000, 2)

# Convert to tensor and run through the model to predict pressure and velocity (ux, uy)
grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
model.eval()  # Set the model to evaluation mode

# Predicting pressure, ux, uy for the grid
with torch.no_grad():
    predictions = model(grid_points_tensor)

# Extract the predicted values for pressure, ux, and uy
pressure = predictions[:, 0].numpy().reshape(X.shape)  # Reshape to match grid
ux = predictions[:, 1].numpy().reshape(X.shape)  # Reshape to match grid
uy = predictions[:, 2].numpy().reshape(X.shape)  # Reshape to match grid

# 1. Plot Pressure Field
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, pressure, cmap='coolwarm', levels=50)
plt.colorbar(label='Pressure')
plt.title('Pressure Field')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 2. Plot Velocity Field (using quiver plot)
plt.figure(figsize=(10, 8))
plt.quiver(X, Y, ux, uy, np.sqrt(ux**2 + uy**2), scale=50, cmap='coolwarm')
plt.title('Velocity Field (Ux, Uy)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Velocity Magnitude')
plt.show()




