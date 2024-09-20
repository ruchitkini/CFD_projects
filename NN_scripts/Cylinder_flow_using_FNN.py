""" Using FNN, predicting the solution of flow over a cylinder in 2D using 500 data points """

import torch
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import time

# Set the device (use GPU if available, otherwise fallback to CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

# Set the directory where CSV files are stored
csv_dir = 'Cylinder_flow_data'

# List all CSV files in the directory
csv_files = [os.path.join(csv_dir, file) for file in os.listdir(csv_dir) if file.endswith('.csv')]

input_data = []
output_data = []

# Extract the data from each file
for file in csv_files:
    df = pd.read_csv(file, skiprows=5)

    features = df[['X [ m ]', ' Y [ m ]']].values  # Extracting x and y coordinates
    labels = df[[' Pressure [ Pa ]', ' Velocity u [ m s^-1 ]', ' Velocity v [ m s^-1 ]']].values

    input_data.append(features)
    output_data.append(labels)

# Convert list of arrays to a single numpy array
input_data = np.array(input_data).reshape(-1, 2)
output_data = np.array(output_data).reshape(-1, 3)

scaler = MinMaxScaler()
scaler.fit(output_data)
input_data = scaler.fit_transform(input_data)  # Normalize input features
output_data = scaler.fit_transform(output_data) # Normalize output data

# Now convert numpy arrays to PyTorch tensors and move them to the device (GPU or CPU)
input_data = torch.tensor(input_data).float().to(device)
output_data = torch.tensor(output_data).float().to(device)

# Split into training and testing sets
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(input_data, output_data, test_size=0.2,
                                                                          random_state=42)

# Create PyTorch DataLoader for batch processing
batch_size = 32

# Convert train and test data to TensorDataset
train_dataset = TensorDataset(train_inputs, train_outputs)
test_dataset = TensorDataset(test_inputs, test_outputs)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(FNN, self).__init__()
        # Linear function, first hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # Activation function
        self.tanh = nn.Tanh()
        # Linear function (readout)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)  # Linear transform (input)
        out = self.tanh(out)  # Non-linearity
        # Forward pass through the second hidden layer
        out = self.fc2(out)
        out = self.tanh(out)

        out = self.fc3(out)  # Linear transform (output)

        return out


# Create the model and move it to the device (GPU or CPU)
model = FNN(input_dim=2, hidden_dim1=30, hidden_dim2=30, output_dim=3).to(device)
loss_function = nn.MSELoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr)

# Training loop
iter = 0
num_epochs = 200
print("Training the model ...")
start_time = time.time()
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device (GPU or CPU)

        outputs = model(inputs)  # Forward pass: Compute predicted y by passing inputs to the model
        loss = loss_function(outputs, labels)  # Compute loss
        optimizer.zero_grad()  # Zero gradients before backward pass
        loss.backward()  # Backward pass: Compute gradients
        optimizer.step()  # Update model parameters

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    interval = 1
    # Save the model after a specific epoch
    if (epoch + 1) % interval == 0 and loss < best_loss:
        best_loss = loss
        model_path = f'Cylinder_flow_model/model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model saved at epoch {epoch + 1} to {model_path}')

end_time = time.time()
total_time = end_time - start_time
print(f'Training completed, required {total_time} s')

# Evaluate the model on test data
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_loss = 0.0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move test data to device
        outputs = model(inputs)
        test_loss += loss_function(outputs, labels).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")
