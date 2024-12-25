import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import AutoencoderModel  # Corrected import
from src.data_preprocessing import load_data, preprocess_data

def train_model(data, model, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def main():
    # Load and preprocess data
    processed_data_path = 'd:/Projects/anomaly-detection-project/data/processed/processed_data.csv'
    data = load_data(processed_data_path)

    # Convert to tensor
    data_tensor = torch.FloatTensor(data['value'].values).unsqueeze(1)

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data_tensor, test_size=0.2, random_state=42)

    # Initialize model, criterion, and optimizer
    model = AutoencoderModel(input_size=train_data.shape[1])  # Update input_size as needed
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(train_data, model, criterion, optimizer)

    # Save the trained model
    torch.save(model.state_dict(), 'd:/Projects/anomaly-detection-project/models/autoencoder.pth')

if __name__ == '__main__':
    main()