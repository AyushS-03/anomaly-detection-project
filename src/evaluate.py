import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import AutoencoderModel  # Ensure this matches the class name in model.py
from src.data_preprocessing import load_data

def evaluate_model(model, data, threshold):
    model.eval()
    with torch.no_grad():
        reconstructed = model(data)
        mse = torch.mean((reconstructed - data) ** 2, dim=1)
        anomalies = mse > threshold
    return mse, anomalies, reconstructed

def main():
    # Load the preprocessed data
    processed_data_path = 'd:/Projects/anomaly-detection-project/data/processed/processed_data.csv'
    data = load_data(processed_data_path)
    data_tensor = torch.FloatTensor(data['value'].values).unsqueeze(1)

    # Load the trained model
    model = AutoencoderModel(input_size=data_tensor.shape[1])
    model.load_state_dict(torch.load('d:/Projects/anomaly-detection-project/models/autoencoder.pth'))

    # Set a threshold for anomaly detection
    threshold = 0.001  # Adjust based on your analysis

    # Evaluate the model
    mse, anomalies, reconstructed = evaluate_model(model, data_tensor, threshold)
    print(f'Mean Squared Error: {mse.mean().item():.4f}')

    # Check if 'label' column exists for calculating evaluation metrics
    if 'label' in data.columns:
        # Calculate evaluation metrics
        precision, recall, f1, _ = precision_recall_fscore_support(data['label'], anomalies, average='binary')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(data['label'], mse)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    # Convert anomalies to boolean array for indexing
    anomalies = anomalies.numpy().astype(bool)

    # Save the evaluation results
    results = pd.DataFrame({
        'timestamp': data.index,
        'original_value': data['value'],
        'reconstructed_value': reconstructed.numpy().flatten(),
        'anomaly': anomalies.astype(int)
    })
    results.to_csv('d:/Projects/anomaly-detection-project/data/evaluation_results.csv', index=False)

    # Plot original and reconstructed values with anomalies
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['value'], label='Original')
    plt.plot(data.index, reconstructed.numpy(), label='Reconstructed')
    plt.scatter(data.index[anomalies], data['value'][anomalies], color='red', label='Anomalies')
    plt.legend()
    plt.show()

    # Additional Visualizations

    # Histogram of Reconstruction Errors
    plt.figure(figsize=(10, 6))
    plt.hist(mse.numpy(), bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Reconstruction Errors')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.show()

    # Time Series Plot of Reconstruction Errors
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, mse.numpy(), label='Reconstruction Error')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Time Series of Reconstruction Errors')
    plt.xlabel('Timestamp')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.show()

    # Box Plot of Reconstruction Errors
    plt.figure(figsize=(10, 6))
    plt.boxplot(mse.numpy(), vert=False)
    plt.title('Box Plot of Reconstruction Errors')
    plt.xlabel('Reconstruction Error')
    plt.show()

if __name__ == '__main__':
    main()