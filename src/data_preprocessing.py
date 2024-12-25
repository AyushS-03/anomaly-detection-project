import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    return data

def preprocess_data(data):
    # Normalize the 'value' column
    data['value'] = normalize_data(data[['value']])
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def scale_data(data, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def create_windows(data, window_size):
    import numpy as np
    windows = []
    for i in range(len(data) - window_size):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def main():
    raw_data_path = 'd:/Projects/anomaly-detection-project/data/raw/art_daily_jumpsup.csv'
    processed_data_path = 'd:/Projects/anomaly-detection-project/data/processed/processed_data.csv'

    # Load and preprocess data
    data = load_data(raw_data_path)
    processed_data = preprocess_data(data)

    # Save processed data
    processed_data.to_csv(processed_data_path)

if __name__ == '__main__':
    main()