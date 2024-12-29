# Anomaly Detection Project

This project focuses on detecting anomalies in datasets using various machine learning techniques.

## Project Structure

- `data/`: Contains raw and processed sensor data.
  - `raw/`: Directory for raw sensor data files.
  - `processed/`: Directory for preprocessed sensor data.
  - `README.md`: Documentation on the data structure and contents.

- `models/`: Contains implementations of anomaly detection models.
  - `model.py`: Implementation of models like Isolation Forest, One-Class SVM, Autoencoders, and Variational Autoencoders.
  - `__init__.py`: Initializes the models package.

- `src/`: Contains source code for data processing, training, and evaluation.
  - `data_preprocessing.py`: Functions for data normalization, scaling, and windowing.
  - `train.py`: Responsible for training the models and saving them.
  - `evaluate.py`: Functions to evaluate models using various metrics.
  - `utils.py`: Utility functions for data loading, visualization, and model management.

- `requirements.txt`: Lists the required Python dependencies for the project.

- `.gitignore`: Specifies files and directories to be ignored by version control.

## Requirements

To install the required packages, run:
```
pip install -r requirements.txt
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd anomaly-detection-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your data by placing raw sensor data files in the `data/raw/` directory.

4. Run the data preprocessing script to prepare the data for training:
   ```
   python src/data_preprocessing.py
   ```

5. Train the models using:
   ```
   python src/train.py
   ```

6. Evaluate the trained models:
   ```
   python src/evaluate.py
   ```

## Visualization

![image](https://github.com/user-attachments/assets/6392f671-9502-4223-928a-e4d0fcdceba6)
![image](https://github.com/user-attachments/assets/0eb2ebbf-ad0d-451c-bf32-bb4ab54c84b8)
![image](https://github.com/user-attachments/assets/74f46605-e186-4d71-82ba-2f094cf6942d)
![image](https://github.com/user-attachments/assets/20393495-36f1-4c87-ad57-52eeb039d49a)


### Visualization Types

1. **ROC Curve**: This plot shows the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity) for different threshold values.
2. **Histogram of Reconstruction Errors**: This histogram displays the distribution of reconstruction errors, helping to identify the threshold for anomaly detection.
3. **Time Series Plot of Reconstruction Errors**: This plot shows the reconstruction errors over time, with a threshold line to identify anomalies.
4. **Box Plot of Reconstruction Errors**: This box plot provides a summary of the distribution of reconstruction errors, highlighting outliers.

## Usage

The models can be imported from the `models/` package for further experimentation and deployment.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
