import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Load and preprocess the Auto MPG dataset
    
    Args:
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        X_train: Training features (normalized)
        X_test: Test features (normalized)
        y_train: Training target
        y_test: Test target
        X_train_raw: Training features (not normalized)
        X_test_raw: Test features (not normalized)
        train_dataset: TensorFlow dataset for training
        test_dataset: TensorFlow dataset for testing
        train_stats: Statistics for normalization
        feature_names: List of feature names
    """
    # Load the dataset
    df = pd.read_csv('auto-mpg.csv')
    
    # Clean the data
    # Replace '?' with NaN and convert to numeric
    df = df.replace('?', np.nan)
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Drop the car name column
    df = df.drop('car name', axis=1)
    
    # Split features and target
    X = df.drop('mpg', axis=1)
    y = df['mpg']
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    
    # Split into train and test sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Normalize the features
    train_stats = X_train_raw.describe().transpose()
    
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    
    X_train = norm(X_train_raw)
    X_test = norm(X_test_raw)
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values)).batch(32)
    
    return X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, train_dataset, test_dataset, train_stats, feature_names

def normalize_input_data(input_data, train_stats, feature_names):
    """
    Normalize input data using training statistics
    
    Args:
        input_data: Dictionary of input feature values
        train_stats: Training statistics for normalization
        feature_names: List of feature names
        
    Returns:
        normalized_input: List of normalized feature values
    """
    # Create normalized input list in the correct order
    normalized_input = []
    for feature in feature_names:
        # Normalize using mean and std from training data
        value = (input_data[feature] - train_stats.loc[feature, 'mean']) / train_stats.loc[feature, 'std']
        normalized_input.append(value)
    
    return normalized_input
