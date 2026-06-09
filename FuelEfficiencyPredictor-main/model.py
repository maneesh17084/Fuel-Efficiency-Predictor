import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_and_train_model(train_dataset, test_dataset, epochs=100):
    """
    Build and train a TensorFlow model for MPG prediction
    
    Args:
        train_dataset: TensorFlow dataset for training
        test_dataset: TensorFlow dataset for validation
        epochs: Number of training epochs
        
    Returns:
        model: Trained TensorFlow model
        history: Training history
    """
    # Build the model architecture
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.element_spec[0].shape)]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    # Compile the model
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    
    # Set up early stopping
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, history.history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained TensorFlow model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        test_loss: Test loss value
        test_mae: Mean absolute error
        test_mse: Mean squared error
        y_pred: Predicted values
        y_test_array: Actual test values as array
    """
    # Convert pandas series to numpy array
    if isinstance(y_test, pd.Series):
        y_test_array = y_test.values
    else:
        y_test_array = y_test
        
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Calculate metrics
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    
    return test_loss, test_mae, test_mse, y_pred, y_test_array

def make_prediction(model, normalized_input):
    """
    Make an MPG prediction for a single input sample
    
    Args:
        model: Trained TensorFlow model
        normalized_input: Normalized input features
        
    Returns:
        prediction: Predicted MPG value
    """
    # Reshape input for prediction (add batch dimension)
    input_array = np.array(normalized_input).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_array)[0, 0]
    
    return float(prediction)
