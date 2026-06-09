import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_history(history):
    """
    Plot training and validation metrics
    
    Args:
        history: Training history dictionary
        
    Returns:
        fig: Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot MAE
    ax2.plot(history['mae'], label='Training MAE')
    ax2.plot(history['val_mae'], label='Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error (MPG)')
    ax2.set_title('Training and Validation MAE')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plot actual vs predicted values
    
    Args:
        y_true: True MPG values
        y_pred: Predicted MPG values
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the points
    ax.scatter(y_true, y_pred, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_xlabel('Actual MPG')
    ax.set_ylabel('Predicted MPG')
    ax.set_title('Actual vs. Predicted MPG')
    
    # Calculate and show R^2
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    r_squared = correlation**2
    ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_feature_importance(input_data, train_stats, feature_names):
    """
    Visualize feature values compared to dataset statistics
    
    Args:
        input_data: Dictionary of input feature values
        train_stats: Training statistics
        feature_names: List of feature names
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate normalized values (0-1 scale based on min-max)
    normalized_values = []
    for feature in feature_names:
        min_val = train_stats.loc[feature, 'min']
        max_val = train_stats.loc[feature, 'max']
        # Normalize to 0-1 scale
        norm_value = (input_data[feature] - min_val) / (max_val - min_val)
        normalized_values.append(norm_value)
    
    # Plot as horizontal bars
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, normalized_values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Normalized Value (0-1 scale)')
    ax.set_title('Feature Values Relative to Dataset Range')
    
    # Add value labels
    for i, v in enumerate(normalized_values):
        ax.text(max(v + 0.03, 0.1), i, f"{input_data[feature_names[i]]}", 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_prediction_comparison(X_train, y_train, input_data, predicted_mpg):
    """
    Plot how the prediction compares to similar vehicles
    
    Args:
        X_train: Training features
        y_train: Training targets
        input_data: Dictionary of input feature values
        predicted_mpg: Predicted MPG value
        
    Returns:
        fig: Matplotlib figure
    """
    # Create dataframe from training data
    train_df = X_train.copy()
    train_df['mpg'] = y_train
    
    # Find similar vehicles based on key features (weight and horsepower)
    weight_range = (input_data['weight'] * 0.85, input_data['weight'] * 1.15)
    hp_range = (input_data['horsepower'] * 0.85, input_data['horsepower'] * 1.15)
    
    similar = train_df[
        (train_df['weight'] >= weight_range[0]) & 
        (train_df['weight'] <= weight_range[1]) &
        (train_df['horsepower'] >= hp_range[0]) & 
        (train_df['horsepower'] <= hp_range[1])
    ]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot similar vehicles
    sns.histplot(similar['mpg'], bins=10, kde=True, ax=ax)
    
    # Plot predicted value
    ax.axvline(x=predicted_mpg, color='red', linestyle='--', linewidth=2, 
               label=f'Predicted MPG: {predicted_mpg:.1f}')
    
    # Add mean value line
    mean_mpg = similar['mpg'].mean()
    ax.axvline(x=mean_mpg, color='green', linestyle='-', linewidth=2,
               label=f'Average MPG of similar vehicles: {mean_mpg:.1f}')
    
    ax.set_title('How Your Vehicle Compares to Similar Vehicles')
    ax.set_xlabel('MPG')
    ax.set_ylabel('Number of Vehicles')
    ax.legend()
    
    plt.tight_layout()
    return fig
