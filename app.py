import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Vehicle Fuel Efficiency Predictor",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("Vehicle Fuel Efficiency (MPG) Predictor")
st.markdown("""
This application predicts a vehicle's fuel efficiency (Miles Per Gallon) based on its specifications
using a machine learning model trained on the Auto MPG dataset.
""")

# Function to load and preprocess data
def load_data():
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
    
    return df

# Load the data
df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict MPG", "Model Performance", "Data Exploration"])

# Define features and target
X = df.drop('mpg', axis=1)
y = df['mpg']

# Save feature names for later use
feature_names = X.columns.tolist()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model if not already saved
model_path = 'mpg_model.pkl'
scaler_path = 'scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    # Load existing model and scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
else:
    # Create a new model and train it
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calculate accuracy-like metric for regression
# (percentage of predictions within 2 MPG of actual value)
accuracy_threshold = 2.0  # within 2 MPG
within_threshold = np.mean(np.abs(y_test - y_pred) <= accuracy_threshold)
accuracy_pct = within_threshold * 100

# Function to predict MPG for new data
def predict_mpg(input_data):
    # Convert input data to DataFrame with correct column order
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return prediction

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    # Get feature importance from model
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
    ax.set_title('Feature Importance')
    plt.tight_layout()
    
    return fig

# Function to find similar vehicles
def find_similar_vehicles(input_data, df, n=10):
    # Create a copy of input data as a dataframe
    input_df = pd.DataFrame([input_data])
    
    # Compute numerical distance for all vehicles (simple Euclidean distance)
    # We'll normalize the values to avoid bias from different scales
    numerical_cols = feature_names
    
    # Create a copy of the features
    df_temp = df[numerical_cols].copy()
    
    # Calculate absolute difference between each row and input data
    distances = []
    for i, row in df.iterrows():
        # Calculate Euclidean distance for numerical columns
        dist = 0
        for col in numerical_cols:
            # Normalize values by feature range
            col_range = df[col].max() - df[col].min()
            # Avoid division by zero
            if col_range == 0:
                col_range = 1
            
            dist += ((row[col] - input_data[col]) / col_range) ** 2
            
        distances.append((i, np.sqrt(dist)))
    
    # Get the indices of the most similar vehicles
    similar_indices = [idx for idx, _ in sorted(distances, key=lambda x: x[1])][1:n+1]
    
    # Return the similar vehicles
    return df.iloc[similar_indices]

# Prediction page
if page == "Predict MPG":
    st.header("Predict Fuel Efficiency")
    st.markdown("Enter vehicle specifications to predict its Miles Per Gallon (MPG).")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            cylinders = st.selectbox("Cylinders", options=[3, 4, 5, 6, 8], index=1)
            displacement = st.slider("Displacement (cubic inches)", 
                                    min_value=float(50), 
                                    max_value=float(500), 
                                    value=float(200))
            horsepower = st.slider("Horsepower", 
                                  min_value=float(40), 
                                  max_value=float(250), 
                                  value=float(100))
            weight = st.slider("Weight (lbs)", 
                              min_value=float(1500), 
                              max_value=float(5000), 
                              value=float(3000))
        
        with col2:
            acceleration = st.slider("Acceleration (seconds to reach 60mph)", 
                                    min_value=float(8), 
                                    max_value=float(25), 
                                    value=float(15))
            model_year = st.slider("Model Year (70-82 represents 1970-1982)", 
                                  min_value=70, 
                                  max_value=82, 
                                  value=76)
            origin = st.selectbox("Origin", 
                                 options=[1, 2, 3], 
                                 index=0, 
                                 help="1 = USA, 2 = Europe, 3 = Asia")
        
        submitted = st.form_submit_button("Predict MPG")
    
    if submitted:
        # Create input data dictionary
        input_data = {
            'cylinders': cylinders,
            'displacement': displacement,
            'horsepower': horsepower,
            'weight': weight,
            'acceleration': acceleration,
            'model year': model_year,
            'origin': origin
        }
        
        # Make prediction
        predicted_mpg = predict_mpg(input_data)
        
        # Display prediction
        st.success(f"## Predicted MPG: {predicted_mpg:.1f}")
        
        # Find similar vehicles
        st.subheader("Similar Vehicles in Dataset")
        similar_vehicles = find_similar_vehicles(input_data, df)
        
        # Add prediction to similar vehicles for comparison
        input_data_with_mpg = input_data.copy()
        input_data_with_mpg['mpg'] = predicted_mpg
        input_df = pd.DataFrame([input_data_with_mpg])
        input_df['Type'] = 'Your Vehicle'
        
        # Add type to similar vehicles
        similar_vehicles_with_type = similar_vehicles.copy()
        similar_vehicles_with_type['Type'] = 'Similar Vehicles'
        
        # Combine for display
        comparison_df = pd.concat([input_df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'Type']], 
                                  similar_vehicles_with_type[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'Type']]])
        
        st.dataframe(comparison_df)
        
        # MPG comparison visualization
        st.subheader("MPG Comparison with Similar Vehicles")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distribution of similar vehicles' MPG
        sns.histplot(similar_vehicles['mpg'], kde=True, ax=ax)
        
        # Plot predicted MPG
        ax.axvline(x=predicted_mpg, color='red', linestyle='--', linewidth=2,
                  label=f'Your Vehicle: {predicted_mpg:.1f} MPG')
        
        # Add mean line
        mean_mpg = similar_vehicles['mpg'].mean()
        ax.axvline(x=mean_mpg, color='green', linestyle='-', linewidth=2,
                  label=f'Average of Similar Vehicles: {mean_mpg:.1f} MPG')
        
        ax.set_xlabel('MPG')
        ax.set_ylabel('Number of Vehicles')
        ax.set_title('How Your Vehicle Compares to Similar Vehicles')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show feature importance
        st.subheader("Feature Importance")
        fig = plot_feature_importance(model, feature_names)
        st.pyplot(fig)

# Model Performance page
elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error", f"{mae:.2f} MPG")
        st.metric("Mean Squared Error", f"{mse:.2f}")
    with col2:
        st.metric("Root Mean Squared Error", f"{rmse:.2f} MPG")
        st.metric("R² Score", f"{r2:.3f}")
    with col3:
        st.metric("Accuracy (within 2 MPG)", f"{accuracy_pct:.1f}%")
        st.markdown(f"*{accuracy_pct:.1f}% of predictions are within 2 MPG of actual value*")
    
    # Plot actual vs. predicted values
    st.subheader("Actual vs. Predicted MPG")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the points
    ax.scatter(y_test, y_pred, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_xlabel('Actual MPG')
    ax.set_ylabel('Predicted MPG')
    ax.set_title('Actual vs. Predicted MPG')
    
    # Add metrics to plot
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.05, 0.87, f'Accuracy (±2 MPG) = {accuracy_pct:.1f}%', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show feature importance
    st.subheader("Feature Importance")
    fig = plot_feature_importance(model, feature_names)
    st.pyplot(fig)
    
    # Show model information
    st.subheader("Model Information")
    st.write(f"**Model Type:** Random Forest Regressor")
    st.write(f"**Number of Trees:** {model.n_estimators}")
    st.write(f"**Max Features:** {model.max_features}")
    st.write(f"**Max Depth:** {'None' if model.max_depth is None else model.max_depth}")

# Data Exploration page
elif page == "Data Exploration":
    st.header("Data Exploration")
    
    # Show raw dataset
    st.subheader("Auto MPG Dataset")
    st.dataframe(df.head(10))
    
    # Dataset statistics
    st.subheader("Dataset Statistics")
    st.write(df.describe())
    
    # Show feature distribution
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select Feature", options=feature_names)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f'Distribution of {feature}')
    st.pyplot(fig)
    
    # Show correlation with MPG
    st.subheader("Correlation with MPG")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df[feature], y=df['mpg'], ax=ax)
    ax.set_xlabel(feature)
    ax.set_ylabel('MPG')
    ax.set_title(f'Correlation between {feature} and MPG')
    
    # Add regression line
    sns.regplot(x=df[feature], y=df['mpg'], ax=ax, scatter=False, color='red')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation = df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    
    # MPG by origin
    st.subheader("MPG by Origin")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='origin', y='mpg', data=df, ax=ax)
    ax.set_xlabel('Origin (1=USA, 2=Europe, 3=Asia)')
    ax.set_ylabel('MPG')
    plt.tight_layout()
    st.pyplot(fig)
    
    # MPG by model year
    st.subheader("MPG Trend Over Model Years")
    fig, ax = plt.subplots(figsize=(12, 6))
    yearly_avg = df.groupby('model year')['mpg'].mean().reset_index()
    sns.lineplot(x='model year', y='mpg', data=yearly_avg, marker='o', ax=ax)
    ax.set_xlabel('Model Year (70-82 represents 1970-1982)')
    ax.set_ylabel('Average MPG')
    ax.set_xticks(range(70, 83))
    plt.tight_layout()
    st.pyplot(fig)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.info("""
### About
This application uses a Random Forest machine learning model to predict vehicle fuel efficiency.

The model was trained on the Auto MPG dataset from the UCI Machine Learning Repository, which includes data from vehicles from the 1970s and early 1980s.
""")
