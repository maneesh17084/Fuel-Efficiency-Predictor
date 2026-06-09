# Vehicle Fuel Efficiency (MPG) Predictor

This is a Streamlit application that predicts a vehicle's fuel efficiency (Miles Per Gallon) based on its specifications using a TensorFlow deep learning model.

## Overview

The application uses a neural network model built with TensorFlow to predict the MPG of vehicles based on various specifications such as:

- Number of cylinders
- Displacement
- Horsepower
- Weight
- Acceleration
- Model year
- Origin (USA, Europe, Asia)

The model is trained on the Auto MPG dataset from the UCI Machine Learning Repository, which includes data from vehicles from the 1970s and early 1980s.

## Features

- **MPG Prediction**: Input vehicle specifications and get an estimated MPG
- **Model Performance**: View performance metrics like MAE and MSE
- **Data Exploration**: Explore the dataset with visualizations
- **Feature Importance**: Understand which features contribute most to the prediction
- **Comparison**: See how your vehicle compares to similar vehicles in the dataset

## Technical Implementation

The application is built using:

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **TensorFlow/Keras**: Deep learning library for model building
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **NumPy**: Numerical operations

## Usage

1. Navigate to the "Predict MPG" page
2. Enter the specifications of your vehicle
3. Click "Predict MPG" to get the estimated fuel efficiency
4. Explore the visualizations to understand the prediction

## Model Architecture

The model uses a sequential neural network with:
- Input layer of 7 features
- Two hidden layers with 64 neurons each and ReLU activation
- Output layer with a single neuron for MPG prediction

## Future Improvements

- Support for more modern vehicle specifications
- Integration with real-time vehicle databases
- Additional metrics such as carbon emissions estimates
- Support for different regional fuel efficiency standards (MPG vs. L/100km)
