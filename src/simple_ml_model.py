#!/usr/bin/env python3
"""
Simple Machine Learning Model for Docker Demo
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import pickle

def create_sample_data():
    """Create sample dataset for demonstration"""
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    X = np.random.randn(n_samples, 4)
    
    # Create target variable with some relationship to features
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 - X[:, 2] * 0.5 + X[:, 3] * 0.8 + 
         np.random.randn(n_samples) * 0.1)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    return X, y

def build_model(input_shape):
    """Build a simple neural network model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Main training function"""
    print("Creating sample data...")
    X, y = create_sample_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Building model...")
    model = build_model(X_train.shape[1])
    
    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model and scaler
    os.makedirs('/app/models', exist_ok=True)
    model.save('/app/models/simple_model.h5')
    
    with open('/app/models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model saved successfully!")
    
    return model, history, test_accuracy

if __name__ == "__main__":
    train_model()
