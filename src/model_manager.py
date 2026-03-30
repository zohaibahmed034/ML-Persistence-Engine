#!/usr/bin/env python3
"""
Model Management Script for Docker Volumes
"""

import tensorflow as tf
import numpy as np
import pickle
import os
import json
from datetime import datetime

class ModelManager:
    def __init__(self, models_dir="/app/models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def save_model_with_metadata(self, model, model_name, metadata=None):
        """Save model with metadata to Docker volume"""
        model_path = os.path.join(self.models_dir, f"{model_name}.h5")
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        
        # Save the model
        model.save(model_path)
        
        # Create metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "model_name": model_name,
            "saved_at": datetime.now().isoformat(),
            "model_path": model_path,
            "tensorflow_version": tf.__version__
        })
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved: {model_path}")
        print(f"Metadata saved: {metadata_path}")
    
    def load_model_with_metadata(self, model_name):
        """Load model with metadata from Docker volume"""
        model_path = os.path.join(self.models_dir, f"{model_name}.h5")
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Load metadata
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        print(f"Model loaded: {model_path}")
        print(f"Metadata: {metadata}")
        
        return model, metadata
    
    def list_saved_models(self):
        """List all saved models in the volume"""
        models = []
        for file in os.listdir(self.models_dir):
            if file.endswith('.h5'):
                model_name = file.replace('.h5', '')
                models.append(model_name)
        
        return models
    
    def save_preprocessing_objects(self, objects_dict, name):
        """Save preprocessing objects like scalers"""
        file_path = os.path.join(self.models_dir, f"{name}_preprocessing.pkl")
        
        with open(file_path, 'wb') as f:
            pickle.dump(objects_dict, f)
        
        print(f"Preprocessing objects saved: {file_path}")
    
    def load_preprocessing_objects(self, name):
        """Load preprocessing objects"""
        file_path = os.path.join(self.models_dir, f"{name}_preprocessing.pkl")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessing objects not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            objects = pickle.load(f)
        
        print(f"Preprocessing objects loaded: {file_path}")
        return objects

def demo_model_persistence():
    """Demonstrate model persistence with Docker volumes"""
    print("=== Model Persistence Demo ===")
    
    # Initialize model manager
    manager = ModelManager()
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create dummy data for demonstration
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randint(0, 2, 100)
    
    # Train briefly
    print("Training demo model...")
    model.fit(X_dummy, y_dummy, epochs=5, verbose=0)
    
    # Save model with metadata
    metadata = {
        "description": "Demo model for Docker volume persistence",
        "input_shape": [10],
        "output_shape": [1],
        "task_type": "binary_classification"
    }
    
    manager.save_model_with_metadata(model, "demo_model", metadata)
    
    # Save preprocessing objects
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    
    preprocessing_objects = {
        "scaler": scaler,
        "feature_names": [f"feature_{i}" for i in range(10)]
    }
    
    manager.save_preprocessing_objects(preprocessing_objects, "demo_model")
    
    # List saved models
    print("\nSaved models:")
    for model_name in manager.list_saved_models():
        print(f"  - {model_name}")
    
    # Load model back
    print("\nLoading model...")
    loaded_model, loaded_metadata = manager.load_model_with_metadata("demo_model")
    loaded_preprocessing = manager.load_preprocessing_objects("demo_model")
    
    # Test loaded model
    print("\nTesting loaded model...")
    test_input = np.random.randn(1, 10)
    prediction = loaded_model.predict(test_input, verbose=0)
    print(f"Prediction: {prediction[0][0]:.4f}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demo_model_persistence()
