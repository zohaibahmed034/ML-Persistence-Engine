#!/usr/bin/env python3
"""
Flask API for serving machine learning models
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import os
import json
from datetime import datetime

app = Flask(__name__)

# Global variables for model and preprocessing
model = None
scaler = None
model_metadata = None

def load_model_and_preprocessing():
    """Load model and preprocessing objects at startup"""
    global model, scaler, model_metadata
    
    models_dir = "/app/models"
    
    try:
        # Load the iris model if it exists
        model_path = os.path.join(models_dir, "iris_model.h5")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        
        # Load preprocessing objects
        preprocessing_path = os.path.join(models_dir, "iris_preprocessing.pkl")
        if os.path.exists(preprocessing_path):
            with open(preprocessing_path, 'rb') as f:
                preprocessing_objects = pickle.load(f)
                scaler = preprocessing_objects.get('scaler')
            print(f"Preprocessing objects loaded from {preprocessing_path}")
        
        # Load metadata
        metadata_path = os.path.join(models_dir, "iris_model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            print(f"Metadata loaded from {metadata_path}")
    
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 400
    
    info = {
        "model_loaded": True,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "metadata": model_metadata
    }
    
    return jsonify(info)

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the loaded model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 400
    
    try:
        # Get input data from request
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400
        
        features = np.array(data['features'])
        
        # Ensure correct shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Apply preprocessing if scaler is available
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make prediction
        predictions = model.predict(features)
        
        # Convert to list for JSON serialization
        predictions_list = predictions.tolist()
        
        # For classification, also return predicted classes
        predicted_classes = np.argmax(predictions, axis=1).tolist()
        
        response = {
            "predictions": predictions_list,
            "predicted_classes": predicted_classes,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/iris', methods=['POST'])
def predict_iris():
    """Specific endpoint for iris classification"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 400
    
    try:
        data = request.get_json()
        
        # Expected features: sepal_length, sepal_width, petal_length, petal_width
        required_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        if not all(feature in data for feature in required_features):
            return jsonify({
                "error": f"Missing required features: {required_features}"
            }), 400
        
        # Extract features in correct order
        features = np.array([[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]])
        
        # Apply preprocessing
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make prediction
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Map class to species name
        species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        predicted_species = species_map.get(predicted_class, 'unknown')
        
        response = {
            "predicted_species": predicted_species,
            "predicted_class": int(predicted_class),
            "confidence": confidence,
            "probabilities": {
                "setosa": float(predictions[0][0]),
                "versicolor": float(predictions[0][1]),
                "virginica": float(predictions[0][2])
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load model at startup
    load_model_and_preprocessing()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
