import numpy as np
import tensorflow as tf
from preprocessing.utils import preprocess_image
from inference_utils import predict_with_uncertainty, generate_scene_description
import os

def load_model(model_path):
    """Load the trained model"""
    return tf.keras.models.load_model(model_path)

def predict_image(image_path, model):
    """Make prediction on a single image"""
    # Preprocess image
    processed_image = preprocess_image(image_path)
    
    # Get prediction with uncertainty
    class_idx, confidence, uncertainty = predict_with_uncertainty(model, processed_image)
    
    # Map index to class name
    class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    predicted_class = class_names[class_idx]
    
    # Generate description (using environment variable for API key)
    api_key = os.getenv("OPENAI_API_KEY", "")
    description = generate_scene_description(predicted_class, api_key)
    
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "uncertainty": float(uncertainty),
        "description": description,
        "model_name": "EfficientNetB0 Transfer Learning"
    }