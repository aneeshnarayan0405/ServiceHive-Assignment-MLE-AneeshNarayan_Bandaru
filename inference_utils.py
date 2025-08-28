import numpy as np
from openai import OpenAI
import os
import requests
from PIL import Image
import tensorflow as tf

# Monte Carlo Dropout for uncertainty estimation
def predict_with_uncertainty(model, image, n_iter=50):
    """
    Make predictions with Monte Carlo Dropout for uncertainty estimation
    """
    # Enable dropout at test time
    predictions = np.stack([model(image, training=True) for _ in range(n_iter)])
    mean_prediction = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)
    
    predicted_class = np.argmax(mean_prediction, axis=1)[0]
    confidence = mean_prediction[0, predicted_class]
    uncertainty_score = uncertainty[0, predicted_class]
    
    return predicted_class, confidence, uncertainty_score

# LLM integration for scene description
def generate_scene_description(predicted_class, api_key):
    """
    Generate a scene description using OpenAI API
    """
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Generate a very short, single-sentence description of a {predicted_class} scene, 
    suitable for an alt-text. Do not use the word 'image' or 'photo'. Description:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates concise image descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Fallback descriptions
        fallback_descriptions = {
            'buildings': 'A cityscape with tall structures and architectural details.',
            'forest': 'A dense woodland with various trees and vegetation.',
            'glacier': 'A vast icy landscape with snow-covered formations.',
            'mountain': 'Majestic peaks rising against the sky.',
            'sea': 'Expansive body of water with waves and horizon.',
            'street': 'Urban pathway with buildings and potential activity.'
        }
        return fallback_descriptions.get(predicted_class, 'A natural scene.')

# Image preprocessing
def preprocess_image(image_path, target_size=(150, 150)):
    """
    Preprocess image for model prediction
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array