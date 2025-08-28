import tensorflow as tf
import numpy as np

def preprocess_image(image_path, target_size=(150, 150)):
    """Preprocess image for model prediction"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array