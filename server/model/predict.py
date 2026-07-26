import os
import json
import numpy as np
import cv2
import tensorflow as tf

class ModelPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "model.keras")
        self.classes_path = os.path.join(model_dir, "class_indices.json")
        self.model = None
        self.class_mapping = {"0": "NORMAL", "1": "PNEUMONIA"} # Default fallback
        self.load_model()

    def load_model(self):
        """Loads Keras model and class indices JSON"""
        if not os.path.exists(self.model_path):
            print(f"Model file not found at {self.model_path}. Model needs training.")
            return False
            
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("Successfully loaded Keras model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

        if os.path.exists(self.classes_path):
            try:
                with open(self.classes_path, "r") as f:
                    indices = json.load(f)
                    # Reverse map indices {class_name: index} to {index: class_name}
                    self.class_mapping = {str(v): k for k, v in indices.items()}
                print(f"Loaded class mapping: {self.class_mapping}")
            except Exception as e:
                print(f"Error loading class mapping file: {e}")
        return True

    def predict_image(self, img_path):
        """
        Preprocesses and predicts the image.
        Returns:
            predicted_class (str): label of predicted class
            confidence (float): float confidence score [0, 1]
            probabilities (dict): label to confidence percentage mapping
        """
        if self.model is None:
            # Try to re-load model
            loaded = self.load_model()
            if not loaded:
                raise ValueError("Model is not trained or loaded. Inference aborted.")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Input image not found: {img_path}")

        # Preprocessing: Read grayscale (matching model's input channel of 1)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to decode image: {img_path}")
            
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_input = np.expand_dims(np.expand_dims(img_normalized, axis=-1), axis=0) # Shape: (1, 224, 224, 1)

        # Run forward pass
        preds = self.model.predict(img_input)[0]
        
        pred_idx = int(np.argmax(preds))
        predicted_class = self.class_mapping.get(str(pred_idx), "UNKNOWN")
        confidence = float(preds[pred_idx])

        probabilities = {}
        for idx, score in enumerate(preds):
            class_name = self.class_mapping.get(str(idx), f"CLASS_{idx}")
            probabilities[class_name] = float(score)

        return predicted_class, confidence, probabilities

# Singleton instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = ModelPredictor()
    return _predictor
