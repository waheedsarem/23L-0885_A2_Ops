"""Startup-loaded model inference with training-matched preprocessing."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class InvalidImageError(ValueError):
    pass


class InferenceService:
    def __init__(self, model_path: str | Path, threshold: float = 0.825):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.threshold = float(threshold)
        if not 0 < self.threshold < 1:
            raise ValueError("MODEL_THRESHOLD must be between 0 and 1")
        # Loaded exactly once when the Flask process starts.
        from tensorflow import keras
        self.model = keras.models.load_model(self.model_path)

    @staticmethod
    def preprocess(image_bytes: bytes) -> np.ndarray:
        import tensorflow as tf
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(BytesIO(image_bytes)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("The uploaded file cannot be decoded as a valid image.") from exc
        tensor = tf.convert_to_tensor(rgb)
        # Matches training: RGB float values in [0,255], 160x160 resize-with-pad,
        # bilinear interpolation and antialiasing. Model-specific scaling is inside the model.
        tensor = tf.image.resize_with_pad(tensor, 160, 160, antialias=True)
        return tf.expand_dims(tensor, 0).numpy()

    def predict_bytes(self, image_bytes: bytes) -> dict:
        batch = self.preprocess(image_bytes)
        ai_probability = float(self.model.predict(batch, verbose=0).reshape(-1)[0])
        is_ai = ai_probability >= self.threshold
        return {
            "label": "AI-generated" if is_ai else "Real",
            "confidence": ai_probability if is_ai else 1.0 - ai_probability,
            "ai_probability": ai_probability,
            "threshold": self.threshold,
        }
