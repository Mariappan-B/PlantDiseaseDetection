"""Model loading and image inference functions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from utils import MODEL_DIR, disease_details, load_class_names, readable_label


MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
IMAGE_SIZE = 224


@lru_cache(maxsize=1)
def get_model() -> tf.keras.Model:
    """Load the saved Keras model only once per Flask process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train.py before making predictions."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path: str | Path) -> dict[str, str | float]:
    """Predict a PlantVillage class and return API-ready disease information."""
    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    probabilities = get_model().predict(batch, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    class_name = load_class_names()[class_index]
    plant, disease = readable_label(class_name)
    details = disease_details(plant, disease)
    return {
        "plant": plant,
        "disease": disease,
        "confidence": round(float(probabilities[class_index] * 100), 2),
        **details,
    }
