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
    """Load the trained model only once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path: str | Path) -> dict[str, str | float]:
    """Run prediction on a leaf image."""

    image = (
        Image.open(image_path)
        .convert("RGB")
        .resize((IMAGE_SIZE, IMAGE_SIZE))
    )

    image = np.asarray(image, dtype=np.float32)

    # Same preprocessing used during EfficientNet training
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    batch = np.expand_dims(image, axis=0)

    model = get_model()

    probabilities = model.predict(batch, verbose=0)[0]

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