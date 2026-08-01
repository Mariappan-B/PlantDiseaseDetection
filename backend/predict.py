"""TensorFlow Lite model loading and image inference."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from utils import MODEL_DIR, disease_details, load_class_names, readable_label


MODEL_PATH = MODEL_DIR / "plant_disease_model.tflite"
IMAGE_SIZE = 224


@lru_cache(maxsize=1)
def get_interpreter():
    """Load the TensorFlow Lite model only once."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def predict_image(image_path: str | Path) -> dict[str, str | float]:
    """Predict plant disease using TensorFlow Lite."""

    image = (
        Image.open(image_path)
        .convert("RGB")
        .resize((IMAGE_SIZE, IMAGE_SIZE))
    )

    image = np.asarray(image, dtype=np.float32)

    image = tf.keras.applications.efficientnet.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    interpreter, input_details, output_details = get_interpreter()

    interpreter.set_tensor(
        input_details[0]["index"],
        image,
    )

    interpreter.invoke()

    probabilities = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    class_index = int(np.argmax(probabilities))

    class_name = load_class_names()[class_index]

    plant, disease = readable_label(class_name)

    details = disease_details(plant, disease)

    return {
        "plant": plant,
        "disease": disease,
        "confidence": round(
            float(probabilities[class_index] * 100),
            2,
        ),
        **details,
    }