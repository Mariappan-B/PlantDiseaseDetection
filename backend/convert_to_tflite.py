from pathlib import Path
import tensorflow as tf

MODEL_DIR = Path("saved_model")

keras_model = tf.keras.models.load_model(
    MODEL_DIR / "plant_disease_model.keras"
)

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# Optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

output = MODEL_DIR / "plant_disease_model.tflite"

with open(output, "wb") as f:
    f.write(tflite_model)

print("Saved:", output)