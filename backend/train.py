"""Train the CNN using PlantVillage folders as class labels.

Expected dataset layout:
dataset/PlantVillage/<class_folder>/<image files>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from model import build_model
from utils import MODEL_DIR, save_class_names


IMAGE_SIZE = 224
BATCH_SIZE = 32


def create_datasets(dataset_path: Path):
    """Create reproducible training and validation datasets."""

    common = {
        "validation_split": 0.2,
        "seed": 42,
        "image_size": (IMAGE_SIZE, IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "label_mode": "int",
    }

    train_data = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        subset="training",
        **common,
    )

    validation_data = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        subset="validation",
        **common,
    )

    # Save class names before caching/prefetching
    class_names = train_data.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_data = (
        train_data
        .cache()
        .shuffle(1000)
        .prefetch(AUTOTUNE)
    )

    validation_data = (
        validation_data
        .cache()
        .prefetch(AUTOTUNE)
    )

    return train_data, validation_data, class_names


def main():

    parser = argparse.ArgumentParser(
        description="Train Plant Disease Detection Model"
    )

    parser.add_argument(
        "--dataset",
        default="../dataset/PlantVillage",
        help="Path to dataset",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    train_data, validation_data, class_names = create_datasets(dataset_path)

    save_class_names(class_names)

    model = build_model(
        num_classes=len(class_names),
        image_size=IMAGE_SIZE,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            verbose=1,
            min_lr=1e-6,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_DIR / "plant_disease_model.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("\nTraining completed successfully.")
    print(f"Model saved to: {MODEL_DIR / 'plant_disease_model.keras'}")


if __name__ == "__main__":
    main()