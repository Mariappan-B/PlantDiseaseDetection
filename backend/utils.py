"""Shared helpers for dataset labels and disease guidance."""

from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"
CLASS_NAMES_FILE = MODEL_DIR / "class_names.json"


def readable_label(class_name: str) -> tuple[str, str]:
    """Convert PlantVillage folder names into friendly plant and disease names."""
    label = class_name.replace("___", "|").replace("_", " ")
    plant, separator, disease = label.partition("|")
    return plant.strip(), (disease if separator else "Healthy").strip()


def load_class_names() -> list[str]:
    """Load the class order stored alongside the trained model."""
    if not CLASS_NAMES_FILE.exists():
        raise FileNotFoundError(
            "Class metadata is missing. Train the model first with train.py."
        )
    with CLASS_NAMES_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_class_names(class_names: list[str]) -> None:
    """Persist model output ordering for reliable inference."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with CLASS_NAMES_FILE.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)


def disease_details(plant: str, disease: str) -> dict[str, str]:
    """Return practical, readable guidance for a classified condition."""
    if disease.lower() == "healthy":
        return {
            "description": f"The uploaded {plant} leaf appears healthy.",
            "causes": "No disease symptoms were identified by the model.",
            "prevention": "Continue regular monitoring, balanced nutrition, and good field sanitation.",
            "treatment": "No treatment is needed. Maintain normal crop care practices.",
        }

    disease_name = disease.replace("_", " ")
    return {
        "description": f"The image shows symptoms consistent with {disease_name} on {plant}.",
        "causes": "This condition can be associated with plant pathogens, humid conditions, infected crop residue, or pest activity.",
        "prevention": "Use disease-free planting material, improve airflow, avoid wet foliage, rotate crops, and remove infected leaves.",
        "treatment": "Consult a local agriculture extension officer for a crop-approved treatment. Follow label directions and local regulations before applying any product.",
    }
