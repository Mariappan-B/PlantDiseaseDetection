# Plant Doctor — AI-Based Plant Disease Detection

Plant Doctor is a final-year project application that classifies plant-leaf images using a TensorFlow/Keras convolutional neural network (CNN). A Flask API serves predictions to a responsive Bootstrap interface.

## Features

- Upload, drag-and-drop, preview, and reset leaf images
- CNN-based PlantVillage class prediction with confidence percentage
- Disease description, causes, prevention, and treatment guidance
- Image validation, file-size limits, loading feedback, and API error handling
- Training workflow with augmentation, Batch Normalization, pooling, dropout, `EarlyStopping`, and `ModelCheckpoint`

## Technologies Used

- Python, TensorFlow, Keras, NumPy, Pillow
- Flask and Flask-CORS
- HTML5, CSS3, Bootstrap 5, and vanilla JavaScript

## Folder Structure

```text
PlantDiseaseDetection/
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   ├── utils.py
│   ├── requirements.txt
│   └── saved_model/          # Generated after training
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   ├── js/script.js
│   └── images/
├── dataset/
│   └── PlantVillage/         # Add dataset class folders here
└── README.md
```

## Installation

1. Open a terminal in the `backend` directory.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the PlantVillage dataset and place its class folders under `dataset/PlantVillage`. Each class should be a separate directory such as `Tomato___Early_blight`.

## Training

From `backend`, run:

```bash
python train.py --dataset ../dataset/PlantVillage --epochs 25
```

The best model is saved as `backend/saved_model/plant_disease_model.keras`. Its class order is saved in `class_names.json`; do not remove this file, because inference uses it to interpret model outputs.

## Running Flask

After a model has been trained, start the application from `backend`:

```bash
python app.py
```

Open `http://localhost:5000` in a browser. Flask serves the frontend and the `/predict` API from the same origin.

## API Documentation

### `POST /predict`

Submit `multipart/form-data` with an `image` field. PNG, JPG, JPEG, and WEBP files up to 8 MB are supported.

Example response:

```json
{
  "plant": "Tomato",
  "disease": "Early blight",
  "confidence": 98.7,
  "description": "The image shows symptoms consistent with Early blight on Tomato.",
  "causes": "This condition can be associated with plant pathogens...",
  "prevention": "Use disease-free planting material...",
  "treatment": "Consult a local agriculture extension officer..."
}
```

If a model is not yet available, the API responds with HTTP 503 and a clear setup message.
