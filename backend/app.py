"""Flask API for plant disease prediction."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import UnidentifiedImageError
from werkzeug.utils import secure_filename

from predict import predict_image


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
CORS(app)


def is_allowed_file(filename: str) -> bool:
    """Check an uploaded filename against accepted image extensions."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    """Serve the single-page interface."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.post("/predict")
def predict():
    """Validate an uploaded image and return CNN prediction JSON."""
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please choose an image file."}), 400
    if not is_allowed_file(uploaded_file.filename):
        return jsonify({"error": "Only PNG, JPG, JPEG, and WEBP images are allowed."}), 400

    suffix = Path(secure_filename(uploaded_file.filename)).suffix.lower()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            uploaded_file.save(temp_path)
        result = predict_image(temp_path)
        return jsonify(result)
    except UnidentifiedImageError:
        return jsonify({"error": "The selected file is not a valid image."}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Image is too large. Maximum file size is 8 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)