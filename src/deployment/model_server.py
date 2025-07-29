#!/usr/bin/env python3
"""
Model deployment script for MODIS land cover classification.
Provides REST API endpoint for land cover prediction.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from flask import Flask, jsonify, request

from train_random_forest import LandCoverClassifier, MODISFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global model variable
model_data = None
feature_extractor = None


def load_model(model_path: str) -> Dict:
    """Load the trained model."""
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        return model_data
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def initialize_model():
    """Initialize the model and feature extractor."""
    global model_data, feature_extractor

    model_path = os.getenv("MODEL_PATH", "random_forest_full_dataset_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_data = load_model(model_path)
    feature_extractor = MODISFeatureExtractor(Path("data/processed"))

    logger.info("Model and feature extractor initialized successfully")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model_data is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """Predict land cover class for given features."""
    try:
        if model_data is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get features from request
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Features not provided"}), 400

        features = np.array(data["features"])

        # Validate feature dimensions
        expected_features = len(model_data["feature_names"])
        if features.shape[-1] != expected_features:
            return (
                jsonify(
                    {"error": f"Expected {expected_features} features, got {features.shape[-1]}"}
                ),
                400,
            )

        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale features
        features_scaled = model_data["scaler"].transform(features)

        # Make prediction
        predictions = model_data["model"].predict(features_scaled)
        probabilities = model_data["model"].predict_proba(features_scaled)

        # Convert predictions back to original labels
        original_predictions = model_data["label_encoder"].inverse_transform(predictions)

        # Get class names
        classifier = LandCoverClassifier()
        class_names = [
            classifier.class_names.get(pred, f"Class_{pred}") for pred in original_predictions
        ]

        # Prepare response
        response = {
            "predictions": original_predictions.tolist(),
            "class_names": class_names,
            "probabilities": probabilities.tolist(),
            "n_samples": len(predictions),
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_from_coordinates", methods=["POST"])
def predict_from_coordinates():
    """Predict land cover for given coordinates (requires MODIS data)."""
    try:
        if model_data is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        if "lat" not in data or "lon" not in data or "year" not in data:
            return jsonify({"error": "lat, lon, and year are required"}), 400

        lat = float(data["lat"])
        lon = float(data["lon"])
        year = int(data["year"])

        # Load MODIS data for the specified year
        tiff_file = Path(f"data/processed/{year}.tif")

        if not tiff_file.exists():
            return jsonify({"error": f"Data not available for year {year}"}), 404

        # Extract features from coordinates
        with rasterio.open(tiff_file) as src:
            # Convert lat/lon to pixel coordinates
            row, col = src.index(lon, lat)

            # Extract patch around coordinates
            patch_size = 5
            r_start = max(0, row - patch_size // 2)
            r_end = min(src.height, row + patch_size // 2 + 1)
            c_start = max(0, col - patch_size // 2)
            c_end = min(src.width, col + patch_size // 2 + 1)

            patch = src.read(1, window=((r_start, r_end), (c_start, c_end)))

            if patch.size == 0:
                return jsonify({"error": "No data at specified coordinates"}), 404

            # Extract features
            spectral_features = feature_extractor.extract_spectral_features(patch)
            spatial_features = feature_extractor.extract_spatial_features(patch)
            texture_features = feature_extractor.extract_texture_features(patch)

            # Combine features
            features = np.concatenate(
                [
                    spectral_features,
                    spatial_features,
                    texture_features,
                    [row / src.height, col / src.width],  # Normalized coordinates
                ]
            ).reshape(1, -1)

        # Make prediction using existing predict logic
        features_scaled = model_data["scaler"].transform(features)
        prediction = model_data["model"].predict(features_scaled)[0]
        probabilities = model_data["model"].predict_proba(features_scaled)[0]

        original_prediction = model_data["label_encoder"].inverse_transform([prediction])[0]

        classifier = LandCoverClassifier()
        class_name = classifier.class_names.get(original_prediction, f"Class_{original_prediction}")

        response = {
            "coordinates": {"lat": lat, "lon": lon},
            "year": year,
            "prediction": int(original_prediction),
            "class_name": class_name,
            "confidence": float(np.max(probabilities)),
            "all_probabilities": probabilities.tolist(),
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Coordinate prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    """Get model information."""
    if model_data is None:
        return jsonify({"error": "Model not loaded"}), 500

    classifier = LandCoverClassifier()

    info = {
        "algorithm": model_data["algorithm"],
        "feature_names": model_data["feature_names"],
        "n_features": len(model_data["feature_names"]),
        "n_classes": len(model_data["label_encoder"].classes_),
        "class_names": {int(k): v for k, v in classifier.class_names.items()},
        "model_type": type(model_data["model"]).__name__,
    }

    return jsonify(info)


if __name__ == "__main__":
    # Initialize model
    initialize_model()

    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("DEBUG", "False").lower() == "true"

    logger.info(f"Starting model server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
