#!/usr/bin/env python3
"""
Random Forest based land cover classification for MODIS MCD12C1 dataset.
Uses traditional machine learning approach with feature extraction from MODIS data.

This approach is more efficient than deep learning and suitable for land cover classification
with good interpretability and faster training times.

Author: NASA Land Cover Classification Team
Date: January 2025
"""

import argparse
import logging
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. XGBoost algorithm will be disabled.")

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not available. Using PIL instead.")
    from PIL import Image

try:
    import mlflow
    import mlflow.sklearn

    HAS_MLFLOW = True  # Enable MLflow tracking
except ImportError:
    HAS_MLFLOW = False
    print("Warning: MLflow not available. Experiment tracking will be disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: Plotting libraries not available. Plots will be skipped.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MODISFeatureExtractor:
    """
    Feature extraction from MODIS land cover data for traditional ML algorithms.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.features = []
        self.labels = []
        self.feature_names = []

    def extract_spectral_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract spectral features from MODIS data.

        Args:
            data: Input data array of shape (H, W) for single band

        Returns:
            Feature vector with spectral statistics
        """
        features = []

        # Basic statistics
        features.extend(
            [
                np.mean(data),
                np.std(data),
                np.median(data),
                np.min(data),
                np.max(data),
                np.percentile(data, 25),
                np.percentile(data, 75),
                np.var(data),
            ]
        )

        return np.array(features)

    def extract_spatial_features(self, data: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Extract spatial features using local neighborhood statistics.

        Args:
            data: Input data array
            window_size: Size of the neighborhood window

        Returns:
            Spatial feature statistics
        """
        from scipy import ndimage

        features = []

        # Local variance
        local_var = ndimage.generic_filter(data.astype(float), np.var, size=window_size)
        features.extend([np.mean(local_var), np.std(local_var)])

        # Edge detection
        sobel_x = ndimage.sobel(data, axis=0)
        sobel_y = ndimage.sobel(data, axis=1)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        features.extend([np.mean(gradient_magnitude), np.std(gradient_magnitude)])

        return np.array(features)

    def extract_texture_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract texture features using GLCM (Gray Level Co-occurrence Matrix).
        Simplified version for computational efficiency.
        """
        features = []

        # Simple texture measures
        # Local standard deviation
        from scipy import ndimage

        local_std = ndimage.generic_filter(data.astype(float), np.std, size=3)
        features.extend([np.mean(local_std), np.std(local_std)])

        # Range filter
        local_range = ndimage.maximum_filter(data, size=3) - ndimage.minimum_filter(data, size=3)
        features.extend([np.mean(local_range), np.std(local_range)])

        return np.array(features)

    def create_feature_names(self) -> List[str]:
        """Create descriptive names for all features."""
        names = []

        # Spectral features
        spectral_names = [
            "mean",
            "std",
            "median",
            "min",
            "max",
            "q25",
            "q75",
            "variance",
        ]
        names.extend([f"spectral_{name}" for name in spectral_names])

        # Spatial features
        spatial_names = [
            "local_var_mean",
            "local_var_std",
            "gradient_mean",
            "gradient_std",
        ]
        names.extend(spatial_names)

        # Texture features
        texture_names = [
            "texture_std_mean",
            "texture_std_std",
            "texture_range_mean",
            "texture_range_std",
        ]
        names.extend(texture_names)

        return names

    def process_single_file(
        self, tiff_file: Path, sample_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single TIFF file and extract features.

        Args:
            tiff_file: Path to TIFF file
            sample_ratio: Ratio of pixels to sample (for computational efficiency)

        Returns:
            Tuple of (features, labels)
        """
        try:
            # Try to load with rasterio first, fallback to PIL
            if HAS_RASTERIO:
                with rasterio.open(tiff_file) as src:
                    data = src.read(1)  # Read first band (land cover labels)
            else:
                # Use PIL as fallback
                with Image.open(tiff_file) as img:
                    data = np.array(img)

            # Remove no-data values
            valid_mask = (data != 255) & (data != 0)  # Exclude no-data and water
            valid_indices = np.where(valid_mask)

            if len(valid_indices[0]) == 0:
                logger.warning(f"No valid data in {tiff_file}")
                return np.array([]), np.array([])

            # Sample pixels for computational efficiency
            n_valid = len(valid_indices[0])
            n_sample = int(n_valid * sample_ratio)
            if n_sample < 1000:  # Minimum sample size
                n_sample = min(1000, n_valid)

            sample_idx = np.random.choice(n_valid, size=n_sample, replace=False)
            sample_rows = valid_indices[0][sample_idx]
            sample_cols = valid_indices[1][sample_idx]

            # Extract features for each sampled pixel
            features_list = []
            labels_list = []

            for i, (row, col) in enumerate(zip(sample_rows, sample_cols)):
                # Extract local patch around pixel
                patch_size = 5
                r_start = max(0, row - patch_size // 2)
                r_end = min(data.shape[0], row + patch_size // 2 + 1)
                c_start = max(0, col - patch_size // 2)
                c_end = min(data.shape[1], col + patch_size // 2 + 1)

                patch = data[r_start:r_end, c_start:c_end]

                if patch.size == 0:
                    continue

                # Extract features
                spectral_features = self.extract_spectral_features(patch)
                spatial_features = self.extract_spatial_features(patch)
                texture_features = self.extract_texture_features(patch)

                # Combine all features
                pixel_features = np.concatenate(
                    [
                        spectral_features,
                        spatial_features,
                        texture_features,
                        [
                            row / data.shape[0],
                            col / data.shape[1],
                        ],  # Normalized coordinates
                    ]
                )

                features_list.append(pixel_features)
                labels_list.append(data[row, col])

            if not features_list:
                return np.array([]), np.array([])

            return np.array(features_list), np.array(labels_list)

        except Exception as e:
            logger.error(f"Error processing {tiff_file}: {e}")
            return np.array([]), np.array([])

    def load_data(
        self, years: List[int], sample_ratio: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process MODIS data for specified years.

        Args:
            years: List of years to process
            sample_ratio: Ratio of pixels to sample from each file

        Returns:
            Tuple of (features, labels)
        """
        logger.info(f"Loading data for years: {years}")

        all_features = []
        all_labels = []

        for year in years:
            tiff_file = self.data_dir / f"{year}.tif"

            if not tiff_file.exists():
                logger.warning(f"File not found: {tiff_file}")
                continue

            logger.info(f"Processing {year}...")
            features, labels = self.process_single_file(tiff_file, sample_ratio)

            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
                logger.info(f"Extracted {len(features)} samples from {year}")

        if not all_features:
            raise ValueError("No valid data found!")

        # Combine all data
        X = np.vstack(all_features)
        y = np.hstack(all_labels)

        # Create feature names
        self.feature_names = self.create_feature_names() + ["coord_row", "coord_col"]

        logger.info(f"Total samples: {len(X)}, Features: {X.shape[1]}")
        logger.info(f"Class distribution: {np.bincount(y.astype(int))}")

        return X, y


class LandCoverClassifier:
    """
    Land cover classification using traditional machine learning algorithms.
    """

    def __init__(self, algorithm: str = "random_forest"):
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []

        # IGBP Land Cover Classes
        self.class_names = {
            1: "Evergreen Needleleaf Forests",
            2: "Evergreen Broadleaf Forests",
            3: "Deciduous Needleleaf Forests",
            4: "Deciduous Broadleaf Forests",
            5: "Mixed Forests",
            6: "Closed Shrublands",
            7: "Open Shrublands",
            8: "Woody Savannas",
            9: "Savannas",
            10: "Grasslands",
            11: "Permanent Wetlands",
            12: "Croplands",
            13: "Urban and Built-up Lands",
            14: "Cropland/Natural Vegetation Mosaics",
            15: "Permanent Snow and Ice",
            16: "Barren",
            17: "Water Bodies",
        }

    def create_model(self, **kwargs) -> object:
        """Create the specified machine learning model."""

        if self.algorithm == "random_forest":
            model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 20),
                min_samples_split=kwargs.get("min_samples_split", 10),
                min_samples_leaf=kwargs.get("min_samples_leaf", 5),
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )

        elif self.algorithm == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
            )

        elif self.algorithm == "xgboost":
            if not HAS_XGBOOST:
                raise ValueError(
                    "XGBoost is not available. Please install xgboost package or choose a different algorithm."
                )
            model = xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 8),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
                n_jobs=-1,
                eval_metric="mlogloss",
            )

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return model

    def train(self, X: np.ndarray, y: np.ndarray, **model_params) -> Dict:
        """
        Train the land cover classification model.

        Args:
            X: Feature matrix
            y: Labels
            **model_params: Model-specific parameters

        Returns:
            Training metrics
        """
        logger.info(f"Training {self.algorithm} model...")

        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Create and train model
        self.model = self.create_model(**model_params)

        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Evaluate on test set
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "training_time": training_time,
        }

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()

        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"CV Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")

        # Generate classification report
        report = classification_report(
            y_test,
            y_pred,
            target_names=[
                self.class_names.get(i + 1, f"Class_{i+1}")
                for i in range(len(np.unique(y_encoded)))
            ],
            output_dict=True,
        )

        return metrics, report

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if hasattr(self.model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            return importance_df
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()

    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "algorithm": self.algorithm,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data["feature_names"]
        self.algorithm = model_data["algorithm"]

        logger.info(f"Model loaded from {filepath}")


def plot_results(metrics: Dict, feature_importance: pd.DataFrame, save_dir: str = "."):
    """Plot training results and feature importance."""

    if not HAS_PLOTTING:
        logger.warning("Plotting libraries not available. Skipping plots.")
        return

    # Create plots directory
    plots_dir = Path(save_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Feature importance plot
    if not feature_importance.empty:
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        sns.barplot(data=top_features, x="importance", y="feature")
        plt.title("Top 20 Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Plots saved to {plots_dir}")
    else:
        logger.info("No feature importance data to plot")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest for land cover classification"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "xgboost"],
        help="Machine learning algorithm to use",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing MODIS data",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.01,
        help="Ratio of pixels to sample from each file",
    )
    parser.add_argument("--n_estimators", type=int, default=200, help="Number of estimators")
    parser.add_argument("--max_depth", type=int, default=20, help="Maximum depth of trees")
    parser.add_argument(
        "--use_all_years",
        action="store_true",
        help="Use all available years for training (no test set)",
    )

    args = parser.parse_args()

    # Set up MLflow tracking only if available
    if HAS_MLFLOW:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("MODIS_Land_Cover_Traditional_ML")
        mlflow.start_run()

        # Log parameters
        mlflow.log_params(
            {
                "algorithm": args.algorithm,
                "sample_ratio": args.sample_ratio,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "use_all_years": args.use_all_years,
            }
        )

    # Load and extract features
    extractor = MODISFeatureExtractor(Path(args.data_dir))

    # Use ALL years for training
    all_years = list(range(2001, 2023))  # All available years

    if args.use_all_years:
        train_years = all_years
        logger.info(f"Training on ALL {len(train_years)} years: {train_years}")
    else:
        # Use most years for training, last 2 for testing
        train_years = all_years[:-2]
        test_years = all_years[-2:]
        logger.info(
            f"Training years: {len(train_years)} years from {train_years[0]} to {train_years[-1]}"
        )
        logger.info(f"Test years: {test_years}")

    # Extract features
    logger.info("🚀 Starting feature extraction from MODIS data...")
    X, y = extractor.load_data(train_years, args.sample_ratio)

    # Create and train classifier
    classifier = LandCoverClassifier(args.algorithm)
    classifier.feature_names = extractor.feature_names

    # Train model
    logger.info("🔥 Starting model training...")
    metrics, report = classifier.train(
        X, y, n_estimators=args.n_estimators, max_depth=args.max_depth
    )

    # Log metrics to MLflow if available
    if HAS_MLFLOW:
        mlflow.log_metrics(metrics)

    # Get feature importance
    feature_importance = classifier.get_feature_importance()

    # Plot results
    plot_results(metrics, feature_importance)

    # Save model
    model_filename = f"{args.algorithm}_full_dataset_model.pkl"
    classifier.save_model(model_filename)

    # Log model to MLflow if available
    if HAS_MLFLOW:
        mlflow.sklearn.log_model(classifier.model, "model")
        mlflow.end_run()

    # Print comprehensive results
    print("\n" + "=" * 70)
    print("🎯 LAND COVER CLASSIFICATION TRAINING RESULTS")
    print("=" * 70)
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Training Years: {len(train_years)} years ({train_years[0]}-{train_years[-1]})")
    print(f"Total Samples: {len(X):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Sample Ratio: {args.sample_ratio}")
    print("-" * 70)
    print(f"Validation Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"Cross-validation: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"Training time: {metrics['training_time']:.2f}s")
    print("-" * 70)

    if not feature_importance.empty:
        print(f"🔍 TOP 10 MOST IMPORTANT FEATURES:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")

    print("-" * 70)
    print(f"✅ Model saved to: {model_filename}")
    print("=" * 70)

    logger.info("🎉 Training completed successfully!")


if __name__ == "__main__":
    main()
