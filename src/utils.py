"""Utility functions for MODIS land cover data processing and analysis."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


def setup_logging(
    log_file: str = "logs/modis_processing.log", level: str = "INFO"
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file
        level: Logging level

    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger(__name__)


def load_config(config_path: str = "config/data_config.yaml") -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return {}


def get_land_cover_classes() -> Dict[int, str]:
    """
    Get IGBP land cover class definitions.

    Returns:
        Dictionary mapping class IDs to names
    """
    return {
        0: "Water Bodies",
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
        255: "Unclassified",
    }


def get_color_mapping() -> Dict[int, str]:
    """
    Get color mapping for land cover classes.

    Returns:
        Dictionary mapping class IDs to hex colors
    """
    return {
        0: "#4169E1",  # Water - Royal Blue
        1: "#006400",  # Evergreen Needleleaf - Dark Green
        2: "#228B22",  # Evergreen Broadleaf - Forest Green
        3: "#32CD32",  # Deciduous Needleleaf - Lime Green
        4: "#90EE90",  # Deciduous Broadleaf - Light Green
        5: "#00FF00",  # Mixed Forests - Lime
        6: "#8FBC8F",  # Closed Shrublands - Dark Sea Green
        7: "#9ACD32",  # Open Shrublands - Yellow Green
        8: "#DAA520",  # Woody Savannas - Goldenrod
        9: "#FFD700",  # Savannas - Gold
        10: "#ADFF2F",  # Grasslands - Green Yellow
        11: "#00CED1",  # Wetlands - Dark Turquoise
        12: "#FF8C00",  # Croplands - Dark Orange
        13: "#FF0000",  # Urban - Red
        14: "#FF69B4",  # Cropland/Vegetation - Hot Pink
        15: "#FFFFFF",  # Snow/Ice - White
        16: "#D2691E",  # Barren - Chocolate
        255: "#696969",  # Unclassified - Dim Gray
    }


def validate_year_range(start_year: int, end_year: int) -> bool:
    """
    Validate year range for MODIS data availability.

    Args:
        start_year: Starting year
        end_year: Ending year

    Returns:
        True if valid range, False otherwise
    """
    modis_start = 2001
    modis_end = 2022  # Update as new data becomes available

    if start_year < modis_start or end_year > modis_end:
        print(f"Invalid year range. MODIS MCD12C1 data available: {modis_start}-{modis_end}")
        return False

    if start_year > end_year:
        print("Start year must be <= end year")
        return False

    return True


def calculate_pixel_area(resolution_deg: float = 0.05) -> float:
    """
    Calculate approximate pixel area in km².

    Args:
        resolution_deg: Pixel resolution in degrees

    Returns:
        Pixel area in km²
    """
    # Approximate conversion: 1 degree ≈ 111 km at equator
    km_per_degree = 111.0
    area_km2 = (resolution_deg * km_per_degree) ** 2
    return area_km2


def calculate_statistics(
    data: np.ndarray, classes: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Calculate land cover statistics from data array.

    Args:
        data: Land cover data array
        classes: Class ID to name mapping

    Returns:
        DataFrame with statistics
    """
    if classes is None:
        classes = get_land_cover_classes()

    # Flatten and clean data
    flat_data = data.flatten()
    valid_data = flat_data[~np.isnan(flat_data)]
    valid_data = valid_data[valid_data != 255]  # Remove unclassified

    # Count classes
    unique, counts = np.unique(valid_data, return_counts=True)
    total_pixels = counts.sum()
    pixel_area = calculate_pixel_area()

    # Build statistics
    stats = []
    for cls_id, count in zip(unique, counts):
        cls_id_int = int(cls_id)
        stats.append(
            {
                "class_id": cls_id_int,
                "class_name": classes.get(cls_id_int, f"Unknown ({cls_id_int})"),
                "pixel_count": count,
                "percentage": (count / total_pixels) * 100,
                "area_km2": count * pixel_area,
            }
        )

    return pd.DataFrame(stats).sort_values("percentage", ascending=False)


def create_sample_data(
    output_dir: str = "data/processed",
    years: List[int] = [2001, 2022],
    shape: Tuple[int, int] = (100, 200),
) -> List[Path]:
    """
    Create sample land cover data for testing (when real data unavailable).

    Args:
        output_dir: Output directory
        years: Years to create data for
        shape: Array shape (height, width)

    Returns:
        List of created file paths
    """
    from osgeo import gdal, osr

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    created_files = []

    for year in years:
        filename = output_path / f"{year}.tif"

        # Generate synthetic land cover data with temporal evolution
        np.random.seed(year)  # Reproducible random data

        # Create realistic land cover distribution
        base_classes = [0, 1, 2, 6, 7, 10, 12, 13, 16]  # Common classes
        base_probs = [
            0.1,
            0.15,
            0.1,
            0.1,
            0.1,
            0.2,
            0.15,
            0.05,
            0.05,
        ]  # Base probabilities

        # Add temporal changes (simulate land cover evolution)
        year_factor = (year - 2001) / 21.0  # Normalize to 0-1 range

        # Simulate urbanization (class 13 increases over time)
        urban_increase = year_factor * 0.02
        # Simulate deforestation (forest classes decrease slightly)
        forest_decrease = year_factor * 0.01
        # Simulate agricultural expansion (class 12 increases)
        crop_increase = year_factor * 0.015

        # Adjust probabilities based on year
        adjusted_probs = base_probs.copy()
        adjusted_probs[7] += urban_increase  # Urban increase
        adjusted_probs[1] -= forest_decrease / 2  # Forest decrease
        adjusted_probs[2] -= forest_decrease / 2  # Forest decrease
        adjusted_probs[6] += crop_increase  # Cropland increase

        # Normalize probabilities
        adjusted_probs = np.array(adjusted_probs)
        adjusted_probs = adjusted_probs / adjusted_probs.sum()

        # Generate land cover data
        data = np.random.choice(base_classes, size=shape, p=adjusted_probs).astype(np.uint8)

        # Add some spatial clustering (make it more realistic)
        from scipy import ndimage

        try:
            # Apply slight smoothing to create spatial clusters
            data_smooth = ndimage.median_filter(data, size=3)
            # Mix original and smoothed data
            mask = np.random.random(shape) < 0.7
            data = np.where(mask, data_smooth, data)
        except ImportError:
            # If scipy not available, use original data
            pass

        # Create GeoTIFF
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            str(filename),
            shape[1],
            shape[0],
            1,
            gdal.GDT_Byte,  # width  # height  # bands
        )

        # Set geotransform (global extent, 0.05 degree resolution)
        geotransform = (-180.0, 0.05, 0.0, 90.0, 0.0, -0.05)
        dataset.SetGeoTransform(geotransform)

        # Set projection (WGS84)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())

        # Write data
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(255)

        # Close dataset
        dataset = None

        created_files.append(filename)
        print(f"Created sample data: {filename}")

    return created_files


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are available.

    Returns:
        Dictionary showing availability of each dependency
    """
    dependencies = {
        "gdal": False,
        "xarray": False,
        "rasterio": False,
        "requests": False,
        "pandas": False,
        "numpy": False,
        "matplotlib": False,
        "yaml": False,
    }

    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False

    return dependencies


def print_system_info():
    """Print system information and dependency status."""
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    print("\nDEPENDENCY STATUS:")
    print("-" * 20)
    deps = check_dependencies()
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"{dep:<15}: {status}")

    missing = [dep for dep, available in deps.items() if not available]
    if missing:
        print(f"\nMissing dependencies: {missing}")
        print("Install with: conda env create -f environment.yml")
    else:
        print("\n✅ All dependencies available!")

    print("=" * 50)


if __name__ == "__main__":
    # Run system check
    print_system_info()

    # Test configuration loading
    config = load_config()
    if config:
        print(f"✅ Configuration loaded: {len(config)} sections")
    else:
        print("❌ Failed to load configuration")

    # Test sample data creation
    print("\nTesting sample data creation...")
    try:
        sample_files = create_sample_data(years=[2001, 2002], shape=(50, 100))
        print(f"✅ Created {len(sample_files)} sample files")
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
