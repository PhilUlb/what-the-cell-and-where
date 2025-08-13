"""I/O utilities for the wtcell package."""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

from wtcell.utils.logging import get_logger

logger = get_logger(__name__)

def safe_path(path: Union[str, Path]) -> Path:
    """Convert path to Path object and ensure it's safe.
    
    Args:
        path: Path string or Path object
        
    Returns:
        Path object
        
    Raises:
        ValueError: If path contains unsafe characters
    """
    path = Path(path)
    
    # Check for path traversal attempts
    try:
        path.resolve()
    except (RuntimeError, OSError):
        raise ValueError(f"Invalid path: {path}")
    
    return path

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = safe_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_array(
    array: np.ndarray,
    path: Union[str, Path],
    format: str = "npz",
    **kwargs: Any,
) -> None:
    """Save numpy array to file.
    
    Args:
        array: Array to save
        path: Output file path
        format: File format (npz, npy, png, jpg)
        **kwargs: Additional arguments for the save function
    """
    path = safe_path(path)
    ensure_dir(path.parent)
    
    if format == "npz":
        np.savez_compressed(path, array=array, **kwargs)
    elif format == "npy":
        np.save(path, array, **kwargs)
    elif format in ["png", "jpg", "jpeg"]:
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        Image.fromarray(array).save(path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.debug(f"Saved array to {path}")

def load_array(
    path: Union[str, Path],
    format: Optional[str] = None,
) -> np.ndarray:
    """Load numpy array from file.
    
    Args:
        path: Input file path
        format: File format (auto-detected if None)
        
    Returns:
        Loaded array
    """
    path = safe_path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if format is None:
        format = path.suffix.lower().lstrip(".")
    
    if format == "npz":
        data = np.load(path)
        if "array" in data:
            return data["array"]
        else:
            # Return first array found
            return data[data.files[0]]
    elif format == "npy":
        return np.load(path)
    elif format in ["png", "jpg", "jpeg"]:
        return np.array(Image.open(path))
    else:
        raise ValueError(f"Unsupported format: {format}")

def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        path: Output file path
    """
    path = safe_path(path)
    ensure_dir(path.parent)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.debug(f"Saved JSON to {path}")

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Loaded data
    """
    path = safe_path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "r") as f:
        return json.load(f)

def download_kaggle_dataset(
    dataset_name: str,
    output_dir: Union[str, Path],
    username: Optional[str] = None,
    key: Optional[str] = None,
) -> bool:
    """Download dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset name (e.g., 'sartorius-cell-instance-segmentation')
        output_dir: Directory to save dataset
        username: Kaggle username (from env if None)
        key: Kaggle API key (from env if None)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import kaggle
    except ImportError:
        logger.error("Kaggle package not installed. Install with: pip install kaggle")
        return False
    
    # Get credentials from environment if not provided
    if username is None:
        username = os.getenv("KAGGLE_USERNAME")
    if key is None:
        key = os.getenv("KAGGLE_KEY")
    
    if not username or not key:
        logger.error("Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        return False
    
    try:
        # Configure Kaggle
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        
        # Download dataset
        output_dir = ensure_dir(output_dir)
        kaggle.api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=True,
        )
        
        logger.info(f"Successfully downloaded dataset {dataset_name} to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False

def get_image_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
) -> List[Path]:
    """Get list of image files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (default: common image formats)
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]
    
    directory = safe_path(directory)
    
    if not directory.exists():
        return []
    
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files) 