"""Tests for mask generators."""

import numpy as np
import pytest
from pathlib import Path

from wtcell.masks import RLEMaskGenerator, KMeansMaskGenerator, SIFTMaskGenerator

def test_mask_generator_base():
    """Test base mask generator functionality."""
    # Test configuration
    config = {"csv_path": "./test.csv"}
    
    # Test RLE mask generator
    rle_generator = RLEMaskGenerator(config)
    
    # Test configuration methods
    assert rle_generator.get_config() == config
    
    # Test config update
    rle_generator.update_config({"new_param": "value"})
    assert rle_generator.get_config()["new_param"] == "value"

def test_mask_validation():
    """Test mask validation."""
    config = {"csv_path": "./test.csv"}
    generator = RLEMaskGenerator(config)
    
    # Valid mask
    valid_mask = np.zeros((64, 64), dtype=np.uint8)
    valid_mask[32:48, 32:48] = 1
    assert generator.validate_mask(valid_mask) == True
    
    # Invalid mask - wrong dimensions
    invalid_mask_3d = np.zeros((64, 64, 3), dtype=np.uint8)
    assert generator.validate_mask(invalid_mask_3d) == False
    
    # Invalid mask - wrong dtype
    invalid_mask_float = np.zeros((64, 64), dtype=np.float32)
    assert generator.validate_mask(invalid_mask_float) == False
    
    # Invalid mask - values > 1
    invalid_mask_values = np.zeros((64, 64), dtype=np.uint8)
    invalid_mask_values[0, 0] = 2
    assert generator.validate_mask(invalid_mask_values) == False

def test_kmeans_generator_config():
    """Test K-Means mask generator configuration."""
    config = {
        "n_clusters": 50,
        "n_init": 3,
        "max_iter": 30,
        "kernel_size": 8,
        "threshold_offset": 1.5
    }
    
    generator = KMeansMaskGenerator(config)
    
    # Check that parameters are set correctly
    assert generator.n_clusters == 50
    assert generator.n_init == 3
    assert generator.max_iter == 30
    assert generator.kernel_size == 8
    assert generator.threshold_offset == 1.5

def test_sift_generator_config():
    """Test SIFT mask generator configuration."""
    config = {
        "n_octave_layers": 30,
        "n_features": 15000,
        "contrast_threshold": 0.03,
        "edge_threshold": 0.02,
        "sigma": 1.2,
        "n_clusters": 60,
        "kernel_size": 10,
        "threshold_offset": 2.5
    }
    
    generator = SIFTMaskGenerator(config)
    
    # Check that parameters are set correctly
    assert generator.n_octave_layers == 30
    assert generator.n_features == 15000
    assert generator.contrast_threshold == 0.03
    assert generator.edge_threshold == 0.02
    assert generator.sigma == 1.2
    assert generator.n_clusters == 60
    assert generator.kernel_size == 10
    assert generator.threshold_offset == 2.5

def test_parameter_updates():
    """Test parameter update methods."""
    config = {"n_clusters": 80, "kernel_size": 12}
    generator = KMeansMaskGenerator(config)
    
    # Update K-Means parameters
    generator.update_kmeans_params(n_clusters=100, max_iter=60)
    assert generator.n_clusters == 100
    assert generator.max_iter == 60
    
    # Update filter parameters
    generator.update_filter_params(kernel_size=16, threshold_offset=3.0)
    assert generator.kernel_size == 16
    assert generator.threshold_offset == 3.0

def test_sift_parameter_updates():
    """Test SIFT parameter update methods."""
    config = {"n_features": 20000, "n_clusters": 80}
    generator = SIFTMaskGenerator(config)
    
    # Update SIFT parameters
    generator.update_sift_params(n_features=25000, contrast_threshold=0.04)
    assert generator.n_features == 25000
    assert generator.contrast_threshold == 0.04
    
    # Update clustering parameters
    generator.update_clustering_params(n_clusters=100, kernel_size=14)
    assert generator.n_clusters == 100
    assert generator.kernel_size == 14 