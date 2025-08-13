"""Tests for dataset module."""

import numpy as np
import pytest
import torch
from pathlib import Path

from wtcell.data.dataset import CellSegmentationDataset, create_data_splits

def test_dataset_initialization():
    """Test dataset initialization."""
    # Create dummy image paths
    image_paths = ["image1.png", "image2.png", "image3.png"]
    
    # Create dummy mask paths
    mask_paths = ["mask1.png", "mask2.png", "mask3.png"]
    
    # This will fail since files don't exist, but we can test the initialization
    with pytest.raises(ValueError):
        dataset = CellSegmentationDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            image_size=(64, 64)
        )

def test_data_splits():
    """Test data splitting functionality."""
    # Create dummy paths
    image_paths = [f"image{i}.png" for i in range(10)]
    mask_paths = [f"mask{i}.png" for i in range(10)]
    
    # Test splitting
    splits = create_data_splits(
        image_paths=image_paths,
        mask_paths=mask_paths,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
    )
    
    # Check that all images are accounted for
    total_images = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total_images == 10
    
    # Check ratios (approximate due to rounding)
    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 2

def test_data_splits_no_masks():
    """Test data splitting without mask paths."""
    # Create dummy paths
    image_paths = [f"image{i}.png" for i in range(10)]
    
    # Test splitting without masks
    splits = create_data_splits(
        image_paths=image_paths,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    # Check that all images are accounted for
    total_images = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total_images == 10
    
    # Check that mask keys don't exist
    assert "train_masks" not in splits
    assert "val_masks" not in splits
    assert "test_masks" not in splits

def test_invalid_split_ratios():
    """Test that invalid split ratios raise errors."""
    image_paths = [f"image{i}.png" for i in range(10)]
    
    # Test ratios that don't sum to 1.0
    with pytest.raises(ValueError):
        create_data_splits(
            image_paths=image_paths,
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.3,  # Sum > 1.0
            seed=42
        )
    
    with pytest.raises(ValueError):
        create_data_splits(
            image_paths=image_paths,
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.1,  # Sum < 1.0
            seed=42
        ) 