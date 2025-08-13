#!/usr/bin/env python3
"""Simple test script to validate the wtcell package structure."""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing package imports...")
    
    try:
        # Test core imports
        from wtcell import UNet, CellSegmentationDataset
        print("✓ Core imports successful")
        
        # Test mask generators
        from wtcell.masks import RLEMaskGenerator, KMeansMaskGenerator, SIFTMaskGenerator
        print("✓ Mask generator imports successful")
        
        # Test data modules
        from wtcell.data import CellSegmentationDataset, get_transforms
        print("✓ Data module imports successful")
        
        # Test models
        from wtcell.models import UNet, VGG16Encoder, MobileNetV2Encoder
        print("✓ Model imports successful")
        
        # Test training modules
        from wtcell.training import CellSegmentationTrainer, BCELoss, DiceLoss
        print("✓ Training module imports successful")
        
        # Test evaluation modules
        from wtcell.eval import CellSegmentationEvaluator, CellSegmentationVisualizer
        print("✓ Evaluation module imports successful")
        
        # Test utilities
        from wtcell.utils import setup_logging, get_logger, safe_path
        print("✓ Utility imports successful")
        
        # Test CLI
        from wtcell.cli import main
        print("✓ CLI imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test logging setup
        from wtcell.utils.logging import setup_logging, get_logger
        setup_logging(level="INFO")
        logger = get_logger(__name__)
        logger.info("Logging test successful")
        print("✓ Logging functionality works")
        
        # Test path utilities
        from wtcell.utils.io import safe_path, ensure_dir
        test_path = safe_path("./test_dir")
        ensure_dir(test_path)
        print("✓ Path utilities work")
        
        # Test RLE utilities
        from wtcell.data.rle import rle_encode, rle_decode
        import numpy as np
        
        # Create test mask
        test_mask = np.zeros((4, 4), dtype=np.uint8)
        test_mask[1:3, 1:3] = 1
        
        # Test encoding/decoding
        rle_string = rle_encode(test_mask)
        decoded_mask = rle_decode(rle_string, (4, 4))
        
        if np.array_equal(test_mask, decoded_mask):
            print("✓ RLE utilities work")
        else:
            print("✗ RLE utilities failed")
            return False
        
        # Test mask generator creation
        from wtcell.masks import RLEMaskGenerator
        config = {"csv_path": "./test.csv"}
        generator = RLEMaskGenerator(config)
        print("✓ Mask generator creation works")
        
        # Test dataset creation
        from wtcell.data.dataset import create_data_splits
        test_paths = [f"image{i}.png" for i in range(10)]
        splits = create_data_splits(test_paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        if len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 10:
            print("✓ Data splitting works")
        else:
            print("✗ Data splitting failed")
            return False
        
        # Test model creation
        from wtcell.models import UNet
        model = UNet(in_channels=1, out_channels=1, encoder="scratch")
        print("✓ Model creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def test_configuration():
    """Test configuration file loading."""
    print("\nTesting configuration...")
    
    try:
        config_path = Path("configs/data.yaml")
        if config_path.exists():
            print("✓ Configuration files exist")
        else:
            print("✗ Configuration files missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("WTCell Package Validation Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Package validation failed: Import errors")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Package validation failed: Functionality errors")
        sys.exit(1)
    
    # Test configuration
    if not test_configuration():
        print("\n❌ Package validation failed: Configuration errors")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Package is ready to use.")
    print("=" * 50)
    
    # Cleanup test directory
    try:
        import shutil
        if Path("./test_dir").exists():
            shutil.rmtree("./test_dir")
    except:
        pass

if __name__ == "__main__":
    main() 