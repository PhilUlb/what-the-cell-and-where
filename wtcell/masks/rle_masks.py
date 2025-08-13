"""RLE (Run Length Encoding) mask generator for Kaggle annotations."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from wtcell.masks.base import MaskGenerator
from wtcell.utils.io import safe_path, load_array

logger = get_logger(__name__)

class RLEMaskGenerator(MaskGenerator):
    """Generate binary masks from Kaggle RLE annotations."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize RLE mask generator.
        
        Args:
            config: Configuration dictionary with RLE settings
        """
        super().__init__(config)
        self.csv_path = safe_path(config.get("csv_path", "./data/raw/train.csv"))
        self.annotations_df = None
        self._load_annotations()
    
    def _load_annotations(self) -> None:
        """Load annotations from CSV file."""
        try:
            self.annotations_df = pd.read_csv(self.csv_path)
            self.logger.info(f"Loaded {len(self.annotations_df)} annotations from {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Failed to load annotations: {e}")
            self.annotations_df = pd.DataFrame()
    
    def _rle_decode(self, rle_string: str, shape: tuple) -> np.ndarray:
        """Decode RLE string to binary mask.
        
        Args:
            rle_string: RLE encoded string
            shape: Image shape (height, width)
            
        Returns:
            Binary mask as numpy array
        """
        if pd.isna(rle_string) or rle_string == "":
            return np.zeros(shape, dtype=np.uint8)
        
        # Parse RLE string
        numbers = [int(num) for num in rle_string.split()]
        
        # Create mask
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        
        start = 0
        for i in range(0, len(numbers), 2):
            start += numbers[i]
            end = start + numbers[i + 1]
            mask[start:end] = 1
        
        # Reshape to image dimensions
        mask = mask.reshape(shape)
        
        return mask
    
    def _get_image_annotations(self, image_id: str) -> List[str]:
        """Get all RLE annotations for a specific image.
        
        Args:
            image_id: Image ID (filename without extension)
            
        Returns:
            List of RLE strings for the image
        """
        if self.annotations_df is None or self.annotations_df.empty:
            return []
        
        # Filter annotations for this image
        image_annotations = self.annotations_df[
            self.annotations_df["id"] == image_id
        ]["annotation"].tolist()
        
        return image_annotations
    
    def generate(
        self,
        image_path: Union[str, Path],
        **kwargs: any,
    ) -> np.ndarray:
        """Generate binary mask from RLE annotations.
        
        Args:
            image_path: Path to the input image
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Binary segmentation mask
        """
        image_path = safe_path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Get image ID from filename
        image_id = image_path.stem
        
        # Load image to get dimensions
        try:
            image = Image.open(image_path)
            height, width = image.size
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise
        
        # Get RLE annotations for this image
        rle_annotations = self._get_image_annotations(image_id)
        
        if not rle_annotations:
            self.logger.warning(f"No annotations found for image {image_id}")
            return np.zeros((height, width), dtype=np.uint8)
        
        # Combine all annotations into single mask
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        for rle_string in rle_annotations:
            if pd.isna(rle_string) or rle_string == "":
                continue
            
            try:
                mask = self._rle_decode(rle_string, (height, width))
                combined_mask = np.logical_or(combined_mask, mask)
            except Exception as e:
                self.logger.warning(f"Failed to decode RLE for {image_id}: {e}")
                continue
        
        # Ensure binary values
        combined_mask = combined_mask.astype(np.uint8)
        
        # Validate mask
        if not self.validate_mask(combined_mask):
            self.logger.warning(f"Generated invalid mask for {image_id}")
        
        return combined_mask
    
    def generate_batch(
        self,
        image_paths: List[Union[str, Path]],
        **kwargs: any,
    ) -> List[np.ndarray]:
        """Generate masks for a batch of images.
        
        Args:
            image_paths: List of paths to input images
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of binary segmentation masks
        """
        masks = []
        
        for i, image_path in enumerate(image_paths):
            if i % 100 == 0:
                self.logger.info(f"Processing image {i+1}/{len(image_paths)}")
            
            try:
                mask = self.generate(image_path, **kwargs)
                masks.append(mask)
            except Exception as e:
                self.logger.error(f"Failed to generate mask for {image_path}: {e}")
                # Create empty mask as fallback
                masks.append(np.zeros((128, 128), dtype=np.uint8))
        
        return masks
    
    def get_cell_type(self, image_id: str) -> Optional[str]:
        """Get cell type for a specific image.
        
        Args:
            image_id: Image ID
            
        Returns:
            Cell type string or None if not found
        """
        if self.annotations_df is None or self.annotations_df.empty:
            return None
        
        # Get cell type for this image
        cell_type_data = self.annotations_df[
            self.annotations_df["id"] == image_id
        ]["cell_type"].iloc[0] if len(self.annotations_df[
            self.annotations_df["id"] == image_id
        ]) > 0 else None
        
        return cell_type_data
    
    def get_all_cell_types(self) -> List[str]:
        """Get list of all available cell types.
        
        Returns:
            List of unique cell types
        """
        if self.annotations_df is None or self.annotations_df.empty:
            return []
        
        return self.annotations_df["cell_type"].unique().tolist()
    
    def get_image_ids_by_cell_type(self, cell_type: str) -> List[str]:
        """Get list of image IDs for a specific cell type.
        
        Args:
            cell_type: Cell type to filter by
            
        Returns:
            List of image IDs
        """
        if self.annotations_df is None or self.annotations_df.empty:
            return []
        
        filtered_df = self.annotations_df[
            self.annotations_df["cell_type"] == cell_type
        ]
        
        return filtered_df["id"].unique().tolist() 