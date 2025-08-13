"""Base interface for mask generators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = get_logger(__name__)

class MaskGenerator(ABC):
    """Abstract base class for mask generators.
    
    All mask generators must implement the generate method to create
    segmentation masks from input images.
    """
    
    def __init__(self, config: Dict) -> None:
        """Initialize the mask generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger
    
    @abstractmethod
    def generate(
        self,
        image_path: Union[str, Path],
        **kwargs: any,
    ) -> np.ndarray:
        """Generate a segmentation mask from an image.
        
        Args:
            image_path: Path to the input image
            **kwargs: Additional arguments for mask generation
            
        Returns:
            Binary segmentation mask as numpy array
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        image_paths: List[Union[str, Path]],
        **kwargs: any,
    ) -> List[np.ndarray]:
        """Generate masks for a batch of images.
        
        Args:
            image_paths: List of paths to input images
            **kwargs: Additional arguments for mask generation
            
        Returns:
            List of binary segmentation masks
        """
        pass
    
    def save_mask(
        self,
        mask: np.ndarray,
        output_path: Union[str, Path],
        format: str = "png",
    ) -> None:
        """Save a generated mask to file.
        
        Args:
            mask: Binary mask to save
            output_path: Output file path
            format: Output format (png, npz, npy)
        """
        from wtcell.utils.io import save_array
        
        save_array(mask, output_path, format=format)
        self.logger.debug(f"Saved mask to {output_path}")
    
    def save_masks_batch(
        self,
        masks: List[np.ndarray],
        output_paths: List[Union[str, Path]],
        format: str = "png",
    ) -> None:
        """Save a batch of masks to files.
        
        Args:
            masks: List of binary masks to save
            output_paths: List of output file paths
            format: Output format (png, npz, npy)
        """
        for mask, output_path in zip(masks, output_paths):
            self.save_mask(mask, output_path, format)
    
    def validate_mask(self, mask: np.ndarray) -> bool:
        """Validate that a generated mask is valid.
        
        Args:
            mask: Mask to validate
            
        Returns:
            True if mask is valid, False otherwise
        """
        if not isinstance(mask, np.ndarray):
            return False
        
        if mask.ndim != 2:
            return False
        
        if mask.dtype not in [np.uint8, np.bool_]:
            return False
        
        if mask.min() < 0 or mask.max() > 1:
            return False
        
        return True
    
    def get_config(self) -> Dict:
        """Get the current configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def update_config(self, updates: Dict) -> None:
        """Update the configuration.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self.config.update(updates)
        self.logger.info(f"Updated configuration: {updates}") 