"""K-Means clustering-based mask generator."""

from pathlib import Path
from typing import Dict, List, Union, Optional

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from wtcell.masks.base import MaskGenerator
from wtcell.utils.io import safe_path
from wtcell.utils.logging import get_logger

logger = get_logger(__name__)

class KMeansMaskGenerator(MaskGenerator):
    """Generate binary masks using K-Means clustering and Gaussian filtering."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize K-Means mask generator.
        
        Args:
            config: Configuration dictionary with K-Means settings
        """
        super().__init__(config)
        
        # K-Means parameters
        self.n_clusters = config.get("n_clusters", 80)
        self.n_init = config.get("n_init", 5)
        self.max_iter = config.get("max_iter", 50)
        
        # Gaussian filtering parameters
        self.kernel_size = config.get("kernel_size", 12)
        self.threshold_offset = config.get("threshold_offset", 2)
        
        self.logger.info(f"Initialized K-Means mask generator with {self.n_clusters} clusters")
    
    def _apply_kmeans(self, image: np.ndarray) -> np.ndarray:
        """Apply K-Means clustering to image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Clustered image
        """
        # Reshape image for clustering
        height, width = image.shape
        image_reshaped = image.reshape(height * width, 1)
        
        # Apply K-Means
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=42
        )
        
        labels = kmeans.fit_predict(image_reshaped)
        
        # Reshape back to image dimensions
        clustered_image = labels.reshape(height, width).astype(np.uint8)
        
        return clustered_image
    
    def _apply_gaussian_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian filter to clustered image.
        
        Args:
            image: Clustered image
            
        Returns:
            Filtered image
        """
        # Create kernel
        kernel = np.ones((self.kernel_size, self.kernel_size), np.float32) / (self.kernel_size ** 2)
        
        # Apply filter
        filtered_image = cv2.filter2D(image.astype(np.float32), -1, kernel)
        
        return filtered_image
    
    def _threshold_mask(self, filtered_image: np.ndarray) -> np.ndarray:
        """Apply thresholding to create binary mask.
        
        Args:
            filtered_image: Filtered image
            
        Returns:
            Binary mask
        """
        # Calculate threshold
        threshold = filtered_image.mean() + self.threshold_offset
        
        # Create binary mask
        mask = (filtered_image > threshold).astype(np.uint8)
        
        return mask
    
    def generate(
        self,
        image_path: Union[str, Path],
        **kwargs: any,
    ) -> np.ndarray:
        """Generate binary mask using K-Means clustering.
        
        Args:
            image_path: Path to the input image
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Binary segmentation mask
        """
        image_path = safe_path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to grayscale if needed
            if image.mode != "L":
                image = image.convert("L")
            
            # Convert to numpy array
            image_array = np.array(image)
            
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise
        
        # Apply K-Means clustering
        clustered_image = self._apply_kmeans(image_array)
        
        # Apply Gaussian filtering
        filtered_image = self._apply_gaussian_filter(clustered_image)
        
        # Create binary mask
        mask = self._threshold_mask(filtered_image)
        
        # Validate mask
        if not self.validate_mask(mask):
            self.logger.warning(f"Generated invalid mask for {image_path}")
        
        return mask
    
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
    
    def update_kmeans_params(
        self,
        n_clusters: Optional[int] = None,
        n_init: Optional[int] = None,
        max_iter: Optional[int] = None,
    ) -> None:
        """Update K-Means parameters.
        
        Args:
            n_clusters: Number of clusters
            n_init: Number of initializations
            max_iter: Maximum iterations
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters
        if n_init is not None:
            self.n_init = n_init
        if max_iter is not None:
            self.max_iter = max_iter
        
        self.logger.info(f"Updated K-Means parameters: clusters={self.n_clusters}, init={self.n_init}, max_iter={self.max_iter}")
    
    def update_filter_params(
        self,
        kernel_size: Optional[int] = None,
        threshold_offset: Optional[float] = None,
    ) -> None:
        """Update filtering parameters.
        
        Args:
            kernel_size: Gaussian kernel size
            threshold_offset: Threshold offset from mean
        """
        if kernel_size is not None:
            self.kernel_size = kernel_size
        if threshold_offset is not None:
            self.threshold_offset = threshold_offset
        
        self.logger.info(f"Updated filter parameters: kernel_size={self.kernel_size}, threshold_offset={self.threshold_offset}") 