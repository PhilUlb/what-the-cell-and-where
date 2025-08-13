"""SIFT keypoint-based mask generator."""

from pathlib import Path
from typing import Dict, List, Union, Optional

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from wtcell.masks.base import MaskGenerator
from wtcell.utils.io import safe_path

logger = get_logger(__name__)

class SIFTMaskGenerator(MaskGenerator):
    """Generate binary masks using SIFT keypoints and clustering."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize SIFT mask generator.
        
        Args:
            config: Configuration dictionary with SIFT settings
        """
        super().__init__(config)
        
        # SIFT parameters
        self.n_octave_layers = config.get("n_octave_layers", 40)
        self.n_features = config.get("n_features", 20000)
        self.contrast_threshold = config.get("contrast_threshold", 0.02)
        self.edge_threshold = config.get("edge_threshold", 0.01)
        self.sigma = config.get("sigma", 1.0)
        
        # Clustering parameters
        self.n_clusters = config.get("n_clusters", 80)
        self.kernel_size = config.get("kernel_size", 12)
        self.threshold_offset = config.get("threshold_offset", 2)
        
        self.logger.info(f"Initialized SIFT mask generator with {self.n_features} features")
    
    def _extract_sift_keypoints(self, image: np.ndarray) -> np.ndarray:
        """Extract SIFT keypoints from image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Keypoint map
        """
        try:
            # Create SIFT detector
            sift = cv2.xfeatures2d.SIFT_create(
                nfeatures=self.n_features,
                nOctaveLayers=self.n_octave_layers,
                contrastThreshold=self.contrast_threshold,
                edgeThreshold=self.edge_threshold,
                sigma=self.sigma
            )
            
            # Detect keypoints
            keypoints, _ = sift.detectAndCompute(image, None)
            
            # Create keypoint map
            height, width = image.shape
            keypoint_map = np.zeros((height, width), dtype=np.uint8)
            
            if keypoints:
                # Extract keypoint coordinates
                keypoint_coords = []
                for kp in keypoints:
                    x, y = kp.pt
                    keypoint_coords.append([int(x), int(y)])
                
                # Round coordinates
                keypoint_coords = np.array(keypoint_coords)
                
                # Set keypoint pixels to 1
                for x, y in keypoint_coords:
                    if 0 <= x < width and 0 <= y < height:
                        keypoint_map[y, x] = 1
            
            return keypoint_map
            
        except Exception as e:
            self.logger.warning(f"SIFT extraction failed: {e}")
            # Fallback: return empty map
            return np.zeros(image.shape, dtype=np.uint8)
    
    def _apply_kmeans_clustering(self, keypoint_map: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """Apply K-Means clustering to keypoint-enhanced image.
        
        Args:
            keypoint_map: Binary keypoint map
            original_image: Original grayscale image
            
        Returns:
            Clustered image
        """
        # Combine keypoint map with original image
        # Replace pixel values where keypoints exist
        enhanced_image = original_image.copy()
        enhanced_image[keypoint_map == 1] = 255
        
        # Reshape for clustering
        height, width = enhanced_image.shape
        image_reshaped = enhanced_image.reshape(height * width, 1)
        
        # Apply K-Means
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=5,
            max_iter=50,
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
        """Generate binary mask using SIFT keypoints and clustering.
        
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
        
        # Extract SIFT keypoints
        keypoint_map = self._extract_sift_keypoints(image_array)
        
        # Apply K-Means clustering
        clustered_image = self._apply_kmeans_clustering(keypoint_map, image_array)
        
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
    
    def update_sift_params(
        self,
        n_features: Optional[int] = None,
        n_octave_layers: Optional[int] = None,
        contrast_threshold: Optional[float] = None,
        edge_threshold: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> None:
        """Update SIFT parameters.
        
        Args:
            n_features: Number of features to detect
            n_octave_layers: Number of octave layers
            contrast_threshold: Contrast threshold
            edge_threshold: Edge threshold
            sigma: Sigma value
        """
        if n_features is not None:
            self.n_features = n_features
        if n_octave_layers is not None:
            self.n_octave_layers = n_octave_layers
        if contrast_threshold is not None:
            self.contrast_threshold = contrast_threshold
        if edge_threshold is not None:
            self.edge_threshold = edge_threshold
        if sigma is not None:
            self.sigma = sigma
        
        self.logger.info(f"Updated SIFT parameters: features={self.n_features}, octaves={self.n_octave_layers}")
    
    def update_clustering_params(
        self,
        n_clusters: Optional[int] = None,
        kernel_size: Optional[int] = None,
        threshold_offset: Optional[float] = None,
    ) -> None:
        """Update clustering parameters.
        
        Args:
            n_clusters: Number of clusters
            kernel_size: Gaussian kernel size
            threshold_offset: Threshold offset from mean
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters
        if kernel_size is not None:
            self.kernel_size = kernel_size
        if threshold_offset is not None:
            self.threshold_offset = threshold_offset
        
        self.logger.info(f"Updated clustering parameters: clusters={self.n_clusters}, kernel_size={self.kernel_size}") 