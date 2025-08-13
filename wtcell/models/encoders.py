"""Encoder architectures for U-Net models."""

from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as models

logger = get_logger(__name__)

class VGG16Encoder(nn.Module):
    """VGG16 encoder for U-Net architecture."""
    
    def __init__(self, pretrained: bool = True) -> None:
        """Initialize VGG16 encoder.
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(pretrained=pretrained)
        
        # Extract features
        self.features = vgg16.features
        
        # Register hooks to capture intermediate features
        self._register_hooks()
        self._captured_features = {}
        
        self.logger.info(f"Initialized VGG16 encoder (pretrained: {pretrained})")
    
    def _register_hooks(self) -> None:
        """Register forward hooks to capture intermediate features."""
        # Define hook points (after each maxpool)
        hook_points = [1, 3, 6, 10, 12]
        
        def hook_fn(name: str):
            def hook(module, input, output):
                self._captured_features[name] = output
            return hook
        
        # Register hooks
        for i, point in enumerate(hook_points):
            self.features[point].register_forward_hook(hook_fn(f"feature_{i}"))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through VGG16 encoder.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature tensors at different scales
        """
        # Clear previous features
        self._captured_features.clear()
        
        # Forward pass
        _ = self.features(x)
        
        # Return features in order
        features = []
        for i in range(5):
            feature_key = f"feature_{i}"
            if feature_key in self._captured_features:
                features.append(self._captured_features[feature_key])
            else:
                # Fallback: create zero tensor
                features.append(torch.zeros_like(x))
        
        return features
    
    def freeze_encoder(self, freeze: bool = True) -> None:
        """Freeze or unfreeze encoder parameters.
        
        Args:
            freeze: Whether to freeze parameters
        """
        for param in self.features.parameters():
            param.requires_grad = not freeze
        
        self.logger.info(f"{'Froze' if freeze else 'Unfroze'} VGG16 encoder parameters")
    
    def get_feature_channels(self) -> List[int]:
        """Get number of channels for each feature level.
        
        Returns:
            List of channel counts
        """
        return [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 encoder for U-Net architecture."""
    
    def __init__(self, pretrained: bool = True) -> None:
        """Initialize MobileNetV2 encoder.
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Extract features
        self.features = mobilenet.features
        
        # Register hooks to capture intermediate features
        self._register_hooks()
        self._captured_features = {}
        
        self.logger.info(f"Initialized MobileNetV2 encoder (pretrained: {pretrained})")
    
    def _register_hooks(self) -> None:
        """Register forward hooks to capture intermediate features."""
        # Define hook points (after each expansion layer)
        hook_points = [1, 3, 5, 7, 8]
        
        def hook_fn(name: str):
            def hook(module, input, output):
                self._captured_features[name] = output
            return hook
        
        # Register hooks
        for i, point in enumerate(hook_points):
            self.features[point].register_forward_hook(hook_fn(f"feature_{i}"))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through MobileNetV2 encoder.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature tensors at different scales
        """
        # Clear previous features
        self._captured_features.clear()
        
        # Forward pass
        _ = self.features(x)
        
        # Return features in order
        features = []
        for i in range(5):
            feature_key = f"feature_{i}"
            if feature_key in self._captured_features:
                features.append(self._captured_features[feature_key])
            else:
                # Fallback: create zero tensor
                features.append(torch.zeros_like(x))
        
        return features
    
    def freeze_encoder(self, freeze: bool = True) -> None:
        """Freeze or unfreeze encoder parameters.
        
        Args:
            freeze: Whether to freeze parameters
        """
        for param in self.features.parameters():
            param.requires_grad = not freeze
        
        self.logger.info(f"{'Froze' if freeze else 'Unfroze'} MobileNetV2 encoder parameters")
    
    def get_feature_channels(self) -> List[int]:
        """Get number of channels for each feature level.
        
        Returns:
            List of channel counts
        """
        return [32, 16, 24, 32, 64, 96, 160, 320, 1280] 