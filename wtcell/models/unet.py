"""U-Net architecture for cell segmentation."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from wtcell.models.encoders import VGG16Encoder, MobileNetV2Encoder

logger = get_logger(__name__)

class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and dropout."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout: float = 0.3,
        batch_norm: bool = True,
    ) -> None:
        """Initialize double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of intermediate channels (default: out_channels)
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=not batch_norm),
        ]
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(mid_channels))
        
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm),
        ])
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution block."""
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling block with max pooling and double convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.3,
        batch_norm: bool = True,
    ) -> None:
        """Initialize downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout, batch_norm=batch_norm)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through downsampling block."""
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling block with transposed convolution and double convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = False,
        dropout: float = 0.3,
        batch_norm: bool = True,
    ) -> None:
        """Initialize upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bilinear: Whether to use bilinear upsampling
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2,
                dropout=dropout, batch_norm=batch_norm
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(
                in_channels, out_channels, dropout=dropout, batch_norm=batch_norm
            )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass through upsampling block.
        
        Args:
            x1: Input from previous layer
            x2: Skip connection from encoder
            
        Returns:
            Upsampled and convolved tensor
        """
        x1 = self.up(x1)
        
        # Handle different input sizes
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize output convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through output convolution."""
        return self.conv(x)

class UNet(nn.Module):
    """U-Net architecture for semantic segmentation."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = None,
        encoder: str = "scratch",
        pretrained: bool = False,
        dropout: float = 0.3,
        batch_norm: bool = True,
        bilinear: bool = False,
    ) -> None:
        """Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: List of feature dimensions for each level
            encoder: Encoder type ('scratch', 'vgg16', 'mobilenet_v2')
            pretrained: Whether to use pretrained encoder
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            bilinear: Whether to use bilinear upsampling
        """
        super().__init__()
        
        if features is None:
            features = [32, 64, 128, 256, 512]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.encoder = encoder
        self.pretrained = pretrained
        self.bilinear = bilinear
        
        # Initialize encoder
        if encoder == "scratch":
            self._init_scratch_encoder(dropout, batch_norm)
        elif encoder == "vgg16":
            self._init_vgg16_encoder(pretrained, dropout, batch_norm)
        elif encoder == "mobilenet_v2":
            self._init_mobilenet_encoder(pretrained, dropout, batch_norm)
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        
        # Initialize decoder
        self._init_decoder(dropout, batch_norm)
        
        # Output layer
        self.outc = OutConv(features[0], out_channels)
        
        self.logger.info(f"Initialized U-Net with {encoder} encoder")
    
    def _init_scratch_encoder(self, dropout: float, batch_norm: bool) -> None:
        """Initialize scratch encoder."""
        self.inc = DoubleConv(self.in_channels, self.features[0], dropout=dropout, batch_norm=batch_norm)
        
        self.down1 = Down(self.features[0], self.features[1], dropout=dropout, batch_norm=batch_norm)
        self.down2 = Down(self.features[1], self.features[2], dropout=dropout, batch_norm=batch_norm)
        self.down3 = Down(self.features[2], self.features[3], dropout=dropout, batch_norm=batch_norm)
        self.down4 = Down(self.features[3], self.features[4], dropout=dropout, batch_norm=batch_norm)
        
        factor = 2 if self.bilinear else 1
        self.down5 = Down(self.features[4], self.features[4] // factor, dropout=dropout, batch_norm=batch_norm)
    
    def _init_vgg16_encoder(self, pretrained: bool, dropout: float, batch_norm: bool) -> None:
        """Initialize VGG16 encoder."""
        self.vgg_encoder = VGG16Encoder(pretrained=pretrained)
        
        # Adjust input layer for grayscale
        if self.in_channels == 1:
            self.vgg_encoder.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        # Get feature dimensions from VGG16
        vgg_features = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        
        # Create downsampling layers
        self.down1 = Down(vgg_features[1], self.features[1], dropout=dropout, batch_norm=batch_norm)
        self.down2 = Down(vgg_features[3], self.features[2], dropout=dropout, batch_norm=batch_norm)
        self.down3 = Down(vgg_features[6], self.features[3], dropout=dropout, batch_norm=batch_norm)
        self.down4 = Down(vgg_features[10], self.features[4], dropout=dropout, batch_norm=batch_norm)
        
        factor = 2 if self.bilinear else 1
        self.down5 = Down(self.features[4], self.features[4] // factor, dropout=dropout, batch_norm=batch_norm)
    
    def _init_mobilenet_encoder(self, pretrained: bool, dropout: float, batch_norm: bool) -> None:
        """Initialize MobileNetV2 encoder."""
        self.mobilenet_encoder = MobileNetV2Encoder(pretrained=pretrained)
        
        # Adjust input layer for grayscale
        if self.in_channels == 1:
            self.mobilenet_encoder.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Get feature dimensions from MobileNetV2
        mobilenet_features = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        
        # Create downsampling layers
        self.down1 = Down(mobilenet_features[1], self.features[1], dropout=dropout, batch_norm=batch_norm)
        self.down2 = Down(mobilenet_features[3], self.features[2], dropout=dropout, batch_norm=batch_norm)
        self.down3 = Down(mobilenet_features[5], self.features[3], dropout=dropout, batch_norm=batch_norm)
        self.down4 = Down(mobilenet_features[7], self.features[4], dropout=dropout, batch_norm=batch_norm)
        
        factor = 2 if self.bilinear else 1
        self.down5 = Down(self.features[4], self.features[4] // factor, dropout=dropout, batch_norm=batch_norm)
    
    def _init_decoder(self, dropout: float, batch_norm: bool) -> None:
        """Initialize decoder layers."""
        factor = 2 if self.bilinear else 1
        
        self.up1 = Up(self.features[4], self.features[3], self.bilinear, dropout=dropout, batch_norm=batch_norm)
        self.up2 = Up(self.features[3], self.features[2], self.bilinear, dropout=dropout, batch_norm=batch_norm)
        self.up3 = Up(self.features[2], self.features[1], self.bilinear, dropout=dropout, batch_norm=batch_norm)
        self.up4 = Up(self.features[1], self.features[0], self.bilinear, dropout=dropout, batch_norm=batch_norm)
        self.up5 = Up(self.features[0], self.features[0], self.bilinear, dropout=dropout, batch_norm=batch_norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        if self.encoder == "scratch":
            return self._forward_scratch(x)
        elif self.encoder == "vgg16":
            return self._forward_vgg16(x)
        elif self.encoder == "mobilenet_v2":
            return self._forward_mobilenet(x)
        else:
            raise ValueError(f"Unknown encoder: {self.encoder}")
    
    def _forward_scratch(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with scratch encoder."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        
        return self.outc(x)
    
    def _forward_vgg16(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with VGG16 encoder."""
        # Extract features from VGG16
        features = self.vgg_encoder(x)
        
        # Use features for skip connections
        x1 = features[1]   # After first conv block
        x2 = features[3]   # After second conv block
        x3 = features[6]   # After third conv block
        x4 = features[10]  # After fourth conv block
        x5 = features[12]  # After fifth conv block
        
        x6 = self.down5(x5)
        
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        
        return self.outc(x)
    
    def _forward_mobilenet(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MobileNetV2 encoder."""
        # Extract features from MobileNetV2
        features = self.mobilenet_encoder(x)
        
        # Use features for skip connections
        x1 = features[1]   # After first expansion
        x2 = features[3]   # After second expansion
        x3 = features[5]   # After third expansion
        x4 = features[7]   # After fourth expansion
        x5 = features[8]   # Final features
        
        x6 = self.down5(x5)
        
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        
        return self.outc(x)
    
    def get_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get intermediate features from encoder for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature tensors at different scales
        """
        if self.encoder == "scratch":
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            return [x1, x2, x3, x4, x5]
        elif self.encoder == "vgg16":
            features = self.vgg_encoder(x)
            return [features[1], features[3], features[6], features[10], features[12]]
        elif self.encoder == "mobilenet_v2":
            features = self.mobilenet_encoder(x)
            return [features[1], features[3], features[5], features[7], features[8]]
        else:
            raise ValueError(f"Unknown encoder: {self.encoder}")
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb 