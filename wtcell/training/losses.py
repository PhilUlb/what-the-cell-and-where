"""Loss functions for cell segmentation training."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = get_logger(__name__)

class BCELoss(nn.Module):
    """Binary Cross Entropy loss for binary segmentation."""
    
    def __init__(self, reduction: str = "mean") -> None:
        """Initialize BCE loss.
        
        Args:
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss.
        
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Target masks (B, 1, H, W)
            
        Returns:
            BCE loss value
        """
        return self.bce(pred, target)

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean") -> None:
        """Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.
        
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Target masks (B, 1, H, W)
            
        Returns:
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - Dice)
        loss = 1.0 - dice
        
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss."""
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """Initialize combined BCE-Dice loss.
        
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            smooth: Smoothing factor for Dice loss
            reduction: Reduction method
        """
        super().__init__()
        
        if abs(bce_weight + dice_weight - 1.0) > 1e-6:
            raise ValueError("BCE and Dice weights must sum to 1.0")
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = BCELoss(reduction=reduction)
        self.dice_loss = DiceLoss(smooth=smooth, reduction=reduction)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined BCE-Dice loss.
        
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Target masks (B, 1, H, W)
            
        Returns:
            Combined loss value
        """
        bce_loss = self.bce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize Focal loss.
        
        Args:
            alpha: Weight for positive class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal loss.
        
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Target masks (B, 1, H, W)
            
        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Calculate focal weight
        pt = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Calculate focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

class IoULoss(nn.Module):
    """Intersection over Union (IoU) loss."""
    
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean") -> None:
        """Initialize IoU loss.
        
        Args:
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss.
        
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Target masks (B, 1, H, W)
            
        Returns:
            IoU loss value
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - IoU)
        loss = 1.0 - iou
        
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

def get_loss_function(
    loss_name: str,
    **kwargs: any,
) -> nn.Module:
    """Get loss function by name.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function instance
    """
    loss_functions = {
        "bce": BCELoss,
        "dice": DiceLoss,
        "bce_dice": BCEDiceLoss,
        "focal": FocalLoss,
        "iou": IoULoss,
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name](**kwargs) 