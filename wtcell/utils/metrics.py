"""Metrics computation utilities for the wtcell package."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torchmetrics import JaccardIndex, Dice, Precision, Recall, F1Score, Accuracy

logger = get_logger(__name__)

def compute_iou(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    num_classes: int = 2,
) -> float:
    """Compute Intersection over Union (IoU) between prediction and target.
    
    Args:
        pred: Prediction array/tensor (logits or probabilities)
        target: Target array/tensor (binary)
        threshold: Threshold for binarization (if pred contains probabilities)
        num_classes: Number of classes (default: 2 for binary segmentation)
        
    Returns:
        IoU score
    """
    # Convert to torch tensors if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Ensure tensors are on the same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Binarize predictions if needed
    if pred.dtype == torch.float32 and pred.max() <= 1.0:
        pred = (pred > threshold).float()
    
    # Ensure binary values
    pred = pred.bool().float()
    target = target.bool().float()
    
    # Compute IoU
    jaccard = JaccardIndex(task="binary", num_classes=num_classes)
    iou = jaccard(pred, target)
    
    return iou.item()

def compute_dice(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute Dice coefficient between prediction and target.
    
    Args:
        pred: Prediction array/tensor (logits or probabilities)
        target: Target array/tensor (binary)
        threshold: Threshold for binarization (if pred contains probabilities)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice score
    """
    # Convert to torch tensors if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Ensure tensors are on the same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Binarize predictions if needed
    if pred.dtype == torch.float32 and pred.max() <= 1.0:
        pred = (pred > threshold).float()
    
    # Ensure binary values
    pred = pred.bool().float()
    target = target.bool().float()
    
    # Compute Dice
    dice = Dice(average="micro")
    dice_score = dice(pred, target)
    
    return dice_score.item()

def compute_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    num_classes: int = 2,
) -> Dict[str, float]:
    """Compute multiple metrics for segmentation evaluation.
    
    Args:
        pred: Prediction array/tensor (logits or probabilities)
        target: Target array/tensor (binary)
        threshold: Threshold for binarization (if pred contains probabilities)
        num_classes: Number of classes (default: 2 for binary segmentation)
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Convert to torch tensors if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Ensure tensors are on the same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Binarize predictions if needed
    if pred.dtype == torch.float32 and pred.max() <= 1.0:
        pred = (pred > threshold).float()
    
    # Ensure binary values
    pred = pred.bool().float()
    target = target.bool().float()
    
    # Initialize metric calculators
    jaccard = JaccardIndex(task="binary", num_classes=num_classes)
    dice = Dice(average="micro")
    precision = Precision(task="binary", num_classes=num_classes)
    recall = Recall(task="binary", num_classes=num_classes)
    f1 = F1Score(task="binary", num_classes=num_classes)
    accuracy = Accuracy(task="binary", num_classes=num_classes)
    
    # Compute metrics
    metrics = {
        "iou": jaccard(pred, target).item(),
        "dice": dice(pred, target).item(),
        "precision": precision(pred, target).item(),
        "recall": recall(pred, target).item(),
        "f1": f1(pred, target).item(),
        "accuracy": accuracy(pred, target).item(),
    }
    
    return metrics

def compute_metrics_batch(
    preds: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    num_classes: int = 2,
    average: str = "mean",
) -> Dict[str, float]:
    """Compute metrics for a batch of predictions and targets.
    
    Args:
        preds: Batch of predictions (B, H, W) or (B, 1, H, W)
        targets: Batch of targets (B, H, W) or (B, 1, H, W)
        threshold: Threshold for binarization
        num_classes: Number of classes
        average: Averaging method ('mean', 'median', 'std')
        
    Returns:
        Dictionary containing averaged metrics
    """
    batch_metrics = []
    
    # Ensure 4D tensors
    if preds.dim() == 3:
        preds = preds.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    
    # Compute metrics for each sample in batch
    for i in range(preds.shape[0]):
        sample_metrics = compute_metrics(
            preds[i], targets[i], threshold, num_classes
        )
        batch_metrics.append(sample_metrics)
    
    # Aggregate metrics
    aggregated_metrics = {}
    for metric_name in batch_metrics[0].keys():
        values = [m[metric_name] for m in batch_metrics]
        
        if average == "mean":
            aggregated_metrics[metric_name] = np.mean(values)
        elif average == "median":
            aggregated_metrics[metric_name] = np.median(values)
        elif average == "std":
            aggregated_metrics[metric_name] = np.std(values)
        else:
            raise ValueError(f"Unknown averaging method: {average}")
    
    return aggregated_metrics

def compute_per_class_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    cell_types: List[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for each cell type separately.
    
    Args:
        preds: Predictions tensor
        targets: Targets tensor
        cell_types: List of cell type names
        threshold: Threshold for binarization
        
    Returns:
        Dictionary with metrics per cell type
    """
    per_class_metrics = {}
    
    for i, cell_type in enumerate(cell_types):
        if preds.shape[1] > 1:  # Multi-class case
            pred_class = preds[:, i:i+1]
            target_class = targets[:, i:i+1]
        else:  # Binary case
            pred_class = preds
            target_class = targets
        
        metrics = compute_metrics_batch(pred_class, target_class, threshold)
        per_class_metrics[cell_type] = metrics
    
    return per_class_metrics 