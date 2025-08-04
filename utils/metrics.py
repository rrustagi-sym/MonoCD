import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class DetectionMetrics:
    """
    A comprehensive metrics calculator for object detection tasks.
    Supports precision, recall, accuracy, F1-score, and IoU-based metrics.
    """
    
    def __init__(self, iou_threshold: float = 0.5, class_names: Optional[List[str]] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            iou_threshold: IoU threshold for considering a detection as correct
            class_names: List of class names for per-class metrics
        """
        self.iou_threshold = iou_threshold
        self.class_names = class_names or ['Car', 'Pedestrian', 'Cyclist']
        self.reset()
    
    def reset(self):
        """Reset all metrics to zero."""
        self.tp = defaultdict(int)  # True positives per class
        self.fp = defaultdict(int)  # False positives per class
        self.fn = defaultdict(int)  # False negatives per class
        self.total_predictions = defaultdict(int)  # Total predictions per class
        self.total_ground_truth = defaultdict(int)  # Total ground truth per class
        self.iou_scores = defaultdict(list)  # IoU scores for each detection
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format
            
        Returns:
            IoU score between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, predictions: List[Dict], ground_truth: List[Dict]):
        """
        Update metrics with new predictions and ground truth.
        
        Args:
            predictions: List of prediction dictionaries with 'bbox', 'class_id', 'score'
            ground_truth: List of ground truth dictionaries with 'bbox', 'class_id'
        """
        # Group predictions and ground truth by class
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)
        
        for pred in predictions:
            class_id = pred['class_id']
            pred_by_class[class_id].append(pred)
            self.total_predictions[class_id] += 1
        
        for gt in ground_truth:
            class_id = gt['class_id']
            gt_by_class[class_id].append(gt)
            self.total_ground_truth[class_id] += 1
        
        # Calculate metrics for each class
        for class_id in set(list(pred_by_class.keys()) + list(gt_by_class.keys())):
            class_preds = pred_by_class[class_id]
            class_gts = gt_by_class[class_id]
            
            # Sort predictions by confidence score (descending)
            class_preds.sort(key=lambda x: x['score'], reverse=True)
            
            # Track which ground truth boxes have been matched
            gt_matched = [False] * len(class_gts)
            
            for pred in class_preds:
                pred_bbox = pred['bbox']
                best_iou = 0.0
                best_gt_idx = -1
                
                # Find the ground truth box with highest IoU
                for gt_idx, gt in enumerate(class_gts):
                    if gt_matched[gt_idx]:
                        continue
                    
                    gt_bbox = gt['bbox']
                    iou = self.calculate_iou(pred_bbox, gt_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Update metrics based on IoU threshold
                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    self.tp[class_id] += 1
                    gt_matched[best_gt_idx] = True
                    self.iou_scores[class_id].append(best_iou)
                else:
                    self.fp[class_id] += 1
            
            # Count unmatched ground truth boxes as false negatives
            for gt_idx, matched in enumerate(gt_matched):
                if not matched:
                    self.fn[class_id] += 1
    
    def get_precision(self, class_id: Optional[int] = None) -> Union[float, Dict[int, float]]:
        """
        Calculate precision for a specific class or all classes.
        
        Args:
            class_id: Class ID to calculate precision for, or None for all classes
            
        Returns:
            Precision value(s)
        """
        if class_id is not None:
            tp = self.tp[class_id]
            fp = self.fp[class_id]
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        precisions = {}
        for class_id in self.class_names:
            tp = self.tp[class_id]
            fp = self.fp[class_id]
            precisions[class_id] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return precisions
    
    def get_recall(self, class_id: Optional[int] = None) -> Union[float, Dict[int, float]]:
        """
        Calculate recall for a specific class or all classes.
        
        Args:
            class_id: Class ID to calculate recall for, or None for all classes
            
        Returns:
            Recall value(s)
        """
        if class_id is not None:
            tp = self.tp[class_id]
            fn = self.fn[class_id]
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        recalls = {}
        for class_id in self.class_names:
            tp = self.tp[class_id]
            fn = self.fn[class_id]
            recalls[class_id] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return recalls
    
    def get_accuracy(self, class_id: Optional[int] = None) -> Union[float, Dict[int, float]]:
        """
        Calculate accuracy for a specific class or all classes.
        
        Args:
            class_id: Class ID to calculate accuracy for, or None for all classes
            
        Returns:
            Accuracy value(s)
        """
        if class_id is not None:
            tp = self.tp[class_id]
            fp = self.fp[class_id]
            fn = self.fn[class_id]
            total = tp + fp + fn
            return tp / total if total > 0 else 0.0
        
        accuracies = {}
        for class_id in self.class_names:
            tp = self.tp[class_id]
            fp = self.fp[class_id]
            fn = self.fn[class_id]
            total = tp + fp + fn
            accuracies[class_id] = tp / total if total > 0 else 0.0
        
        return accuracies
    
    def get_f1_score(self, class_id: Optional[int] = None) -> Union[float, Dict[int, float]]:
        """
        Calculate F1-score for a specific class or all classes.
        
        Args:
            class_id: Class ID to calculate F1-score for, or None for all classes
            
        Returns:
            F1-score value(s)
        """
        if class_id is not None:
            precision = self.get_precision(class_id)
            recall = self.get_recall(class_id)
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        f1_scores = {}
        for class_id in self.class_names:
            precision = self.get_precision(class_id)
            recall = self.get_recall(class_id)
            f1_scores[class_id] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1_scores
    
    def get_mean_iou(self, class_id: Optional[int] = None) -> Union[float, Dict[int, float]]:
        """
        Calculate mean IoU for a specific class or all classes.
        
        Args:
            class_id: Class ID to calculate mean IoU for, or None for all classes
            
        Returns:
            Mean IoU value(s)
        """
        if class_id is not None:
            iou_scores = self.iou_scores[class_id]
            return np.mean(iou_scores) if iou_scores else 0.0
        
        mean_ious = {}
        for class_id in self.class_names:
            iou_scores = self.iou_scores[class_id]
            mean_ious[class_id] = np.mean(iou_scores) if iou_scores else 0.0
        
        return mean_ious
    
    def get_all_metrics(self) -> Dict[str, Union[float, Dict[int, float]]]:
        """
        Get all calculated metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        return {
            'precision': self.get_precision(),
            'recall': self.get_recall(),
            'accuracy': self.get_accuracy(),
            'f1_score': self.get_f1_score(),
            'mean_iou': self.get_mean_iou(),
            'total_predictions': dict(self.total_predictions),
            'total_ground_truth': dict(self.total_ground_truth),
            'true_positives': dict(self.tp),
            'false_positives': dict(self.fp),
            'false_negatives': dict(self.fn)
        }
    
    def print_summary(self):
        """Print a summary of all metrics."""
        metrics = self.get_all_metrics()
        
        logger.info("=" * 60)
        logger.info("DETECTION METRICS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"IoU Threshold: {self.iou_threshold}")
        logger.info("-" * 60)
        
        for class_id in self.class_names:
            logger.info(f"Class: {class_id}")
            logger.info(f"  Precision: {metrics['precision'][class_id]:.4f}")
            logger.info(f"  Recall:    {metrics['recall'][class_id]:.4f}")
            logger.info(f"  Accuracy:  {metrics['accuracy'][class_id]:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score'][class_id]:.4f}")
            logger.info(f"  Mean IoU:  {metrics['mean_iou'][class_id]:.4f}")
            logger.info(f"  TP: {metrics['true_positives'][class_id]}, "
                       f"FP: {metrics['false_positives'][class_id]}, "
                       f"FN: {metrics['false_negatives'][class_id]}")
            logger.info("-" * 60)
        
        # Calculate macro averages
        macro_precision = np.mean(list(metrics['precision'].values()))
        macro_recall = np.mean(list(metrics['recall'].values()))
        macro_accuracy = np.mean(list(metrics['accuracy'].values()))
        macro_f1 = np.mean(list(metrics['f1_score'].values()))
        macro_iou = np.mean(list(metrics['mean_iou'].values()))
        
        logger.info("MACRO AVERAGES:")
        logger.info(f"  Precision: {macro_precision:.4f}")
        logger.info(f"  Recall:    {macro_recall:.4f}")
        logger.info(f"  Accuracy:  {macro_accuracy:.4f}")
        logger.info(f"  F1-Score:  {macro_f1:.4f}")
        logger.info(f"  Mean IoU:  {macro_iou:.4f}")
        logger.info("=" * 60)


def calculate_detection_metrics(predictions: List[Dict], 
                              ground_truth: List[Dict], 
                              iou_threshold: float = 0.5,
                              class_names: Optional[List[str]] = None) -> Dict:
    """
    Convenience function to calculate detection metrics in one call.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for considering a detection as correct
        class_names: List of class names
        
    Returns:
        Dictionary containing all metrics
    """
    metrics_calculator = DetectionMetrics(iou_threshold=iou_threshold, class_names=class_names)
    metrics_calculator.update(predictions, ground_truth)
    return metrics_calculator.get_all_metrics()


def format_metrics_for_tensorboard(metrics: Dict) -> Dict[str, float]:
    """
    Format metrics for TensorBoard logging.
    
    Args:
        metrics: Dictionary of metrics from DetectionMetrics
        
    Returns:
        Flattened dictionary suitable for TensorBoard
    """
    tensorboard_metrics = {}
    
    for metric_name, metric_values in metrics.items():
        if isinstance(metric_values, dict):
            for class_name, value in metric_values.items():
                tensorboard_metrics[f"{metric_name}_{class_name}"] = value
        else:
            tensorboard_metrics[metric_name] = metric_values
    
    return tensorboard_metrics 