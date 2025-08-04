#!/usr/bin/env python3
"""
Example script demonstrating how to use the DetectionMetrics class.
This shows how to calculate precision, recall, and accuracy for object detection.
"""

import numpy as np
from utils.metrics import DetectionMetrics, calculate_detection_metrics

def create_sample_data():
    """Create sample predictions and ground truth data for demonstration."""
    
    # Sample ground truth boxes (format: [x1, y1, x2, y2])
    ground_truth = [
        {'bbox': np.array([100, 100, 200, 200]), 'class_id': 0},  # Car
        {'bbox': np.array([300, 150, 400, 250]), 'class_id': 1},  # Pedestrian
        {'bbox': np.array([500, 200, 600, 300]), 'class_id': 0},  # Car
    ]
    
    # Sample predictions (format: [x1, y1, x2, y2, score, class_id])
    predictions = [
        {'bbox': np.array([105, 105, 195, 195]), 'class_id': 0, 'score': 0.95},  # Correct car detection
        {'bbox': np.array([310, 155, 390, 245]), 'class_id': 1, 'score': 0.88},  # Correct pedestrian detection
        {'bbox': np.array([480, 190, 580, 290]), 'class_id': 0, 'score': 0.92},  # Correct car detection
        {'bbox': np.array([700, 300, 800, 400]), 'class_id': 0, 'score': 0.75},  # False positive car
        {'bbox': np.array([150, 150, 250, 250]), 'class_id': 1, 'score': 0.65},  # False positive pedestrian
    ]
    
    return predictions, ground_truth

def example_basic_usage():
    """Demonstrate basic usage of DetectionMetrics."""
    print("=" * 60)
    print("BASIC METRICS CALCULATION EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    predictions, ground_truth = create_sample_data()
    
    # Initialize metrics calculator
    metrics_calculator = DetectionMetrics(
        iou_threshold=0.5,
        class_names=['Car', 'Pedestrian', 'Cyclist']
    )
    
    # Update metrics with data
    metrics_calculator.update(predictions, ground_truth)
    
    # Get all metrics
    metrics = metrics_calculator.get_all_metrics()
    
    # Print results
    print("\nPer-class metrics:")
    for class_name in ['Car', 'Pedestrian']:
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision'][class_name]:.4f}")
        print(f"  Recall:    {metrics['recall'][class_name]:.4f}")
        print(f"  Accuracy:  {metrics['accuracy'][class_name]:.4f}")
        print(f"  F1-Score:  {metrics['f1_score'][class_name]:.4f}")
        print(f"  Mean IoU:  {metrics['mean_iou'][class_name]:.4f}")
    
    print("\nCounts:")
    for class_name in ['Car', 'Pedestrian']:
        print(f"{class_name}:")
        print(f"  True Positives:  {metrics['true_positives'][class_name]}")
        print(f"  False Positives: {metrics['false_positives'][class_name]}")
        print(f"  False Negatives: {metrics['false_negatives'][class_name]}")
        print(f"  Total Predictions: {metrics['total_predictions'][class_name]}")
        print(f"  Total Ground Truth: {metrics['total_ground_truth'][class_name]}")

def example_convenience_function():
    """Demonstrate the convenience function."""
    print("\n" + "=" * 60)
    print("CONVENIENCE FUNCTION EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    predictions, ground_truth = create_sample_data()
    
    # Use convenience function
    metrics = calculate_detection_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=0.5,
        class_names=['Car', 'Pedestrian', 'Cyclist']
    )
    
    print("\nResults from convenience function:")
    for class_name in ['Car', 'Pedestrian']:
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision'][class_name]:.4f}")
        print(f"  Recall:    {metrics['recall'][class_name]:.4f}")
        print(f"  F1-Score:  {metrics['f1_score'][class_name]:.4f}")

def example_different_iou_thresholds():
    """Demonstrate how IoU threshold affects metrics."""
    print("\n" + "=" * 60)
    print("DIFFERENT IOU THRESHOLDS EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    predictions, ground_truth = create_sample_data()
    
    # Test different IoU thresholds
    iou_thresholds = [0.3, 0.5, 0.7]
    
    for iou_thresh in iou_thresholds:
        metrics = calculate_detection_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            iou_threshold=iou_thresh,
            class_names=['Car', 'Pedestrian', 'Cyclist']
        )
        
        print(f"\nIoU Threshold: {iou_thresh}")
        print("Car - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
            metrics['precision']['Car'],
            metrics['recall']['Car'],
            metrics['f1_score']['Car']
        ))
        print("Pedestrian - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
            metrics['precision']['Pedestrian'],
            metrics['recall']['Pedestrian'],
            metrics['f1_score']['Pedestrian']
        ))

def example_tensorboard_format():
    """Demonstrate TensorBoard formatting."""
    print("\n" + "=" * 60)
    print("TENSORBOARD FORMATTING EXAMPLE")
    print("=" * 60)
    
    from utils.metrics import format_metrics_for_tensorboard
    
    # Create sample data and calculate metrics
    predictions, ground_truth = create_sample_data()
    metrics = calculate_detection_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=0.5,
        class_names=['Car', 'Pedestrian', 'Cyclist']
    )
    
    # Format for TensorBoard
    tensorboard_metrics = format_metrics_for_tensorboard(metrics)
    
    print("\nTensorBoard formatted metrics:")
    for key, value in tensorboard_metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_convenience_function()
    example_different_iou_thresholds()
    example_tensorboard_format()
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED")
    print("=" * 60)
    print("\nTo use this in your training:")
    print("1. Import DetectionMetrics from utils.metrics")
    print("2. Initialize with your IoU threshold and class names")
    print("3. Call update() with your predictions and ground truth")
    print("4. Get metrics using get_precision(), get_recall(), etc.")
    print("5. Use format_metrics_for_tensorboard() for logging") 