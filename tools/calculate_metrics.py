#!/usr/bin/env python3
"""
Script to calculate precision, recall, and accuracy metrics for MonoCD detections.
This script can be used to evaluate model performance on validation/test sets.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from utils.metrics import DetectionMetrics, calculate_detection_metrics
from engine.inference import inference
from data import make_data_loader
from model.detector import build_detection_model
from utils.check_point import DetectronCheckpointer
from config import cfg

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate detection metrics for MonoCD")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for considering a detection as correct"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./metrics_output",
        help="Directory to save metrics results"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed metrics results to file"
    )
    
    return parser.parse_args()

def main():
    """Main function to calculate detection metrics."""
    args = parse_args()
    logger = setup_logging()
    
    # Load configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    logger.info("Loaded config from {}".format(args.config_file))
    logger.info("Configuration:\n{}".format(cfg))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(cfg.MODEL.DEVICE)
    logger.info("Using device: {}".format(device))
    
    # Build model
    model = build_detection_model(cfg)
    model.to(device)
    
    # Load checkpoint
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    
    # Create data loader
    data_loader = make_data_loader(
        cfg,
        is_train=False,
        is_distributed=False,
    )
    
    logger.info("Starting metrics calculation...")
    
    # Initialize metrics calculator
    metrics_calculator = DetectionMetrics(
        iou_threshold=args.iou_threshold,
        class_names=['Car', 'Pedestrian', 'Cyclist']
    )
    
    model.eval()
    total_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch["images"].to(device)
            targets = [target.to(device) for target in batch["targets"]]
            
            # Get predictions
            output, eval_utils, _ = model(images, targets)
            
            # Convert predictions and ground truth to format expected by metrics calculator
            predictions = []
            ground_truth = []
            
            # Process each image in the batch
            for i in range(len(targets)):
                target = targets[i]
                pred_boxes = output[i] if output.dim() > 1 else output
                
                # Extract ground truth boxes
                if hasattr(target, 'bbox') and target.bbox is not None:
                    gt_boxes = target.bbox.cpu().numpy()
                    gt_labels = target.get_field('labels').cpu().numpy() if hasattr(target, 'get_field') else []
                    
                    for j, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
                        ground_truth.append({
                            'bbox': box,
                            'class_id': label
                        })
                
                # Extract predicted boxes
                if pred_boxes is not None and len(pred_boxes) > 0:
                    pred_boxes_np = pred_boxes.cpu().numpy()
                    
                    # Assuming pred_boxes format is [x1, y1, x2, y2, score, class_id, ...]
                    for j in range(len(pred_boxes_np)):
                        box = pred_boxes_np[j][:4]  # First 4 elements are bbox coordinates
                        score = pred_boxes_np[j][4] if len(pred_boxes_np[j]) > 4 else 1.0
                        class_id = pred_boxes_np[j][5] if len(pred_boxes_np[j]) > 5 else 0
                        
                        predictions.append({
                            'bbox': box,
                            'class_id': class_id,
                            'score': score
                        })
            
            # Update metrics
            metrics_calculator.update(predictions, ground_truth)
            total_batches += 1
            
            if total_batches % 10 == 0:
                logger.info(f"Processed {total_batches} batches...")
    
    # Get final metrics
    metrics = metrics_calculator.get_all_metrics()
    
    # Print summary
    logger.info("=" * 80)
    logger.info("FINAL METRICS SUMMARY")
    logger.info("=" * 80)
    metrics_calculator.print_summary()
    
    # Save results if requested
    if args.save_results:
        import json
        from datetime import datetime
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Prepare results for saving
        results = {
            'timestamp': datetime.now().isoformat(),
            'config_file': args.config_file,
            'iou_threshold': args.iou_threshold,
            'total_batches': total_batches,
            'metrics': convert_numpy_types(metrics)
        }
        
        # Save to file
        output_file = output_dir / f"metrics_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
    
    # Print key metrics
    logger.info("\nKEY METRICS:")
    logger.info(f"IoU Threshold: {args.iou_threshold}")
    logger.info(f"Total Batches Processed: {total_batches}")
    
    # Calculate macro averages
    precision_values = list(metrics['precision'].values())
    recall_values = list(metrics['recall'].values())
    accuracy_values = list(metrics['accuracy'].values())
    f1_values = list(metrics['f1_score'].values())
    
    if precision_values:
        logger.info(f"Macro Precision: {np.mean(precision_values):.4f}")
        logger.info(f"Macro Recall: {np.mean(recall_values):.4f}")
        logger.info(f"Macro Accuracy: {np.mean(accuracy_values):.4f}")
        logger.info(f"Macro F1-Score: {np.mean(f1_values):.4f}")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    main() 