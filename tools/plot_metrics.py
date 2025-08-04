#!/usr/bin/env python3
"""
Script to plot KITTI AP metrics from CSV file showing how metrics vary with iterations.
This script creates various plots to visualize training progress.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_and_clean_data(csv_file):
    """Load CSV data and clean it."""
    logger = logging.getLogger(__name__)
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Successfully loaded CSV with {len(df)} rows")
        
        # Remove rows with missing values
        df_clean = df.dropna()
        logger.info(f"After cleaning: {len(df_clean)} rows")
        
        return df_clean
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_file}")
        return None
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None

def plot_ap_vs_iterations(df, output_dir, ap_type='Car AP@0.70, 0.70, 0.70'):
    """Plot AP metrics vs iterations for a specific AP type."""
    logger = logging.getLogger(__name__)
    
    # Filter for the specified AP type
    df_filtered = df[df['AP_Type'] == ap_type].copy()
    
    if len(df_filtered) == 0:
        logger.warning(f"No data found for AP type: {ap_type}")
        return False
    
    # Sort by iteration
    df_filtered = df_filtered.sort_values('iteration')
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot different metrics - organized by difficulty level
    metrics_to_plot = [
        # 3D metrics
        ('3d_easy', '3D AP Easy', 'o', 'lightblue'),
        ('3d_moderate', '3D AP Moderate', 's', 'blue'),
        ('3d_hard', '3D AP Hard', '^', 'darkblue'),
        # 2D BBox metrics
        ('bbox_easy', '2D BBox AP Easy', 'd', 'lightcoral'),
        ('bbox_moderate', '2D BBox AP Moderate', 'v', 'red'),
        ('bbox_hard', '2D BBox AP Hard', '<', 'darkred'),
        # BEV metrics
        ('bev_easy', 'BEV AP Easy', '>', 'lightgreen'),
        ('bev_moderate', 'BEV AP Moderate', 'p', 'green'),
        ('bev_hard', 'BEV AP Hard', '*', 'darkgreen')
    ]
    
    for metric, label, marker, color in metrics_to_plot:
        if metric in df_filtered.columns:
            plt.plot(df_filtered['iteration'], df_filtered[metric], 
                    marker=marker, label=label, color=color, linewidth=2, markersize=6)
    
    plt.title(f'KITTI AP Metrics Over Training Iterations\n{ap_type}', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Precision (AP)', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_dir) / f'ap_metrics_{ap_type.replace(" ", "_").replace(",", "")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot to: {output_file}")
    return True

def plot_all_ap_types(df, output_dir):
    """Plot all AP types in separate subplots."""
    logger = logging.getLogger(__name__)
    
    # Get unique AP types
    ap_types = df['AP_Type'].unique()
    
    if len(ap_types) == 0:
        logger.warning("No AP types found in data")
        return False
    
    # Create subplots
    fig, axes = plt.subplots(len(ap_types), 1, figsize=(12, 4*len(ap_types)))
    if len(ap_types) == 1:
        axes = [axes]
    
    for i, ap_type in enumerate(ap_types):
        df_filtered = df[df['AP_Type'] == ap_type].sort_values('iteration')
        
        # Plot metrics for this AP type
        metrics_to_plot = [
            ('3d_moderate', '3D Moderate', 'o', 'blue'),
            ('bbox_moderate', '2D BBox Moderate', 's', 'red'),
            ('bev_moderate', 'BEV Moderate', '^', 'green')
        ]
        
        for metric, label, marker, color in metrics_to_plot:
            if metric in df_filtered.columns:
                axes[i].plot(df_filtered['iteration'], df_filtered[metric], 
                           marker=marker, label=label, color=color, linewidth=2, markersize=4)
        
        axes[i].set_title(f'{ap_type}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('AP', fontsize=10)
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=9)
        
        if i == len(ap_types) - 1:  # Only last subplot gets x-label
            axes[i].set_xlabel('Iteration', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_dir) / 'all_ap_types.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved all AP types plot to: {output_file}")
    return True

def plot_heatmap(df, output_dir):
    """Create a heatmap showing AP values across iterations and metrics."""
    logger = logging.getLogger(__name__)
    
    # Filter for standard AP type
    ap_type = 'Car AP@0.70, 0.70, 0.70'
    df_filtered = df[df['AP_Type'] == ap_type].sort_values('iteration')
    
    if len(df_filtered) == 0:
        logger.warning(f"No data found for AP type: {ap_type}")
        return False
    
    # Select metrics for heatmap
    metrics = ['3d_easy', '3d_moderate', '3d_hard', 'bbox_easy', 'bbox_moderate', 'bbox_hard']
    available_metrics = [m for m in metrics if m in df_filtered.columns]
    
    if not available_metrics:
        logger.warning("No suitable metrics found for heatmap")
        return False
    
    # Create heatmap data
    heatmap_data = df_filtered[available_metrics].values.T
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create heatmap
    im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Set labels
    plt.yticks(range(len(available_metrics)), available_metrics)
    plt.xticks(range(0, len(df_filtered), max(1, len(df_filtered)//10)), 
               df_filtered['iteration'].iloc[::max(1, len(df_filtered)//10)])
    plt.xlabel('Iteration')
    plt.title(f'AP Metrics Heatmap - {ap_type}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Average Precision (AP)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_dir) / 'ap_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to: {output_file}")
    return True

def plot_best_performance(df, output_dir):
    """Plot the best performance achieved for each metric."""
    logger = logging.getLogger(__name__)
    
    # Filter for standard AP type
    ap_type = 'Car AP@0.70, 0.70, 0.70'
    df_filtered = df[df['AP_Type'] == ap_type]
    
    if len(df_filtered) == 0:
        logger.warning(f"No data found for AP type: {ap_type}")
        return False
    
    # Find best performance for each metric
    metrics = ['3d_easy', '3d_moderate', '3d_hard', 'bbox_easy', 'bbox_moderate', 'bbox_hard']
    available_metrics = [m for m in metrics if m in df_filtered.columns]
    
    best_values = []
    best_iterations = []
    metric_labels = []
    
    for metric in available_metrics:
        best_idx = df_filtered[metric].idxmax()
        best_values.append(df_filtered.loc[best_idx, metric])
        best_iterations.append(df_filtered.loc[best_idx, 'iteration'])
        metric_labels.append(metric.replace('_', ' ').title())
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(len(available_metrics)), best_values, 
                   color=['blue', 'darkblue', 'lightblue', 'red', 'darkred', 'lightcoral'])
    
    # Add value labels on bars
    for i, (bar, value, iteration) in enumerate(zip(bars, best_values, best_iterations)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}\n(iter {iteration})', 
                ha='center', va='bottom', fontsize=9)
    
    plt.title(f'Best AP Performance Achieved - {ap_type}', fontsize=14, fontweight='bold')
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Best Average Precision (AP)', fontsize=12)
    plt.xticks(range(len(available_metrics)), metric_labels, rotation=45, ha='right')
    plt.ylim(0, max(best_values) * 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_dir) / 'best_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved best performance plot to: {output_file}")
    return True

def main():
    """Main function to create all plots."""
    parser = argparse.ArgumentParser(description="Plot KITTI AP metrics from CSV")
    parser.add_argument("--csv-file", type=str, required=False, default="/home/rrustagi_symbotic_com/MonoCD/output_finetuned/eval_metrics_with_iterations.csv", help="Path to CSV file")
    parser.add_argument("--output-dir", type=str, default="./plots", help="Output directory for plots")
    parser.add_argument("--ap-type", type=str, default="Car AP@0.70, 0.70, 0.70", 
                       help="AP type to plot (default: Car AP@0.70, 0.70, 0.70)")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from: {args.csv_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    df = load_and_clean_data(args.csv_file)
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Print data summary
    logger.info(f"Data summary:")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  AP types: {df['AP_Type'].unique()}")
    logger.info(f"  Iteration range: {df['iteration'].min()} - {df['iteration'].max()}")
    
    # Create plots
    success_count = 0
    
    # Plot 1: AP vs iterations for specific AP type
    if plot_ap_vs_iterations(df, output_dir, args.ap_type):
        success_count += 1
    
    # Plot 2: All AP types
    if plot_all_ap_types(df, output_dir):
        success_count += 1
    
    # Plot 3: Heatmap
    if plot_heatmap(df, output_dir):
        success_count += 1
    
    # Plot 4: Best performance
    if plot_best_performance(df, output_dir):
        success_count += 1
    
    logger.info(f"Successfully created {success_count} plots in {output_dir}")
    
    # Print some statistics
    if len(df) > 0:
        logger.info("\nPerformance Statistics:")
        ap_type_data = df[df['AP_Type'] == args.ap_type]
        if len(ap_type_data) > 0:
            for metric in ['3d_easy', 'bbox_easy', 'bev_easy']:
                if metric in ap_type_data.columns:
                    max_val = ap_type_data[metric].max()
                    max_iter = ap_type_data.loc[ap_type_data[metric].idxmax(), 'iteration']
                    logger.info(f"  {metric}: max = {max_val:.3f} at iteration {max_iter}")

if __name__ == "__main__":
    main() 