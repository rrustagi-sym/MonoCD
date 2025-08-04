#!/usr/bin/env python3
"""
Simple script to plot KITTI AP metrics from CSV file.
This fixes the issues with the original plotting script.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(csv_file, output_file):
    """Plot AP metrics from CSV file."""
    
    # Load CSV
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")
    
    # Filter for standard AP type
    ap_type_standard = 'Car AP@0.70, 0.70, 0.70'
    df_standard = df[df['AP_Type'] == ap_type_standard].copy()
    
    if len(df_standard) == 0:
        print(f"No data found for AP type: {ap_type_standard}")
        return False
    
    # Sort by iteration
    df_standard = df_standard.sort_values('iteration')
    print(f"Found {len(df_standard)} rows for {ap_type_standard}")
    
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
        if metric in df_standard.columns:
            # Remove any NaN values
            valid_data = df_standard[['iteration', metric]].dropna()
            if len(valid_data) > 0:
                plt.plot(valid_data['iteration'], valid_data[metric], 
                        marker=marker, label=label, color=color, linewidth=2, markersize=6)
                print(f"Plotted {metric}: {len(valid_data)} points")
    
    plt.title(f'KITTI AP Metrics Over Training Iterations\n{ap_type_standard}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Precision (AP)', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    # Save figure
    print(f"Saving plot to: {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved plot to: {output_file}")
    return True

if __name__ == "__main__":
    # Use the same paths as your original script
    csv_file = '/home/rrustagi_symbotic_com/MonoCD/output_finetuned/eval_metrics_with_iterations.csv'
    output_file = '/home/rrustagi_symbotic_com/MonoCD/output_finetuned/eval_metrics_finetuned.png'
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"Error: CSV file not found: {csv_file}")
        print("Please check the file path.")
    else:
        success = plot_metrics(csv_file, output_file)
        if success:
            print("Plotting completed successfully!")
        else:
            print("Plotting failed.") 