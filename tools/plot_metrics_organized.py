#!/usr/bin/env python3
"""
Organized script to plot KITTI AP metrics from CSV file.
Shows easy, moderate, and hard metrics grouped by type (3D, 2D BBox, BEV).
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics_organized(csv_file, output_file):
    """Plot AP metrics organized by type with all difficulty levels."""
    
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
    
    # Create subplots for each metric type
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'KITTI AP Metrics Over Training Iterations\n{ap_type_standard}', 
                 fontsize=16, fontweight='bold')
    
    # Define metric groups
    metric_groups = [
        {
            'title': '3D Detection AP',
            'metrics': [
                ('3d_easy', 'Easy', 'o', 'lightblue'),
                ('3d_moderate', 'Moderate', 's', 'blue'),
                ('3d_hard', 'Hard', '^', 'darkblue')
            ],
            'axis': axes[0]
        },
        {
            'title': '2D BBox Detection AP',
            'metrics': [
                ('bbox_easy', 'Easy', 'd', 'lightcoral'),
                ('bbox_moderate', 'Moderate', 'v', 'red'),
                ('bbox_hard', 'Hard', '<', 'darkred')
            ],
            'axis': axes[1]
        },
        {
            'title': 'BEV Detection AP',
            'metrics': [
                ('bev_easy', 'Easy', '>', 'lightgreen'),
                ('bev_moderate', 'Moderate', 'p', 'green'),
                ('bev_hard', 'Hard', '*', 'darkgreen')
            ],
            'axis': axes[2]
        }
    ]
    
    # Plot each group
    for group in metric_groups:
        ax = group['axis']
        ax.set_title(group['title'], fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Precision (AP)', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        for metric, label, marker, color in group['metrics']:
            if metric in df_standard.columns:
                # Remove any NaN values
                valid_data = df_standard[['iteration', metric]].dropna()
                if len(valid_data) > 0:
                    ax.plot(valid_data['iteration'], valid_data[metric], 
                           marker=marker, label=label, color=color, 
                           linewidth=2, markersize=6)
                    print(f"Plotted {metric}: {len(valid_data)} points")
        
        ax.legend(fontsize=10, loc='best')
        
        # Only add x-label to the bottom subplot
        if group == metric_groups[-1]:
            ax.set_xlabel('Iteration', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    print(f"Saving plot to: {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved organized plot to: {output_file}")
    return True

def plot_metrics_comparison(csv_file, output_file):
    """Plot comparison of easy, moderate, hard for each metric type."""
    
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
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Difficulty Level Comparison - {ap_type_standard}', 
                 fontsize=16, fontweight='bold')
    
    # Define metric types and their difficulty levels
    metric_types = [
        {
            'name': '3D Detection',
            'prefix': '3d',
            'axis': axes[0],
            'color': 'blue'
        },
        {
            'name': '2D BBox Detection',
            'prefix': 'bbox',
            'axis': axes[1],
            'color': 'red'
        },
        {
            'name': 'BEV Detection',
            'prefix': 'bev',
            'axis': axes[2],
            'color': 'green'
        }
    ]
    
    for metric_type in metric_types:
        ax = metric_type['axis']
        prefix = metric_type['prefix']
        name = metric_type['name']
        base_color = metric_type['color']
        
        # Plot easy, moderate, hard for this metric type
        difficulties = [
            ('easy', 'Easy', 'light' + base_color, 'o'),
            ('moderate', 'Moderate', base_color, 's'),
            ('hard', 'Hard', 'dark' + base_color, '^')
        ]
        
        for difficulty, label, color, marker in difficulties:
            metric = f"{prefix}_{difficulty}"
            if metric in df_standard.columns:
                valid_data = df_standard[['iteration', metric]].dropna()
                if len(valid_data) > 0:
                    ax.plot(valid_data['iteration'], valid_data[metric], 
                           marker=marker, label=label, color=color, 
                           linewidth=2, markersize=6)
        
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Precision (AP)', fontsize=12)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # Save figure
    comparison_output = output_file.replace('.png', '_comparison.png')
    print(f"Saving comparison plot to: {comparison_output}")
    plt.savefig(comparison_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved comparison plot to: {comparison_output}")
    return True

if __name__ == "__main__":
    # Use the same paths as your original script
    csv_file = '/home/rrustagi_symbotic_com/MonoCD/output_finetuned/eval_metrics_with_iterations.csv'
    output_file = '/home/rrustagi_symbotic_com/MonoCD/output_finetuned/eval_metrics_organized.png'
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"Error: CSV file not found: {csv_file}")
        print("Please check the file path.")
    else:
        # Create organized plot
        success1 = plot_metrics_organized(csv_file, output_file)
        
        # Create comparison plot
        success2 = plot_metrics_comparison(csv_file, output_file)
        
        if success1 and success2:
            print("All plotting completed successfully!")
        else:
            print("Some plotting failed.") 