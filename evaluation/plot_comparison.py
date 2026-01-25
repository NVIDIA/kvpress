#!/usr/bin/env python3
"""
Plot comparison of KV cache compression methods.
Edit the data below and run: python plot_comparison.py
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# =============================================================================
# EDIT YOUR DATA HERE
# =============================================================================

# Format: { "Method Name": { compression_ratio: score, ... }, ... }
DATA = {
    "KVzip (no InnerPress)": {
        0.5: 93,
        0.75: 93.9,
        0.88: 94.4,
    },
    "KVSquared + KeyDiff top 0.5": {
        0.5: 93,
        0.75: 93.9,
        0.88: 89.8,
    },
    
    #"KVSquared + KeyDiff top 0.2": {
    #    0.5: 90.9,
    #    0.75: 87.6,
    #    0.88: 79.4,
    #},
    "KVSquared + KeyDiff top 0.02": {
        0.5: 86.2, # 88.7
        0.75: 79.3, # 76.3
        0.88: 72.8, # 70.3
    },
    #"KVSquared + Random top 0.5": {
    #    0.5: 93,
    #    0.75: 93.7,
    #    0.88: 75.1,
    #},
    #"KVSquared + Random top 0.02": {
    #    0.5: 89.2,
    #    0.75: 73.7,
    #    0.88: 55.1,
    #},
    "KeyDiff": {
        0.25: 91.6,
        0.5: 85.5,
        0.75: 72.9,
        0.88: 63.3,
    },
    "RandomPress": {
        0.25: 61.9,
        0.5: 6.88,
        0.75: 0.56,
        0.88: 0.18,
    },
}

# =============================================================================
# PLOT SETTINGS
# =============================================================================

TITLE = "KV² Compression Comparison (chunk_size=2048)"
XLABEL = "Compression Ratio"
YLABEL = "Score (%)"
FIGSIZE = (10, 6)
OUTPUT_FILE = "kv_comparison.png"  # Set to None to only display, not save

# Colors and markers for each method (matching plot_chunk_size.py where applicable)
# Order: KVzip=blue, KVSquared+KeyDiff 0.5=red, 0.02=green, KeyDiff=black, RandomPress=black
COLORS = ["#4A90D9", "#E74C3C", "#27AE60", "#000000", "#555555", "#9B59B6", "#f39c12", "#1abc9c", "#e91e63", "#795548"]
MARKERS = ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o"]

# =============================================================================
# PLOTTING CODE (no need to edit below)
# =============================================================================


def plot_comparison():
    plt.figure(figsize=FIGSIZE)
    
    for idx, (method_name, scores) in enumerate(DATA.items()):
        # Sort by compression ratio
        sorted_items = sorted(scores.items())
        x_vals = [item[0] for item in sorted_items]
        y_vals = [item[1] for item in sorted_items]
        
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        
        plt.plot(
            x_vals, 
            y_vals, 
            label=method_name,
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2,
        )
    
    # Add baseline (no compression) horizontal dotted line
    plt.axhline(y=95.7, color='gray', linestyle='--', linewidth=1.5, label='No compression')
    
    plt.xlabel(XLABEL, fontsize=12)
    plt.ylabel(YLABEL, fontsize=12)
    plt.title(TITLE, fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis as percentages (e.g., 50%, 75%)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
    
    # Set axis limits
    plt.xlim(0.0, 1.0)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    
    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {OUTPUT_FILE}")
    
    plt.show()


if __name__ == "__main__":
    plot_comparison()
