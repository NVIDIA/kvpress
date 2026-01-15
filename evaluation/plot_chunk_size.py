# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np

# Data - each method has values for each chunk size
chunk_sizes = ["0.5k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]

# Compute times for each method (use "OOM" for out-of-memory cases)
method_1_times  = [138, 131, 116, 121, 132, "OOM", "OOM", "OOM", "OOM"]
method_2_times = [108, 80, 72, 70, 73, 80, "OOM", "OOM", "OOM"]  
method_3_times = [98, 60, 41, 32, 27, 25, 25, 25, 26]

def process_data(data):
    """Convert data list, replacing 'OOM' with 0 and tracking OOM positions."""
    numeric_data = []
    oom_indices = []
    for i, val in enumerate(data):
        if val == "OOM":
            numeric_data.append(0)
            oom_indices.append(i)
        else:
            numeric_data.append(val)
    return numeric_data, oom_indices

# Define 3 different colors with transparency (fill) and solid edge colors
# Format: (fill_color_with_alpha, edge_color)
colors_fill = [
    (0.529, 0.808, 0.922, 0.5),   # Light blue with 50% opacity
    (1.0, 0.6, 0.6, 0.5),         # Light red/coral with 50% opacity  
    (0.565, 0.933, 0.565, 0.5),   # Light green with 50% opacity
]
colors_edge = [
    "#4A90D9",  # Darker blue edge
    "#E74C3C",  # Darker red edge
    "#27AE60",  # Darker green edge
]
method_names = ["kvzip", "kvsquared_keydiff_0.5", "kvsquared_keydiff_0.02"]

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Bar positioning
x = np.arange(len(chunk_sizes))
bar_width = 0.25
offsets = [-bar_width, 0, bar_width]

# Create grouped bars
all_data_raw = [method_1_times, method_2_times, method_3_times]
bars_list = []

for i, (data_raw, fill_color, edge_color, name) in enumerate(
    zip(all_data_raw, colors_fill, colors_edge, method_names)
):
    # Process data to handle OOM values
    data, oom_indices = process_data(data_raw)
    
    bars = ax.bar(
        x + offsets[i],
        data,
        bar_width,
        color=fill_color,
        edgecolor=edge_color,
        linewidth=1.5,
        label=name,
    )
    bars_list.append(bars)
    
    # Add value labels on top of bars
    for j, (bar, val) in enumerate(zip(bars, data)):
        if val > 0:  # Only show label for non-zero values
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{int(val) if val == int(val) else val}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    
    # Add OOM annotations at method's typical height
    bar_centers = x + offsets[i]
    oom_y_positions = [130, 90, None]  # Blue at 130, red at 90, green doesn't OOM
    if oom_y_positions[i] is not None:
        for oom_idx in oom_indices:
            ax.text(
                bar_centers[oom_idx],
                oom_y_positions[i],
                "OOM",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=edge_color,
            )
    
    # Draw line connecting the bars (only through non-zero values)
    non_zero_mask = [v > 0 for v in data]
    line_x = [bar_centers[j] for j in range(len(data)) if non_zero_mask[j]]
    line_y = [data[j] for j in range(len(data)) if non_zero_mask[j]]
    
    if len(line_x) > 1:  # Only draw line if we have at least 2 points
        ax.plot(line_x, line_y, color=edge_color, linewidth=2, marker='o', markersize=5, zorder=5)

# Add horizontal dashed line at 20s
ax.axhline(y=20, color="black", linestyle="--", linewidth=1.5)

# Customize axes
ax.set_xticks(x)
ax.set_xticklabels(chunk_sizes)
ax.set_xlabel("Repeat chunk size", fontsize=12)
ax.set_ylabel("Compute time (s)", fontsize=12)

# Set y-axis limits and ticks
ax.set_ylim(0, 110)
ax.set_yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])

# Add grid lines (horizontal only, dashed)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add legend
ax.legend(loc="upper center", fontsize=10, ncol=3)

plt.tight_layout()

# Save figure
plt.savefig("chunk_size_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: chunk_size_comparison.png and chunk_size_comparison.pdf")

plt.show()
