# Databricks notebook source

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("deep")

# Set white background
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Data
categories = ['', '']  # Empty categories for x-axis
isolation_forest = [560, 11]  # Processing time and cost for Isolation Forest
daxs = [2, 1]  # Processing time and cost for DAXS

# Calculate bar positions
x = np.arange(len(categories))
width = 0.35

# Create figure and axis with larger size
fig, ax = plt.subplots(figsize=(12, 8))

# Set logarithmic scale for y-axis
ax.set_yscale('log')

# Create bars
rects1 = ax.bar(x - width/2, isolation_forest, width, label='Isolation Forest', color='#2C394B', zorder=3)
rects2 = ax.bar(x + width/2, daxs, width, label='DAXS', color='#FF6B6B', zorder=3)

# Customize the plot
plt.suptitle('DAXS improves performance and cost', fontsize=20, y=0.95, weight='bold')

# Add grid with light gray color
ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#ddd', zorder=0)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if x[int((rect.get_x() - x[0] + width/2) / 1)] == 1:  # Cost column
            label = f'${height}'
        else:  # Time column
            label = f'{int(height)}'
            
        ax.text(rect.get_x() + rect.get_width()/2, height,
                label,
                ha='center', va='bottom',
                fontsize=12, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Add speedup annotations
annotations = ['280x\nspeed-up', '11x\ncost reduction']
y_positions = [6, 6]  # Same height for both annotations
for i in range(len(categories)):
    middle_x = x[i]
    ax.text(middle_x, y_positions[i], annotations[i],
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='rarrow,pad=0.8',
                     fc='white', ec='gray', alpha=1.0),
            zorder=100)

# Add labels below the bars
ax.text(x[0], 0.5, 'Processing Time\n(minutes)', ha='center', va='top', fontsize=14, fontweight='bold')
ax.text(x[1], 0.5, 'Cost\n($)', ha='center', va='top', fontsize=14, fontweight='bold')

# Set legend
ax.legend(fontsize=14, loc='upper right', bbox_to_anchor=(0.98, 0.98))

# Remove x-axis ticks and labels
ax.set_xticks([])
ax.set_xticklabels([])

# Add subtitle at the bottom
fig.text(0.5, 0.02, 'Cost and time to train and inference 10000 models on 1 Billion records', 
         fontsize=14, color='#666666', ha='center')

# Adjust layout with more bottom space for subtitle
plt.subplots_adjust(bottom=0.15)

# Save the plot
plt.savefig('daxs_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# COMMAND ----------
