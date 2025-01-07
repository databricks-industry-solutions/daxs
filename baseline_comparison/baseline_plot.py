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
categories = ['Processing Time\n(minutes)', 'Cost\n($)']
isolation_forest = [560, 11]  # Processing time and cost for Isolation Forest
daxs = [2, 1]  # Processing time and cost for DAXS

# Calculate bar positions
x = np.arange(len(categories))
width = 0.35

# Create figure and axis with larger size
fig, ax = plt.subplots(figsize=(12, 8))

# Set logarithmic scale for y-axis
ax.set_yscale('log')

# Add speedup annotations first
annotations = ['280x\nspeed-up', '11x\ncost reduction']
y_positions = [6, 6]  # Lower height for both annotations
for i in range(len(categories)):
    middle_x = x[i]
    ax.text(middle_x, y_positions[i], annotations[i],
            ha='center',
            va='center',  
            fontsize=14,  
            fontweight='bold',
            bbox=dict(boxstyle='rarrow,pad=0.8',  
                     fc='white',
                     ec='gray',
                     alpha=1.0),
            zorder=100)  

# Create bars with higher zorder
rects1 = ax.bar(x - width/2, isolation_forest, width, label='Isolation Forest', color='#2C394B', zorder=3)
rects2 = ax.bar(x + width/2, daxs, width, label='DAXS', color='#FF6B6B', zorder=3)

# Customize the plot
ax.set_title('DAXS Performance Comparison', fontsize=18, pad=20)  
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14)  
ax.legend(fontsize=14, loc='upper right', bbox_to_anchor=(0.98, 0.98))  

# Add grid with light gray color
ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#ddd', zorder=0)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels on top of each bar with increased font size
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', 
                    va='bottom',
                    fontsize=12)  

autolabel(rects1)
autolabel(rects2)

# Increase y-axis tick label font size
ax.tick_params(axis='y', labelsize=14)

# Adjust layout and display
plt.tight_layout()

# Save the plot
plt.savefig('daxs_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# COMMAND ----------
