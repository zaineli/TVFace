import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib import cm
from scipy.interpolate import PchipInterpolator

# Create assets directory if it doesn't exist
os.makedirs('assets', exist_ok=True)

# Set up visualization defaults with minimal white space
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1  # Minimal padding
plt.rcParams['axes.titlesize'] = 0  # No titles
plt.rcParams['axes.labelpad'] = 8  # Reduced label padding

# Vibrant color palettes - much brighter and more visually appealing
GENDER_COLORS = ['#00A4EF', '#FF5252']  # Bright blue and red
ETHNICITY_COLORS = ['#FF5252', '#FF9E00', '#00C853', '#00A4EF', '#6200EA', '#D500F9', '#FFD600']
EXPRESSION_COLORS = ['#FF5252', '#FF9E00', '#00C853', '#00A4EF', '#6200EA', '#D500F9', '#FFD600']
POSE_COLORS = ['#00A4EF', '#FF9E00', '#00C853']

def create_gender_chart():
    """Create a vibrant vertical bar chart for gender distribution without title"""
    labels = ['Male', 'Female']
    percentages = [65.71, 34.29]
    
    # Create figure with minimal margins
    plt.figure(figsize=(6, 5))
    
    # Create vertical bars with vibrant colors
    bars = plt.bar(
        labels, 
        percentages, 
        color=GENDER_COLORS,
        width=0.6,
        edgecolor='white',
        linewidth=1
    )
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    # Set axis labels with minimal padding
    plt.ylabel('Percentage (%)', fontsize=12, labelpad=8)
    
    # Set y-axis limits with minimal padding
    plt.ylim(0, max(percentages) * 1.15)
    
    # Remove spines to reduce visual clutter
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout(pad=0.5)
    plt.savefig('assets/gender_distribution.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_ethnicity_chart():
    """Create a vibrant vertical bar chart for ethnicity distribution"""
    ethnicity = ['White', 'Middle Eastern', 'Black', 'East Asian', 
                'Indian', 'Latino Hispanic', 'Southeast Asian']
    percentages = [52.49, 18.04, 11.12, 9.36, 4.09, 3.48, 1.43]
    
    # Sort in descending order
    sorted_indices = np.argsort(percentages)[::-1]
    ethnicity = [ethnicity[i] for i in sorted_indices]
    percentages = [percentages[i] for i in sorted_indices]
    
    # Create figure with minimal margins
    plt.figure(figsize=(10, 6))
    
    # Create vertical bars with vibrant colors
    bars = plt.bar(
        ethnicity, 
        percentages, 
        color=ETHNICITY_COLORS[:len(ethnicity)],
        width=0.7,
        edgecolor='white',
        linewidth=1
    )
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.7,  # Reduced padding
            f'{height:.2f}%', 
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Set axis labels with minimal padding
    plt.ylabel('Percentage (%)', fontsize=12, labelpad=8)
    
    # Set y-axis limits with minimal padding
    plt.ylim(0, max(percentages) * 1.15)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=30, ha='right', fontsize=10)
    
    # Remove spines to reduce visual clutter
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout(pad=0.5)
    plt.savefig('assets/ethnicity_distribution.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_identity_distribution():
    """Create a vibrant vertical bar chart for identity distribution clusters"""
    # Create percentile groups for visualization
    percentiles = ['Top 1%', 'Top 5%', 'Top 10%', 'Top 25%', 'Average', 'Bottom 25%', 'Bottom 10%', 'Bottom 1%']
    images_count = [21983, 500, 250, 150, 90, 40, 20, 10]
    
    # Create figure with minimal margins
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with gradient colors
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(percentiles)))
    bars = ax.bar(
        percentiles, 
        images_count, 
        color=colors,
        width=0.7,
        edgecolor='white',
        linewidth=1
    )
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + (height * 0.03),  # Proportional padding
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Configure axes with minimal padding
    ax.set_ylabel('Number of Images per Identity', fontsize=12, labelpad=8)
    
    # Set log scale for better visualization of the range
    ax.set_yscale('log')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=30, ha='right', fontsize=10)
    
    # Remove spines to reduce visual clutter
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid for readability but with minimal visual impact
    ax.grid(True, linestyle='--', axis='y', alpha=0.2)
    
    # Ensure proper spacing with minimal margins
    fig.tight_layout(pad=0.5)
    plt.savefig('assets/identity_distribution.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == "__main__":
    print("Generating vibrant visualizations with minimal white space...")
    create_gender_chart()
    create_ethnicity_chart()
    # create_expression_chart()
    # create_head_pose_chart()
    create_identity_distribution()
    print("All visualizations have been generated in the 'assets' directory.")