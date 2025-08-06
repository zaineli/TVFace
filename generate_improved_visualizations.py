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
    """Create a vibrant pie chart for gender distribution without title"""
    labels = ['Male', 'Female']
    sizes = [65.71, 34.29]
    
    # Create a figure with minimal margins
    plt.figure(figsize=(7, 7))
    
    # Create pie chart with vibrant colors
    wedges, texts = plt.pie(
        sizes, 
        labels=None,
        colors=GENDER_COLORS,
        wedgeprops={
            'width': 0.6,            # Make it a donut chart
            'edgecolor': 'white',
            'linewidth': 1,
        },
        startangle=90,
    )
    
    # Add percentage labels inside wedges
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta1 + wedge.theta2) / 2
        x = wedge.r * 0.75 * np.cos(np.radians(angle))
        y = wedge.r * 0.75 * np.sin(np.radians(angle))
        
        plt.text(
            x, y,
            f"{labels[i]}\n{sizes[i]}%",
            ha='center',
            va='center',
            fontsize=14,
            fontweight='bold',
            color='white'
        )
    
    # Set equal aspect ratio for perfect circular pie
    plt.axis('equal')
    plt.tight_layout(pad=0.5)
    
    # Save with minimal white space
    plt.savefig('assets/gender_distribution.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_ethnicity_chart():
    """Create a vibrant horizontal bar chart for ethnicity distribution"""
    ethnicity = ['White', 'Middle Eastern', 'Black', 'East Asian', 
                'Indian', 'Latino Hispanic', 'Southeast Asian']
    percentages = [52.49, 18.04, 11.12, 9.36, 4.09, 3.48, 1.43]
    
    # Sort in descending order
    sorted_indices = np.argsort(percentages)[::-1]
    ethnicity = [ethnicity[i] for i in sorted_indices]
    percentages = [percentages[i] for i in sorted_indices]
    
    # Create figure with minimal margins
    plt.figure(figsize=(8, 6))
    
    # Create horizontal bars with vibrant colors
    bars = plt.barh(
        ethnicity, 
        percentages, 
        color=ETHNICITY_COLORS[:len(ethnicity)],
        height=0.7,
        edgecolor='white',
        linewidth=1
    )
    
    # Add value labels with proper spacing
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.7,  # Reduced padding
            bar.get_y() + bar.get_height()/2, 
            f'{percentages[i]}%', 
            va='center',
            ha='left',
            fontsize=11,
            fontweight='bold',
            color='#333333'
        )
    
    # Set axis labels with minimal padding
    plt.xlabel('Percentage (%)', fontsize=12, labelpad=8)
    plt.ylabel('Ethnic Group', fontsize=12, labelpad=8)
    
    # Set x-axis limits with minimal padding
    plt.xlim(0, max(percentages) * 1.15)
    
    # Remove spines to reduce visual clutter
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout(pad=0.5)
    plt.savefig('assets/ethnicity_distribution.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_expression_chart():
    """Create a vibrant bar chart for expression distribution"""
    expressions = ['Neutral', 'Sad', 'Happy', 'Angry', 'Fear', 'Surprise', 'Disgust']
    percentages = [39.39, 17.44, 17.42, 13.64, 10.08, 1.8, 0.23]
    confidence = [0.790, 0.684, 0.867, 0.717, 0.674, 0.716, 0.640]
    
    # Sort in descending order by percentage
    sorted_indices = np.argsort(percentages)[::-1]
    expressions = [expressions[i] for i in sorted_indices]
    percentages = [percentages[i] for i in sorted_indices]
    confidence = [confidence[i] for i in sorted_indices]
    
    # Create figure with minimal margins
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    # Create bars with vibrant colors
    bars = ax1.bar(
        expressions, 
        percentages, 
        color=EXPRESSION_COLORS[:len(expressions)],
        width=0.7,
        edgecolor='white',
        linewidth=1
    )
    
    # Add percentage labels on top of bars with minimal padding
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.5,  # Minimal padding
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Configure primary y-axis for percentages
    ax1.set_ylabel('Percentage (%)', fontsize=12, labelpad=8)
    ax1.set_ylim(0, max(percentages) * 1.1)  # Reduced space
    
    # Create secondary y-axis for confidence scores
    ax2 = ax1.twinx()
    ax2.set_ylabel('Confidence', fontsize=12, labelpad=8, color='#FF5252')
    ax2.tick_params(axis='y', labelcolor='#FF5252')
    ax2.set_ylim(0.6, 0.9)  # Tighter y-limits
    
    # Plot confidence line with bright color
    line = ax2.plot(
        expressions, 
        confidence, 
        marker='o', 
        markersize=6,
        linewidth=2, 
        color='#FF5252',
        label='Confidence'
    )
    
    # Add confidence value labels with minimal space
    for i, (x, y) in enumerate(zip(expressions, confidence)):
        ax2.text(
            i, 
            y + 0.01,  # Minimal padding
            f'{y:.2f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#FF5252'
        )
    
    # Properly rotate and align x-axis labels
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right', fontsize=10)
    
    # Remove spines to reduce visual clutter
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Ensure proper spacing with minimal margins
    fig.tight_layout(pad=0.5)
    plt.savefig('assets/expression_distribution.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_head_pose_chart():
    """Create a vibrant bar chart for head pose statistics"""
    pose_types = ['Yaw', 'Pitch', 'Roll']
    means = [0.43, -4.27, -0.997]
    std_devs = [23.47, 9.25, 7.98]
    
    # Create figure with minimal margins
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Bar positions and width
    x = np.arange(len(pose_types))
    width = 0.35
    
    # Create bars with vibrant colors
    mean_bars = ax.bar(
        x - width/2, 
        means, 
        width,
        label='Mean', 
        color=POSE_COLORS[0],
        edgecolor='white',
        linewidth=1
    )
    
    std_bars = ax.bar(
        x + width/2, 
        std_devs, 
        width,
        label='Standard Deviation', 
        color=POSE_COLORS[1],
        edgecolor='white',
        linewidth=1
    )
    
    # Add zero line for reference
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add value labels with minimal space
    for bar in mean_bars:
        height = bar.get_height()
        ypos = height + 0.5 if height > 0 else height - 2
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f'{height:.2f}°',
            ha='center',
            va='bottom' if height > 0 else 'top',
            fontsize=9,
            fontweight='bold'
        )
    
    for bar in std_bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f'{height:.2f}°',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    # Configure axes with minimal space
    ax.set_xlabel('Pose Component', fontsize=12, labelpad=8)
    ax.set_ylabel('Degrees', fontsize=12, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(pose_types)
    
    # Set y-axis limits with minimal padding
    y_min = min(min(means) - 2, -6)
    y_max = max(max(std_devs) * 1.1, 26)
    ax.set_ylim(y_min, y_max)
    
    # Add legend with minimal space
    ax.legend(frameon=False, loc='upper right')
    
    # Remove spines to reduce visual clutter
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Ensure proper spacing with minimal margins
    fig.tight_layout(pad=0.5)
    plt.savefig('assets/head_pose_distribution.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_identity_distribution():
    """Create a vibrant distribution chart for identity clusters"""
    # Create simulated long-tail distribution based on dataset statistics
    num_identities = 28955
    min_images = 10
    max_images = 21983
    avg_images = 90
    
    # Define key points for accurate long-tail shape
    x = np.arange(1, 101)  # Percentiles
    y = np.zeros(100)
    
    # Define distribution landmarks
    y[0] = max_images    # Top 1%
    y[4] = 500           # Top 5%
    y[9] = 250           # Top 10%
    y[24] = 150          # Top 25%
    y[49] = avg_images   # Median (average)
    y[74] = 40           # Bottom 25%
    y[89] = 20           # Bottom 10%
    y[99] = min_images   # Bottom 1%
    
    # Create smooth interpolation between known points
    known_indices = [0, 4, 9, 24, 49, 74, 89, 99]
    known_values = [y[i] for i in known_indices]
    interp = PchipInterpolator(known_indices, known_values)
    for i in range(100):
        if i not in known_indices:
            y[i] = interp(i)
    
    # Create figure with minimal margins
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Plot distribution with vibrant color
    ax.semilogy(x, y, '-', linewidth=3, color='#00A4EF')
    ax.fill_between(x, y, alpha=0.3, color='#00A4EF')
    
    # Add key annotations with minimal space
    annotations = [
        (1, max_images, f'Max: {max_images}'),
        (50, avg_images, f'Avg: {avg_images}'),
        (99, min_images, f'Min: {min_images}')
    ]
    
    for x_pos, y_pos, text in annotations:
        ax.annotate(
            text,
            xy=(x_pos, y_pos),
            xytext=(x_pos + 3, y_pos * (1.2 if x_pos != 99 else 0.7)),
            arrowprops=dict(
                arrowstyle='->', 
                color='#333333',
                lw=1,
                connectionstyle='arc3,rad=0'
            ),
            fontsize=10,
            fontweight='bold'
        )
    
    # Configure axes with minimal padding
    ax.set_xlabel('Identity Percentile', fontsize=12, labelpad=8)
    ax.set_ylabel('Images per Identity (log)', fontsize=12, labelpad=8)
    
    # Set x-ticks at meaningful percentiles
    ax.set_xticks([1, 25, 50, 75, 99])
    ax.set_xticklabels(['1%', '25%', '50%', '75%', '99%'])
    
    # Remove spines to reduce visual clutter
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid for readability but with minimal visual impact
    ax.grid(True, linestyle='--', alpha=0.2)
    
    # Ensure proper spacing with minimal margins
    fig.tight_layout(pad=0.5)
    plt.savefig('assets/identity_distribution.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == "__main__":
    print("Generating vibrant visualizations with minimal white space...")
    create_gender_chart()
    create_ethnicity_chart()
    create_expression_chart()
    create_head_pose_chart()
    create_identity_distribution()
    print("All visualizations have been generated in the 'assets' directory.")