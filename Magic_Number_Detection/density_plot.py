import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_data_density(data_arr, col_name, plot_graphs = True):

    if plot_graphs == True:
        # Professional settings
        plt.rcParams.update({
            'font.size': 14,
            'font.family': 'serif',
            'figure.figsize': (10, 6),
            'axes.linewidth': 1.5
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))

        density = gaussian_kde(data_arr)  # Estimate the density
        x_vals = np.linspace(min(data_arr), max(data_arr), 1000)  # Create a range of x values
        y_vals = density(x_vals)  # Calculate the density for each x value

        ax.plot(x_vals, y_vals, color='#A23B72', linewidth=3, label='Density')  # density curve

        ax.hist(data_arr, bins=100, density=True, alpha=0.5, color='#2E86AB', label='Histogram')

        # Professional labels and title
        ax.set_xlabel(col_name, fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title('Density-Based Distribution of Data', fontsize=16, fontweight='bold')

        # Legend and grid
        ax.legend(framealpha=0.9, edgecolor='black', loc='upper right')
        ax.grid(True, alpha=0.2, linestyle='-')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()