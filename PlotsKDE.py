import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Elsevier Publication Figure Settings
# ==========================================
fig_width = 3.54 
# Height halved since we are now exporting two separate images
fig_height = 2.75 

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,             
    'axes.labelsize': 9,        
    'axes.labelweight': 'bold', 
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,          
    'axes.linewidth': 0.5
})

# ==========================================
# Data Generation
# ==========================================
np.random.seed(42)

dist_type = 'lognormal' 

if dist_type == 'normal':
    base_data = np.random.normal(loc=0, scale=1, size=1000)
    x_limit = (-5, 13)
elif dist_type == 'lognormal':
    base_data = np.random.lognormal(mean=0, sigma=1.0, size=1000)
    x_limit = (-13, 13)

base_n = len(base_data)
placeholder_value = -10.0

outliers_10 = np.full(10, placeholder_value)
data_10 = np.concatenate([base_data, outliers_10])

outliers_100 = np.full(100, placeholder_value)
data_100 = np.concatenate([base_data, outliers_100])

# ==========================================
# Plotting & Exporting
# ==========================================
color_hist = '#4F94B4'  
color_kde = '#A33B76'   
color_vline = '#F29C07' 

# Map the datasets to the 'a' and 'b' filenames required by LaTeX
datasets = [
    (data_10, 10, 'a'), 
    (data_100, 100, 'b')
]

for data, added_n, file_suffix in datasets:
    # Create a fresh figure for each dataset
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), tight_layout=True)
    
    ax.set_facecolor('#EAEAF2')
    ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.7)
    
    # 1. Histogram
    sns.histplot(
        data, 
        bins=50, 
        stat='density', 
        color=color_hist, 
        edgecolor='white',
        linewidth=0.5,
        alpha=1.0,
        ax=ax
    )
    
    # 2. KDE
    sns.kdeplot(
        data, 
        color=color_kde, 
        linewidth=2.5, 
        ax=ax,
        label='Kernel Density Estimate'
    )
    
    # 3. Artificial Outliers Line
    ax.axvline(
        x=placeholder_value, 
        color=color_vline, 
        linestyle='--', 
        linewidth=2.5, 
        label='Artificial outliers'
    )
    
    # Formatting
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(x_limit)
    
    
    # Legend moved to upper left
    legend = ax.legend(
        loc='upper left', 
        frameon=True, 
        facecolor='white', 
        edgecolor='gray',
        fancybox=False
    )
    legend.get_frame().set_linewidth(0.5)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Save each figure individually
    filename = f'artificial_outliers_lognormal{file_suffix}.png'
    plt.savefig(filename, format='png', bbox_inches='tight') 
    plt.show()