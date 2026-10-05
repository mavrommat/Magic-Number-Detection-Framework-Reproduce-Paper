import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os


def find_center_of_excellence(df, metric, params, tolerance=0.05, alpha=0.9):
    max_val = df[metric].max()
    top_tier = df[df[metric] >= max_val * (1.0 - tolerance)]
    if top_tier.empty:
        return df.loc[df[metric].idxmax()][params]
    return top_tier[params].mean()


def plot_corner_metric_continuous(df, file_name=None, metrics=['f1_score', 'recall', 'precision'], f1_threshold=0.9, dpi=300, save_fig=False):

    print("Generating Corner Plots with Continuous Metric and Consistent KDE Coloring...")

    cmap_name = 'turbo'
    kde_colors = {'good': '#1f77b4', 'bad': '#d62728'}

    cmap_dict = {
        'f1_score': cmap_name,
        'recall': cmap_name,
        'precision': cmap_name
    }
    
    # determine the expected count
    expected_mn = df['expected_count'].iloc[0] if 'expected_count' in df.columns else 2
    
    # Base parameters for both setups
    param_names = {
        'std': r'Sigma ($\sigma$)',
        'center_distance': 'Center Distance',
        'gap_distance': 'Pair Gap',
        'num_samples': 'Number of Samples',
        'gauss_threshold': 'Gauss Threshold',
        'overlap_threshold': 'Overlap Threshold',
        'contamination_rate': 'Contamination Ratio'  
    }
    
    # inject the gap distance if there are 2 magic numbers
    if expected_mn == 2:
        param_names['gap_distance'] = 'Pair Gap (g)'
            
    params = [p for p in param_names if p in df.columns and df[p].notna().any()]

    for p in params:
        if p in df.columns:
            try:
                df[p] = df[p].astype(float)
            except ValueError:
                print(f"Warning: Could not convert parameter '{p}' to float. Check data types.")

    # Diagonal Plot Function (KDE with F1 Threshold) 
    def plot_kde_for_param(ax, df, param, metric, threshold):

        # Apply the threshold classification metrics
        if metric in ['f1_score', 'precision', 'recall']:
            df_good = df[df[metric] >= threshold]
            df_bad = df[df[metric] < threshold]
        else:
            median_val = df[metric].median()
            df_good = df[df[metric] >= median_val]
            df_bad = df[df[metric] < median_val]

        if len(df_bad) > 1:
            sns.kdeplot(
                x=df_bad[param],
                ax=ax,
                color=kde_colors['bad'],
                fill=True,
                alpha=0.5,
                linewidth=1.5,
                label=rf'Low {metric.upper()}',
                bw_method=0.1
            )

        if len(df_good) > 1:
            sns.kdeplot(
                x=df_good[param],
                ax=ax,
                color=kde_colors['good'],
                fill=True,
                alpha=0.5,
                linewidth=1.5,
                label=rf'High {metric.upper()}',
                bw_method=0.1
            )

        ax.set_ylabel("Density")
        
        if len(df_bad) > 1 or len(df_good) > 1:
            ax.legend(fontsize=7, loc="lower right")
            
        ax.set_title(param_names.get(param, param), fontsize=10, fontweight='bold')

    #  Off-Diagonal Plot Function 
    def plot_scatter_pair(ax, df, x_param, y_param, metric, best_params, cmap, norm, f1_threshold):
        
        # define df_good 
        if metric in ['f1_score', 'precision', 'recall']:
            df_good = df[df[metric] >= f1_threshold]
        else:
            df_good = df[df[metric] >= df[metric].median()]

        # Fallback if the threshold left us with 0 points or all points
        if len(df_good) == len(df) or len(df_good) == 0:
            df_good = df[df[metric] >= df[metric].median()]

        scatter = ax.scatter(df[x_param], df[y_param],
                             c=df[metric], cmap=cmap, norm=norm,
                             alpha=0.7, s=10, edgecolors='black', linewidth=0.2)
        
        return scatter


    #  Axis Setup Function 
    def setup_axes(ax, i, j, n_params, params, param_names):
        if i == n_params - 1:
            ax.set_xlabel(param_names.get(params[j], params[j]), fontsize=10, fontweight='bold')
        else:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)

        if j == 0 and i != 0: 
            ax.set_ylabel(param_names.get(params[i], params[i]), fontsize=10, fontweight='bold')
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False)

        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    #  Main Loop 
    for metric in metrics:
        print(f"  Plotting {metric}...")
        
        best_params = find_center_of_excellence(df, metric, params, tolerance=0.05, alpha=0.9)
        n_params = len(params)
        fig, axes = plt.subplots(n_params, n_params, figsize=(18, 16))

        # Titles based on magic number count
        if expected_mn == 2:
            fig.suptitle(f'Visualizing Hyperparameter Interaction in Optimal Performance Regions\n' f'Two magic numbers\n' 
                         f'Metric: {metric.upper()}' , fontsize=18, fontweight='bold', y=0.98)
        elif expected_mn == 1:
            fig.suptitle(f'Visualizing Hyperparameter Interaction in Optimal Performance Regions\n' f'One magic number\n' 
                         f'Metric: {metric.upper()}' , fontsize=18, fontweight='bold', y=0.98)
        
        cmap_name_metric = cmap_dict.get(metric, cmap_name)
        cmap_obj = plt.colormaps[cmap_name_metric]
        
        if metric in ['f1_score', 'precision', 'recall']:
            norm = Normalize(vmin=0.0, vmax=1.0)
        else:
            norm = Normalize(vmin=df[metric].min(), vmax=df[metric].max())

        last_scatter = None 

        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                
                if i == j:  
                    plot_kde_for_param(ax, df, params[i], metric, f1_threshold)
                    
                    ax.set_ylabel('Density', fontsize=9, fontweight='bold')
                    ax.yaxis.set_label_position("left")
                    ax.tick_params(axis='y', labelleft=False, left=True) 
                    
                    if i == n_params - 1:
                        ax.set_xlabel(param_names.get(params[j], params[j]), fontsize=10, fontweight='bold')

                elif i > j: 
                    last_scatter = plot_scatter_pair(ax, df, params[j], params[i], metric, best_params, cmap_obj, norm, f1_threshold)
                    setup_axes(ax, i, j, n_params, params, param_names)
                else: 
                    ax.axis('off')

        if last_scatter:
            cbar_ax = fig.add_axes([0.94, 0.1, 0.02, 0.8])
            cbar = fig.colorbar(last_scatter, cax=cbar_ax)
            cbar.set_label(f'{metric.upper()} Value', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

        plt.subplots_adjust(left=0.07, right=0.93, bottom=0.07, top=0.9, wspace=0.1, hspace=0.1)

        if save_fig:
            # Change output directory path
            output_dir = os.path.join("Synthetic Experiments", "Plots")
            os.makedirs(output_dir, exist_ok=True)
            
            filepath = os.path.join(output_dir, f"CP_{file_name}_dpi{dpi}_{metric}.png")
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            
        plt.close(fig)

from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Standalone KDEs
# ---------------------------------------------
def plot_standalone_kdes(csv_path, f1_threshold=0.95):
    print("Loading data and generating standalone KDE distributions...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return
        
    df = df.dropna(subset=['f1_score'])
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # determine the expected count
    expected_mn = df['expected_count'].iloc[0] if 'expected_count' in df.columns else 2
    
    if expected_mn == 2:
        target_params = [
            'std', 'center_distance', 'gap_distance', 
            'gauss_threshold', 'overlap_threshold', 'num_samples',
            'contamination_rate'  
        ]
        param_labels = [
            r'Standard Deviation ($\sigma$)', 'Center Distance (D)', 'Pair Gap (g)', 
            'Gaussian Threshold', 'Overlap Threshold', 'Number of Samples',
            'Contamination Ratio'  
        ]
    else: 
        target_params = [
            'std', 'center_distance', 
            'gauss_threshold', 'overlap_threshold', 'num_samples',
            'contamination_rate'  
        ]
        param_labels = [
            r'Standard Deviation ($\sigma$)', 'Center Distance (D)', 
            'Gaussian Threshold', 'Overlap Threshold', 'Number of Samples',
            'Contamination Ratio'  
        ]

    # Single column width: 90 mm (3.54 in), standard 4:3 aspect ratio
    fig_width = 3.54 
    fig_height = 2.65 
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        # Increased base DPI for crisp raster output
        'figure.dpi': 600
    })

    def get_kde_data(data):
        if len(data) > 1:
            kde = gaussian_kde(data, bw_method=0.1)
            x_vals = np.linspace(data.min(), data.max(), 500)
            y_vals = kde.evaluate(x_vals)
            return x_vals, y_vals
        return None, None

    for param, label in zip(target_params, param_labels):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        
        df_good = df[df['f1_score'] >= f1_threshold]
        df_bad = df[df['f1_score'] < f1_threshold]

        x_bad, y_bad = get_kde_data(df_bad[param]) if not df_bad.empty else (None, None)
        x_good, y_good = get_kde_data(df_good[param]) if not df_good.empty else (None, None)

        max_y = 0
        if y_bad is not None: max_y = max(max_y, y_bad.max())
        if y_good is not None: max_y = max(max_y, y_good.max())

        if y_bad is not None:
            y_scaled_bad = y_bad / max_y
            ax.plot(x_bad, y_scaled_bad, color='#e74c3c', label=f'Low $F_1$ (< {f1_threshold})')
            ax.fill_between(x_bad, y_scaled_bad, color='#e74c3c', alpha=0.6)
            
        if y_good is not None:
            y_scaled_good = y_good / max_y
            ax.plot(
                x_good, 
                y_scaled_good, 
                color='#3498db', 
                label=rf'Optimal $F_1$ ($\geq$ {f1_threshold})'
            )
            ax.fill_between(x_good, y_scaled_good, color='#3498db', alpha=0.6)

        ax.set_ylim(0, 1.1)

        ax.set_title(f'Relative Density Profile: {label}')
        ax.set_xlabel(label)
        ax.set_ylabel('Relative Density')
        
        ax.legend(loc='upper right', frameon=False)
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

        output_dir = os.path.join("Synthetic Experiments", "Plots")
        os.makedirs(output_dir, exist_ok=True)
        
        # Outputting as high-resolution PNG
        save_filename = os.path.join(output_dir, f"{base_name}_{param}_kde_scaled.png")
        plt.savefig(save_filename, 
                    format='png',
                    dpi=600, # 600 DPI ensures line plots remain sharp in print 
                    bbox_inches='tight',
                    pad_inches=0.02)
        
        plt.close(fig)
# ---------------------------------------------
# Script Execution
# ---------------------------------------------
if __name__ == "__main__":
    output_filename = "Synthetic Experiments/Data/2_magic_numbers_results.csv" 
    
    try:
        df_export = pd.read_csv(output_filename)
        print(f"Successfully loaded {len(df_export)} runs from {output_filename}")
        """
        plot_corner_metric_continuous(
                    df=df_export, 
                    file_name=f"Monte_Carlo_Grid_{df_export['expected_count'].iloc[0]}_MN", 
                    metrics=['f1_score', "recall", "precision"],
                    f1_threshold=0.95, 
                    dpi=300, 
                    save_fig=True
                )
        """
        
       
        plot_standalone_kdes(csv_path=output_filename)
        
    except FileNotFoundError:
        print(f"Error: '{output_filename}' not found. Make sure the CSV has been exported.")