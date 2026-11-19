import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_artificial_examples(dist_type='normal'):
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
        'lines.linewidth': 1.5,
        'figure.dpi': 600,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
    })

    np.random.seed(42)

    if dist_type == 'normal':
        base_data = np.random.normal(loc=0, scale=1, size=1000)
        x_limit = (-5, 13)
    elif dist_type == 'lognormal':
        base_data = np.random.lognormal(mean=0, sigma=1.0, size=1000)
        x_limit = (-13, 13)
    else:
        raise ValueError("dist_type must be 'normal' or 'lognormal'")

    placeholder_value = 10.0

    data_10 = np.concatenate([base_data, np.full(10, placeholder_value)])
    data_100 = np.concatenate([base_data, np.full(100, placeholder_value)])

    color_hist = '#3498db'  
    color_kde = '#e74c3c'   
    color_vline = '#2c3e50'  

    datasets = [
        (data_10, 10, 'a'),
        (data_100, 100, 'b')
    ]

    data_dir = os.path.join("Synthetic Experiments", "Plots")
    os.makedirs(data_dir, exist_ok=True)

    for data, added_n, file_suffix in datasets:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        ax.hist(
            data,
            bins=50,
            density=True,
            color=color_hist,
            alpha=0.25,
            edgecolor=color_hist,
            linewidth=0.6,
            histtype='stepfilled',
            label='Data histogram'
        )
        sns.kdeplot(
            data,
            color=color_kde,
            alpha=0.9,
            linewidth=1.5,
            ax=ax,
            label='KDE'
        )
        ax.axvline(
            x=placeholder_value,
            color=color_vline,
            linestyle='--',
            linewidth=1.2,
            alpha=0.85,
            label='Artificial outliers'
        )
        dist_label = 'Normal' if dist_type == 'normal' else 'Log-normal'
        ax.set_title(
            rf'{dist_label} Distribution with $N_{{\mathrm{{out}}}} = {added_n}$ Magic Numbers',
            pad=6
        )
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.set_xlim(x_limit)
        ax.set_ylim(bottom=0)
        ax.minorticks_on()
        ax.legend(loc='upper right', frameon=False, handlelength=1.8)
        filename = f'artificial_outliers_{file_suffix}.pdf'
        fig.savefig(os.path.join(data_dir, filename), format='pdf', bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    plot_artificial_examples()