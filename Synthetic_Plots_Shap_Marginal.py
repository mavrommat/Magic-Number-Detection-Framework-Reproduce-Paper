import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import shap
import warnings
import os

warnings.filterwarnings("ignore")
F1_SUCCESS_THRESHOLD = 0.95
# 1. Set the base style and general scale FIRST
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4) 

# 2. Force specific font sizes globally using matplotlib's rcParams
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    
    # --- FONT SIZES (Increase these numbers as needed) ---
    "font.size": 12,          # Base font size
    "axes.labelsize": 22,     # X and Y axis labels (e.g., "Standard Deviation")
    "axes.titlesize": 22,     # Plot titles
    "xtick.labelsize": 16,    # X tick numbers
    "ytick.labelsize": 16,    # Y tick numbers
    "legend.fontsize": 12,
    "legend.title_fontsize": 14,
    
    # --- AESTHETICS ---
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def load_and_prep_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['f1_score'])
    expected_mn = df['expected_count'].iloc[0] if 'expected_count' in df.columns else 2
    
    df['std_bin'] = pd.cut(df['std'], bins=10, precision=1)
    df['dist_bin'] = pd.cut(df['center_distance'], bins=10, precision=1)
    
    if expected_mn == 2 and 'gap_distance' in df.columns:
        df['gap_bin'] = pd.cut(df['gap_distance'], bins=10, precision=1)
        
    df['samples_bin'] = pd.cut(df['num_samples'], bins=10)
    df['success'] = (df['f1_score'] >= F1_SUCCESS_THRESHOLD).astype(int)
    return df, expected_mn

def plot_shap_importance(df, ax, expected_mn):
    features = ['std', 'center_distance', 'gauss_threshold', 'overlap_threshold', 'num_samples', 'contamination_rate']
    if expected_mn == 2:
        features.append('gap_distance')
        
    features = [f for f in features if f in df.columns]
    
    rename_dict = {
        'std': r'$\sigma$ (Standard Deviation)',
        'center_distance': 'D (Center Distance)',
        'gauss_threshold': r'$G_{th}$ (Gauss Threshold)',
        'overlap_threshold': r'$O_{th}$ (Overlap Threshold)',
        'num_samples': 'N (Number of Samples)',
        'gap_distance': 'g (Pair Gap)',
        'contamination_rate': 'Contamination Ratio' 
    }
    
    X = df[features]
    y = df['f1_score']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    explainer = shap.TreeExplainer(rf)
    X_sample = X.sample(n=min(2000, len(X)), random_state=42) 
    shap_values = explainer.shap_values(X_sample)
    
    X_sample_renamed = X_sample.rename(columns=rename_dict)
    
    plt.sca(ax) 
    original_tight_layout = plt.tight_layout
    plt.tight_layout = lambda *args, **kwargs: None
    try:
        shap.summary_plot(shap_values, X_sample_renamed, show=False, plot_size=None, alpha=0.6)
    finally:
        plt.tight_layout = original_tight_layout
    
    ax.set_xlabel(r"$\Delta$ in $F_1$ Score (Impact on Model Output)")
    ax.set_title("SHAP Value Parameter Impact", pad=15, fontweight='bold')


def plot_marginal_response(df, ax):
    sns.lineplot(data=df, x='std', y='f1_score', ax=ax, color='#e74c3c', errorbar=('ci', 95), linewidth=2)
    ax.set_title("Marginal Response & Robustness", fontweight='bold')
    ax.set_xlabel(r"Standard Deviation ($\sigma$)")
    ax.set_ylabel(r"Mean $F_1$ Score (with 95% CI)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_num_samples_marginal(df, ax):
    # main marginal line
    sns.lineplot(
        data=df, x='num_samples', y='f1_score', ax=ax, 
        color='#2980b9', errorbar=('ci', 95), linewidth=2, label=r'Mean $F_1$'
    )
    
    # 1. Calculate average F1 in the range N of [50, 150]
    mask_50_150 = (df['num_samples'] >= 50) & (df['num_samples'] <= 150)
    avg_f1 = df.loc[mask_50_150, 'f1_score'].mean()
    target_f1 = 0.95 * avg_f1

    # horizontal line: <F1> (50 <= N <= 150)
    ax.axhline(
        y=avg_f1, color='#27ae60', linestyle='--', linewidth=1.2, alpha=0.85,
        label=rf'$\langle F_1 \rangle_{{30-200}} = {avg_f1:.3f}$'
    )

    # horizontal line: 0.95 * <F1>
    ax.axhline(
        y=target_f1, color='#e67e22', linestyle=':', linewidth=1.5, alpha=0.9,
        label=rf'$0.95 \times \langle F_1 \rangle = {target_f1:.3f}$'
    )

    # Find intersection: <F1> with 0.95 * <F1>
    mean_curve = df.groupby('num_samples')['f1_score'].mean().reset_index().sort_values('num_samples')
    x_vals = mean_curve['num_samples'].values
    y_vals = mean_curve['f1_score'].values

    intersect_n = None
    for i in range(len(y_vals) - 1):
        # Check for crossing 
        if (y_vals[i] - target_f1) * (y_vals[i + 1] - target_f1) <= 0 and y_vals[i] != y_vals[i + 1]:
            # find N: linear interpolation 
            intersect_n = x_vals[i] + (target_f1 - y_vals[i]) * (x_vals[i + 1] - x_vals[i]) / (y_vals[i + 1] - y_vals[i])
            break

    # Mark intersection
    if intersect_n is not None:
        ax.axvline(x=intersect_n, color='#e67e22', linestyle='-.', linewidth=1.0, alpha=0.7)
        ax.plot(
            intersect_n, target_f1, marker='o', color='#d35400', markersize=6, zorder=5,
            label=rf'Intersection ($N \approx {intersect_n:.1f}$)'
        )
        ax.annotate(
            rf'$N \approx {intersect_n:.1f}$',
            xy=(intersect_n, target_f1),
            xytext=(intersect_n + (x_vals.max() - x_vals.min()) * 0.04, target_f1 - 0.1),
            arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.0),
            fontsize=8.5, fontweight='bold', color='#d35400'
        )

    ax.set_title("Marginal Response: Sample Size (N)", fontweight='bold')
    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel(r"Mean $F_1$ Score (with 95% CI)")
    ax.set_xlim(0, 200)
    ax.set_ylim(-0.05, 0.6)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)



def generate_all_plots():
    target_files = [
        os.path.join("Synthetic Experiments", "Data", "1_magic_numbers_results.csv"),
        os.path.join("Synthetic Experiments", "Data", "2_magic_numbers_results.csv")
    ]
    
    output_dir = os.path.join("Synthetic Experiments", "Plots")
    os.makedirs(output_dir, exist_ok=True)

    for csv_file in target_files:
        try:
            print(f"\nLoading data from {csv_file}...")
            df_results, exp_mn = load_and_prep_data(csv_file)
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            print(f"Generating and saving individual plots for {base_name}...")


            '''
            # Shap
            fig, ax = plt.subplots(figsize=(8, 5.0), constrained_layout=True)
            plot_shap_importance(df_results, ax, exp_mn)
            plt.savefig(os.path.join(output_dir, f"{base_name}_SHAP_Importance.pdf"), dpi=600, bbox_inches='tight')
            plt.close(fig)
            '''

            # Marginal Sigma
            fig, ax = plt.subplots(figsize=(8, 5.0), constrained_layout=True)
            plot_marginal_response(df_results, ax)
            plt.savefig(os.path.join(output_dir, f"{base_name}_Marginal_Sigma.pdf"), dpi=600, bbox_inches='tight')
            plt.close(fig)

            # Marginal Number of samples
            fig, ax = plt.subplots(figsize=(8, 5.0), constrained_layout=True)
            plot_num_samples_marginal(df_results, ax)
            plt.savefig(os.path.join(output_dir, f"{base_name}_Marginal_Samples.pdf"), dpi=600, bbox_inches='tight')
            plt.close(fig)

            print(f"All standalone figures successfully generated for {base_name}.")
            
        except FileNotFoundError:
            print(f"Error: Could not find {csv_file}.")