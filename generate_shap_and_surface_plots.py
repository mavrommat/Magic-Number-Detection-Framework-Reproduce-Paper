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
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.1)

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

def plot_response_surface_D_sigma(df, ax):
    pivot = df.pivot_table(index="std_bin", columns="dist_bin", values="f1_score", aggfunc="mean", observed=False)
    pivot.index = [f"{i.mid:.1f}" for i in pivot.index]
    pivot.columns = [f"{c.mid:.1f}" for c in pivot.columns]
    
    sns.heatmap(pivot, cmap="viridis", ax=ax, vmin=0, vmax=1, 
                cbar_kws={'label': 'Mean $F_1$ Score', 'pad': 0.02}, linewidths=0.5)
    ax.invert_yaxis()
    ax.set_title(r"Response Surface: $\sigma$ vs Distance", fontweight='bold')
    ax.set_xlabel("Center Distance (D) [Bin Midpoint]")
    ax.set_ylabel(r"Standard Deviation ($\sigma$) [Bin Midpoint]")
    ax.tick_params(axis='x', rotation=45)

def plot_marginal_response(df, ax):
    sns.lineplot(data=df, x='std', y='f1_score', ax=ax, color='#e74c3c', errorbar=('ci', 95), linewidth=2)
    ax.set_title("Marginal Response & Robustness", fontweight='bold')
    ax.set_xlabel(r"Standard Deviation ($\sigma$)")
    ax.set_ylabel(r"Mean $F_1$ Score (with 95% CI)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_num_samples_marginal(df, ax):
    sns.lineplot(data=df, x='num_samples', y='f1_score', ax=ax, color='#2980b9', errorbar=('ci', 95), linewidth=2)
    ax.set_title("Marginal Response: Sample Size (N)", fontweight='bold')
    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel(r"Mean $F_1$ Score (with 95% CI)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.95, color='black', linestyle=':', alpha=0.6, label='0.95 Threshold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_gap_vs_distance_surface(df, ax):
    if 'gap_distance' not in df.columns:
        return
    pivot = df.pivot_table(index="gap_bin", columns="dist_bin", values="f1_score", aggfunc="mean", observed=False)
    pivot.index = [f"{i.mid:.1f}" for i in pivot.index]
    pivot.columns = [f"{c.mid:.1f}" for c in pivot.columns]
    
    sns.heatmap(pivot, cmap="viridis", ax=ax, vmin=0, vmax=1, 
                cbar_kws={'label': 'Mean $F_1$ Score', 'pad': 0.02}, linewidths=0.5)
    ax.invert_yaxis()
    ax.set_title(r"Response Surface: Gap ($g$) vs Distance ($D$)", fontweight='bold')
    ax.set_xlabel("Center Distance (D) [Bin Midpoint]")
    ax.set_ylabel("Pair Gap (g) [Bin Midpoint]")
    ax.tick_params(axis='x', rotation=45)

def plot_gap_ratio_marginal(df, ax):
    if 'gap_distance' not in df.columns:
        return
    df_plot = df.copy()
    df_plot['gap_ratio'] = df_plot['gap_distance'] / (df_plot['center_distance'] + 1e-5)
    df_filtered = df_plot[df_plot['gap_ratio'] <= 5.0] 
    
    sns.lineplot(data=df_filtered, x='gap_ratio', y='f1_score', ax=ax, color='#8e44ad', errorbar=('ci', 95), linewidth=2)
    ax.set_title("Marginal Response: Gap-to-Distance Ratio", fontweight='bold')
    ax.set_xlabel(r"Cluster Isolation Ratio ($g / D$)")
    ax.set_ylabel(r"Mean $F_1$ Score (with 95% CI)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.95, color='black', linestyle=':', alpha=0.6, label='0.95 Threshold')
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.6)

def generate_all_plots():
    """Loops through generated Data folders and creates plots."""
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

            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            plot_shap_importance(df_results, ax, exp_mn)
            plt.savefig(os.path.join(output_dir, f"{base_name}_SHAP_Importance.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            plot_response_surface_D_sigma(df_results, ax)
            plt.savefig(os.path.join(output_dir, f"{base_name}_Response_Surface_Sigma.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            plot_marginal_response(df_results, ax)
            plt.savefig(os.path.join(output_dir, f"{base_name}_Marginal_Sigma.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            plot_num_samples_marginal(df_results, ax)
            plt.savefig(os.path.join(output_dir, f"{base_name}_Marginal_Samples.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

            if exp_mn == 2 and 'gap_distance' in df_results.columns:
                fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
                plot_gap_vs_distance_surface(df_results, ax)
                plt.savefig(os.path.join(output_dir, f"{base_name}_Response_Surface_Gap.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
                plot_gap_ratio_marginal(df_results, ax)
                plt.savefig(os.path.join(output_dir, f"{base_name}_Marginal_Gap_Ratio.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            print(f"All standalone figures successfully generated for {base_name}.")
            
        except FileNotFoundError:
            print(f"Error: Could not find {csv_file}.")