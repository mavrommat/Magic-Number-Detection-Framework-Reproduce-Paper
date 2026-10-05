import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_robustness_proof(json_filepath: str):
    print("Loading full grid search data for Robustness Proof...")
    
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Extract ALL Parameter Combinations
    flattened_data = []
    for catalog_name, runs in data.items():
        for run in runs:
            params = run.get("Run_Parameters", {})
            for col_name, metrics in run.get("Column_Metrics", {}).items():
                if metrics.get("Processed by Algorithm") == "Yes":
                    flattened_data.append({
                        "Sign_Threshold": str(params.get("sign_violation_threshold")),
                        "Gauss_Threshold": str(params.get("gauss_threshold")),
                        "Overlap_Threshold": str(params.get("overlap_threshold")),
                        "Alg_F1": metrics.get("Metrics", {}).get("F1 Score", 0.0)
                    })

    df = pd.DataFrame(flattened_data)
    print(f"Extracted {len(df)} total column evaluations across all configurations.\n")

    # 2. Setup the Visualization
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Helper function for Violin + Stripplot layering
    def plot_dense_metric(x_col, ax, title, color, order):
        # The Violin shows the density (cut=0 keeps it within actual data bounds)
        sns.violinplot(data=df, x=x_col, y="Alg_F1", ax=ax, color=color, 
                       order=order, inner=None, cut=0, alpha=0.4)
        
        # The Stripplot shows the actual individual runs
        sns.stripplot(data=df, x=x_col, y="Alg_F1", ax=ax, color="black", 
                      order=order, alpha=0.05, size=2.5, jitter=True)
        
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel(x_col.replace("_", " "), fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis='x', labelsize=11)

    # Panel A: Sign Violation Threshold
    plot_dense_metric("Sign_Threshold", axes[0], "A: Sensitivity to Sign Threshold", "skyblue", 
                      ['1', '2', '3', '4', '5'])
    axes[0].set_ylabel("Algorithm F1 Score", fontsize=13)

    # Panel B: Gauss Threshold
    plot_dense_metric("Gauss_Threshold", axes[1], "B: Sensitivity to Gauss Threshold", "lightgreen", 
                      [0.01, 0.025, 0.05, 0.075, 0.1])
    axes[1].set_ylabel("") 

    # Panel C: Overlap Threshold
    plot_dense_metric("Overlap_Threshold", axes[2], "C: Sensitivity to Overlap Threshold", "salmon", 
                      [0.1, 2.0, 4.0, 6.0, 8.0])
    axes[2].set_ylabel("") 

    # Final Formatting
    plt.suptitle("Algorithm Robustness Across Hyperparameter Grid", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("Hyperparameter_Robustness_Proof.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: Hyperparameter_Robustness_Proof.png")
    plt.show()

# --- Execution ---
generate_robustness_proof("grid_search_results.json")