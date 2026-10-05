import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def Hyperparameter_Sensitivity(json_filepath: str):
    print("Loading and parsing full grid search data...")

    output_dir = Path(json_filepath).parent / "Plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{json_filepath}'.")
        return

    # every run across the grid
    records = []
    for catalog_name, runs in data.items():
        if not isinstance(runs, list): continue
        
        for run in runs:
            p = run.get("Run_Parameters", {})
            sign = p.get("sign_violation_threshold")
            gauss = p.get("gauss_threshold")
            overlap = p.get("overlap_threshold")

            if sign is None or gauss is None or overlap is None: 
                continue

            for col_name, metrics in run.get("Column_Metrics", {}).items():
                if metrics.get("Processed by Algorithm") != "Yes": 
                    continue
                
                f1 = metrics.get("Metrics", {}).get("F1 Score", 0.0)

                records.append({
                    "Sign Violation": sign,
                    "Gauss Threshold": gauss,
                    "Overlap Threshold": overlap,
                    "F1 Score": f1
                })

    df = pd.DataFrame(records)
    if df.empty:
        print("Error: No valid grid search data found.")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",      
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "font.size": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",         
        "ytick.direction": "in",
        "xtick.top": True,               
        "ytick.right": True,
        "axes.linewidth": 1.2,           
    })

    params = [
        ("Sign Violation", "Sign Violation Threshold"),
        ("Gauss Threshold", "Gaussian Threshold"),
        ("Overlap Threshold", "Overlap Threshold")
    ]
    
    styles = [
        {"color": "#000000", "marker": "o"}, 
        {"color": "#b22222", "marker": "s"}, 
        {"color": "#000080", "marker": "^"}  
    ]

    for (col_name, title), style in zip(params, styles):
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.grid(True, which='major', linestyle=':', linewidth=0.8, color='gray', alpha=0.5, zorder=0)

        # mean and 95% CI
        grouped = df.groupby(col_name)["F1 Score"].agg(['mean', 'std', 'count'])
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
        grouped['ci'] = 1.96 * grouped['se'] 
        
        ax.errorbar(
            x=grouped.index,
            y=grouped['mean'],
            yerr=grouped['ci'],
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],     
            elinewidth=1.2,            
            capsize=4,                 
            capthick=1.2,
            markersize=7,
            markerfacecolor="white",   
            markeredgewidth=1.2,
            linestyle="--",            
            linewidth=1.0,
            alpha=0.9,
            zorder=3  
        )
        
        ax.set_title(title, pad=15)
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Mean Framework F1 Score") 
            
        padding = (grouped.index.max() - grouped.index.min()) * 0.1
        ax.set_xlim(grouped.index.min() - padding, grouped.index.max() + padding)

        plt.tight_layout()
        safe_name = col_name.replace(" ", "_")
        out_path = output_dir / f"Sensitivity_{safe_name}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Plot: {out_path}")

if __name__ == "__main__":
    Hyperparameter_Sensitivity("grid_search_results.json")