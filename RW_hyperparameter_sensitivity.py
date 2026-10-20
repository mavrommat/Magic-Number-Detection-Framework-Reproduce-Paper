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

    records = []
    for catalog_name, runs in data.items():
        if not isinstance(runs, list): 
            continue
        
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
                    "Sign Violation": float(sign),
                    "Gauss Threshold": float(gauss),
                    "Overlap Threshold": float(overlap),
                    "F1 Score": float(f1)
                })

    df = pd.DataFrame(records)
    if df.empty:
        print("Error: No valid grid search data found.")
        return

    # Exact 90 mm (3.54 in) width, Arial/Helvetica 8 pt font, Type 42 vector PDF
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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

    # Find the global optimal parameter combination
    param_cols = ["Sign Violation", "Gauss Threshold", "Overlap Threshold"]
    grid_means = df.groupby(param_cols)["F1 Score"].mean()
    best_combo = grid_means.idxmax()
    best_params = dict(zip(param_cols, best_combo))
    print(f"Optimal Grid Configuration: {best_params} (Mean F1 = {grid_means.max():.4f})")

    plots_config = [
        {
            "col": "Sign Violation",
            "xlabel": r"Sign Violation Threshold ($k$)",
            "filename": "Sensitivity_Sign_Violation",
            "color": "#0072B2",  # Okabe-Ito Blue
            "marker": "o"
        },
        {
            "col": "Gauss Threshold",
            "xlabel": r"Gaussian Threshold ($G_{\mathrm{th}}$)",
            "filename": "Sensitivity_Gauss_Threshold",
            "color": "#D55E00",  
            "marker": "s"
        },
        {
            "col": "Overlap Threshold",
            "xlabel": r"Overlap Threshold ($O_{\mathrm{th}}$)",
            "filename": "Sensitivity_Overlap_Threshold",
            "color": "#009E73",  
            "marker": "^"
        }
    ]

    # Pre calculate  y-axis limits 
    global_y_min, global_y_max = 1.0, 0.0
    for p in plots_config:
        col = p["col"]
        other_cols = [c for c in param_cols if c != col]
        
        marg = df.groupby(col)["F1 Score"].agg(["mean", "std", "count"])
        marg["ci"] = 1.96 * (marg["std"] / np.sqrt(marg["count"].clip(lower=1)))
        
        mask = (df[other_cols[0]] == best_params[other_cols[0]]) & \
               (df[other_cols[1]] == best_params[other_cols[1]])
        cond = df[mask].groupby(col)["F1 Score"].agg(["mean", "std", "count"])
        cond["ci"] = 1.96 * (cond["std"] / np.sqrt(cond["count"].clip(lower=1)))

        low = min((marg["mean"] - marg["ci"]).min(), (cond["mean"] - cond["ci"]).min())
        high = max((marg["mean"] + marg["ci"]).max(), (cond["mean"] + cond["ci"]).max())
        global_y_min = min(global_y_min, low)
        global_y_max = max(global_y_max, high)

    y_Span = global_y_max - global_y_min
    shared_ylim = (
        max(0.0, global_y_min - 0.08 * y_Span),
        min(1.02, global_y_max + 0.12 * y_Span)
    )

    
    mm_to_inch = 1 / 25.4
    fig_width = 90 * mm_to_inch   # 3.54 in (Elsevier single-column width)
    fig_height = 70 * mm_to_inch  # 2.76 in

    for p in plots_config:
        col = p["col"]
        other_cols = [c for c in param_cols if c != col]

        # A. Global Marginal Stats
        marg = df.groupby(col)["F1 Score"].agg(["mean", "std", "count"]).sort_index()
        marg["ci"] = 1.96 * (marg["std"] / np.sqrt(marg["count"].clip(lower=1)))

        # B. Optimal Conditional Slice
        mask = (df[other_cols[0]] == best_params[other_cols[0]]) & \
               (df[other_cols[1]] == best_params[other_cols[1]])
        cond = df[mask].groupby(col)["F1 Score"].agg(["mean", "std", "count"]).sort_index()
        cond["ci"] = 1.96 * (cond["std"] / np.sqrt(cond["count"].clip(lower=1)))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.grid(True, linestyle="--", linewidth=0.5, color="#CCCCCC", alpha=0.7, zorder=0)

        # 1. Marginal Mean + 95% CI Shaded Band
        ax.plot(
            marg.index, marg["mean"],
            linestyle="--", linewidth=1.0, color="#555555",
            label="Marginal Mean", zorder=2
        )
        ax.fill_between(
            marg.index,
            marg["mean"] - marg["ci"],
            marg["mean"] + marg["ci"],
            color="#888888", alpha=0.18, edgecolor="none",
            label="Marginal 95% CI", zorder=1
        )

        # 2. Optimal Slice + Error Bars
        if not cond.empty:
            ax.errorbar(
                x=cond.index,
                y=cond["mean"],
                yerr=cond["ci"],
                fmt=p["marker"] + "-",
                color=p["color"],
                ecolor=p["color"],
                elinewidth=0.9,
                capsize=2.5,
                capthick=0.9,
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.1,
                linewidth=1.3,
                label=r"Optimal Slice ($\pm$95% CI)",
                zorder=4
            )

        # 3. Vertical line for optimal parameter value
        opt_val = best_params[col]
        ax.axvline(
            opt_val, color=p["color"], linestyle=":", linewidth=1.0, alpha=0.9,
            label=f"Optimal ({opt_val:g})", zorder=3
        )

        ax.set_xlabel(p["xlabel"], labelpad=4)
        ax.set_ylabel("Mean Framework $F_1$ Score", labelpad=4)
        ax.set_ylim(shared_ylim)

        # X-axis padding
        x_span = marg.index.max() - marg.index.min()
        pad = x_span * 0.08 if x_span > 0 else 0.1
        ax.set_xlim(marg.index.min() - pad, marg.index.max() + pad)

        # Compact 2x2 legend above the plot canvas so it never blocks data
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.4
        )

        # Save both Vector PDF and 600 DPI TIFF for each separate plot
        pdf_path = output_dir / f"{p['filename']}.pdf"
        tiff_path = output_dir / f"{p['filename']}.tiff"
        
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(tiff_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        
        print(f"Saved Separate Plot: {pdf_path} & {tiff_path}")


if __name__ == "__main__":
    Hyperparameter_Sensitivity("Real_World_Analysis/grid_search_results.json")