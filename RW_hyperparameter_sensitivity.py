import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42


def _catalog_bootstrap_ci(
    frame: pd.DataFrame,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return a percentile CI for the column-weighted mean, resampling catalogs."""
    catalog_stats = frame.groupby("Catalog")["F1 Score"].agg(["sum", "count"])
    if len(catalog_stats) < 2:
        raise ValueError(
            "Catalog-bootstrap confidence intervals require at least two catalogs."
        )
    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")

    if rng is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
    sums = catalog_stats["sum"].to_numpy(dtype=float)
    counts = catalog_stats["count"].to_numpy(dtype=float)
    sampled_catalogs = rng.integers(
        0,
        len(catalog_stats),
        size=(n_resamples, len(catalog_stats)),
    )
    bootstrap_means = sums[sampled_catalogs].sum(axis=1) / counts[sampled_catalogs].sum(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def _summarize_by_parameter(
    frame: pd.DataFrame,
    parameter: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"No data available to summarize for {parameter}.")

    rows = []
    for value, group in frame.groupby(parameter, sort=True):
        lower, upper = _catalog_bootstrap_ci(group, rng=rng)
        rows.append(
            {
                parameter: value,
                "mean": group["F1 Score"].mean(),
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    return pd.DataFrame(rows).set_index(parameter)


def Hyperparameter_Sensitivity(json_filepath: str):
    """Plot row-instance F1 sensitivity with catalog-cluster bootstrap intervals."""
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
                row_metrics = metrics.get("Row-Instance Metrics")
                if not isinstance(row_metrics, dict):
                    raise KeyError(
                        f"Missing 'Row-Instance Metrics' for "
                        f"{catalog_name}/{col_name}; regenerate the results."
                    )

                f1 = float(row_metrics["F1 Score"])

                records.append({
                    "Catalog": catalog_name,
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
    print(
        f"Pointwise 95% percentile confidence intervals use "
        f"{BOOTSTRAP_RESAMPLES:,} catalog-cluster bootstrap resamples across "
        f"{df['Catalog'].nunique()} catalogs."
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)

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

    # Calculate shared y-axis limits from the bootstrap intervals.
    global_y_min, global_y_max = 1.0, 0.0
    for p in plots_config:
        col = p["col"]
        other_cols = [c for c in param_cols if c != col]
        
        marg = _summarize_by_parameter(df, col, rng)
        
        mask = (df[other_cols[0]] == best_params[other_cols[0]]) & \
               (df[other_cols[1]] == best_params[other_cols[1]])
        cond = _summarize_by_parameter(df[mask], col, rng)

        low = min(marg["ci_lower"].min(), cond["ci_lower"].min())
        high = max(marg["ci_upper"].max(), cond["ci_upper"].max())
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

        # A. Global marginal mean with catalog-cluster bootstrap CI
        marg = _summarize_by_parameter(df, col, rng).sort_index()

        # B. Optimal Conditional Slice
        mask = (df[other_cols[0]] == best_params[other_cols[0]]) & \
               (df[other_cols[1]] == best_params[other_cols[1]])
        cond = _summarize_by_parameter(df[mask], col, rng).sort_index()

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.grid(True, linestyle="--", linewidth=0.5, color="#CCCCCC", alpha=0.7, zorder=0)

        # 1. Marginal mean + catalog-bootstrap 95% confidence interval
        ax.plot(
            marg.index, marg["mean"],
            linestyle="--", linewidth=1.0, color="#555555",
            label="Marginal Mean", zorder=2
        )
        ax.fill_between(
            marg.index,
            marg["ci_lower"],
            marg["ci_upper"],
            color="#888888", alpha=0.18, edgecolor="none",
            label="Marginal catalog-bootstrap percentile 95% CI", zorder=1
        )

        # 2. Optimal slice + asymmetric catalog-bootstrap confidence intervals
        if not cond.empty:
            ax.errorbar(
                x=cond.index,
                y=cond["mean"],
                yerr=np.vstack(
                    (
                        cond["mean"] - cond["ci_lower"],
                        cond["ci_upper"] - cond["mean"],
                    )
                ),
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
                label="Optimal slice (catalog-bootstrap percentile 95% CI)",
                zorder=4
            )

        # 3. Vertical line for optimal parameter value
        opt_val = best_params[col]
        ax.axvline(
            opt_val, color=p["color"], linestyle=":", linewidth=1.0, alpha=0.9,
            label=f"Optimal ({opt_val:g})", zorder=3
        )

        ax.set_xlabel(p["xlabel"], labelpad=4)
        ax.set_ylabel("Mean row-instance Framework $F_1$ Score", labelpad=4)
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