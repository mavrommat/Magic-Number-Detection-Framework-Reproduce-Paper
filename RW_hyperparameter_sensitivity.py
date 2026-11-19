import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42

AA_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "figure.dpi": 600,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _catalog_bootstrap_ci(
    frame: pd.DataFrame,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    
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


def Hyperparameter_Sensitivity(json_filepath: str, column_type: str = "single"):

    print("Loading and parsing full grid search data...")
    rng = np.random.default_rng(BOOTSTRAP_SEED)

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

                has_gt = bool(
                    metrics.get(
                        "Has_Ground_Truth",
                        len(metrics.get("Ground Truth Magic Numbers") or []) > 0,
                    )
                )

                records.append({
                    "Catalog": catalog_name,
                    "Sign Violation": float(sign),
                    "Gauss Threshold": float(gauss),
                    "Overlap Threshold": float(overlap),
                    "F1 Score": float(f1),
                    "Has Ground Truth": has_gt,
                })

    df = pd.DataFrame(records)
    if df.empty:
        print("Error: No valid grid search data found.")
        return

    param_cols = ["Sign Violation", "Gauss Threshold", "Overlap Threshold"]
    contam_df = df[df["Has Ground Truth"]].copy()
    if contam_df.empty:
        raise ValueError("No columns with ground-truth magic numbers found.")

    oof_df = _run_loco_cv(df, param_cols, max_clean_fpr=0.05)
    oof_contam = oof_df[oof_df["Has Ground Truth"]]
    oof_clean = oof_df[~oof_df["Has Ground Truth"]]

    fold_params = oof_df.groupby("Catalog")[param_cols].first()
    best_params = dict(zip(param_cols, fold_params.value_counts().index[0]))
    print(
        "Most common LOCO-selected configuration "
        f"(for visualization only): {best_params}"
    )
    print(
        f"LOCO-CV Out-of-Fold Contaminated F1 (Catalog-Macro): "
        f"{oof_contam.groupby('Catalog')['F1 Score'].mean().mean():.4f} | "
        f"Column-Macro: {oof_contam['F1 Score'].mean():.4f}"
    )
    if not oof_clean.empty:
        print(
            f"LOCO-CV Out-of-Fold Clean-Column FPR: "
            f"{(oof_clean['F1 Score'] < 1.0).mean() * 100:.2f}%"
        )

    plots_config = [
        {
            "col": "Sign Violation",
            "title": r"Sensitivity to Sign Violation Threshold ($k$)",
            "xlabel": r"Sign violation threshold ($k$)",
            "filename": "Sensitivity_Sign_Violation",
            "color": "#3498db",
            "marker": "o",
        },
        {
            "col": "Gauss Threshold",
            "title": r"Sensitivity to Gaussian Threshold ($G_{\mathrm{th}}$)",
            "xlabel": r"Gaussian threshold ($G_{\mathrm{th}}$)",
            "filename": "Sensitivity_Gauss_Threshold",
            "color": "#e74c3c",
            "marker": "s",
        },
        {
            "col": "Overlap Threshold",
            "title": r"Sensitivity to Overlap Threshold ($O_{\mathrm{th}}$)",
            "xlabel": r"Overlap threshold ($O_{\mathrm{th}}$)",
            "filename": "Sensitivity_Overlap_Threshold",
            "color": "#2c3e50",
            "marker": "^",
        },
    ]

    # Calculate shared y-axis limits from the bootstrap intervals
    global_y_min, global_y_max = 1.0, 0.0
    for p in plots_config:
        col = p["col"]
        other_cols = [c for c in param_cols if c != col]

        marg = _summarize_by_parameter(contam_df, col, rng)
        mask = (
            (contam_df[other_cols[0]] == best_params[other_cols[0]])
            & (contam_df[other_cols[1]] == best_params[other_cols[1]])
        )
        cond = _summarize_by_parameter(contam_df[mask], col, rng)

        low = min(marg["ci_lower"].min(), cond["ci_lower"].min())
        high = max(marg["ci_upper"].max(), cond["ci_upper"].max())
        global_y_min = min(global_y_min, low)
        global_y_max = max(global_y_max, high)

    y_span = global_y_max - global_y_min
    shared_ylim = (
        max(0.0, global_y_min - 0.08 * y_span),
        min(1.02, global_y_max + 0.12 * y_span),
    )

    fig_width = 3.54 if column_type == "single" else 7.09
    fig_height = 2.65 if column_type == "single" else 4.2

    with plt.rc_context(AA_RCPARAMS):
        for p in plots_config:
            col = p["col"]
            other_cols = [c for c in param_cols if c != col]

            marg = _summarize_by_parameter(contam_df, col, rng).sort_index()

            mask = (
                (contam_df[other_cols[0]] == best_params[other_cols[0]])
                & (contam_df[other_cols[1]] == best_params[other_cols[1]])
            )
            cond = _summarize_by_parameter(contam_df[mask], col, rng).sort_index()

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.25, zorder=0)

            # Contaminated column marginal mean and catalogue-bootstrap CI
            ax.plot(
                marg.index,
                marg["mean"],
                linestyle="--",
                linewidth=1.0,
                color="#7f8c8d",
                label="Contaminated-column marginal mean",
                zorder=2,
            )
            ax.fill_between(
                marg.index,
                marg["ci_lower"],
                marg["ci_upper"],
                color="#7f8c8d",
                alpha=0.18,
                edgecolor="none",
                label="Marginal 95% CI",
                zorder=1,
            )

            # LOCO-selected slice with asymmetric catalogue - bootstrap CI
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
                    capsize=2.0,
                    capthick=0.9,
                    markersize=3.8,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    linewidth=1.5,
                    label="Most common LOCO-selected slice (95% CI)",
                    zorder=4,
                )

            # the most common LOCO-selected parameter value
            opt_val = best_params[col]
            ax.axvline(
                opt_val,
                color=p["color"],
                linestyle=":",
                linewidth=1.0,
                alpha=0.9,
                label=f"LOCO-selected mode ({opt_val:g})",
                zorder=3,
            )
            ax.set_title(p["title"], pad=24)
            ax.set_xlabel(p["xlabel"], labelpad=4)
            ax.set_ylabel(r"Mean contaminated-column row-instance $F_1$", labelpad=4)
            ax.set_ylim(shared_ylim)
            ax.minorticks_on()
            x_span = marg.index.max() - marg.index.min()
            pad = x_span * 0.08 if x_span > 0 else 0.1
            ax.set_xlim(marg.index.min() - pad, marg.index.max() + pad)
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=2,
                frameon=False,
                columnspacing=0.9,
                handletextpad=0.4,
            )
            pdf_path = output_dir / f"{p['filename']}.pdf"
            fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
            plt.close(fig)

            print(f"Saved Separate Plot: {pdf_path}")

def _select_constrained_params(
    frame: pd.DataFrame,
    param_cols: list[str],
    max_clean_fpr: float = 0.05,
) -> tuple[float, float, float]:
    
    contam = frame[frame["Has Ground Truth"]]
    clean = frame[~frame["Has Ground Truth"]]

    contam_f1 = (
        contam.groupby([*param_cols, "Catalog"])["F1 Score"]
        .mean()
        .groupby(param_cols)
        .mean()
        .rename("contam_f1")
    )
    if contam_f1.empty:
        raise ValueError("No contaminated-column data is available for parameter selection.")

    if clean.empty:
        return contam_f1.idxmax()

    # On clean columns F1 < 1.0 means at least one false positive occurred (F1 = 0)
    clean_fpr = (
        clean.assign(false_alarm=clean["F1 Score"] < 1.0)
        .groupby([*param_cols, "Catalog"])["false_alarm"]
        .mean()
        .groupby(param_cols)
        .mean()
        .rename("clean_fpr")
    )

    grid = pd.concat([contam_f1, clean_fpr], axis=1)
    valid_grid = grid.dropna(subset=["contam_f1", "clean_fpr"])
    feasible = valid_grid[valid_grid["clean_fpr"] <= max_clean_fpr]
    if feasible.empty:
        if valid_grid.empty:
            raise ValueError("No clean-column false-alarm data is available for parameter selection.")
        min_clean_fpr = float(valid_grid["clean_fpr"].min())
        fallback = valid_grid[valid_grid["clean_fpr"] == min_clean_fpr]
        warnings.warn(
            f"No parameter configuration satisfies the clean-column false-alarm "
            f"limit of {max_clean_fpr:.1%}; using the lowest achievable rate "
            f"({min_clean_fpr:.1%}) and maximizing contaminated-column F1 among "
            "configurations at that rate.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback["contam_f1"].idxmax()

    return feasible["contam_f1"].idxmax()


def _run_loco_cv(
    df: pd.DataFrame,
    param_cols: list[str],
    max_clean_fpr: float = 0.05,
) -> pd.DataFrame:
    
    if df["Catalog"].nunique() < 2:
        raise ValueError("Leave-one-catalog-out evaluation requires at least two catalogs.")

    oof_slices = []
    for held_out in sorted(df["Catalog"].unique()):
        train_df = df[df["Catalog"] != held_out]
        best_sign, best_gauss, best_overlap = _select_constrained_params(
            train_df, param_cols, max_clean_fpr=max_clean_fpr
        )
        test_slice = df[
            (df["Catalog"] == held_out)
            & (df["Sign Violation"] == best_sign)
            & np.isclose(df["Gauss Threshold"], best_gauss)
            & np.isclose(df["Overlap Threshold"], best_overlap)
        ].copy()
        test_slice["Selected_Params"] = f"({best_sign:g}, {best_gauss:g}, {best_overlap:g})"
        oof_slices.append(test_slice)

    return pd.concat(oof_slices, ignore_index=True)

if __name__ == "__main__":
    Hyperparameter_Sensitivity("Real_World_Analysis/grid_search_results.json")