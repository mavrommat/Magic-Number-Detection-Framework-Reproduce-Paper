import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import shap

warnings.filterwarnings("ignore")

F1_SUCCESS_THRESHOLD = 0.95

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

COLOR_BLUE = "#3498db"
COLOR_RED = "#e74c3c"
COLOR_DARK = "#2c3e50"


def load_and_prep_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["f1_score"])
    expected_mn = df["expected_count"].iloc[0] if "expected_count" in df.columns else 2

    df["std_bin"] = pd.cut(df["std"], bins=10, precision=1)
    df["dist_bin"] = pd.cut(df["center_distance"], bins=10, precision=1)

    if expected_mn == 2 and "gap_distance" in df.columns:
        df["gap_bin"] = pd.cut(df["gap_distance"], bins=10, precision=1)

    df["samples_bin"] = pd.cut(df["num_samples"], bins=10)
    df["success"] = (df["f1_score"] >= F1_SUCCESS_THRESHOLD).astype(int)
    return df, expected_mn


def plot_shap_importance(df, ax, expected_mn):
    features = [
        "std",
        "center_distance",
        "gauss_threshold",
        "overlap_threshold",
        "num_samples",
        "contamination_rate",
    ]
    if expected_mn == 2:
        features.append("gap_distance")

    features = [f for f in features if f in df.columns]

    rename_dict = {
        "std": r"$\sigma$ (Std. Dev.)",
        "center_distance": "D (Center Dist.)",
        "gauss_threshold": r"$G_{\mathrm{th}}$ (Gauss Thresh.)",
        "overlap_threshold": r"$O_{\mathrm{th}}$ (Overlap Thresh.)",
        "num_samples": "N (Sample Size)",
        "gap_distance": "g (Pair Gap)",
        "contamination_rate": "Contamination Ratio",
    }

    X = df[features]
    y = df["f1_score"]

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
        shap.summary_plot(
            shap_values,
            X_sample_renamed,
            show=False,
            plot_size=None,
            alpha=0.6,
        )
    finally:
        plt.tight_layout = original_tight_layout

    # Re-enforce A&A typography after SHAP's internal formatting
    ax.set_xlabel(r"$\Delta F_1$ score (SHAP value)", fontsize=8, labelpad=4)
    ax.set_title("SHAP Parameter Impact on Detection Performance", fontsize=8, pad=6)
    ax.tick_params(axis="both", labelsize=7, direction="in")

    # Match SHAP colorbar font sizes
    fig = ax.get_figure()
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]
        cbar_ax.tick_params(labelsize=7)
        cbar_ax.set_ylabel(cbar_ax.get_ylabel(), fontsize=7)


def plot_marginal_response(df, ax):
    sns.lineplot(
        data=df,
        x="std",
        y="f1_score",
        ax=ax,
        color=COLOR_RED,
        errorbar=("ci", 95),
        linewidth=1.5,
    )
    ax.set_title(r"Marginal $F_1$ Response to Distribution Dispersion", pad=6)
    ax.set_xlabel(r"Standard deviation ($\sigma$)", labelpad=4)
    ax.set_ylabel(r"Mean $F_1$ score (95% CI)", labelpad=4)
    ax.set_ylim(-0.05, 1.05)
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.25)


def plot_num_samples_marginal(df, ax):
    sns.lineplot(
        data=df,
        x="num_samples",
        y="f1_score",
        ax=ax,
        color=COLOR_BLUE,
        errorbar=("ci", 95),
        linewidth=1.5,
        label=r"Mean $F_1$",
    )

    # Calculate average F1 in the range 30 <= N <= 200
    mask_30_200 = (df["num_samples"] >= 30) & (df["num_samples"] <= 200)
    avg_f1 = df.loc[mask_30_200, "f1_score"].mean()
    target_f1 = 0.95 * avg_f1

    ax.axhline(
        y=avg_f1,
        color=COLOR_DARK,
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        label=rf"$\langle F_1 \rangle_{{30-200}} = {avg_f1:.3f}$",
    )

    ax.axhline(
        y=target_f1,
        color=COLOR_RED,
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
        label=rf"$0.95 \times \langle F_1 \rangle = {target_f1:.3f}$",
    )

    # Find intersection with 0.95 * <F1>
    mean_curve = (
        df.groupby("num_samples")["f1_score"]
        .mean()
        .reset_index()
        .sort_values("num_samples")
    )
    x_vals = mean_curve["num_samples"].values
    y_vals = mean_curve["f1_score"].values

    intersect_n = None
    for i in range(len(y_vals) - 1):
        if (y_vals[i] - target_f1) * (y_vals[i + 1] - target_f1) <= 0 and y_vals[i] != y_vals[i + 1]:
            intersect_n = x_vals[i] + (target_f1 - y_vals[i]) * (x_vals[i + 1] - x_vals[i]) / (
                y_vals[i + 1] - y_vals[i]
            )
            break

    if intersect_n is not None:
        ax.axvline(x=intersect_n, color=COLOR_RED, linestyle="-.", linewidth=0.9, alpha=0.7)
        ax.plot(
            intersect_n,
            target_f1,
            marker="o",
            color=COLOR_RED,
            markersize=3.5,
            zorder=5,
            label=rf"Intersection ($N \approx {intersect_n:.1f}$)",
        )
        ax.annotate(
            rf"$N \approx {intersect_n:.1f}$",
            xy=(intersect_n, target_f1),
            xytext=(intersect_n + (x_vals.max() - x_vals.min()) * 0.05, target_f1 - 0.12),
            arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=0.8),
            fontsize=6.5,
            color=COLOR_RED,
        )
    ax.set_title("Marginal $F_1$ Response to Sample Size", pad=6)
    ax.set_xlabel("Number of samples ($N$)", labelpad=4)
    ax.set_ylabel(r"Mean $F_1$ score (95% CI)", labelpad=4)
    ax.set_xlim(0, 200)
    ax.set_ylim(-0.05, 0.6)
    ax.minorticks_on()
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4, borderaxespad=0.4)
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.25)


def generate_all_plots(column_type="single"):
    fig_width = 3.54 if column_type == "single" else 7.09
    fig_height = 2.65 if column_type == "single" else 4.2

    target_files = [
        os.path.join("Synthetic Experiments", "Data", "1_magic_numbers_results.csv"),
        os.path.join("Synthetic Experiments", "Data", "2_magic_numbers_results.csv"),
    ]

    output_dir = os.path.join("Synthetic Experiments", "Plots")
    os.makedirs(output_dir, exist_ok=True)

    with plt.rc_context(AA_RCPARAMS):
        for csv_file in target_files:
            try:
                print(f"\nLoading data from {csv_file}...")
                df_results, exp_mn = load_and_prep_data(csv_file)
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                print(f"Generating and saving individual plots for {base_name}...")

                # 1. SHAP Importance
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                plot_shap_importance(df_results, ax, exp_mn)
                fig.tight_layout()
                fig.savefig(
                    os.path.join(output_dir, f"{base_name}_SHAP_Importance.pdf"),
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0.03,
                    facecolor="white",
                )
                plt.close(fig)

                # Marginal Sigma
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                plot_marginal_response(df_results, ax)
                fig.tight_layout()
                fig.savefig(
                    os.path.join(output_dir, f"{base_name}_Marginal_Sigma.pdf"),
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0.03,
                    facecolor="white",
                )
                plt.close(fig)

                # Marginal Number of Samples
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                plot_num_samples_marginal(df_results, ax)
                fig.tight_layout()
                fig.savefig(
                    os.path.join(output_dir, f"{base_name}_Marginal_Samples.pdf"),
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0.03,
                    facecolor="white",
                )
                plt.close(fig)

                print(f"All standalone figures successfully generated for {base_name}.")

            except FileNotFoundError:
                print(f"Error: Could not find {csv_file}.")


if __name__ == "__main__":
    generate_all_plots(column_type="single")