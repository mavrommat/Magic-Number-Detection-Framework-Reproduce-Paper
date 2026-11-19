import logging
from pathlib import Path
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import truststore
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from Magic_Number_Detection.get_magic_numbers import MagicNumberDetector

truststore.inject_into_ssl()
logging.getLogger("matplotlib").setLevel(logging.WARNING)
plt.set_loglevel("warning")

CATALOG_ID = "V/162/dr11sl"
VIZIER_SERVER = "vizier.cfa.harvard.edu"
ROW_LIMIT = -1

FIELD_NAME = "LAMOST DR11 (M67 field)"
FIELD_RA_DEG = 132.845833
FIELD_DEC_DEG = 11.813889
SEARCH_RADIUS_DEG = 1.5

PHOTOMETRIC_COLS = ["gmag", "rmag", "imag", "ymag", "zmag"]
SPECTROSCOPIC_COLS = ["Teff", "logg", "[Fe/H]"]
DETECT_COLS = SPECTROSCOPIC_COLS + PHOTOMETRIC_COLS

BASE_DIR = Path("Synthetic Experiments")
PLOT_DIR = BASE_DIR / "Plots"

OUTPUT_PLOT = PLOT_DIR / "LAMOST_DR11_M67_log_flux_z_i_vs_g_r.pdf"

MAGIC_KEYS = {
    "magic_number", "magic_numbers", "magic_value", "magic_values",
    "value", "values", "candidate", "candidates", "sentinel", "sentinels",
}


def extract_numbers(obj):
    if isinstance(obj, (int, float, np.integer, np.floating)) and not isinstance(obj, (bool, np.bool_)):
        yield float(obj)
    elif isinstance(obj, dict):
        matched = [v for k, v in obj.items() if str(k).lower().replace(" ", "_") in MAGIC_KEYS]
        if matched:
            for value in matched:
                yield from extract_numbers(value)
        else:
            for key in obj:
                try:
                    yield float(key)
                except (ValueError, TypeError):
                    pass
    elif isinstance(obj, (list, tuple, set, np.ndarray, pd.Series)):
        for value in obj:
            yield from extract_numbers(value)


def get_magic_values(results, column, atol=1e-6):
    if not isinstance(results, dict):
        return []

    unique = []
    for val in extract_numbers(results.get(column, [])):
        if np.isfinite(val) and not any(np.isclose(val, u_val, rtol=0, atol=atol) for u_val in unique):
            unique.append(val)
    return unique


def match_magic_values(series, magic_values, atol=1e-6):
    values = series.to_numpy(dtype=float)
    if not magic_values:
        return np.zeros(len(values), dtype=bool)
    finite = np.isfinite(values)
    matches = np.isclose(values[:, None], magic_values, rtol=0, atol=atol).any(axis=1)
    return finite & matches


def fetch_lamost_stars():
    vizier = Vizier(vizier_server=VIZIER_SERVER, row_limit=ROW_LIMIT, columns=["**"])
    center = SkyCoord(ra=FIELD_RA_DEG * u.deg, dec=FIELD_DEC_DEG * u.deg, frame="icrs")

    tables = vizier.query_region(center, radius=SEARCH_RADIUS_DEG * u.deg, catalog=CATALOG_ID)
    if not tables:
        raise RuntimeError(f"No tables returned for {CATALOG_ID}.")

    df = tables[0].to_pandas()
    required = ["Class", "FiberType", *DETECT_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Required catalogue columns are missing: {missing}")

    is_star = df["Class"].astype(str).str.strip().str.upper() == "STAR"
    is_obj = df["FiberType"].astype(str).str.strip() == "Obj"
    df = df.loc[is_star & is_obj].copy().reset_index(drop=True)

    for col in DETECT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.empty:
        raise RuntimeError("No stellar object-fibre sources remain in the selected field.")

    return df


def run_magic_detection(df):
    detector = MagicNumberDetector(
        sign_violation_threshold=3,
        gauss_threshold=0.1,
        overlap_threshold=0.1,
        plot_graphs=False,
        min_unique_values=20,
    )
    _, cleaned_results = detector.run_magic_detection(df[DETECT_COLS].copy())

    for col in DETECT_COLS:
        magic_vals = get_magic_values(cleaned_results, col)
        is_magic = match_magic_values(df[col], magic_vals)
        df[f"is_{col}_magic"] = is_magic

    phot_flag_cols = [f"is_{col}_magic" for col in PHOTOMETRIC_COLS]
    df["photometric_magic_count"] = df[phot_flag_cols].sum(axis=1)

    return df


def plot_flux_ratios(df, output_path, column_type="single"):
    fig_width = 3.54 if column_type == "single" else 7.09
    fig_height = 2.65 if column_type == "single" else 4.5

    x = df["log10_Fz_over_Fi"].to_numpy(dtype=float)
    y = df["log10_Fg_over_Fr"].to_numpy(dtype=float)

    finite_xy = np.isfinite(x) & np.isfinite(y)
    if not finite_xy.any():
        raise RuntimeError("No sources have finite z/i and g/r flux-ratio values.")

    x_magic = (df["is_zmag_magic"] | df["is_imag_magic"]).to_numpy(dtype=bool)
    y_magic = (df["is_gmag_magic"] | df["is_rmag_magic"]).to_numpy(dtype=bool)

    clean = finite_xy & ~x_magic & ~y_magic
    contaminated = finite_xy & (x_magic | y_magic)
    total_points = int(finite_xy.sum())
    contaminated_points = int(contaminated.sum())
    contaminated_pct = 100.0 * contaminated_points / total_points
    print(
        f"Contaminated plot points: {contaminated_points:,} / {total_points:,} "
        f"({contaminated_pct:.2f}%)"
    )

    aa_rcparams = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
    }

    layers = [
        (
            clean,
            dict(s=5, alpha=0.45, color="#3498db", marker="o", zorder=1),
            lambda count: f"Clean photometry (N={count:,})",
        ),
        (
            finite_xy & x_magic & ~y_magic,
            dict(s=14, alpha=0.80, color="#f39c12", marker="^", zorder=3),
            lambda count: f"Magic value in z/i only (N={count:,})",
        ),
        (
            finite_xy & ~x_magic & y_magic,
            dict(s=14, alpha=0.80, color="#2c3e50", marker="v", zorder=3),
            lambda count: f"Magic value in g/r only (N={count:,})",
        ),
        (
            finite_xy & x_magic & y_magic,
            dict(s=18, alpha=0.85, color="#e74c3c", marker="s", zorder=4),
            lambda count: f"Magic values in both (N={count:,})",
        ),
    ]

    with plt.rc_context(aa_rcparams):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        for mask, style, label_fn in layers:
            ax.scatter(
                x[mask],
                y[mask],
                edgecolors="none",
                rasterized=True,
                label=label_fn(int(mask.sum())),
                **style,
            )

        ax.set_xlim(-0.05, 0.15)
        ax.set_ylim(-0.5, 0.05)
        ax.minorticks_on()
        ax.set_title(f"{FIELD_NAME}: Photometric Flux-Ratio Contamination", pad=6)
        ax.set_xlabel(r"$\log_{10}(F_z/F_i) = -0.4(z - i)$", labelpad=4)
        ax.set_ylabel(r"$\log_{10}(F_g/F_r) = -0.4(g - r)$", labelpad=4)
        ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.25)
        ax.legend(
            loc="best",
            frameon=False,
            handletextpad=0.4,
            markerscale=1.1,
            borderaxespad=0.4,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        plt.close(fig)


def Bias_analysis():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_lamost_stars()
    df = run_magic_detection(df)
    df["g_minus_r"] = df["gmag"] - df["rmag"]
    df["log10_Fg_over_Fr"] = -0.4 * df["g_minus_r"]
    df["z_minus_i"] = df["zmag"] - df["imag"]
    df["log10_Fz_over_Fi"] = -0.4 * df["z_minus_i"]

    plot_flux_ratios(df, OUTPUT_PLOT, column_type="single")


if __name__ == "__main__":
    Bias_analysis()