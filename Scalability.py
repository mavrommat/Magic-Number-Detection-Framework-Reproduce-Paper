import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import logging
import os

logging.basicConfig(level=logging.ERROR)

from SyntheticDataGenerator import SyntheticData
from Magic_Number_Detection.get_magic_numbers import MagicNumberDetector


class ScalabilityBenchmark:
    def __init__(self, sizes, iterations):
        self.sizes = sizes
        self.iterations = iterations
        self.results = []

    def run_benchmark(self):
        print("Initiating Computational Scalability Benchmark...")
        print(f"Testing {len(self.sizes)} dataset scales, {self.iterations} iterations each.\n")

        for n_rows in self.sizes:
            times_for_size = []

            for i in tqdm(range(self.iterations), desc=f"Testing {n_rows:,} rows"):
                synth = SyntheticData(
                    data_type="log_normal",
                    variables={"std": 1.0, "num_samples": n_rows}
                )
                synth.generate_data()
                synth.clipping_data(SIGMA_LIMIT=3.0)

                synth.Magic_Number_Ingestion([998, 999, 1000], [100, 100, 100])
                df_test = synth.create_dataframe()

                detector = MagicNumberDetector(
                    sign_violation_threshold=3,
                    gauss_threshold=0.1,
                    overlap_threshold=0.1,
                    plot_graphs=False,
                    min_unique_values=20
                )

                start_time = time.perf_counter()
                master_results, cleaned_results = detector.run_magic_detection(df_test)
                end_time = time.perf_counter()

                exec_time = end_time - start_time
                times_for_size.append(exec_time)

            self.results.append({
                "num_samples": n_rows,
                "mean_time_seconds": np.mean(times_for_size),
                "std_time_seconds": np.std(times_for_size),
                "min_time_seconds": np.min(times_for_size),
                "max_time_seconds": np.max(times_for_size)
            })

        print("\nBenchmarking complete.")
        return pd.DataFrame(self.results)


def plot_scalability(df, save_path, column_type='single'):
    fig_width = 3.54 if column_type == 'single' else 7.09
    fig_height = 2.65 if column_type == 'single' else 4.2

    aa_rcparams = {
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
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'text.color': 'black',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    }

    color_empirical = '#2c3e50'
    color_band = '#3498db'
    color_theo = '#e74c3c'

    with plt.rc_context(aa_rcparams):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Empirical Execution Time
        ax.plot(
            df['num_samples'],
            df['mean_time_seconds'],
            marker='o',
            markersize=3.5,
            linestyle='-',
            linewidth=1.5,
            color=color_empirical,
            label='Empirical execution time'
        )

        lower_bound = np.maximum(df['mean_time_seconds'] - df['std_time_seconds'], 1e-4)
        upper_bound = df['mean_time_seconds'] + df['std_time_seconds']
        ax.fill_between(
            df['num_samples'],
            lower_bound,
            upper_bound,
            color=color_band,
            alpha=0.25,
            linewidth=0,
            label=r'$\pm 1\sigma$ (CPU variance)'
        )

        # Theoretical O(n log n) scaling
        n_base = df['num_samples'].iloc[0]
        t_base = df['mean_time_seconds'].iloc[0]

        x_theo = np.logspace(3, 8, 500)
        c_nlogn = t_base / (n_base * np.log(n_base))
        theoretical_nlogn = c_nlogn * (x_theo * np.log(x_theo))

        ax.plot(
            x_theo,
            theoretical_nlogn,
            linestyle='--',
            linewidth=1.5,
            color=color_theo,
            label=r'Theoretical $\mathcal{O}(n\,\log n)$'
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(left=800, right=2e7)
        ax.set_ylim(bottom=0.01, top=400)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_title("Computational Scalability of Magic Number Detection", pad=6)
        ax.set_xlabel("Dataset size (number of rows)", labelpad=4)
        ax.set_ylabel("Execution time (s)", labelpad=4)
        ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.25)
        ax.legend(loc='upper left', frameon=False)
        fig.tight_layout()
        fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.03, facecolor='white')
        plt.close(fig)


def run_scalability_benchmark(DATASET_SIZES, ITERATIONS_PER_SIZE):
    data_dir = os.path.join("Synthetic Experiments", "Data")
    plots_dir = os.path.join("Synthetic Experiments", "Plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "scalability_metrics.csv")
    plot_path = os.path.join(plots_dir, "Scalability_Proof.pdf")

    if os.path.exists(csv_path):
        print(f"Found existing data at {csv_path}. Skipping benchmark simulation...")
        df_metrics = pd.read_csv(csv_path)
    else:
        benchmark = ScalabilityBenchmark(sizes=DATASET_SIZES, iterations=ITERATIONS_PER_SIZE)
        df_metrics = benchmark.run_benchmark()
        df_metrics.to_csv(csv_path, index=False)

    plot_scalability(df_metrics, save_path=plot_path)

    print("\nScalability Summary:")
    print(df_metrics[['num_samples', 'mean_time_seconds', 'std_time_seconds']].to_string(index=False))


if __name__ == "__main__":
    run_scalability_benchmark(
        DATASET_SIZES=[1000, 10000, 100000, 1000000, 10000000],
        ITERATIONS_PER_SIZE=3
    )