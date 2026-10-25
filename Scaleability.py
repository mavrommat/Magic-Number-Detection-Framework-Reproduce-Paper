import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import os

logging.basicConfig(level=logging.ERROR) 

from SyntheticDataGenerator import SyntheticData
from Magic_Number_Detection.get_magic_numbers import MagicNumberDetector


plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4, rc={
    "axes.titlesize": 14,      
    "axes.labelsize": 12,      
    "xtick.labelsize": 10,     
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 12
})

class ScalabilityBenchmark:
    def __init__(self, sizes, iterations):
        self.sizes = sizes
        self.iterations = iterations
        self.results = []

    def run_benchmark(self):
        print(f"Initiating Computational Scalability Benchmark...")
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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_scalability(df, save_path, column_type='single'):
    fig_width = 3.54 if column_type == 'single' else 7.48
    fig_height = 2.6 if column_type == 'single' else 4.5

    with plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,          
        'axes.labelsize': 10,     
        'xtick.labelsize': 8,    
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'text.color': 'black',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    }):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Plot Data
        ax.plot(df['num_samples'], df['mean_time_seconds'], 
                marker='o', markersize=4, linestyle='-', linewidth=1.5, color='#2c3e50', 
                label='Empirical Execution Time')
        lower_bound = np.maximum(df['mean_time_seconds'] - df['std_time_seconds'], 1e-4)
        upper_bound = df['mean_time_seconds'] + df['std_time_seconds']
        ax.fill_between(df['num_samples'], lower_bound, upper_bound, 
                        color='#2c3e50', alpha=0.2, label='1 Std Dev (CPU Variance)')
        
        # Theoretical O(n log n)
        n_base = df['num_samples'].iloc[0]
        t_base = df['mean_time_seconds'].iloc[0]

        x_theo = np.logspace(3, 8, 500)
        C_nlogn = t_base / (n_base * np.log(n_base))
        theoretical_nlogn = C_nlogn * (x_theo * np.log(x_theo))

        ax.plot(x_theo, theoretical_nlogn, 
                linestyle='--', linewidth=1.5, color='#e74c3c', 
                label=r'Theoretical $O(n \log n)$')
        
        # 3. Scales & Limits
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(left=9000, right=2e7) 
        ax.set_ylim(bottom=0.1, top=400) 
        
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        
        # 4. Labels & Grid (labelpad ensures distance from tick exponents)
        ax.set_xlabel("Dataset Size (Number of Rows)", fontweight='bold', labelpad=6)
        ax.set_ylabel("Execution Time (Seconds)", fontweight='bold', labelpad=6)
        
        ax.grid(True, which="major", ls="-", alpha=0.3)
        ax.grid(True, which="minor", ls=":", alpha=0.2)
        ax.legend(loc='upper left', frameon=False)
        
        fig.tight_layout()
        fig.savefig(save_path, dpi=1000, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        plt.close(fig)


def run_scalability_benchmark(DATASET_SIZES, ITERATIONS_PER_SIZE):
    
    data_dir = os.path.join("Synthetic Experiments", "Data")
    plots_dir = os.path.join("Synthetic Experiments", "Plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "scalability_metrics.csv")
    plot_path = os.path.join(plots_dir, "Scalability_Proof.pdf")

    # Check if the metrics file already exists
    if os.path.exists(csv_path):
        print(f"Found existing data at {csv_path}. Skipping benchmark simulation...")
        df_metrics = pd.read_csv(csv_path)
    else:
        # Run Benchmark if no data exists
        benchmark = ScalabilityBenchmark(sizes=DATASET_SIZES, iterations=ITERATIONS_PER_SIZE)
        df_metrics = benchmark.run_benchmark()
        
        # Export CSV
        df_metrics.to_csv(csv_path, index=False)
    
    # Export Plot
    plot_scalability(df_metrics, save_path=plot_path)
    
    print("\nScalability Summary:")
    print(df_metrics[['num_samples', 'mean_time_seconds', 'std_time_seconds']].to_string(index=False))

if __name__ == "__main__":

    DATASET_SIZES = [
                    10000,
                    50000, 
                    100000,
                    500000,
                    1000000,
                    5000000,
                    10000000
                    ]
    ITERATIONS_PER_SIZE = 3
    
    run_scalability_benchmark(DATASET_SIZES, ITERATIONS_PER_SIZE)