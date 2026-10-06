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

# 1. Configuration for Scalability Test
DATASET_SIZES = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]

ITERATIONS_PER_SIZE = 5 

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)

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

# Plotting the theoretical - empirical curve
def plot_scalability(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(df['num_samples'], df['mean_time_seconds'], 
            marker='o', linestyle='-', linewidth=2.5, color='#2c3e50', 
            label='Empirical Execution Time (Magic Number Framework)')
    
    ax.fill_between(df['num_samples'], 
                    df['mean_time_seconds'] - df['std_time_seconds'], 
                    df['mean_time_seconds'] + df['std_time_seconds'], 
                    color='#2c3e50', alpha=0.2, label='1 Std Dev (CPU Variance)')
    
    n_base = df['num_samples'].iloc[0]
    t_base = df['mean_time_seconds'].iloc[0]
    C = t_base / (n_base * np.log(n_base))
    theoretical_time = C * (df['num_samples'] * np.log(df['num_samples']))
    
    ax.plot(df['num_samples'], theoretical_time, 
            linestyle='--', linewidth=2, color='#e74c3c', 
            label=r'Theoretical $O(n \log n)$ Complexity')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Computational Scalability of the Detection Framework", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Dataset Size (Number of Rows)", fontsize=13)
    ax.set_ylabel("Execution Time (Seconds)", fontsize=13)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Publication figure saved to {save_path}")
    plt.close(fig) 


def run_scalability_benchmark():
    data_dir = os.path.join("Synthetic Experiments", "Data")
    plots_dir = os.path.join("Synthetic Experiments", "Plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Run Benchmark
    benchmark = ScalabilityBenchmark(sizes=DATASET_SIZES, iterations=ITERATIONS_PER_SIZE)
    df_metrics = benchmark.run_benchmark()
    
    # Export CSV
    csv_path = os.path.join(data_dir, "scalability_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    
    # Export Plot
    plot_path = os.path.join(plots_dir, "Scalability_Proof.png")
    plot_scalability(df_metrics, save_path=plot_path)
    
    print("\nScalability Summary:")
    print(df_metrics[['num_samples', 'mean_time_seconds', 'std_time_seconds']].to_string(index=False))

if __name__ == "__main__":
    run_scalability_benchmark()