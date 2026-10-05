import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    def __init__(self, json_filepath: str, sign_opt: int, gauss_opt: float, overlap_opt: float, avg_rows: int = 100000, plots_dir: str = "Plots", data_dir: str = "data"):
        self.filepath = json_filepath
        self.sign_opt = sign_opt
        self.gauss_opt = gauss_opt
        self.overlap_opt = overlap_opt
        self.avg_rows = avg_rows
        self.prefix = f"sign{sign_opt}_gauss{gauss_opt}_overlap{overlap_opt}"
        self.warned_catalogs = set()
        self.df = pd.DataFrame()

        # path for plots
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # path for csv and data
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def run_pipeline(self):
        print(f"\nLoading Configuration: Sign={self.sign_opt}, Gauss={self.gauss_opt}, Overlap={self.overlap_opt}")
        
        raw_data = self._load_json()
        if not raw_data: return None
        
        self.df = self._extract_data(raw_data)
        if self.df.empty:
            print("\nError: No data found for specific hyperparameters.")
            return None

        # aggregation
        metrics = self._calculate_metrics()
        
        self._print_terminal_report(metrics)
        self._generate_plots(metrics)
        self._export_data(metrics)
        
        return metrics

    def _load_json(self):
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"\nError: Could not find the file '{self.filepath}'.")
            return None

    def _get_num_samples(self, run, metrics, alg):
        keys = ["Num Samples", "num_samples", "N", "n_samples", "Total Rows", "total_rows", "Number of Rows", "number_of_rows"]
        for container in [alg, metrics, run]:
            if isinstance(container, dict):
                for key in keys:
                    if key in container and container[key] is not None:
                        try:
                            val = int(container[key])
                            if val > 0: return val
                        except (TypeError, ValueError):
                            pass
        return None

    def _extract_data(self, data):
        flattened = []
        for catalog_name, runs in data.items():
            if not isinstance(runs, list): continue

            for run in runs:
                p = run.get("Run_Parameters", {})
                if not (p.get("sign_violation_threshold") == self.sign_opt and 
                        p.get("gauss_threshold") == self.gauss_opt and 
                        p.get("overlap_threshold") == self.overlap_opt):
                    continue

                for col_name, metrics in run.get("Column_Metrics", {}).items():
                    if metrics.get("Processed by Algorithm") != "Yes": continue

                    alg = metrics.get("Metrics", {})
                    base = metrics.get("Baseline Analysis", {})

                    tp, fp, fn = len(alg.get("True Positives (TP)", [])), len(alg.get("False Positives (FP)", [])), len(alg.get("False Negatives (FN)", []))
                    
                    num_samples = self._get_num_samples(run, metrics, alg)
                    if num_samples is None:
                        num_samples = self.avg_rows
                        if catalog_name not in self.warned_catalogs:
                            warnings.warn(f"\nNo explicit sample count for '{catalog_name}'. Using {self.avg_rows}.")
                            self.warned_catalogs.add(catalog_name)

                    if tp + fp + fn > num_samples:
                        raise ValueError(f"Invalid confusion matrix for {catalog_name}/{col_name}: TP+FP+FN={tp+fp+fn}, N={num_samples}")

                    tn = num_samples - (tp + fp + fn)
                    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator != 0 else (1.0 if (fp + fn) == 0 else 0.0)
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                    flattened.append({
                        "Catalog": catalog_name,
                        "Column": col_name,
                        "Category": metrics.get("Category", "Unknown").replace("_", " "),
                        "Framework_Precision": alg.get("Precision", 0.0),
                        "Framework_Recall": alg.get("Recall", 0.0),
                        "Framework_F1": alg.get("F1 Score", 0.0),
                        "ZScore_F1": base.get("Z-Score", {}).get("Metrics", {}).get("F1 Score", 0.0),
                        "IQR_F1": base.get("IQR", {}).get("Metrics", {}).get("F1 Score", 0.0),
                        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "N": num_samples,
                        "MCC": mcc, "FPR": fpr, "Zero_FP": fp == 0,
                        "String_Detections": set(alg.get("Magic Numbers String Process", []) or []),
                        "Sign_Detections": set(alg.get("Magic Numbers Sign Violation", []) or []),
                        "Delta_Detections": set(alg.get("Magic Numbers Delta", []) or []),
                        "Total_Magic": set(alg.get("Total Magic", []) or []),
                    })
        return pd.DataFrame(flattened)

    def _calculate_metrics(self):
        df = self.df
        total_tp, total_fp, total_fn = df["TP"].sum(), df["FP"].sum(), df["FN"].sum()

        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

        # Baseline Winners
        df["Best_Baseline_F1"] = df[["ZScore_F1", "IQR_F1"]].max(axis=1)
        df["Baseline_Winner"] = np.select(
            [(df["ZScore_F1"] > df["IQR_F1"]) & (df["ZScore_F1"] > df["Framework_F1"]),
             (df["IQR_F1"] > df["ZScore_F1"]) & (df["IQR_F1"] > df["Framework_F1"])],
            ["Z-Score", "IQR"], default="Framework"
        )
        
        category_summary = df.groupby("Category").agg(
            Columns=("Category", "size"), Precision=("Framework_Precision", "mean"),
            Recall=("Framework_Recall", "mean"), F1=("Framework_F1", "mean"),
            MCC=("MCC", "mean"), FPR=("FPR", "mean"), Zero_FP_Rate=("Zero_FP", "mean")
        ).sort_values("F1", ascending=False)
        category_summary["Zero_FP_Rate"] *= 100

        return {
            "df": df,
            "category_summary": category_summary,
            "baseline_wins": df[df["Best_Baseline_F1"] > df["Framework_F1"]].copy(),
            "totals": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
            "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
            "macro": {
                "precision": df["Framework_Precision"].mean(), "recall": df["Framework_Recall"].mean(),
                "f1": df["Framework_F1"].mean(), "mcc": df["MCC"].mean(),
                "fpr": df["FPR"].mean(), "zero_fp_rate": df["Zero_FP"].mean() * 100
            },
            "baselines": {"zscore_f1": df["ZScore_F1"].mean(), "iqr_f1": df["IQR_F1"].mean()}
        }

    def _print_terminal_report(self, m):
        report = f"""
                {'=' * 70}
                FINAL OPTIMAL CONFIGURATION REPORT
                {'=' * 70}
                Configuration:
                Sign violation threshold : {self.sign_opt}
                Gaussian threshold       : {self.gauss_opt}
                Overlap threshold        : {self.overlap_opt}

                Dataset:
                Total columns evaluated  : {len(self.df)}
                Total catalogs           : {self.df['Catalog'].nunique()}
                Categories               : {self.df['Category'].nunique()}

                Framework Performance:
                Macro Precision          : {m['macro']['precision']:.4f}
                Macro Recall             : {m['macro']['recall']:.4f}
                Macro F1                 : {m['macro']['f1']:.4f}
                Micro Precision          : {m['micro']['precision']:.4f}
                Micro Recall             : {m['micro']['recall']:.4f}
                Micro F1                 : {m['micro']['f1']:.4f}

                Baseline Comparison:
                Z-Score Mean F1          : {m['baselines']['zscore_f1']:.4f}
                IQR Mean F1              : {m['baselines']['iqr_f1']:.4f}

                Aggregate Confusion Counts:
                Total TP                 : {m['totals']['tp']}
                Total FP                 : {m['totals']['fp']}
                Total FN                 : {m['totals']['fn']}

                Safety / Imbalance Metrics:
                Macro MCC                : {m['macro']['mcc']:.4f}
                Macro FPR                : {m['macro']['fpr']:.6f}
                Zero-FP Column Rate      : {m['macro']['zero_fp_rate']:.2f}%
                """
        print(report)

        # Low-Performance  Block
        low_perf = self.df[self.df["Framework_F1"] < 0.5]
        
        print(f"\n{'=' * 70}\nLOW-PERFORMANCE ANALYSIS\n{'=' * 70}")
        if not low_perf.empty:
            print(f"\nNumber of columns: {len(low_perf)}")
            print("\nBreakdown by Category:")
            print(low_perf["Category"].value_counts().to_string())
            print("\nAffected Columns:")
            print(low_perf[["Catalog", "Column", "Category", "Framework_F1", "FPR"]]
                  .sort_values("Framework_F1").to_string(index=False))
        else:
            print("\nNo columns with F1 < 0.5.")

        # 3. Baseline-Win Analysis Block
        baseline_wins = m["baseline_wins"]
        
        print(f"\n{'=' * 70}\nBASELINE-WIN ANALYSIS\n{'=' * 70}")
        print(f"\nColumns where a baseline outperforms the framework: {len(baseline_wins)}")
        
        if not baseline_wins.empty:
            print("\nWinning baseline:")
            print(baseline_wins["Baseline_Winner"].value_counts().to_string())
            print("\nBreakdown by Category:")
            print(baseline_wins["Category"].value_counts().to_string())
            print("\nWorst baseline-win cases:")
            
            baseline_wins["Framework_Baseline_Gap"] = (baseline_wins["Framework_F1"] - baseline_wins["Best_Baseline_F1"])
            print(baseline_wins[["Catalog", "Column", "Category", "Framework_F1", "ZScore_F1", 
                                 "IQR_F1", "Baseline_Winner", "Framework_Baseline_Gap"]]
                  .sort_values("Framework_Baseline_Gap").to_string(index=False))
        else:
            print("\nNo baseline outperforms the framework.")

    def _generate_plots(self, m):
        sns.set_theme(style="whitegrid", context="talk")
        
        # Global F1 Comparison
        plt.figure(figsize=(9, 7))
        f1_data = pd.DataFrame({
            "Method": ["Framework", "IQR", "Z-Score"],
            "Mean F1": [m['macro']['f1'], m['baselines']['iqr_f1'], m['baselines']['zscore_f1']]
        })
        ax1 = sns.barplot(data=f1_data, x="Method", y="Mean F1", palette=["#2ca02c", "#9e9e9e", "#9e9e9e"])
        
        for container in ax1.containers:
            ax1.bar_label(container, fmt="%.3f", padding=5, fontsize=14, fontweight="bold", color="#0d47a1")

        plt.title("Global F1 Score Comparison", pad=20, fontweight="bold")
        plt.ylim(0, 1.1)
        sns.despine()
        
        plot1_path = self.plots_dir / f"{self.prefix}_Global_F1.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Plot: {plot1_path}")

        # F1 Distribution
        distribution_long = pd.DataFrame({
            "Framework": self.df["Framework_F1"],
            "IQR": self.df["IQR_F1"],
            "Z-Score": self.df["ZScore_F1"],
        }).melt(var_name="Method", value_name="F1 Score")

        plt.figure(figsize=(10, 7))
        violin_colors = {
            "Framework": "#2ca02c", 
            "IQR": "#ff9800",       
            "Z-Score": "#2196f3",   
        }

        ax2 = sns.violinplot(
            data=distribution_long, x="Method", y="F1 Score",
            palette=violin_colors, cut=0, inner="stick"
        )

        means = [m['macro']['f1'], m['baselines']['iqr_f1'], m['baselines']['zscore_f1']]
        for i, mean_val in enumerate(means):
            ax2.text(i, 1.02, f"{mean_val:.3f}", ha="center", va="bottom", 
                     fontsize=14, fontweight="bold", color="#0d47a1")

        plt.title("F1 Score Distribution Across Evaluated Columns", pad=20, fontweight="bold")
        plt.ylabel("F1 Score")
        plt.xlabel("")
        plt.ylim(0, 1.1) 
        sns.despine()
        plt.tight_layout()

        plot2_path = self.plots_dir / f"{self.prefix}_F1_Distribution.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Plot: {plot2_path}")

        # Category Level F1
        agg_df = self.df.groupby("Category")[["Framework_F1", "ZScore_F1", "IQR_F1"]].mean().reset_index()

        melted_df = pd.melt(
            agg_df, id_vars="Category", value_vars=["Framework_F1", "ZScore_F1", "IQR_F1"],
            var_name="Detection_Method", value_name="Mean_F1_Score"
        )
        melted_df["Detection_Method"] = melted_df["Detection_Method"].replace({
            "Framework_F1": "Framework", "IQR_F1": "IQR", "ZScore_F1": "Z-Score"
        })

        plt.figure(figsize=(10, 8))
        ax3 = sns.barplot(
            data=melted_df, x="Mean_F1_Score", y="Category",
            hue="Detection_Method", palette=violin_colors
        )

        plt.title("F1 Score by Astrophysical Category", pad=20, fontweight="bold")
        plt.xlim(0, 1.15)
        plt.xlabel("Mean F1 Score", fontweight="bold")
        plt.ylabel("")
        plt.legend(title="Method", bbox_to_anchor=(1.01, 1), loc="upper left")

        for container in ax3.containers:
            ax3.bar_label(container, fmt="%.2f", padding=3, fontsize=10)

        sns.despine()
        plt.tight_layout()

        plot3_path = self.plots_dir / f"{self.prefix}_Category_Performance.png"
        plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Plot: {plot3_path}")

        # Macro vs Micro F1 
        total_n = self.df["N"].sum()
        zscore_micro_f1 = (self.df["ZScore_F1"] * self.df["N"]).sum() / total_n if total_n > 0 else 0.0
        iqr_micro_f1 = (self.df["IQR_F1"] * self.df["N"]).sum() / total_n if total_n > 0 else 0.0

        macro_micro_df = pd.DataFrame([
            {"Method": "Framework", "Metric": "Macro F1", "Score": m['macro']['f1']},
            {"Method": "Framework", "Metric": "Micro F1", "Score": m['micro']['f1']},
            {"Method": "IQR", "Metric": "Macro F1", "Score": m['baselines']['iqr_f1']},
            {"Method": "IQR", "Metric": "Micro F1", "Score": iqr_micro_f1},
            {"Method": "Z-Score", "Metric": "Macro F1", "Score": m['baselines']['zscore_f1']},
            {"Method": "Z-Score", "Metric": "Micro F1", "Score": zscore_micro_f1},
        ])

        plt.figure(figsize=(10, 6))
        macro_micro_colors = {"Macro F1": "#9e9e9e", "Micro F1": "#424242"}

        ax4 = sns.barplot(
            data=macro_micro_df, x="Method", y="Score",
            hue="Metric", palette=macro_micro_colors
        )

        plt.ylim(0, 1.1)
        plt.ylabel("F1 Score")
        plt.xlabel("")
        plt.title("Macro vs Micro F1 comparison across detection methods", pad=20, fontweight="bold")
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

        for container in ax4.containers:
            ax4.bar_label(container, fmt="%.3f", padding=3, fontsize=11, fontweight="bold")

        sns.despine()
        plt.tight_layout()

        plot4_path = self.plots_dir / f"{self.prefix}_Macro_vs_Micro_F1.png"
        plt.savefig(plot4_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Plot: {plot4_path}")

    def _export_data(self, m):
        df_export = self.df.copy()
        
        list_cols = ["String_Detections", "Sign_Detections", "Delta_Detections", "Total_Magic"]
        for col in list_cols:
            if col in df_export.columns:
                df_export[col] = df_export[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, (set, list, tuple)) else str(x))

        exports = {
            self.data_dir / f"{self.prefix}_Full_Evaluation.csv": df_export,
            self.data_dir / f"{self.prefix}_Category_Summary.csv": m["category_summary"],
            self.data_dir / f"{self.prefix}_Baseline_Wins.csv": m["baseline_wins"]
        }

        for path, data in exports.items():
            data.to_csv(path, index=(path.name.endswith("Summary.csv")))
            print(f"Saved Data: {path}")


if __name__ == "__main__":
    evaluator = ModelEvaluator(
        json_filepath="grid_search_results.json",
        sign_opt=3,
        gauss_opt=0.1,
        overlap_opt=0.1
    )
    results = evaluator.run_pipeline()