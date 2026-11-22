import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    def __init__(
        self,
        json_filepath: str = "Real_World_Analysis/grid_search_results.json",
        sign_opt: int = 3,
        gauss_opt: float = 0.1,
        overlap_opt: float = 0.1,
        plots_dir: str = "Plots",
        data_dir: str = "data",
    ):
        self.filepath = Path(json_filepath)
        # Fallback if user placed grid_search_results.json in current directory
        if not self.filepath.exists() and Path("grid_search_results.json").exists():
            self.filepath = Path("grid_search_results.json")

        self.sign_opt = sign_opt
        self.gauss_opt = gauss_opt
        self.overlap_opt = overlap_opt
        self.prefix = f"sign{sign_opt}_gauss{gauss_opt}_overlap{overlap_opt}"
        self.df = pd.DataFrame()
        self.baseline_names: list[str] = []

        # Path for plots
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Path for csv and data
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self):
        print(f"\nLoading Configuration: Sign={self.sign_opt}, Gauss={self.gauss_opt}, Overlap={self.overlap_opt}")

        raw_data = self._load_json()
        if not raw_data:
            return None

        self.df = self._extract_data(raw_data)
        if self.df.empty:
            print("\nError: No data found for specific hyperparameters.")
            return None

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

    @staticmethod
    def _is_processed(value) -> bool:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"yes", "true", "1"}:
            return True
        if normalized in {"no", "false", "0"}:
            return False
        raise ValueError(f"Invalid algorithm processing status: {value!r}")

    def _extract_data(self, data):
        flattened = []
        for catalog_name, runs in data.items():
            if not isinstance(runs, list):
                continue

            for run in runs:
                params = run.get("Run_Parameters", {})
                if not (
                    params.get("sign_violation_threshold") == self.sign_opt
                    and np.isclose(float(params.get("gauss_threshold", -1.0)), float(self.gauss_opt))
                    and np.isclose(float(params.get("overlap_threshold", -1.0)), float(self.overlap_opt))
                ):
                    continue

                for col_name, column_metrics in run.get("Column_Metrics", {}).items():
                    processing_status = column_metrics.get(
                        "Processed_By_Algorithm",
                        column_metrics.get("Processed by Algorithm"),
                    )
                    if processing_status is None:
                        raise KeyError(
                            f"Missing processing status for {catalog_name}/{col_name}"
                        )

                    alg = column_metrics.get("Row-Instance Metrics")
                    if not isinstance(alg, dict):
                        raise KeyError(
                            f"Missing 'Row-Instance Metrics' for {catalog_name}/{col_name}; "
                            "regenerate the results with the updated PaperWorkflowMagicNumbers.py."
                        )

                    # Support top-level Num_Samples (and fallback to Unique-Constant Metrics if needed)
                    num_samples = column_metrics.get("Num_Samples")
                    if num_samples is None:
                        num_samples = column_metrics.get("Unique-Constant Metrics", {}).get("Num_Samples")
                    if num_samples is None:
                        raise KeyError(
                            f"Missing 'Num_Samples' for {catalog_name}/{col_name}; "
                            "regenerate the results with the updated PaperWorkflowMagicNumbers.py."
                        )
                    num_samples = int(num_samples)

                    unique_vals = column_metrics.get("Unique_Values")
                    if unique_vals is None:
                        unique_vals = column_metrics.get("Unique-Constant Metrics", {}).get("Unique_Values", 0)

                    tp, fp, tn, fn = (
                        int(alg.get("TP", 0)),
                        int(alg.get("FP", 0)),
                        int(alg.get("TN", 0)),
                        int(alg.get("FN", 0)),
                    )
                    if tp + fp + tn + fn != num_samples:
                        raise ValueError(
                            f"Invalid row-instance confusion matrix for {catalog_name}/{col_name}: "
                            f"TP+FP+TN+FN={tp + fp + tn + fn}, N={num_samples}"
                        )

                    base = column_metrics.get("Baseline Analysis", {})
                    if not base:
                        raise KeyError(f"Missing 'Baseline Analysis' for {catalog_name}/{col_name}")

                    if not self.baseline_names:
                        self.baseline_names = list(base.keys())

                    denom = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
                    mcc = (
                        ((float(tp) * float(tn)) - (float(fp) * float(fn))) / np.sqrt(denom)
                        if denom > 0
                        else (1.0 if fp + fn == 0 else 0.0)
                    )

                    row_entry = {
                        "Catalog": catalog_name,
                        "Column": col_name,
                        "Category": column_metrics.get("Category", "Unknown").replace("_", " "),
                        "Num_Samples": num_samples,
                        "Unique_Values": int(unique_vals),
                        "Framework_Precision": alg.get("Precision", 0.0),
                        "Framework_Recall": alg.get("Recall", 0.0),
                        "Framework_F1": alg.get("F1 Score", 0.0),
                        "TP": tp,
                        "FP": fp,
                        "FN": fn,
                        "TN": tn,
                        "N": num_samples,
                        "MCC": mcc,
                        "FPR": fp / (fp + tn) if fp + tn > 0 else 0.0,
                        "Zero_FP": fp == 0,
                        "Has_Ground_Truth": bool(
                            column_metrics.get(
                                "Has_Ground_Truth",
                                len(column_metrics.get("Ground Truth Magic Numbers") or []) > 0,
                            )
                        ),
                        "Processed_By_Algorithm": self._is_processed(processing_status),
                        "String_Detections": set(column_metrics.get("Magic Numbers String Process", []) or []),
                        "Sign_Detections": set(column_metrics.get("Magic Numbers Sign Violation", []) or []),
                        "Delta_Detections": set(column_metrics.get("Magic Numbers Delta", []) or []),
                        "Total_Magic": set(column_metrics.get("Total Magic", []) or []),
                    }

                    for b_name in self.baseline_names:
                        b_row = base.get(b_name, {}).get("Row-Instance Metrics", {})
                        if not b_row:
                            raise KeyError(f"Missing {b_name} Row-Instance Metrics for {catalog_name}/{col_name}")
                        b_counts = sum(int(b_row.get(k, 0)) for k in ("TP", "FP", "TN", "FN"))
                        if b_counts != num_samples:
                            raise ValueError(f"Invalid {b_name} confusion matrix for {catalog_name}/{col_name}")

                        row_entry[f"{b_name}_F1"] = b_row.get("F1 Score", 0.0)
                        row_entry[f"{b_name}_TP"] = int(b_row.get("TP", 0))
                        row_entry[f"{b_name}_FP"] = int(b_row.get("FP", 0))
                        row_entry[f"{b_name}_FN"] = int(b_row.get("FN", 0))

                    flattened.append(row_entry)
        return pd.DataFrame(flattened)

    @staticmethod
    def _pooled_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        if tp == 0 and fp == 0 and fn == 0:
            return 1.0, 1.0, 1.0

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    @classmethod
    def _performance_summary(cls, df):
        if df.empty:
            return {
                "columns": 0,
                "macro_precision": np.nan,
                "macro_recall": np.nan,
                "macro_f1": np.nan,
                "micro_precision": np.nan,
                "micro_recall": np.nan,
                "micro_f1": np.nan,
            }

        tp, fp, fn = int(df["TP"].sum()), int(df["FP"].sum()), int(df["FN"].sum())
        micro_precision, micro_recall, micro_f1 = cls._pooled_f1(tp, fp, fn)
        return {
            "columns": len(df),
            "macro_precision": float(df["Framework_Precision"].mean()),
            "macro_recall": float(df["Framework_Recall"].mean()),
            "macro_f1": float(df["Framework_F1"].mean()),
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "micro_f1": float(micro_f1),
        }

    @classmethod
    def _build_processing_summary(cls, df):
        rows = []
        groups = [("Overall", "All", df)]
        groups.extend(("Category", str(name), group) for name, group in df.groupby("Category"))
        groups.extend(("Catalog", str(name), group) for name, group in df.groupby("Catalog"))

        for grouping, name, group in groups:
            processed = group[group["Processed_By_Algorithm"]]
            bypassed = group[~group["Processed_By_Algorithm"]]
            ground_truth = group[group["Has_Ground_Truth"]]
            processed_ground_truth = processed[processed["Has_Ground_Truth"]]
            bypassed_ground_truth = bypassed[bypassed["Has_Ground_Truth"]]

            row = {
                "Grouping": grouping,
                "Group": name,
                "Columns": len(group),
                "Processed_Columns": len(processed),
                "Bypassed_Columns": len(bypassed),
                "Processing_Coverage_Pct": len(processed) / len(group) * 100 if len(group) else np.nan,
                "Ground_Truth_Columns": len(ground_truth),
                "Processed_Ground_Truth_Columns": len(processed_ground_truth),
                "Bypassed_Ground_Truth_Columns": len(bypassed_ground_truth),
                "Ground_Truth_Processing_Coverage_Pct": (
                    len(processed_ground_truth) / len(ground_truth) * 100 if len(ground_truth) else np.nan
                ),
            }
            for status, subset in (
                ("Overall", group),
                ("Processed", processed),
                ("Bypassed", bypassed),
            ):
                for metric, value in cls._performance_summary(subset).items():
                    if metric != "columns":
                        row[f"{status}_{metric}"] = value
            rows.append(row)

        return pd.DataFrame(rows)

    def _calculate_metrics(self):
        df = self.df

        baselines_summary = {}
        f1_cols = [f"{b}_F1" for b in self.baseline_names]

        for b in self.baseline_names:
            b_tp, b_fp, b_fn = (
                int(df[f"{b}_TP"].sum()),
                int(df[f"{b}_FP"].sum()),
                int(df[f"{b}_FN"].sum()),
            )
            b_prec, b_rec, b_micro_f1 = self._pooled_f1(b_tp, b_fp, b_fn)
            baselines_summary[b] = {
                "macro_f1": float(df[f"{b}_F1"].mean()),
                "micro_f1": float(b_micro_f1),
                "micro_precision": float(b_prec),
                "micro_recall": float(b_rec),
                "totals": {"tp": b_tp, "fp": b_fp, "fn": b_fn},
            }

        df["Best_Baseline_F1"] = df[f1_cols].max(axis=1)
        best_baseline_idx = df[f1_cols].values.argmax(axis=1)
        best_baseline_names = np.array(self.baseline_names)[best_baseline_idx]
        df["Baseline_Winner"] = np.where(
            df["Best_Baseline_F1"] > df["Framework_F1"], best_baseline_names, "Framework"
        )

        total_tp, total_fp, total_fn = int(df["TP"].sum()), int(df["FP"].sum()), int(df["FN"].sum())
        micro_prec, micro_rec, micro_f1 = self._pooled_f1(total_tp, total_fp, total_fn)

        contam_df = df[df["Has_Ground_Truth"]]
        clean_df = df[~df["Has_Ground_Truth"]]
        bypassed_contam = contam_df[~contam_df["Processed_By_Algorithm"]]

        c_tp, c_fp, c_fn = int(contam_df["TP"].sum()), int(contam_df["FP"].sum()), int(contam_df["FN"].sum())
        c_micro_prec, c_micro_rec, c_micro_f1 = self._pooled_f1(c_tp, c_fp, c_fn)

        stratified = {
            "contam_cols": len(contam_df),
            "contam_macro_prec": float(contam_df["Framework_Precision"].mean()),
            "contam_macro_rec": float(contam_df["Framework_Recall"].mean()),
            "contam_macro_f1": float(contam_df["Framework_F1"].mean()),
            "contam_catalog_macro_f1": float(contam_df.groupby("Catalog")["Framework_F1"].mean().mean()),
            "contam_micro_f1": float(c_micro_f1),
            "bypassed_contam_penalized": len(bypassed_contam),
            "clean_cols": len(clean_df),
            "clean_col_false_alarm_incidence": (
                float((clean_df["FP"] > 0).mean() * 100.0) if not clean_df.empty else 0.0
            ),
            "clean_zero_fp_rate": float((clean_df["FP"] == 0).mean() * 100.0) if not clean_df.empty else 100.0,
            
        }

        category_summary = (
            df.groupby("Category")
            .agg(
                Columns=("Category", "size"),
                Precision=("Framework_Precision", "mean"),
                Recall=("Framework_Recall", "mean"),
                F1=("Framework_F1", "mean"),
                MCC=("MCC", "mean"),
                FPR=("FPR", "mean"),
                Zero_FP_Rate=("Zero_FP", "mean"),
            )
            .sort_values("F1", ascending=False)
        )

        category_summary["Zero_FP_Rate"] *= 100

        processing_summary = self._build_processing_summary(df)

        return {
            "df": df,
            "category_summary": category_summary,
            "processing_summary": processing_summary,
            "baseline_wins": df[df["Best_Baseline_F1"] > df["Framework_F1"]].copy(),
            "totals": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
            "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
            "macro": {
                "precision": df["Framework_Precision"].mean(),
                "recall": df["Framework_Recall"].mean(),
                "f1": df["Framework_F1"].mean(),
                "mcc": df["MCC"].mean(),
                "fpr": df["FPR"].mean(),
                "zero_fp_rate": df["Zero_FP"].mean() * 100,
            },
            "stratified": stratified,
            "baselines": baselines_summary,
        }

    def _print_terminal_report(self, m):
        baseline_lines = "\n".join(
            f"                {b:<24} : Macro F1 = {stats['macro_f1']:.4f} | Micro F1 = {stats['micro_f1']:.4f}"
            for b, stats in m["baselines"].items()
        )

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

                Algorithm Processing:
                Processed columns        : {m['processing_summary'].iloc[0]['Processed_Columns']} ({m['processing_summary'].iloc[0]['Processing_Coverage_Pct']:.2f}% coverage)
                Bypassed columns         : {m['processing_summary'].iloc[0]['Bypassed_Columns']}
                Ground-truth columns     : {m['processing_summary'].iloc[0]['Ground_Truth_Columns']}
                Ground-truth coverage    : {m['processing_summary'].iloc[0]['Ground_Truth_Processing_Coverage_Pct']:.2f}%

                Framework Performance:
                Macro Precision          : {m['macro']['precision']:.4f}
                Macro Recall             : {m['macro']['recall']:.4f}
                Macro F1                 : {m['macro']['f1']:.4f}
                Micro Precision          : {m['micro']['precision']:.4f}
                Micro Recall             : {m['micro']['recall']:.4f}
                Micro F1                 : {m['micro']['f1']:.4f}

                Baseline Comparison:
{baseline_lines}

                Aggregate Confusion Counts:
                Total TP                 : {m['totals']['tp']}
                Total FP                 : {m['totals']['fp']}
                Total FN                 : {m['totals']['fn']}

                Safety / Imbalance Metrics:
                Macro MCC                : {m['macro']['mcc']:.4f}
                Macro Row-Level FPR      : {m['macro']['fpr']:.6f}
                Zero-FP Column Rate      : {m['macro']['zero_fp_rate']:.2f}%

                Stratified Evaluation (Contaminated vs. Clean):
                Contaminated Columns     : {m['stratified']['contam_cols']} (Bypassed w/ F1=0: {m['stratified']['bypassed_contam_penalized']})
                Contam. Macro Precision  : {m['stratified']['contam_macro_prec']:.4f}
                Contam. Macro Recall     : {m['stratified']['contam_macro_rec']:.4f}
                Contam. Macro F1         : {m['stratified']['contam_macro_f1']:.4f}
                Contam. Catalog-Macro F1 : {m['stratified']['contam_catalog_macro_f1']:.4f}
                Contam. Micro F1         : {m['stratified']['contam_micro_f1']:.4f}
                Clean Columns            : {m['stratified']['clean_cols']}
                Clean-Column False-Alarm Incidence (%) : {m['stratified']['clean_col_false_alarm_incidence']:.2f}
                Clean Zero-FP Rate       : {m['stratified']['clean_zero_fp_rate']:.2f}%
                """
        print(report)

        print("\nPerformance by Processing Status:")
        status_columns = [
            "Grouping",
            "Group",
            "Columns",
            "Processed_Columns",
            "Processing_Coverage_Pct",
            "Overall_macro_f1",
            "Processed_macro_f1",
            "Bypassed_macro_f1",
            "Overall_micro_f1",
            "Processed_micro_f1",
            "Bypassed_micro_f1",
        ]
        print(
            m["processing_summary"][status_columns]
            .to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )

        # Low Performance
        low_perf = self.df[self.df["Framework_F1"] < 0.5]

        print(f"\n{'=' * 70}\nLOW-PERFORMANCE ANALYSIS\n{'=' * 70}")
        if not low_perf.empty:
            print(f"\nNumber of columns: {len(low_perf)}")
            print("\nBreakdown by Category:")
            print(low_perf["Category"].value_counts().to_string())
            print("\nAffected Columns:")
            print(
                low_perf[["Catalog", "Column", "Category", "Framework_F1", "FPR"]]
                .sort_values("Framework_F1")
                .to_string(index=False)
            )
        else:
            print("\nNo columns with F1 < 0.5.")

        # Baseline Win Analysis
        baseline_wins = m["baseline_wins"]

        print(f"\n{'=' * 70}\nBASELINE-WIN ANALYSIS\n{'=' * 70}")
        print(f"\nColumns where a baseline outperforms the framework: {len(baseline_wins)}")

        if not baseline_wins.empty:
            print("\nWinning baseline:")
            print(baseline_wins["Baseline_Winner"].value_counts().to_string())
            print("\nBreakdown by Category:")
            print(baseline_wins["Category"].value_counts().to_string())
            print("\nWorst baseline-win cases:")

            baseline_wins["Framework_Baseline_Gap"] = (
                baseline_wins["Framework_F1"] - baseline_wins["Best_Baseline_F1"]
            )
            f1_cols = [f"{b}_F1" for b in self.baseline_names]
            display_cols = [
                "Catalog",
                "Column",
                "Category",
                "Framework_F1",
                *f1_cols,
                "Baseline_Winner",
                "Framework_Baseline_Gap",
            ]
            print(
                baseline_wins[display_cols]
                .sort_values("Framework_Baseline_Gap")
                .to_string(index=False)
            )
        else:
            print("\nNo baseline outperforms the framework.")

    def _generate_plots(self, m, include_titles=False):
        SINGLE_COL_W = 3.54
        DOUBLE_COL_W = 7.48

        elsevier_rc = {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#cccccc",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }

        default_colors = ["#e74c3c", "#95a5a6", "#e67e22", "#3498db", "#9b59b6", "#1abc9c"]
        method_palette = {"Framework": "#2c3e50"}
        for idx, b_name in enumerate(self.baseline_names):
            method_palette[b_name] = default_colors[idx % len(default_colors)]

        f1_cols = [f"{b}_F1" for b in self.baseline_names]
        rename_map = {"Framework_F1": "Framework", **{f"{b}_F1": b for b in self.baseline_names}}

        sns.reset_defaults()
        with plt.rc_context(elsevier_rc):

            # 1. Category Level F1
            agg_df = (
                self.df.groupby("Category")[["Framework_F1", *f1_cols]]
                .mean()
                .sort_values("Framework_F1", ascending=False)
                .reset_index()
            )

            melted_df = pd.melt(
                agg_df,
                id_vars="Category",
                value_vars=["Framework_F1", *f1_cols],
                var_name="Detection_Method",
                value_name="Mean_F1_Score",
            )
            melted_df["Detection_Method"] = melted_df["Detection_Method"].replace(rename_map)

            num_cats = len(agg_df)
            num_methods = 1 + len(self.baseline_names)
            row_height = max(0.55, 0.14 * num_methods)
            fig3_height = max(4.2, num_cats * row_height)
            fig3, ax3 = plt.subplots(figsize=(DOUBLE_COL_W, fig3_height))
            ax3.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)

            sns.barplot(
                data=melted_df,
                x="Mean_F1_Score",
                y="Category",
                hue="Detection_Method",
                palette=method_palette,
                ax=ax3,
                zorder=3,
            )

            if include_titles:
                ax3.set_title("F1 Score by Astrophysical Category", pad=22, fontsize=12, fontweight="bold")

            ax3.set_xlim(0, 1.15)
            ax3.set_xlabel("Mean F1 Score", fontsize=11, fontweight="bold", labelpad=6)
            ax3.set_ylabel("Astrophysical Category", fontsize=11, fontweight="bold", labelpad=6)
            ax3.tick_params(axis="y", labelsize=10)
            ax3.tick_params(axis="x", labelsize=10)
            ax3.legend(
                title=None,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=min(4, num_methods),
                fontsize=8.5,
                frameon=True,
                facecolor="white",
                edgecolor="#cccccc",
            )

            for container in ax3.containers:
                ax3.bar_label(container, fmt="%.2f", padding=3, fontsize=7.0, color="black")

            sns.despine(ax=ax3)
            fig3.tight_layout()

            plot3_path = self.plots_dir / f"{self.prefix}_Category_Performance.pdf"
            fig3.savefig(plot3_path, dpi=1000, bbox_inches="tight", pad_inches=0.05, facecolor="white")
            plt.close(fig3)
            print(f"Saved Plot: {plot3_path}")

            # 2. Macro - Micro F1
            baseline_rows = [
                {"Method": name, "Metric": "Macro F1", "Score": summary["macro_f1"]}
                for name, summary in m["baselines"].items()
            ] + [
                {"Method": name, "Metric": "Micro F1", "Score": summary["micro_f1"]}
                for name, summary in m["baselines"].items()
            ]

            macro_micro_df = pd.DataFrame(
                [
                    {"Method": "Framework", "Metric": "Macro F1", "Score": m["macro"]["f1"]},
                    {"Method": "Framework", "Metric": "Micro F1", "Score": m["micro"]["f1"]},
                    *baseline_rows,
                ]
            )

            fig4_width = DOUBLE_COL_W if len(self.baseline_names) > 2 else SINGLE_COL_W
            fig4, ax4 = plt.subplots(figsize=(fig4_width, 3.2))
            ax4.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

            macro_micro_colors = {"Macro F1": "#2c3e50", "Micro F1": "#95a5a6"}

            sns.barplot(
                data=macro_micro_df,
                x="Method",
                y="Score",
                hue="Metric",
                palette=macro_micro_colors,
                width=0.68,
                ax=ax4,
                zorder=3,
            )

            if include_titles:
                ax4.set_title("Macro vs. Micro F1 Comparison", pad=8, fontweight="bold")
            ax4.set_ylim(0, 1.18)
            ax4.set_ylabel("F1 Score", fontweight="bold", labelpad=5)
            ax4.set_xlabel("Detection Method", fontweight="bold", labelpad=5)
            if len(self.baseline_names) > 2:
                ax4.tick_params(axis="x", rotation=15)
            ax4.legend(title=None, loc="upper right", frameon=False)

            for container in ax4.containers:
                ax4.bar_label(container, fmt="%.3f", padding=2, fontsize=7, fontweight="bold", color="black")

            sns.despine(ax=ax4)
            fig4.tight_layout()

            plot4_path = self.plots_dir / f"{self.prefix}_Macro_vs_Micro_F1.pdf"
            fig4.savefig(plot4_path, dpi=1000, bbox_inches="tight", pad_inches=0.05, facecolor="white")
            plt.close(fig4)
            print(f"Saved Plot: {plot4_path}")

    def _export_data(self, m):
        df_export = self.df.copy()

        list_cols = ["String_Detections", "Sign_Detections", "Delta_Detections", "Total_Magic"]
        for col in list_cols:
            if col in df_export.columns:
                df_export[col] = df_export[col].apply(
                    lambda x: ", ".join(map(str, sorted(x, key=str)))
                    if isinstance(x, (set, list, tuple))
                    else str(x)
                )

        exports = {
            self.data_dir / f"{self.prefix}_Full_Evaluation.csv": df_export,
            self.data_dir / f"{self.prefix}_Category_Summary.csv": m["category_summary"],
            self.data_dir / f"{self.prefix}_Processing_Summary.csv": m["processing_summary"],
            self.data_dir / f"{self.prefix}_Baseline_Wins.csv": m["baseline_wins"],
        }

        for path, data in exports.items():
            data.to_csv(path, index=(path.name.endswith("Summary.csv")))
            print(f"Saved Data: {path}")


if __name__ == "__main__":
    evaluator = ModelEvaluator(
        json_filepath="Real_World_Analysis/grid_search_results.json",
        sign_opt=3,
        gauss_opt=0.01,
        overlap_opt=0.1,
    )
    results = evaluator.run_pipeline()