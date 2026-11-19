import numpy as np
import pandas as pd
import time
from typing import Dict, Tuple, List, Callable
from sklearn.ensemble import IsolationForest
from StringValDetector import magic_strings_detection as string_values_process
from scipy.stats import t as student_t
from sklearn.neighbors import KernelDensity

class BaselineEvaluator:
    
    def __init__(self, df: pd.DataFrame, include_ml: bool = True):
        self.df = df
        
        # Map Display Name -> function(1D_float_array) -> boolean_outlier_mask
        self.methods: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "Z-Score": self._z_score_mask,
            "IQR": self._iqr_mask,
            "Modified Z-Score": self._mad_z_score_mask,
            "Quantile (99%)": self._quantile_mask,
            "HBOS": self._hbos_mask,
            #"KDE (1% density tail)": self._kde_mask,
            #"Generalized ESD": self._generalized_esd_mask,
        }
        if include_ml:
            self.methods["Isolation Forest"] = self._isolation_forest_mask

    # Input: 1D float array -> Output: bool mask
    @staticmethod
    def _z_score_mask(arr: np.ndarray) -> np.ndarray:
        std_val = np.std(arr)
        if std_val <= 0:
            return np.zeros(len(arr), dtype=bool)
        return np.abs((arr - np.mean(arr)) / std_val) > 3.0

    @staticmethod
    def _iqr_mask(arr: np.ndarray) -> np.ndarray:
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            return np.zeros(len(arr), dtype=bool)
        return (arr < (q1 - 1.5 * iqr)) | (arr > (q3 + 1.5 * iqr))

    @staticmethod
    def _mad_z_score_mask(arr: np.ndarray) -> np.ndarray:
        # Modified Z-Score using Median Absolute Deviation
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad <= 0:
            return np.zeros(len(arr), dtype=bool)
        mod_z = 0.6745 * np.abs(arr - median) / mad
        return mod_z > 3.5

    @staticmethod
    def _quantile_mask(arr: np.ndarray) -> np.ndarray:
        # Empirical 0.5% - 99.5% tail clipping 
        if len(np.unique(arr)) < 5:
            return np.zeros(len(arr), dtype=bool)
        low, high = np.percentile(arr, [0.5, 99.5])
        if low == high:
            return np.zeros(len(arr), dtype=bool)
        return (arr < low) | (arr > high)

    @staticmethod
    def _hbos_mask(arr: np.ndarray) -> np.ndarray:
        # 1D Histogram-based Outlier Score 
        if len(np.unique(arr)) < 5:
            return np.zeros(len(arr), dtype=bool)
        n_bins = int(np.clip(np.sqrt(len(arr)), 10, 100))
        counts, bin_edges = np.histogram(arr, bins=n_bins)
        bin_indices = np.clip(np.digitize(arr, bin_edges[:-1]) - 1, 0, n_bins - 1)
        
        densities = counts[bin_indices] / len(arr)
        scores = -np.log10(densities + 1e-10)
        # Flag top 1%
        thresh = np.percentile(scores, 99)
        if thresh <= np.median(scores):
            return np.zeros(len(arr), dtype=bool)
        return scores > thresh

    @staticmethod
    def _isolation_forest_mask(arr: np.ndarray) -> np.ndarray:
        # Subsampled Isolation Forest 
        if len(np.unique(arr)) < 5 or len(arr) < 10:
            return np.zeros(len(arr), dtype=bool)
        X = arr.reshape(-1, 1)
        clf = IsolationForest(
            n_estimators=50,
            max_samples=min(256, len(arr)),
            contamination=0.01,
            random_state=42,
            n_jobs=1,
        )
        # Fit on a subsample if column is big >20,000 rows
        if len(arr) > 20000:
            rng = np.random.default_rng(42)
            sub_idx = rng.choice(len(arr), size=20000, replace=False)
            clf.fit(X[sub_idx])
        else:
            clf.fit(X)
        return clf.predict(X) == -1

    @staticmethod
    def _generalized_esd_mask(
        arr: np.ndarray,
        alpha: float = 0.05,
        max_outlier_fraction: float = 0.10,
    ) -> np.ndarray:
        # ESD test for multiple outliers

        x = np.asarray(arr, dtype=float)
        n = len(x)
        mask = np.zeros(n, dtype=bool)

        if n < 10 or len(np.unique(x)) < 5:
            return mask

        max_outliers = min(
            int(np.floor(max_outlier_fraction * n)),
            n - 2,
        )

        if max_outliers < 1:
            return mask

        remaining = np.arange(n)
        work = x.copy()

        removed_indices = []
        statistics = []
        critical_values = []

        for _ in range(max_outliers):
            ni = len(work)

            mean = np.mean(work)
            std = np.std(work, ddof=1)

            if not np.isfinite(std) or std <= 0:
                break

            deviations = np.abs(work - mean)
            local_idx = int(np.argmax(deviations))

            R_i = deviations[local_idx] / std

            # Critical value for the current iteration
            p = 1.0 - alpha / (2.0 * ni)
            t_crit = student_t.ppf(p, df=ni - 2)

            lambda_i = (
                (ni - 1) * t_crit
                / np.sqrt((ni - 2 + t_crit**2) * ni)
            )

            removed_indices.append(remaining[local_idx])
            statistics.append(R_i)
            critical_values.append(lambda_i)

            work = np.delete(work, local_idx)
            remaining = np.delete(remaining, local_idx)

        # Largest iteration at which R_i exceeds its critical value
        significant = [
            i + 1
            for i, (r, critical) in enumerate(
                zip(statistics, critical_values)
            )
            if r > critical
        ]

        if significant:
            n_outliers = max(significant)
            mask[removed_indices[:n_outliers]] = True

        return mask

    @staticmethod
    def _kde_mask(arr: np.ndarray) -> np.ndarray:
        # Kernel Density Estimation: flag the lowest-density 1%

        x = np.asarray(arr, dtype=float)
        n = len(x)

        if n < 20 or len(np.unique(x)) < 5:
            return np.zeros(n, dtype=bool)

        std = np.std(x, ddof=1)
        q1, q3 = np.percentile(x, [25, 75])

        robust_std = (q3 - q1) / 1.349

        # Robust Silverman bandwidth estimate
        scale = robust_std if robust_std > 0 else std

        if not np.isfinite(scale) or scale <= 0:
            return np.zeros(n, dtype=bool)

        bandwidth = 0.9 * scale * n ** (-1.0 / 5.0)

        kde = KernelDensity(
            kernel="gaussian",
            bandwidth=bandwidth,
        )

        kde.fit(x.reshape(-1, 1))

        log_density = kde.score_samples(x.reshape(-1, 1))

        threshold = np.percentile(log_density, 1.0)

        return log_density < threshold

    def _preprocess_array_with_indices(self, data_array: np.ndarray) -> Tuple[np.ndarray, List, np.ndarray]:
        data_arr, magic_strings = string_values_process(data_array)
        mask = (data_arr != '') & (data_arr != ' ') & (~pd.isna(data_arr))

        valid_indices = np.flatnonzero(mask)
        filtered_arr = data_arr[mask]

        if len(filtered_arr) == 0:
            return np.array([], dtype=float), magic_strings, valid_indices

        try:
            filtered_arr = np.asarray(filtered_arr, dtype=float)
            finite_mask = np.isfinite(filtered_arr)
            if not np.all(finite_mask):
                filtered_arr = filtered_arr[finite_mask]
                valid_indices = valid_indices[finite_mask]
        except (ValueError, TypeError):
            filtered_arr = np.asarray(filtered_arr)

        return filtered_arr, magic_strings, valid_indices

    def _preprocess_array(self, data_array: np.ndarray) -> Tuple[np.ndarray, List]:
        filtered_arr, magic_strings, _ = self._preprocess_array_with_indices(data_array)
        return filtered_arr, magic_strings

    def analyze_column(self, data_array: np.ndarray, col_name: str, value_type: str) -> Dict:
        self.magic_strings = []

        if value_type != "N" or len(data_array) == 0:
            return {}

        self.data_arr, self.magic_strings, valid_indices = self._preprocess_array_with_indices(data_array)

        if len(self.data_arr) == 0:
            return {}

        methods_output = {}
        for method_name, mask_fn in self.methods.items():
            started_at = time.perf_counter()
            if method_name == "Isolation Forest":
                print(
                    f"    Running {method_name} for {col_name} "
                    f"({len(self.data_arr):,} values)...",
                    flush=True,
                )
            mask = np.asarray(mask_fn(self.data_arr), dtype=bool)
            outliers = np.unique(self.data_arr[mask]).tolist() if np.any(mask) else []
            outlier_rows = valid_indices[mask].astype(int).tolist() if np.any(mask) else []
            methods_output[method_name] = {
                "outliers": outliers,
                "outlier_rows": outlier_rows,
            }
            if method_name == "Isolation Forest":
                print(
                    f"    {method_name} completed for {col_name} "
                    f"in {time.perf_counter() - started_at:.1f}s.",
                    flush=True,
                )

        return {
            col_name: {
                "methods": methods_output,
                "z_score_outliers": methods_output.get("Z-Score", {}).get("outliers", []),
                "iqr_outliers": methods_output.get("IQR", {}).get("outliers", []),
                "z_score_outlier_rows": methods_output.get("Z-Score", {}).get("outlier_rows", []),
                "iqr_outlier_rows": methods_output.get("IQR", {}).get("outlier_rows", []),
            }
        }

    def _generate_col_info(self, df: pd.DataFrame) -> List[List]:
        extended_col_info = []
        rows = len(df)
        for i, column_header in enumerate(df.columns):
            column_type = df[column_header].dtype
            type_str = "N" if pd.api.types.is_numeric_dtype(column_type) else "S"
            extended_col_info.append([column_header, type_str, i, rows])
        return extended_col_info

    def run_baseline_detection(self) -> Dict:
        master_dict = {}
        extended_col_info = self._generate_col_info(self.df)

        numeric_columns = [
            (col_name, col_idx)
            for col_name, value_type, col_idx, _ in extended_col_info
            if value_type == "N"
        ]
        total_numeric_columns = len(numeric_columns)

        for column_number, (col_name, col_idx) in enumerate(numeric_columns, start=1):
            data_array = self.df.iloc[:, col_idx].values
            print(
                f"  Baseline column {column_number}/{total_numeric_columns}: "
                f"{col_name} ({len(data_array):,} rows)...",
                flush=True,
            )
            started_at = time.perf_counter()
            results_dict = self.analyze_column(data_array, col_name, "N")
            if results_dict:
                master_dict.update(results_dict)
            print(
                f"  Baseline column {column_number}/{total_numeric_columns} "
                f"finished in {time.perf_counter() - started_at:.1f}s.",
                flush=True,
            )

        return master_dict