from __future__ import annotations
import hashlib
import itertools
import json
import numbers
import time
from pathlib import Path
from typing import Any
import os
import numpy as np
import pandas as pd
import yaml
from astropy.io import ascii

from BaselineEvaluator import BaselineEvaluator
from Magic_Number_Detection.get_magic_numbers import MagicNumberDetector


class PaperProcessor:

    CACHE_VERSION = 2

    def __init__(
        self,
        numeric_rtol: float = 1e-9,
        numeric_atol: float = 1e-12,
        refresh_cache: bool = False,
    ) -> None:
        self.all_paper_results: dict[str, list[dict[str, Any]]] = {}
        self.processing_summary: dict[str, Any] = {
            "successful_catalogs": [],
            "failed_catalogs": {},
            "skipped_catalogs": {},
        }
        self.numeric_rtol = numeric_rtol
        self.numeric_atol = numeric_atol
        self.refresh_cache = refresh_cache

        self.sign_violation_threshold: float | int | None = None
        self.gauss_threshold: float | int | None = None
        self.overlap_threshold: float | int | None = None

    def yaml_file_iterator(self, folder_path: str | Path, param_grid: dict[str, list[Any]]) -> None:
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Ground-truth folder does not exist: {self.folder_path}")

        required_keys = {
            "sign_violation_threshold",
            "gauss_threshold",
            "overlap_threshold",
        }
        missing = required_keys - set(param_grid)
        if missing:
            raise ValueError(f"Parameter grid is missing keys: {sorted(missing)}")

        keys, values = zip(*param_grid.items())
        grid_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        if not grid_combinations:
            raise ValueError("Parameter grid produced zero configurations.")

        yaml_files = sorted(self.folder_path.glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(f"No YAML files found in {self.folder_path}")

        cache_dir = self.folder_path / ".paper_processor_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Found {len(yaml_files)} catalog definitions.")
        print(f"Testing {len(grid_combinations)} hyperparameter combinations per catalog.")

        total_runs = len(grid_combinations)

        for yaml_file in yaml_files:
            catalog_name = yaml_file.stem
            print(f"\n--- Processing {catalog_name} ---")

            try:
                yaml_content = self._load_yaml(yaml_file)
                catalog_info = yaml_content.get("catalog_info", {}) or {}
                ground_truth_columns = yaml_content.get("columns", {}) or {}

                fmt = catalog_info.get("format", "N/A")
                readme_url = catalog_info.get("readme_url", "N/A")
                data_url = catalog_info.get("data_url", "N/A")
                data_path = catalog_info.get("data_path", "N/A")

                if not self._has_value(fmt) or not self._has_value(readme_url):
                    reason = "Missing required catalog format or ReadMe_url."
                    self.processing_summary["skipped_catalogs"][catalog_name] = reason
                    print(f"Skipping {catalog_name}: {reason}")
                    continue

                # Load once, with a validated cache.
                self.df = self._load_or_build_cache(
                    catalog_name=catalog_name,
                    cache_dir=cache_dir,
                    fmt=fmt,
                    readme_url=readme_url,
                    data_url=data_url,
                    data_path=data_path,
                )

                # Evaluation once per catalog.
                baseline = BaselineEvaluator(self.df)
                baseline_results = baseline.run_baseline_detection()
                cached_baseline_metrics = self._prepare_baseline_metrics(
                    baseline_results, ground_truth_columns
                )

                self.all_paper_results[catalog_name] = []
                for run_number, params in enumerate(grid_combinations, start=1):
                    self.sign_violation_threshold = params["sign_violation_threshold"]
                    self.gauss_threshold = params["gauss_threshold"]
                    self.overlap_threshold = params["overlap_threshold"]

                    detector = MagicNumberDetector(
                        self.sign_violation_threshold,
                        self.gauss_threshold,
                        self.overlap_threshold,
                        plot_graphs=False,
                        min_unique_values=20
                    )

                    # cleaned_results was unused in the original pipeline.
                    master_results, _ = detector.run_magic_detection(self.df)

                    run_metrics = self.Results_GroundTruth(
                        master_results=master_results,
                        ground_truth_columns=ground_truth_columns,
                        cached_baseline_metrics=cached_baseline_metrics,
                    )
                    run_metrics["Run_Number"] = run_number
                    self.all_paper_results[catalog_name].append(run_metrics)

                 
                    progress_pct = (run_number / total_runs) * 100
                    print(f"\r  Running configurations... {run_number}/{total_runs} ({progress_pct:.1f}%)", end="", flush=True)

                # newline the progress loop is done so  logs are not overwritten
                print() 

                self.processing_summary["successful_catalogs"].append(catalog_name)
                print(f"Completed {catalog_name}: {len(grid_combinations)} configurations.")

            except Exception as exc:
                print()

                self.processing_summary["failed_catalogs"][catalog_name] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(f"ERROR processing {catalog_name}: {type(exc).__name__}: {exc}")

        self._print_processing_summary()

    def YAML_file_Iteratior(self, folder_path, param_grid):
        return self.yaml_file_iterator(folder_path, param_grid)

    # Loading new or use caching
    @staticmethod
    def _has_value(value: Any) -> bool:
        return value is not None and str(value).strip() not in {"", "N/A", "None"}

    @staticmethod
    def _load_yaml(yaml_file: Path) -> dict[str, Any]:
        with yaml_file.open("r", encoding="utf-8") as file:
            content = yaml.safe_load(file)
        if not isinstance(content, dict):
            raise ValueError(f"YAML file does not contain a mapping: {yaml_file}")
        return content

    def _cache_metadata(
        self,
        catalog_name: str,
        fmt: Any,
        readme_url: Any,
        data_url: Any,
        data_path: Any,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "cache_version": self.CACHE_VERSION,
            "catalog_name": catalog_name,
            "format": str(fmt),
            "readme_url": str(readme_url),
            "data_url": str(data_url),
        }

        if self._has_value(data_path):
            path = Path(str(data_path))
            try:
                stat = path.stat()
                metadata["data_path"] = str(path.resolve())
                metadata["data_path_mtime_ns"] = stat.st_mtime_ns
                metadata["data_path_size"] = stat.st_size
            except OSError:
                metadata["data_path"] = str(path)
                metadata["data_path_missing"] = True
        else:
            metadata["data_path"] = None

        return metadata

    @staticmethod
    def _metadata_digest(metadata: dict[str, Any]) -> str:
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_or_build_cache(
        self,
        catalog_name: str,
        cache_dir: Path,
        fmt: Any,
        readme_url: Any,
        data_url: Any,
        data_path: Any,
    ) -> pd.DataFrame:
        cache_path = cache_dir / f"{catalog_name}.parquet"
        metadata_path = cache_dir / f"{catalog_name}.json"
        metadata = self._cache_metadata(catalog_name, fmt, readme_url, data_url, data_path)
        digest = self._metadata_digest(metadata)

        if not self.refresh_cache and cache_path.exists() and metadata_path.exists():
            try:
                cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if cached_metadata.get("metadata_digest") == digest:
                    print(f"Loading {catalog_name} from validated Parquet cache...")
                    return pd.read_parquet(cache_path)
                print(f"Cache for {catalog_name} is stale; rebuilding.")
            except Exception as exc:
                print(f"Cache validation failed for {catalog_name}: {exc}; rebuilding.")

        print(f"Loading {catalog_name} from source via Astropy...")
        if self._has_value(data_url):
            self.Catalog_from_URL(data_url, readme_url, fmt)
        elif self._has_value(data_path):
            if not Path(str(data_path)).is_file():
                raise FileNotFoundError(f"Catalog path does not exist: {data_path}")
            self.Catalog_from_Path(data_path, readme_url, fmt)
        else:
            raise ValueError("Catalog has neither a valid data_url nor a valid data_path.")

        self.df.to_parquet(cache_path, index=False)
        metadata_path.write_text(
            json.dumps({"metadata_digest": digest, "metadata": metadata}, indent=2),
            encoding="utf-8",
        )
        return self.df

    def Catalog_from_URL(self, data_url, ReadMe_url, format):
        table = ascii.read(data_url, readme=ReadMe_url, format="cds")
        self.df = table.to_pandas()

    def Catalog_from_Path(self, file_path, ReadMe_url, format):
        reader = ascii.get_reader(ascii.Cds, readme=ReadMe_url)
        astropy_table = reader.read(file_path)
        self.df = astropy_table.to_pandas()

    # Ground-truth - baseline evaluation
    def _prepare_baseline_metrics(self, baseline_results, ground_truth_columns):
        cached_baseline_metrics = {}
        for col_name, gt_data in ground_truth_columns.items():
            gt_data = gt_data if gt_data is not None else {}
            
            ground_truth = gt_data.get("Ground_Truth_Magic")
            ground_truth = ground_truth if ground_truth is not None else []
            
            col_baselines = baseline_results.get(col_name)
            col_baselines = col_baselines if col_baselines is not None else {}

            z_outliers = col_baselines.get("z_score_outliers")
            z_outliers = z_outliers if z_outliers is not None else []
            
            iqr_outliers = col_baselines.get("iqr_outliers")
            iqr_outliers = iqr_outliers if iqr_outliers is not None else []

            _, _, _, z_p, z_r, z_f1 = self._calc_metrics(z_outliers, ground_truth)
            _, _, _, iqr_p, iqr_r, iqr_f1 = self._calc_metrics(iqr_outliers, ground_truth)

            cached_baseline_metrics[col_name] = {
                "Z-Score": {"Metrics": {"Precision": z_p, "Recall": z_r, "F1 Score": z_f1}},
                "IQR": {"Metrics": {"Precision": iqr_p, "Recall": iqr_r, "F1 Score": iqr_f1}},
            }
        return cached_baseline_metrics

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))

    def _values_equal(self, a: Any, b: Any) -> bool:
        if self._is_numeric(a) and self._is_numeric(b):
            try:
                return bool(
                    np.isclose(
                        float(a),
                        float(b),
                        rtol=self.numeric_rtol,
                        atol=self.numeric_atol,
                        equal_nan=False,
                    )
                )
            except (TypeError, ValueError):
                return False

        return str(a).strip() == str(b).strip()

    def _unique_values(self, values: Any) -> list[Any]:
        if values is None:
            return []
        if isinstance(values, (str, bytes)):
            values = [values]
        else:
            try:
                values = list(values)
            except TypeError:
                values = [values]

        unique: list[Any] = []
        for value in values:
            if not any(self._values_equal(value, existing) for existing in unique):
                unique.append(value)
        return unique

    # Calculate F1 metrics 
    def _calc_metrics(self, detected, ground_truth):
       
        det_values = self._unique_values(detected)
        gt_values = self._unique_values(ground_truth)

        matched_gt_indices: set[int] = set()
        tp: list[Any] = []
        fp: list[Any] = []

        for det in det_values:
            match_index = next(
                (
                    idx
                    for idx, gt in enumerate(gt_values)
                    if idx not in matched_gt_indices and self._values_equal(det, gt)
                ),
                None,
            )
            if match_index is None:
                fp.append(det)
            else:
                matched_gt_indices.add(match_index)
                tp.append(det)

        fn = [gt for idx, gt in enumerate(gt_values) if idx not in matched_gt_indices]

        TP_count = len(tp)
        FP_count = len(fp)
        FN_count = len(fn)

        if not gt_values and not det_values:
            precision = recall = f1_score = 1.0
        else:
            precision = TP_count / (TP_count + FP_count) if (TP_count + FP_count) else 0.0
            recall = TP_count / (TP_count + FN_count) if (TP_count + FN_count) else 0.0
            f1_score = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        return (
            tp,
            fp,
            fn,
            round(precision, 4),
            round(recall, 4),
            round(f1_score, 4),
        )

    def Results_GroundTruth(self, master_results, ground_truth_columns, cached_baseline_metrics):
        column_evaluations = {}

        for col_name, gt_data in ground_truth_columns.items():
            gt_data = gt_data if gt_data is not None else {}
            
            ground_truth = gt_data.get("Ground_Truth_Magic")
            ground_truth = ground_truth if ground_truth is not None else []
            
            category = gt_data.get("Category", "Unknown")
            baseline_analysis_dict = cached_baseline_metrics.get(col_name, {})

            if col_name not in master_results:
                _, _, alg_fn, alg_p, alg_r, alg_f1 = self._calc_metrics([], ground_truth)
                column_evaluations[col_name] = {
                    "Processed by Algorithm": "No",
                    "Category": category,
                    "Ground Truth Magic Numbers": ground_truth,
                    "Warnings": ["Column was completely bypassed/dropped by the detector."],
                    "Magic Numbers String Process": [],
                    "Magic Numbers Sign Violation": [],
                    "Magic Numbers Delta": [],
                    "Total Magic": [],
                    "Metrics": {
                        "True Positives (TP)": [],
                        "False Positives (FP)": [],
                        "False Negatives (FN)": alg_fn,
                        "Precision": alg_p,
                        "Recall": alg_r,
                        "F1 Score": alg_f1,
                    },
                    "Baseline Analysis": baseline_analysis_dict,
                }
                continue

            res = master_results[col_name]
            res = res if res is not None else {}
            warnings = []

            distanced_raw = res.get("magic_distanced_numbers")
            distanced_raw = distanced_raw if distanced_raw is not None else []
            
            if isinstance(distanced_raw, list) and distanced_raw and isinstance(distanced_raw[0], str):
                warnings.append(distanced_raw[0])
                distanced = []
            else:
                distanced = self._unique_values(distanced_raw)

            sign_viol_raw = res.get("magic_sign_violation")
            sign_viol = self._unique_values(sign_viol_raw if sign_viol_raw is not None else [])
            
            strings_raw = res.get("magic_strings")
            strings = self._unique_values(strings_raw if strings_raw is not None else [])

            total_detected = self._unique_values(distanced + sign_viol + strings)

            all_magic = res.get("all_magic_numbers")
            if all_magic is not None:
                all_magic_values = (
                    list(all_magic)
                    if not isinstance(all_magic, (str, bytes)) and hasattr(all_magic, "__iter__")
                    else [all_magic]
                )
                total_detected = self._unique_values(total_detected + all_magic_values)

            alg_tp, alg_fp, alg_fn, alg_p, alg_r, alg_f1 = self._calc_metrics(
                total_detected, ground_truth
            )

            column_evaluations[col_name] = {
                "Processed by Algorithm": "Yes",
                "Category": category,
                "Ground Truth Magic Numbers": ground_truth,
                "Warnings": warnings,
                "Magic Numbers String Process": strings,
                "Magic Numbers Sign Violation": sign_viol,
                "Magic Numbers Delta": distanced,
                "Total Magic": total_detected,
                "Metrics": {
                    "True Positives (TP)": alg_tp,
                    "False Positives (FP)": alg_fp,
                    "False Negatives (FN)": alg_fn,
                    "Precision": alg_p,
                    "Recall": alg_r,
                    "F1 Score": alg_f1,
                },
                "Baseline Analysis": baseline_analysis_dict,
            }

        return {
            "Run_Parameters": {
                "sign_violation_threshold": self.sign_violation_threshold,
                "gauss_threshold": self.gauss_threshold,
                "overlap_threshold": self.overlap_threshold,
            },
            "Column_Metrics": column_evaluations,
        }

    # Report
    def _print_processing_summary(self) -> None:
        successful = self.processing_summary["successful_catalogs"]
        failed = self.processing_summary["failed_catalogs"]
        skipped = self.processing_summary["skipped_catalogs"]

        print("\n--- Processing Summary ---")
        print(f"Successful catalogs: {len(successful)}")
        print(f"Failed catalogs:     {len(failed)}")
        print(f"Skipped catalogs:    {len(skipped)}")
        if failed:
            for name, details in failed.items():
                print(f"  FAILED: {name} -> {details['error_type']}: {details['error']}")
        if skipped:
            for name, reason in skipped.items():
                print(f"  SKIPPED: {name} -> {reason}")

    def save_results(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(self.all_paper_results, outfile, indent=4, default=self._json_default)

    def save_experiment_metadata(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        metadata = {
            "cache_version": self.CACHE_VERSION,
            "numeric_rtol": self.numeric_rtol,
            "numeric_atol": self.numeric_atol,
            "catalogs_with_results": len(self.all_paper_results),
            "successful_catalogs": self.processing_summary["successful_catalogs"],
            "failed_catalogs": self.processing_summary["failed_catalogs"],
            "skipped_catalogs": self.processing_summary["skipped_catalogs"],
            "note": (
                "All grid configurations are evaluated against the supplied ground truth. "
                "The grid-search result is an exploration/evaluation dataset; selecting an "
                "optimal configuration should be performed as a separate analysis step."
            ),
        }
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(metadata, outfile, indent=4, default=self._json_default)

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


# Experiment configurations
hyperparameter_grid = {
    "sign_violation_threshold": [1, 2, 3, 4, 5],
    "gauss_threshold": [0.01, 0.025, 0.05, 0.075, 0.1],
    "overlap_threshold": [0.1, 2.0, 4.0, 6.0, 8.0],
}

# Configuration for quick testing
"""
hyperparameter_grid = {
    "sign_violation_threshold": [3],
    "gauss_threshold": [0.1],
    "overlap_threshold": [0.1],
}
"""

def run_real_world_analysis():
    processor = PaperProcessor(
        numeric_rtol=1e-9,
        numeric_atol=1e-12,
        refresh_cache=False,
    )

    print("\nInitiating Real-World Catalog Grid Search...")
    start_time = time.perf_counter()
    
    processor.yaml_file_iterator(
        folder_path=Path("ground_truth"),
        param_grid=hyperparameter_grid,
    )
    
    elapsed_seconds = time.perf_counter() - start_time

    # output dir
    output_dir = "Real_World_Analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Assign paths
    results_path = os.path.join(output_dir, "grid_search_results.json")
    metadata_path = os.path.join(output_dir, "grid_search_metadata.json")

    # Save files 
    processor.save_results(results_path)
    processor.save_experiment_metadata(metadata_path)

    print("\n--- Grid Search Complete ---")
    print(f"Execution Time: {elapsed_seconds/60:.2f} minutes")

if __name__ == "__main__":
    run_real_world_analysis()