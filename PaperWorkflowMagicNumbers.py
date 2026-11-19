from __future__ import annotations
import hashlib
import itertools
import json
import numbers
import os
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from vizier_mirrors import get_vizier_candidate_urls, resolve_working_vizier_url
try:
    from astropy.io import ascii
    from astropy.utils.data import conf as astropy_data_conf
except ImportError:
    ascii = None
    astropy_data_conf = None

_THIS_DIR = Path(__file__).resolve().parent
_MND_DIR = _THIS_DIR / "Magic_Number_Detection"
for _p in (_THIS_DIR, _MND_DIR):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from astropy.io import ascii
except ImportError:
    ascii = None

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

        self.df: pd.DataFrame = pd.DataFrame()
        self.folder_path: Path = Path("ground_truth")
        self._actual_mask_cache: dict[str, np.ndarray] = {}

    def yaml_file_iterator(self, folder_path: str | Path, param_grid: dict[str, list[Any]]) -> None:
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists() and (_THIS_DIR / self.folder_path).exists():
            self.folder_path = _THIS_DIR / self.folder_path
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
                print(f"Loading catalog data for {catalog_name}...")
                self.df = self._load_or_build_cache(
                    catalog_name=catalog_name,
                    cache_dir=cache_dir,
                    fmt=fmt,
                    readme_url=readme_url,
                    data_url=data_url,
                    data_path=data_path,
                )
                print(
                    f"Loaded {catalog_name}: {len(self.df):,} rows, "
                    f"{len(self.df.columns)} columns."
                )

                # Reset per-catalog ground-truth row mask cache
                self._actual_mask_cache = {}

                # Evaluation once per catalog 
                print(f"Running baseline evaluation for {catalog_name}...")
                baseline = BaselineEvaluator(self.df.copy(deep=True))
                baseline_results = baseline.run_baseline_detection()
                print(f"Preparing baseline metrics for {catalog_name}...", flush=True)
                cached_baseline_metrics = self._prepare_baseline_metrics(
                    baseline_results,
                    ground_truth_columns,
                    registered_methods=list(baseline.methods.keys()),
                )
                print(f"Baseline evaluation complete for {catalog_name}.", flush=True)

                self.all_paper_results[catalog_name] = []
                for run_number, params in enumerate(grid_combinations, start=1):
                    self.sign_violation_threshold = params["sign_violation_threshold"]
                    self.gauss_threshold = params["gauss_threshold"]
                    self.overlap_threshold = params["overlap_threshold"]
                    print(
                        f"  Starting configuration {run_number}/{total_runs}: "
                        f"sign={self.sign_violation_threshold}, "
                        f"gauss={self.gauss_threshold}, "
                        f"overlap={self.overlap_threshold}...",
                        flush=True,
                    )

                    detector = MagicNumberDetector(
                        self.sign_violation_threshold,
                        self.gauss_threshold,
                        self.overlap_threshold,
                        plot_graphs=False,
                        min_unique_values=30,
                    )

                    # Pass a deep copy so in-place array operations inside MagicNumberDetector
                    master_results, _ = detector.run_magic_detection(self.df.copy(deep=True))
                    print(
                        f"  Detection finished for configuration "
                        f"{run_number}/{total_runs}; calculating metrics...",
                        flush=True,
                    )

                    run_metrics = self.Results_GroundTruth(
                        master_results=master_results,
                        ground_truth_columns=ground_truth_columns,
                        cached_baseline_metrics=cached_baseline_metrics,
                    )
                    run_metrics["Run_Number"] = run_number
                    self.all_paper_results[catalog_name].append(run_metrics)
                    print(
                        f"  Metrics finished for configuration "
                        f"{run_number}/{total_runs}.",
                        flush=True,
                    )

                    progress_pct = (run_number / total_runs) * 100
                    print(
                        f"\r  Running configurations... {run_number}/{total_runs} ({progress_pct:.1f}%)",
                        end="",
                        flush=True,
                    )

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

    def _resolve_local_path(self, raw_path: Any) -> Path:
        path = Path(str(raw_path))
        if path.exists():
            return path
        if hasattr(self, "folder_path"):
            for base in (self.folder_path, self.folder_path.parent, _THIS_DIR):
                candidate = base / path
                if candidate.exists():
                    return candidate
        return path

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
            path = self._resolve_local_path(data_path)
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
        resolved_readme = (
            str(self._resolve_local_path(readme_url))
            if not str(readme_url).startswith(("http://", "https://", "ftp://"))
            else readme_url
        )
        if self._has_value(data_url):
            self.Catalog_from_URL(data_url, resolved_readme, fmt)
        elif self._has_value(data_path):
            resolved_data_path = self._resolve_local_path(data_path)
            if not resolved_data_path.is_file():
                raise FileNotFoundError(f"Catalog path does not exist: {data_path}")
            self.Catalog_from_Path(str(resolved_data_path), resolved_readme, fmt)
        else:
            raise ValueError("Catalog has neither a valid data_url nor a valid data_path.")

        try:
            self.df.to_parquet(cache_path, index=False)
            metadata_path.write_text(
                json.dumps({"metadata_digest": digest, "metadata": metadata}, indent=2),
                encoding="utf-8",
            )
        except Exception as cache_exc:
            print(f"  Warning: Could not write Parquet cache for {catalog_name} ({cache_exc}); continuing in-memory.")

        return self.df

    def Catalog_from_URL(self, data_url, ReadMe_url, format, read_timeout: float = 10.0):
        if ascii is None:
            raise ImportError("astropy is required to read catalogs from URL.")

        # Resolve working ReadMe URL across mirrors (if remote)
        working_readme = (
            resolve_working_vizier_url(ReadMe_url)
            if str(ReadMe_url).startswith(("http://", "https://"))
            else ReadMe_url
        )

        # Try candidate data URLs across mirrors (including .gz / non-.gz variants)
        data_candidates = get_vizier_candidate_urls(
            data_url, include_compression_variants=True
        )
        last_exc: Exception | None = None

        for candidate_data_url in data_candidates:
            try:
                print(f"  Reading catalog data from: {candidate_data_url}")
                with astropy_data_conf.set_temp("remote_timeout", read_timeout):
                    table = ascii.read(
                        candidate_data_url, readme=working_readme, format="cds"
                    )
                    self.df = table.to_pandas()
                    return
            except Exception as exc:
                last_exc = exc
                print(f"  Warning: Failed reading data from {candidate_data_url} ({type(exc).__name__}: {exc})")

        raise RuntimeError(
            f"Exhausted all VizieR mirrors for data_url={data_url} (ReadMe={working_readme})"
        ) from last_exc

    def Catalog_from_Path(
        self, file_path, ReadMe_url, format, readme_timeout: float = 8.0
    ):
        if ascii is None:
            raise ImportError("astropy is required to read catalogs from file path.")

        working_readme = (
            resolve_working_vizier_url(
                ReadMe_url,
                per_mirror_timeout=2.5,
                max_total_time=readme_timeout,
            )
            if str(ReadMe_url).startswith(("http://", "https://"))
            else ReadMe_url
        )
        with astropy_data_conf.set_temp("remote_timeout", readme_timeout):
            reader = ascii.get_reader(ascii.Cds, readme=working_readme)
            astropy_table = reader.read(file_path)
            self.df = astropy_table.to_pandas()

    def _resolve_df_column(self, col_name: str) -> str | None:
        if col_name in self.df.columns:
            return col_name
        stripped = str(col_name).strip()
        for existing in self.df.columns:
            if str(existing).strip() == stripped:
                return existing
        return None

    # Ground-truth - baseline evaluation
    def _prepare_baseline_metrics(
        self,
        baseline_results: dict[str, Any],
        ground_truth_columns: dict[str, Any],
        registered_methods: list[str] | None = None,
    ) -> dict[str, Any]:
        if not registered_methods:
            for col_res in baseline_results.values():
                if isinstance(col_res, dict) and "methods" in col_res:
                    registered_methods = list(col_res["methods"].keys())
                    break
            if not registered_methods:
                registered_methods = list(BaselineEvaluator(pd.DataFrame()).methods.keys())

        cached_baseline_metrics = {}
        for col_name, gt_data in ground_truth_columns.items():
            resolved_col = self._resolve_df_column(col_name)
            if resolved_col is None:
                continue

            gt_data = gt_data if gt_data is not None else {}
            ground_truth = gt_data.get("Ground_Truth_Magic") or []
            col_baselines = baseline_results.get(resolved_col) or baseline_results.get(col_name) or {}

            methods_dict = col_baselines.get("methods")
            if methods_dict is None:
                # Fallback for legacy format or string/skipped columns
                methods_dict = {
                    "Z-Score": {
                        "outliers": col_baselines.get("z_score_outliers") or [],
                        "outlier_rows": col_baselines.get("z_score_outlier_rows"),
                    },
                    "IQR": {
                        "outliers": col_baselines.get("iqr_outliers") or [],
                        "outlier_rows": col_baselines.get("iqr_outlier_rows"),
                    },
                }

            col_eval = {}
            for method_name in registered_methods:
                m_data = methods_dict.get(method_name, {})
                outliers = m_data.get("outliers") or []
                has_outlier_rows = "outlier_rows" in m_data and m_data["outlier_rows"] is not None
                outlier_rows = m_data["outlier_rows"] if has_outlier_rows else ([] if not outliers else None)

                _, _, _, p, r, f1 = self._calc_metrics(outliers, ground_truth)
                col_eval[method_name] = {
                    "Metrics": {"Precision": p, "Recall": r, "F1 Score": f1},
                    "Row-Instance Metrics": self._row_instance_metrics(
                        resolved_col,
                        ground_truth,
                        detected_values=outliers if outlier_rows is None else None,
                        detected_row_indices=outlier_rows,
                    ),
                }

            cached_baseline_metrics[col_name] = col_eval
        return cached_baseline_metrics

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None or value is pd.NA:
            return True
        if isinstance(value, (float, np.floating)):
            return bool(np.isnan(value))
        try:
            res = pd.isna(value)
            if isinstance(res, (bool, np.bool_)):
                return bool(res)
        except Exception:
            pass
        return False

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))

    @classmethod
    def _try_parse_finite_float(cls, value: Any) -> float | None:
        if cls._is_missing(value) or isinstance(value, (bool, np.bool_)):
            return None
        if cls._is_numeric(value):
            val = float(value)
            return val if np.isfinite(val) else None
        if isinstance(value, (str, bytes)):
            s = value.decode("utf-8", errors="ignore").strip() if isinstance(value, bytes) else value.strip()
            if not s:
                return None
            try:
                val = float(s)
                return val if np.isfinite(val) else None
            except (TypeError, ValueError):
                return None
        return None

    def _values_equal(self, a: Any, b: Any) -> bool:
        if self._is_missing(a) or self._is_missing(b):
            return False

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

        # Handle numeric string vs numeric float/int comparison (e.g. "-99.00" vs -99.0)
        fa = self._try_parse_finite_float(a)
        fb = self._try_parse_finite_float(b)
        if fa is not None and fb is not None:
            return bool(
                np.isclose(
                    fa,
                    fb,
                    rtol=self.numeric_rtol,
                    atol=self.numeric_atol,
                    equal_nan=False,
                )
            )

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

        if (
            0 <= self.numeric_rtol < 1
            and self.numeric_atol >= 0
            and all(self._is_numeric(value) and np.isfinite(value) for value in values)
        ):
            exact_unique = list(dict.fromkeys(values))
            exact_unique.sort(key=float)
            numeric_unique: list[Any] = []
            for value in exact_unique:
                if not numeric_unique or not self._values_equal(value, numeric_unique[-1]):
                    numeric_unique.append(value)
            return numeric_unique

        unique: list[Any] = []
        for value in values:
            if self._is_missing(value):
                continue
            if not any(self._values_equal(value, existing) for existing in unique):
                unique.append(value)
        return unique

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

    def _build_column_mask(self, col_name: str, target_values: list[Any]) -> np.ndarray:
        """Build a boolean mask of rows in self.df[col_name] matching any value in target_values."""
        n_rows = len(self.df)
        targets = self._unique_values(target_values)
        if not targets or n_rows == 0:
            return np.zeros(n_rows, dtype=bool)

        series = self.df[col_name]
        col_dtype = series.dtype

        # Fast vectorized path for purely numeric columns
        if pd.api.types.is_numeric_dtype(col_dtype) and not pd.api.types.is_bool_dtype(col_dtype):
            num_targets = [
                f for f in (self._try_parse_finite_float(t) for t in targets) if f is not None
            ]
            if not num_targets:
                return np.zeros(n_rows, dtype=bool)
            col_arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            target_arr = np.asarray(num_targets, dtype=float)
            return np.any(
                np.isclose(
                    col_arr[:, None],
                    target_arr[None, :],
                    rtol=self.numeric_rtol,
                    atol=self.numeric_atol,
                    equal_nan=False,
                ),
                axis=1,
            )

        column_values = series.tolist()
        return np.array(
            [
                any(self._values_equal(value, magic_value) for magic_value in targets)
                for value in column_values
            ],
            dtype=bool,
        )

    def _row_instance_metrics(
        self,
        col_name: str,
        ground_truth: Any,
        detected_values: Any = None,
        detected_row_indices: Any = None,
    ) -> dict[str, int | float]:
        """Evaluate detections per dataframe row, rather than per unique value."""
        n_rows = len(self.df)
        ground_truth_values = self._unique_values(ground_truth)

        if col_name in self._actual_mask_cache:
            actual_mask = self._actual_mask_cache[col_name]
        else:
            actual_mask = self._build_column_mask(col_name, ground_truth_values)
            self._actual_mask_cache[col_name] = actual_mask

        if detected_row_indices is not None:
            predicted_mask = np.zeros(n_rows, dtype=bool)
            valid_indices = [
                int(index)
                for index in detected_row_indices
                if isinstance(index, numbers.Integral)
                and not isinstance(index, (bool, np.bool_))
                and 0 <= int(index) < n_rows
            ]
            if valid_indices:
                predicted_mask[valid_indices] = True
        else:
            detected_unique = self._unique_values(detected_values)
            predicted_mask = self._build_column_mask(col_name, detected_unique)

        tp = int(np.count_nonzero(actual_mask & predicted_mask))
        fp = int(np.count_nonzero(~actual_mask & predicted_mask))
        tn = int(np.count_nonzero(~actual_mask & ~predicted_mask))
        fn = int(np.count_nonzero(actual_mask & ~predicted_mask))

        # Per Section 4.1 of Manuscript.tex and _calc_metrics:
        # When no ground-truth magic numbers exist and no false positives are predicted,
        # a correct non-detection is defined as Precision = Recall = F1 = 1.0.
        if tp == 0 and fp == 0 and fn == 0:
            precision = recall = f1_score = 1.0
        else:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1_score, 4),
        }

    def Results_GroundTruth(self, master_results, ground_truth_columns, cached_baseline_metrics):
        column_evaluations = {}

        for col_name, gt_data in ground_truth_columns.items():
            resolved_col = self._resolve_df_column(col_name)
            if resolved_col is None:
                print(f"  Warning: Ground-truth column '{col_name}' not found in dataframe; skipping.")
                continue

            gt_data = gt_data if gt_data is not None else {}

            ground_truth = gt_data.get("Ground_Truth_Magic")
            ground_truth = ground_truth if ground_truth is not None else []

            category = gt_data.get("Category", "Unknown")
            baseline_analysis_dict = cached_baseline_metrics.get(col_name, {})

            try:
                unique_cnt = int(self.df[resolved_col].nunique(dropna=True))
            except TypeError:
                unique_cnt = len(self._unique_values(self.df[resolved_col].tolist()))

            res_key = (
                resolved_col
                if resolved_col in master_results
                else (col_name if col_name in master_results else None)
            )

            column_metadata = {
                "Num_Samples": int(len(self.df)),
                "Unique_Values": unique_cnt,
                "Has_Ground_Truth": len(self._unique_values(ground_truth)) > 0,
                "Processed_By_Algorithm": res_key is not None,
            }
            
            if res_key is None:
                alg_tp, alg_fp, alg_fn, alg_p, alg_r, alg_f1 = self._calc_metrics([], ground_truth)
                unique_metrics = {
                    "True Positives (TP)": alg_tp,
                    "False Positives (FP)": alg_fp,
                    "False Negatives (FN)": alg_fn,
                    "Precision": alg_p,
                    "Recall": alg_r,
                    "F1 Score": alg_f1,
                }
                column_evaluations[col_name] = {
                    **column_metadata,
                    "Processed by Algorithm": "No",
                    "Category": category,
                    "Ground Truth Magic Numbers": ground_truth,
                    "Warnings": ["Column was completely bypassed/dropped by the detector."],
                    "Magic Numbers String Process": [],
                    "Magic Numbers Sign Violation": [],
                    "Magic Numbers Delta": [],
                    "Total Magic": [],
                    "Unique-Constant Metrics": unique_metrics,
                    "Row-Instance Metrics": self._row_instance_metrics(resolved_col, ground_truth, detected_values=[]),
                    "Metrics": unique_metrics,
                    "Baseline Analysis": baseline_analysis_dict,
                }
                continue

            res = master_results[res_key] or {}
            warnings: list[str] = []

            distanced_raw = res.get("magic_distanced_numbers")
            if distanced_raw is None:
                distanced_items = []
            elif isinstance(distanced_raw, (str, bytes)):
                distanced_items = [distanced_raw]
            else:
                try:
                    distanced_items = list(distanced_raw)
                except TypeError:
                    distanced_items = [distanced_raw]

            distanced_numeric = []
            for item in distanced_items:
                if isinstance(item, str):
                    warnings.append(item)
                else:
                    distanced_numeric.append(item)
            distanced = self._unique_values(distanced_numeric)

            sign_viol = self._unique_values(res.get("magic_sign_violation") or [])
            strings = self._unique_values(res.get("magic_strings") or [])

            total_detected = self._unique_values(distanced + sign_viol + strings)

            all_magic = res.get("all_magic_numbers")
            if all_magic is not None:
                all_magic_values = (
                    list(all_magic)
                    if not isinstance(all_magic, (str, bytes)) and hasattr(all_magic, "__iter__")
                    else [all_magic]
                )
                warning_set = set(warnings)
                string_set = {str(s).strip() for s in strings}
                filtered_all_magic = [
                    v
                    for v in all_magic_values
                    if not (isinstance(v, str) and v in warning_set and v.strip() not in string_set)
                ]
                total_detected = self._unique_values(total_detected + filtered_all_magic)

            alg_tp, alg_fp, alg_fn, alg_p, alg_r, alg_f1 = self._calc_metrics(total_detected, ground_truth)
            unique_metrics = {
                "True Positives (TP)": alg_tp,
                "False Positives (FP)": alg_fp,
                "False Negatives (FN)": alg_fn,
                "Precision": alg_p,
                "Recall": alg_r,
                "F1 Score": alg_f1,
            }

            column_evaluations[col_name] = {
                **column_metadata,
                "Processed by Algorithm": "Yes",
                "Category": category,
                "Ground Truth Magic Numbers": ground_truth,
                "Warnings": warnings,
                "Magic Numbers String Process": strings,
                "Magic Numbers Sign Violation": sign_viol,
                "Magic Numbers Delta": distanced,
                "Total Magic": total_detected,
                "Unique-Constant Metrics": unique_metrics,
                "Row-Instance Metrics": self._row_instance_metrics(
                    resolved_col, ground_truth, detected_values=total_detected
                ),
                "Metrics": unique_metrics,
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(self.all_paper_results, outfile, indent=4, default=self._json_default)

    def save_experiment_metadata(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (set, tuple)):
            return list(value)
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

    output_dir = "Real_World_Analysis"
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "grid_search_results.json")
    metadata_path = os.path.join(output_dir, "grid_search_metadata.json")

    processor.save_results(results_path)
    processor.save_experiment_metadata(metadata_path)