import json
from collections import Counter
from pathlib import Path
import pandas as pd

def Magic_Numbers_Patterns(
    json_filepath: str,
    sign_opt: int,
    gauss_opt: float,
    overlap_opt: float,
    export_csv: bool = True
):
    print(f"\nAnalyzing Ground Truth for Config: Sign={sign_opt}, Gauss={gauss_opt}, Overlap={overlap_opt}")

    output_dir = Path(json_filepath).parent / "Plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{json_filepath}'.")
        return

    total_columns = 0
    single_magic_count = 0
    zero_magic_count = 0
    multi_magic_count = 0
    
    all_ground_truth_values = []

    for catalog_name, runs in data.items():
        if not isinstance(runs, list): 
            continue

        for run in runs:
            p = run.get("Run_Parameters", {})
            
            # Isolate configuration
            if not (p.get("sign_violation_threshold") == sign_opt and
                    p.get("gauss_threshold") == gauss_opt and
                    p.get("overlap_threshold") == overlap_opt):
                continue

            for col_name, metrics in run.get("Column_Metrics", {}).items():
                if metrics.get("Processed by Algorithm") != "Yes":
                    continue
                
                total_columns += 1
                
                ground_truth = metrics.get("Ground Truth Magic Numbers", [])
                
                # Standardize to list for counting
                if not isinstance(ground_truth, (list, set)):
                    ground_truth = [ground_truth] if ground_truth is not None else []
                else:
                    ground_truth = list(ground_truth)

                gt_length = len(ground_truth)
                
                if gt_length == 0:
                    zero_magic_count += 1
                elif gt_length == 1:
                    single_magic_count += 1
                else:
                    multi_magic_count += 1

                all_ground_truth_values.extend(ground_truth)

    if total_columns == 0:
        print("\nError: No valid columns found for this configuration.")
        return

    pct_single = (single_magic_count / total_columns) * 100
    pct_zero = (zero_magic_count / total_columns) * 100
    pct_multi = (multi_magic_count / total_columns) * 100

    counter = Counter(all_ground_truth_values)

    # Report
    report = f"""
    {'=' * 70}
    GROUND TRUTH ANALYSIS
    {'=' * 70}
    Total Columns Evaluated : {total_columns}

    Magic Number Presence per Column (from ground truth):
    0 Magic Numbers         : {zero_magic_count} columns ({pct_zero:.1f}%)
    Exactly 1 Magic Number  : {single_magic_count} columns ({pct_single:.1f}%)
    > 1 Magic Numbers       : {multi_magic_count} columns ({pct_multi:.1f}%)
    {'-' * 70}
    >= 1 Magic Numbers       : {single_magic_count + multi_magic_count} columns ({pct_single + pct_multi:.1f}%)


    {'=' * 70}
    MOST COMMON GROUND MAGIC VALUES
    {'=' * 70}
    """
    print(report)
    print(f"{'Magic Value':<20} | {'Dataset Frequency':<15}")
    print("-" * 40)
    
    export_data = []
    for magic_val, count in counter.most_common():
        print(f"{str(magic_val):<20} | {count:<15}")
        export_data.append({"Magic Value": magic_val, "Frequency": count})

    if export_csv and export_data:
        df_export = pd.DataFrame(export_data)
        csv_path = output_dir / f"Ground_Truth_Frequencies_sign{sign_opt}_gauss{gauss_opt}_overlap{overlap_opt}.csv"
        df_export.to_csv(csv_path, index=False)
        print(f"\nSaved frequencies to: {csv_path}")

if __name__ == "__main__":
    Magic_Numbers_Patterns(
        json_filepath="grid_search_results.json",
        sign_opt=1,       
        gauss_opt=0.1,
        overlap_opt=0.1
    )