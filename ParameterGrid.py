import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import logging
logging.basicConfig(level=logging.DEBUG)
from random import seed
from SyntheticDataGenerator import SyntheticData
from Magic_Number_Detection.get_magic_numbers import MagicNumberDetector

import GridVisualisation as gv

class ParameterGrid:
    def __init__(self, variables, param_grid, total_runs=10000):
        self.variables = variables
        self.param_grid = param_grid
        self.total_runs = total_runs
        self.all_results = []

    def enhanced_validation(self, master_dict, expected_magic_numbers, tol=1e-6):
        col_results = master_dict.get('Synthetic_Data', {})
        
        # Extract detections safely 
        dist = col_results.get('magic_distanced_numbers', [])
        opp = col_results.get('magic_sign_violation', [])
        all_mag = col_results.get('all_magic_numbers', None)

        # Clean strings
        clean_dist = [x for x in dist if isinstance(x, (int, float, np.number))] if dist is not None else []
        clean_opp = [x for x in opp if isinstance(x, (int, float, np.number))] if opp is not None else []
        clean_all = [all_mag] if isinstance(all_mag, (int, float, np.number)) else []

        # Normalize to numpy arrays
        dist_arr = np.array(clean_dist, dtype=float)
        opp_arr = np.array(clean_opp, dtype=float)
        all_arr = np.array(clean_all, dtype=float)

        # Merge all three detection pipelines
        detected = np.unique(np.concatenate([dist_arr, opp_arr, all_arr]))
        expected = np.array(expected_magic_numbers, dtype=float)
        
        # Classification metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Count matches (True Positives and False Negatives)
        for exp_num in expected:
            if any(abs(exp_num - det) < tol for det in detected):
                true_positives += 1
            else:
                false_negatives += 1
        
        # Count False Positives
        for det_num in detected:
            if not any(abs(det_num - exp) < tol for exp in expected):
                false_positives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Edge case: If ground truth is empty and we detected nothing, F1 = 1.0
        if len(expected) == 0 and len(detected) == 0:
            f1_score = 1.0
            precision = 1.0
            recall = 1.0

        return {
            'true_positives': true_positives,
            'false_positives': false_positives, 
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'detected_count': len(detected),
            'expected_count': len(expected)
        }

    def run_sweep(self):
        pbar = tqdm(total=self.total_runs, desc="Pure Random Grid Search")

        # Every run generates a unique base dataset AND unique detector thresholds
        for _ in range(self.total_runs):
            
            # Randomly sample ALL parameters for this specific iteration
            std = random.choice(self.param_grid["sigmas"])
            n_samples = random.choice(self.param_grid["num_samples"])
            gauss_thresh = random.choice(self.param_grid["gauss_thresholds"])
            overlap_thresh = random.choice(self.param_grid["overlap_thresholds"])
            m_num_dist = random.choice(self.param_grid["center_distances"])
            gap_dist = random.choice(self.param_grid["gap_distances"])
            contam_rate = random.uniform(0.01, 0.20)

            # Generate Data
            synth = SyntheticData(data_type="log_normal", variables={"std": std, "num_samples": n_samples})
            synth.generate_data()
            synth.clipping_data(SIGMA_LIMIT=3.0)

            # Configure paired or unpaired 
            if self.variables.get("Paired") == True:
                m_nums = [m_num_dist, m_num_dist + gap_dist]
                shared_quant = max(1, int(n_samples * contam_rate))
                quants = [shared_quant, shared_quant]
            else:
                gap_dist = None
                m_nums = [m_num_dist]
                quants = [max(1, int(n_samples * contam_rate))]

            # Inject into the fresh baseline
            synth.Magic_Number_Ingestion(m_nums, quants)
            df_ingested = synth.create_dataframe()

            # Initialize detector with current random hyperparameters
            detector = MagicNumberDetector(
                sign_violation_threshold=self.variables.get("sign_violation_threshold", 3),
                gauss_threshold=gauss_thresh,
                overlap_threshold=overlap_thresh,
                plot_graphs=False,
                min_unique_values=2 # minimum unique values threshold set to a minimum of 2 for testing
            )
            
            master_results, cleaned_results = detector.run_magic_detection(df_ingested)
            
            validation_metrics = self.enhanced_validation(master_results, m_nums)

            self.all_results.append({
                "std": std,
                "num_samples": n_samples,
                "gauss_threshold": gauss_thresh,
                "overlap_threshold": overlap_thresh,
                "center_distance": m_num_dist,   
                "gap_distance": gap_dist,
                "contamination_rate": contam_rate,  
                "injected_magic_numbers": m_nums,
                "quantities": quants,
                "metrics": validation_metrics
            })
            
            pbar.update(1)
        
        pbar.close()
        return self.all_results
    

if __name__ == "__main__":
    # Lock the seed for reproducibility
    seed(42)
    np.random.seed(42)

    # Define the parameter spaces
    hyperparameter_spaces = {
        "sigmas": np.linspace(0.01, 2.5, 1000),
        "gauss_thresholds": np.linspace(0.001, 1.0, 1000),
        "overlap_thresholds": np.linspace(0.001, 100.0, 1000),
        "center_distances": np.linspace(0.00, 100, 10000), 
        "gap_distances": np.linspace(0.01, 100.0, 1000),
        "num_samples": np.linspace(1, 200.0, 1000).astype(int),
        "contamination_rates": [0.05, 0.025] # 5% for first MN, 2.5% for second
    }

    # Define state variables
    framework_variables = {
        "Paired": True,
        "sign_violation_threshold": 3
    }

    # 4. Execute the Grid Search 
    grid_runner = ParameterGrid(
        variables=framework_variables, 
        param_grid=hyperparameter_spaces, 
        total_runs=100000 # (10,000 iterations)
    )
    
    print(f"Initiating Flat Random Grid Search (100,000 unique data & detector combinations)...")
    final_results = grid_runner.run_sweep()

    print("Flattening results for CSV export...")
    csv_ready_results = []
    
    for run in final_results:
        flat_row = {
            "std": run["std"],
            "num_samples": run["num_samples"],
            "gauss_threshold": run["gauss_threshold"],
            "overlap_threshold": run["overlap_threshold"],
            "center_distance": run["center_distance"], 
            "gap_distance": run["gap_distance"],
            "contamination_rate": run.get("contamination_rate", 0.05), 
            "injected_magic_numbers": str(run["injected_magic_numbers"]),
            "quantities": str(run["quantities"])
        }
        flat_row.update(run["metrics"])
        csv_ready_results.append(flat_row)

    output_filename = "2_magic_numbers_results.csv"
    df_export = pd.DataFrame(csv_ready_results)
    df_export.to_csv(output_filename, index=False)
        
    print(f"\nGrid search complete. Flattened metrics successfully saved to {output_filename}")

    # Generate corner plots 
    print("Initiating automatic corner plot generation...")
    gv.plot_corner_metric_continuous(
        df=df_export, 
        file_name=f"Monte_Carlo_Grid_{df_export['expected_count'].iloc[0]}_MN", 
        metrics=['f1_score', 'recall', 'precision'], 
        #metrics=['f1_score'],
        f1_threshold=0.95, 
        dpi=300, 
        save_fig=True
    )                    
    
    # Generate standalone KDE plots 
    print("Initiating automatic standalone KDE generation...")
    gv.plot_standalone_kdes(csv_path=output_filename)