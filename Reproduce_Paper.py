import pandas as pd
import numpy as np
import random
from random import seed
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
import os
from pathlib import Path
import ANOVA
import GridVisualisation as gv
from ParameterGrid import ParameterGrid
from generate_shap_and_surface_plots import generate_all_plots
from ANOVA import run_n_way_anova
from Scaleability import run_scalability_benchmark
from PaperWorkflowMagicNumbers import run_real_world_analysis 
from json_processing import ModelEvaluator
from RW_hyperparameter_sensitivity import Hyperparameter_Sensitivity
from Magic_Numbers_Patterns import Magic_Numbers_Patterns

logging.basicConfig(level=logging.DEBUG)

def run_synthetic_data_grid_search_2_MN():
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
            "Paired": True, # Set to False == 1 magic number injection, True == 2 magic number injection
            "sign_violation_threshold": 3
        }
    
        # 4. Execute the Grid Search (10,000 iterations)
        grid_runner = ParameterGrid(
            variables=framework_variables, 
            param_grid=hyperparameter_spaces, 
            total_runs=100000
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
                "contamination_rate": run.get("contamination_rate", 0.05), # <-- Ensure this is exported
                "injected_magic_numbers": str(run["injected_magic_numbers"]),
                "quantities": str(run["quantities"])
            }
            flat_row.update(run["metrics"])
            csv_ready_results.append(flat_row)

        # 1. Create the Data directory
        data_dir = os.path.join("Synthetic Experiments", "Data")
        os.makedirs(data_dir, exist_ok=True)
        
        # 2. Assign the full path to output_filename
        output_filename = os.path.join(data_dir, "2_magic_numbers_results.csv")
        
        # 3. Export the dataframe
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

def run_synthetic_data_grid_search_1_MN():
        
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
            "Paired": False, # Set to False == 1 magic number injection, True == 2 magic number injection
            "sign_violation_threshold": 3
        }
    
        # 4. Execute the Grid Search (10,000 iterations)
        grid_runner = ParameterGrid(
            variables=framework_variables, 
            param_grid=hyperparameter_spaces, 
            total_runs=100000
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

        # 1. Create the Data directory
        data_dir = os.path.join("Synthetic Experiments", "Data")
        os.makedirs(data_dir, exist_ok=True)
        
        # 2. Assign the full path to output_filename
        output_filename = os.path.join(data_dir, "1_magic_numbers_results.csv")
        
        # 3. Export the dataframe
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

def run_json_processing_plots():
    input_file = Path("Real_World_Analysis") / "grid_search_results.json"
    
    evaluator = ModelEvaluator(
        json_filepath=str(input_file),
        sign_opt=3,
        gauss_opt=0.1,
        overlap_opt=0.1,
        plots_dir=str(Path("Real_World_Analysis") / "Plots"),
        data_dir=str(Path("Real_World_Analysis") / "data")
    )
    
    results = evaluator.run_pipeline()
    

if __name__ == "__main__":

    RUN_SYNTHETIC_ANALYSIS = True
    RUN_REAL_WORLD_ANALYSIS = False

    #==============================
    # Run Sythetic Experiments 
    #==============================
    if RUN_SYNTHETIC_ANALYSIS:
        # Run the grid search for 2 magic numbers
        run_synthetic_data_grid_search_2_MN()
        
        # Run the grid search for 1 magic number
        run_synthetic_data_grid_search_1_MN()

        # Generate the SHAP and marginal response plots
        print("\nInitiating statistical plotting...")
        #generate_all_plots()

        print("\nInitiating N-Way ANOVA Analysis...")
        #data_dir = os.path.join("Synthetic Experiments", "Data")
        #run_n_way_anova(os.path.join(data_dir, "1_magic_numbers_results.csv"), is_dual=False)
        #run_n_way_anova(os.path.join(data_dir, "2_magic_numbers_results.csv"), is_dual=True)

        print("\nInitiating Scalability Benchmark...")
        #run_scalability_benchmark()

    #==============================
    # Run Real-World Analysis 
    #==============================
    if RUN_REAL_WORLD_ANALYSIS:
        print("\nInitiating Real-World Data Analysis...")
        run_real_world_analysis()

        print("\nInitiating JSON Processing and Plot Generation...")
        run_json_processing_plots()

        json_path = Path("Real_World_Analysis") / "grid_search_results.json"

        print("\nInitiating Hyperparameter Sensitivity Analysis...")    
        Hyperparameter_Sensitivity(str(json_path))

        print("\nInitiating Common Magic Number Detection Pattersns Analysis...")
        Magic_Numbers_Patterns(
                json_filepath=str(json_path),
                sign_opt=3,       
                gauss_opt=0.1,
                overlap_opt=0.1
            )



