import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Any

from .string_magic_numbers import magic_strings_detection as string_values_process
from .sign_violation_magic_numbers import sign_violation_magic_numbers as opposite_sign_process
from .delta_magic_numbers import DeltaMagicNumbers
from .identical_magic_numbers import identical_column_magic_numbers as all_values_are_same
from .magic_dictionaries import magic_dictionary, add_to_master_dict, safe_concatenate, clean_magic_results  
from .density_plot import plot_data_density
from .safeguard_small_tables import SmallTablesSafeguard

class MagicNumberDetector:
    def __init__(self, 
                 sign_violation_threshold: int = 3, 
                 gauss_threshold: float = 0.01, 
                 overlap_threshold: float = 1.0, 
                 plot_graphs: bool = True,
                 min_unique_values: int = 20):
      
        self.sign_threshold = sign_violation_threshold
        self.gauss_threshold = gauss_threshold
        self.overlap_threshold = overlap_threshold
        self.plot_graphs = plot_graphs
        self.min_unique_values = min_unique_values
        self.logger = logging.getLogger(__name__)

    def _preprocess_array(self, data_array: np.ndarray) -> Tuple[np.ndarray, List]:
        data_arr, magic_strings = string_values_process(data_array)
        
        mask = (data_arr != '') & (data_arr != ' ') # Remove empty strings
        
        if pd.api.types.is_numeric_dtype(data_arr):
            mask &= ~pd.isna(data_arr)
            
        filtered_arr = data_arr[mask] # Apply the mask
        
        if pd.api.types.is_numeric_dtype(filtered_arr):
            filtered_arr = np.array(filtered_arr, dtype=float)
        else:
            filtered_arr = np.array(filtered_arr)
            
        return filtered_arr, magic_strings
    
    def _generate_col_info(self, df: pd.DataFrame) -> List[List]:
        extended_col_info = []
        rows = len(df)
        
        for i, column_header in enumerate(df.columns):
            column_type = df[column_header].dtype
            
            # Cleaner pandas-native way to check for numeric vs string
            if pd.api.types.is_numeric_dtype(column_type):
                type_str = "N"
            else:
                type_str = "S" 
                
            extended_col_info.append([column_header, type_str, i, rows])
            
        return extended_col_info

    def analyze_column(self, data_array: np.ndarray, col_name: str, value_type: str) -> Dict:
        magic_strings = []
        magic_sign_violation = []
        magic_distanced_numbers = []
        all_magic_numbers = None

        if value_type != "N" or len(data_array) == 0:
            return {}

        # Process 1: Strings & Cleaning
        data_arr, magic_strings = self._preprocess_array(data_array)

        # Process 4: Identical values check
        all_magic_numbers = all_values_are_same(data_arr)

        # Check if unique values are more than 20 to have reliable statistical analysis
        len_unique_values =  len(set(data_arr))
        min_unique_values = self.min_unique_values

        if len_unique_values <= min_unique_values:
            sufficient_unique_vals = False
            magic_distanced_numbers = ["Cannot detect statistical Magic Numbers: Unique values < 20"]
        elif len_unique_values > min_unique_values:
            sufficient_unique_vals = True

        if (all_magic_numbers is None) and sufficient_unique_vals:
            # Process 2: Sign Violations
            magic_sign_violation = opposite_sign_process(data_arr, self.sign_threshold)

            """
            # Safeguard for small tables: stabilisation Process 3
            safeguard = SmallTablesSafeguard(data_arr, 
                                            min_unique_values = self.min_unique_values,
                                            per_inner_distr=90,
                                            num_imput=50)
            
            self.Output_stabilisation = safeguard.Output_stabilisation
            """      

            # Bypass SmallTablesSafeguard class output
            Output_stabilisation = {
                'enabled': False,
                'original_data': data_arr,      
                'sampled_values': [],
                'final_combined_data': data_arr  
            }
            
            # Process 3: Distance/Distribution
            delta_magic = DeltaMagicNumbers(
                Output_stabilisation,
                self.gauss_threshold, 
                self.overlap_threshold, 
                col_name,
                self.plot_graphs
            )
            
            magic_distanced_numbers = safe_concatenate([delta_magic.Non_extreme_magic,
                                                        delta_magic.Extreme_magic])
            
            if self.plot_graphs:
                plot_data_density(data_arr, col_name, self.plot_graphs)
        #else:
            #self.logger.debug(f"Skipping P2 and P3 for Column: {col_name} due to identical values.")

        return magic_dictionary(
            magic_strings, magic_sign_violation, magic_distanced_numbers, all_magic_numbers, col_name
        )

    def run_magic_detection(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        master_dict = {}
        
        extended_col_info = self._generate_col_info(df)

        for col_info in extended_col_info:
            col_name = col_info[0]
            value_type = col_info[1]
            col_idx = col_info[2]
            
            data_array = df.iloc[:, col_idx].values
            
            # Pass the extracted metadata to analysis method
            results_dict = self.analyze_column(data_array, col_name, value_type)
            
            if results_dict:
                master_dict = add_to_master_dict(master_dict, results_dict)

        cleaned_dict = clean_magic_results(master_dict)
        return master_dict, cleaned_dict