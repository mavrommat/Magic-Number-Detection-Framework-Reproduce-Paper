import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from StringValDetector import magic_strings_detection as string_values_process

class BaselineEvaluator:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df

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
    
    def analyze_column(self, data_array: np.ndarray, col_name: str, value_type: str) -> Dict:
        self.magic_strings = []
       
        if value_type != "N" or len(data_array) == 0:
            return {}

        # stage 1: Strings / Cleaning
        self.data_arr, self.magic_strings = self._preprocess_array(data_array)

        if len(self.data_arr) == 0:
            return {}

        z_outliers = []
        iqr_outliers = []

        # Z-Score Baseline (3 STD) 
        mean_val = np.mean(self.data_arr)
        std_val = np.std(self.data_arr)
        
        # Prevent infinity x/0 
        if std_val > 0:
            z_scores = np.abs((self.data_arr - mean_val) / std_val)
            z_outliers = np.unique(self.data_arr[z_scores > 3]).tolist()

        # IQR Baseline (1.5x Interquartile Range) ---
        q1 = np.percentile(self.data_arr, 25)
        q3 = np.percentile(self.data_arr, 75)
        iqr = q3 - q1
        
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_outliers = np.unique(self.data_arr[(self.data_arr < lower_bound) | (self.data_arr > upper_bound)]).tolist()

        return {
            col_name: {
                'z_score_outliers': z_outliers,
                'iqr_outliers': iqr_outliers
            }
        }

    def _generate_col_info(self, df: pd.DataFrame) -> List[List]:
        extended_col_info = []
        rows = len(df)
        
        for i, column_header in enumerate(df.columns):
            column_type = df[column_header].dtype
            
            # check for numeric vs string
            if pd.api.types.is_numeric_dtype(column_type):
                type_str = "N"
            else:
                type_str = "S" 
                
            extended_col_info.append([column_header, type_str, i, rows])
            
        return extended_col_info
    
    def run_baseline_detection(self) -> Dict:
        master_dict = {}
        
        extended_col_info = self._generate_col_info(self.df)

        for col_info in extended_col_info:
            col_name = col_info[0]
            value_type = col_info[1]
            col_idx = col_info[2]
            
            data_array = self.df.iloc[:, col_idx].values
            
            # Pass metadata to analysis 
            results_dict = self.analyze_column(data_array, col_name, value_type)
            
            # If valid numeric data was processed adds it to the master dict
            if results_dict:   
                master_dict.update(results_dict)

        return master_dict