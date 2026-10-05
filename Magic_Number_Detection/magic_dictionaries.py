import pandas as pd
import numpy as np

def magic_dictionary(magic_strings, magic_sign_violation, magic_distanced_numbers, all_magic_numbers, col_name):
    
    return {
        'column_name': col_name,
        'magic_strings': magic_strings,
        'magic_sign_violation': magic_sign_violation,
        'magic_distanced_numbers': magic_distanced_numbers,
        'all_magic_numbers': all_magic_numbers
    }

def add_to_master_dict(master_dict, results_dict):
    col_name = results_dict['column_name']
    master_dict[col_name] = {
        'magic_strings': results_dict['magic_strings'],
        'magic_sign_violation': results_dict['magic_sign_violation'],
        'magic_distanced_numbers': results_dict['magic_distanced_numbers'],
        'all_magic_numbers': results_dict['all_magic_numbers']
    }
    return master_dict

def safe_concatenate(nested_arrays):
    valid_arrays = []
    
    for arr in nested_arrays:
        if arr is None:
            continue
            
        if isinstance(arr, (np.number, int, float, np.float64, np.int64)):
            valid_arrays.append(np.array([arr]))
            
        elif hasattr(arr, '__len__') and len(arr) > 0:
            arr_np = np.array(arr)
            valid_arrays.append(arr_np.ravel())
            
        elif hasattr(arr, '__len__') and len(arr) == 0:
            continue
    
    if len(valid_arrays) == 0:
        return np.array([])
    
    return np.concatenate(valid_arrays)


def clean_magic_results(master_dict):
    cleaned_dict = {}
    
    for col_name, results in master_dict.items():
        # Check if all_magic_numbers has a valid value 
        all_magic = results.get("all_magic_numbers")
        
        if all_magic is not None and all_magic is not False:
            # If a primary magic number exists, use it
            #magic_results = all_magic
            continue
        else:
            # Combine secondary magic indicators
            magic_strings = [val for val in results.get("magic_strings", []) if val not in [None, '']]
            
            # Handle potential numpy arrays in magic_distanced_numbers
            distanced = results.get("magic_distanced_numbers", [])
            distanced_list = distanced.tolist() if isinstance(distanced, np.ndarray) else distanced
            
            sign_violations = results.get("magic_sign_violation", [])

            # Concatenate and find unique values
            total_magic = np.concatenate([magic_strings, distanced_list, sign_violations]) 
            unique_magic = np.unique(total_magic) if total_magic.size > 0 else np.array([])

            magic_results = unique_magic.tolist()
    
        cleaned_dict[col_name] = magic_results

    return cleaned_dict