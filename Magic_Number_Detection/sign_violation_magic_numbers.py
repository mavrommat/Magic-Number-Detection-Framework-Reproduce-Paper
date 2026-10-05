import pandas as pd
import numpy as np

def sign_violation_magic_numbers(data_arr, sign_violation_theshold=3):
    threshold = sign_violation_theshold

    unique_vals, counts = np.unique(data_arr, return_counts=True)
    
    count_map = dict(zip(unique_vals, counts))

    data_arr_unique_sorted = np.sort(unique_vals)

    # identify the bulk of the data vs the extremes
    pos_data_arr_unique_sorted = data_arr_unique_sorted[threshold:]
    neg_data_arr_unique_sorted = data_arr_unique_sorted[:-threshold]

    # candidates for output
    test_values = list(data_arr_unique_sorted[:threshold]) + list(data_arr_unique_sorted[-threshold:])

    sign_violating_magic_numbers = []

    # Check for sign violations, ensure the count > 1
    if all(x > 0 for x in pos_data_arr_unique_sorted):
        sign_violating_magic_numbers.extend([
            val for val in test_values 
            if val < 0 and count_map[val] > 1
        ])
    elif all(x < 0 for x in neg_data_arr_unique_sorted):
        sign_violating_magic_numbers.extend([
            val for val in test_values 
            if val > 0 and count_map[val] > 1
        ])
        
    return sign_violating_magic_numbers