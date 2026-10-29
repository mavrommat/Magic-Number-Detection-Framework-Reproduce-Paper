import pandas as pd
import numpy as np

import numpy as np

def sign_violation_magic_numbers(data_arr, sign_violation_threshold=3):
    k = sign_violation_threshold

    unique_vals, counts = np.unique(data_arr, return_counts=True)

    # Enough values guard
    if len(unique_vals) <= 2 * k:
        return []

    count_map = dict(zip(unique_vals, counts))

    # Separate the tails and the inner core
    lower_tail = unique_vals[:k]
    upper_tail = unique_vals[-k:]
    inner_core = unique_vals[k:-k]

    sign_violating_magic_numbers = []

    # Check for negative magic numbers in a non-negative (>= 0) core distribution
    if np.all(inner_core >= 0):
        sign_violating_magic_numbers.extend([
            val for val in lower_tail 
            if val < 0 and count_map[val] > 1
        ])
    # Check for positive magic numbers in a non-positive (<= 0) core distribution
    elif np.all(inner_core <= 0):
        sign_violating_magic_numbers.extend([
            val for val in upper_tail 
            if val > 0 and count_map[val] > 1
        ])

    return sign_violating_magic_numbers