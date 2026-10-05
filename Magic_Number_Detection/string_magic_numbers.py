import pandas as pd
import numpy as np

def string_magic_values(data_array):
    # Check if there are any string values in the array (including np.str_)
    has_strings = any(isinstance(item, (str, np.str_)) for item in data_array)

    string_values = []
    filtered_array = []

    if not has_strings:
        # If no strings, returns an empty list
        return data_array, string_values

    else:
        for item in data_array:
            if isinstance(item, (str, np.str_)):
                # Try to convert string to numeric value
                try:
                    numeric_value = float(item)  # Convert to float (handles both int and float strings)
                    filtered_array.append(numeric_value)
                except ValueError:
                    string_values.append(item)
            elif isinstance(item, (int, float, np.number)):
                # If it's already a numeric type, add to filtered_array
                filtered_array.append(item)

        filtered_array = np.array(filtered_array)
        string_values = np.array(string_values)

        return filtered_array, string_values


def magic_strings_detection(data_array):
    data_arr, magic_strings = string_magic_values(data_array) 
    magic_strings = list(set(magic_strings))

    return data_arr, magic_strings 

