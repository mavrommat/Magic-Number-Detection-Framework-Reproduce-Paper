import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde
from pprint import pprint
from scipy.integrate import simpson
from collections import Counter
import logging
logging.basicConfig(level=logging.DEBUG)

# Create sequential distances from unique sorted values
def sequential_distances(data_arr):
    data_arr_unique_sorted = np.unique(data_arr)
    seq_dist = np.diff(data_arr_unique_sorted) 
    
    return seq_dist

# Create symmetrical array of sequential distances around zero
def symmetrical_sequential_distances(seq_dist):
    seq_dist_opp = -seq_dist
    symmetrical_seq_dist = np.concatenate((seq_dist, seq_dist_opp))

    return symmetrical_seq_dist

# Find outlier distances based on Gaussian distribution and a threshold 
def distances_outliers_plot(values_dist, col_name, gauss_threshold = 0.01, plot_graphs = True):
    mu, sigma = np.mean(values_dist), np.std(values_dist)  # Calculates mean and standard deviation

    dist = norm(loc=mu, scale=sigma) # Creates the normal distribution

    probabilities = dist.pdf(values_dist) # Calculates probabilities for each value

    threshold_prob = dist.pdf(mu) * gauss_threshold  # threshold for % probability 

    # Identify outliers (values with probability <= threshold)
    is_outlier = probabilities <= threshold_prob
    outliers = values_dist[is_outlier]
    
    # Find the threshold values where probability = ( treshold * 100 ) %
    threshold_values = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
    threshold_points = threshold_values[np.where(dist.pdf(threshold_values) <= threshold_prob)[0]]
    lower_threshold = min(threshold_points)
    upper_threshold = max(threshold_points)
    
    if plot_graphs == True:
        # Plot histogram
        plt.figure(figsize=(12, 7))
        n, bins, patches = plt.hist(values_dist, bins=100, density=True, alpha=0.6, color='#2E86AB', label='Mirroed Sequential Distances', edgecolor='white', linewidth=0.5)
        
        # Professional settings
        plt.rcParams.update({
            'font.size': 14,
            'font.family': 'serif',
            'figure.figsize': (8, 6),
            'axes.linewidth': 1.5
        })

        # Add Gaussian fit
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = dist.pdf(x)
        plt.plot(x, p, '#A23B72', linewidth=3, label='Gaussian fit')

        # Mark threshold boundaries with dotted lines
        plt.axvline(lower_threshold, color='#2E86AB', linestyle=':', linewidth=2, 
                    label='1% probability threshold')
        plt.axvline(upper_threshold, color='#2E86AB', linestyle=':', linewidth=2)

        # Mark outliers with red dots
        outlier_heights = dist.pdf(outliers)  # Height for dots on PDF curve
        plt.scatter(outliers, outlier_heights, color="#F10101", s=50, 
                label='Outliers (≤1% prob)', zorder=5)

        # Professional labels and title
        plt.title('Sequential Distances Distribution with Outliers Marked', fontsize=16, fontweight='bold')
        plt.xlabel(col_name, fontweight='bold')
        plt.ylabel('Probability Density', fontweight='bold')

        # Legend and grid
        plt.legend(framealpha=0.9, edgecolor='black', loc='upper right')
        plt.grid(True, alpha=0.2, linestyle='-')

        # Remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

    # Print outliers
    #print(f"Outliers (≤1% probability, beyond [{lower_threshold:.2f}, {upper_threshold:.2f}]):")
    outliers = np.unique(np.abs(outliers))
    #print(f"Outlier distances: {outliers}")


    return outliers


# Match outlier distances with actual array values
def matching_outlier_dist_with_array_values(outlier_distances, data_arr, seq_dist):
    # Clean and prepare data
    data_arr = np.asarray(data_arr)
    data_arr = data_arr[~np.isnan(data_arr)]
    data_arr_unique_sorted = np.unique(data_arr)
    
    # Handle edge cases
    if len(outlier_distances) == 0 or len(data_arr_unique_sorted) <= 1:
        return [None] * len(outlier_distances) if len(outlier_distances) > 0 else [None]
    
    matched_values = []
    mu = np.median(data_arr_unique_sorted)
    
    # Use middle 90% of data for more robust median (avoid edge effects)
    n = len(data_arr_unique_sorted)
    mu_d = np.median(data_arr_unique_sorted[int(n*0.05):int(n*0.95)])
    
    for dist in outlier_distances:
        found = False
        dist = float(dist)  # Ensure distance is a float
        
        # Find closest index in seq_dist
        closest_idx = np.argmin(np.abs(seq_dist - dist))
        
        if np.isclose(dist, seq_dist[closest_idx], atol=1e-5):
            i = int(closest_idx)  # Ensure integer index
            
            # Safely get neighboring values
            val_left = data_arr_unique_sorted[min(i, len(data_arr_unique_sorted)-1)]
            val_right = data_arr_unique_sorted[min(i+1, len(data_arr_unique_sorted)-1)]
            
            # Determine outlier value based on median position
            if (mu > val_left) and (mu > val_right):
                outlier_value = val_left
            elif (mu < val_left) and (mu < val_right):
                outlier_value = val_right
            else:
                # Use the more robust median for ambiguous cases
                if (mu_d > val_left) and (mu_d > val_right):
                    outlier_value = val_left
                elif (mu_d < val_left) and (mu_d < val_right):
                    outlier_value = val_right
                else:
                    outlier_value = None
            
            matched_values.append(outlier_value)
            found = True
        
        if not found:
            matched_values.append(None)
    
    #print(f"Matched {len([x for x in matched_values if x is not None])} outliers")
    return matched_values


def candidate_numbers_info(data_arr, matched_values, col_name):
    mu = np.mean(data_arr)
    data_arr_unique_sorted = np.unique(data_arr)

    results = {}

    for out_val in matched_values:
        if out_val not in data_arr_unique_sorted:
            results[out_val] = {
                'col_name': col_name,
                'frequency': 0,
                'position': None,
                'is_first': False,
                'is_last': False
            }
            continue
            
        position = 'left' if out_val < mu else 'right'
        
        # Calculate actual frequency in original array
        frequency = np.count_nonzero(data_arr == out_val)
        
        # Check position in unique sorted array
        is_first = (data_arr_unique_sorted[0] == out_val)
        is_last = (data_arr_unique_sorted[-1] == out_val)
        
        results[out_val] = {
            'col_name': col_name,
            'frequency': frequency,
            'position': position,
            'is_first': is_first,
            'is_last': is_last
        }
        
    return results

def print_candidate_info(results):

    for value, info in results.items():
        print(f"\nValue: {value}")
        print(f"Column: {info['col_name']}")
        print(f"Frequency: {info['frequency']}")
        print(f"Position relative to mean: {info['position']}")
        print(f"Is first value in sorted unique array: {'Yes' if info['is_first'] else 'No'}")
        print(f"Is last value in sorted unique array: {'Yes' if info['is_last'] else 'No'}")
        print("-" * 40)
        
def overlap_non_extreme_values(results, data_arr, overlap_threshold, plot_graphs=True):
    # Convert the data array to floats 
    data_arr_clean = np.array([float(x) for x in data_arr if x is not None and isinstance(x, (int, float, np.number))])
    if data_arr_clean.size == 0:
        return [], []
    
    data_arr_sorted = np.sort(data_arr_clean)
    from collections import Counter
    value_counts = Counter(data_arr_sorted)
    
    magic_non_extreme = []
    magic_extreme = []

    # Safely unpack the dictionary
    for val, info in results.items():
        # Skip None keys that were identified in debug log
        if val is None:
            continue

        # Ensures the key 'val' is treated as a float to match data_arr_sorted
        try:
            numeric_val = float(val)
        except (ValueError, TypeError):
            continue

        # Extract info with defaults to prevent KeyErrors
        is_extreme = info.get('is_first', False) or info.get('is_last', False)
        freq = info.get('frequency', 0)

        # Handle Extreme Values immediately
        if is_extreme:
            if freq > 1:
                if numeric_val not in magic_extreme:
                    magic_extreme.append(numeric_val)
                    #print(f" Extreme magic number added: {numeric_val}")
            continue # Move to the next value in the dictionary

        # Handles Non-extreme values
        if not (results[val]['is_first'] or results[val]['is_last']):
            indices = np.argwhere(np.isclose(data_arr_sorted, val)).flatten()
            if len(indices)==0:
                continue

            position = results[val]['position']
            if position not in ('left', 'right'):
                continue

            # Common processing for both left and right cases
            if position == 'left':
                candidate_arr = data_arr_sorted[:indices[-1]+1]
                normal_arr = data_arr_sorted[indices[-1]+1:]
                #print("Ok debug after overlap_non_extreme_values\\")

            else:  # right
                candidate_arr = data_arr_sorted[indices[0]:]
                normal_arr = data_arr_sorted[:indices[0]]
                #print("Ok debug after overlap_non_extreme_values\\")


            counts = [value_counts[x] for x in candidate_arr]
            if all(c == 1 for c in counts): # Check if all values appear exactly once
                #print("All candidate values appear exactly once in the dataset so none can be magic")
                continue
            elif all(c > 1 for c in counts): # Check if all values appear more than   . once
                # Calculate overlap (note: using kde_overlapping consistently)
                #print("All candidate values appear more than once in the dataset so all are magic")
                overlap_percentage = kde_overlapping(
                    small_array=candidate_arr,
                    big_array=normal_arr,
                    plot_graphs = plot_graphs
                )
                #print(f"Overlap percentage ({position}): {overlap_percentage:.2f}%")
            
                if overlap_percentage <= overlap_threshold:
                    magic_non_extreme.append(np.unique(candidate_arr))

    return magic_non_extreme, magic_extreme 

def kde_overlapping(small_array, big_array, plot_graphs = True):
    # Convert both to numpy arrays
    small_array = np.asarray(small_array)
    big_array = np.asarray(big_array)
    
    # Perform KDE for both arrays
    kde_small = gaussian_kde(small_array)
    kde_big = gaussian_kde(big_array)
    
    # Create a range of values covering both distributions
    min_val = min(small_array.min(), big_array.min())
    max_val = max(small_array.max(), big_array.max())
    x = np.linspace(min_val - 1, max_val + 1, 1000)
    
    # Evaluate KDEs
    small_pdf = kde_small(x)
    big_pdf = kde_big(x)
    
    # Calculate overlap percentage
    # The overlap is the minimum of the two PDFs at each point
    overlap = np.minimum(small_pdf, big_pdf)
    total_area = np.trapz(overlap, x)
    
    # Normalize by the average of the two individual areas
    area_small = np.trapz(small_pdf, x)
    area_big = np.trapz(big_pdf, x)
    overlap_percentage = (total_area / ((area_small + area_big) / 2)) * 100

    #print(f"Overlap area: {total_area}, Area small: {area_small}, Area big: {area_big}, Overlap %: {overlap_percentage}")
    
    if plot_graphs == True:
        # Plot the distributions
        plt.figure(figsize=(10, 6))
        
        # Plot small array (green)
        plt.plot(x, small_pdf, color='green', label='Small Array')
        plt.fill_between(x, small_pdf, color='green', alpha=0.2)
        
        # Plot big array (blue)
        plt.plot(x, big_pdf, color='blue', label='Big Array')
        plt.fill_between(x, big_pdf, color='blue', alpha=0.2)
        
        # Plot overlap (red)
        plt.fill_between(x, overlap, color='red', alpha=0.5, label='Overlap')
        
        plt.title(f'KDE Overlap: {overlap_percentage:.2f}%')
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    return overlap_percentage

def delta_distributed_magic_numbers(data_arr, col_name, gauss_threshold = 0.01,  overlap_threshold=5.0, plot_graphs = True):
    delta_magic_numbers = []
    
    seq_dist = sequential_distances(data_arr)
    symmetrical_seq_dist = symmetrical_sequential_distances(seq_dist)

    outlier_distances = distances_outliers_plot(symmetrical_seq_dist, col_name, gauss_threshold = 0.01, plot_graphs = plot_graphs)
    #print(f"Outlier distances identified: {outlier_distances}")

    matched_values = matching_outlier_dist_with_array_values(outlier_distances, data_arr, seq_dist)
    #print(f"Matched outlier values: {matched_values}")

    results = candidate_numbers_info(data_arr, matched_values, col_name)

    #print_candidate_info(results) # Visual Representation of the results.

    #print(f"Type of keys in results: {type(list(results.keys())[0])}")
    #print(f"Sample info for 99999: {results.get(99999) or results.get('99999')}")
    
    magic_non_extreme, magic_extreme = overlap_non_extreme_values(results, data_arr,  overlap_threshold=5.0, plot_graphs = plot_graphs)
    #print("Ok debug after overlap_non_extreme_values")

    delta_magic_numbers = magic_non_extreme + magic_extreme


    return delta_magic_numbers

    