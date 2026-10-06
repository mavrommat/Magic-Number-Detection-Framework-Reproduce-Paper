
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde
from pprint import pprint
from scipy.integrate import simpson
from collections import Counter
import logging
from typing import Dict, List, Any, Tuple
logging.basicConfig(level=logging.DEBUG)
from scipy.integrate import trapezoid
class DeltaMagicNumbers:

    def __init__(self,
                 Output_stabilisation: Dict[str, Any],
                 gauss_threshold: float = 0.01,
                 overlap_threshold: float = 1.0,
                 col_name: str = "Column_name",
                 plot_graphs: bool = False):
        
        self.Output_stabilisation = Output_stabilisation
        self.gauss_threshold = gauss_threshold
        self.overlap_threshold = overlap_threshold
        self.col_name = col_name
        self.plot_graphs = plot_graphs

        # Enable safeguard 
        self.data_arr = self.Active_Safeguard(self.Output_stabilisation)
    
        # Calculate the sequential distances between unique values
        self.seq_dist = self.Calc_seq_distances(self.data_arr) 

        # Create a symmetrical distribution around 0 of the sequential distances
        self.symmetrical_seq_dist = self.Create_symmetrical_seq_distances(self.seq_dist) 

        self.magic_distances = self.Calculate_magic_distances(
            self.symmetrical_seq_dist, 
            self.col_name, 
            gauss_threshold = self.gauss_threshold, 
            plot_graphs = self.plot_graphs
        )
        
        #If safeguard was enabled, we remove the imputed values before matching distances with actual values in the array
        self.data_arr = self.Remove_Safeguard_imputed_values(self.Output_stabilisation)

        # Match the identified outlier distance with actual valuew in the array
        self.matched_values = self.Match_dist_with_values(self.magic_distances, 
                                                          self.data_arr, 
                                                          self.seq_dist)

        # Create a dictionary with the candidate magic number info
        self.Candidate_magic_info_dict = self.Candidate_magic_info(self.data_arr, 
                                                                   self.matched_values, 
                                                                   self.col_name)

        # Print the candidate info 
        #self.print_candidate_info(self, self.Candidate_magic_info_dict)

        # KDE overlap analysis to classify candidates as non-extreme or extreme magic numbers
        self.Non_extreme_magic, self.Extreme_magic = self.Overlap_Cluster_Normal_distr(self.Candidate_magic_info_dict, 
                                                                                       self.data_arr, 
                                                                                       self.overlap_threshold, 
                                                                                       self.plot_graphs)


    def Active_Safeguard(self, Output_stabilisation):
        enable_process = Output_stabilisation['enabled']
        if not enable_process:
            data_arr = Output_stabilisation['original_data']
                 
        else:
            data_arr = Output_stabilisation['final_combined_data']
            
        return data_arr
    
    def Remove_Safeguard_imputed_values(self, Output_stabilisation):
        data_arr = Output_stabilisation['original_data']

        return data_arr

             
    def Calc_seq_distances(self, data_arr: np.ndarray) -> np.ndarray:
        data_arr_unique_sorted = sorted(set(data_arr))
        seq_dist = np.diff(data_arr_unique_sorted) 
        return seq_dist
    
    def Create_symmetrical_seq_distances(self, seq_dist: np.ndarray) -> np.ndarray:
        seq_dist_opp = -seq_dist
        symmetrical_seq_dist = np.concatenate((seq_dist, seq_dist_opp))
        return symmetrical_seq_dist
    
    def Calculate_magic_distances(self, symmetrical_seq_dist, col_name, gauss_threshold, plot_graphs=False) -> np.ndarray:
        symmetrical_seq_dist = np.asarray(symmetrical_seq_dist)
        mu, sigma = np.mean(symmetrical_seq_dist), np.std(symmetrical_seq_dist)
        
        if sigma == 0:
            return np.array([])

        dist = norm(loc=mu, scale=sigma)
        
        # Use gauss_threshold as a percentage of the peak density
        threshold_prob = dist.pdf(mu) * gauss_threshold #
        
        # Now find which points are in that "rare" zone
        is_outlier = dist.pdf(symmetrical_seq_dist) <= threshold_prob #
        outliers = symmetrical_seq_dist[is_outlier] #
        
        # The linspace part is ONLY needed if you want to draw the 
        # vertical dotted lines on your graph at the correct spots.
        if plot_graphs:
            threshold_values = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000) 
            threshold_points = threshold_values[np.where(dist.pdf(threshold_values) <= threshold_prob)[0]] 
            self.lower_threshold = min(threshold_points) 
            self.upper_threshold = max(threshold_points) 

        
        if plot_graphs:
            # We still calculate the normal distribution just for the visual overlay
            dist = norm(loc=mu, scale=sigma)
            
            plt.rcParams.update({
                'font.size': 14,
                'font.family': 'serif',
                'figure.figsize': (8, 6),
                'axes.linewidth': 1.5
            })
            
            # Create figure and plot
            plt.figure(figsize=(12, 7))
            n, bins, patches = plt.hist(symmetrical_seq_dist, bins=100, density=True, alpha=0.6, 
                                        color='#2E86AB', label='Mirrored Sequential Distances', 
                                        edgecolor='white', linewidth=0.5)
            
            # Add Gaussian fit overlay
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = dist.pdf(x)
            plt.plot(x, p, '#A23B72', linewidth=3, label='Gaussian fit')

            # Mark threshold boundaries (using your manual threshold)
            plt.axvline(self.lower_threshold, color='#2E86AB', linestyle=':', linewidth=2, 
                        label=f'Manual Threshold (±{gauss_threshold})')
            plt.axvline(self.upper_threshold, color='#2E86AB', linestyle=':', linewidth=2)

            # Mark outliers
            outlier_heights = dist.pdf(outliers) # Keep dots on the curve for visual consistency
            plt.scatter(outliers, outlier_heights, color="#F10101", s=50, 
                        label='Outliers', zorder=5)

            # Labels and title
            plt.title('Sequential Distances Distribution with Manual Outliers Marked', fontsize=16, fontweight='bold')
            plt.xlabel(col_name, fontweight='bold')
            plt.ylabel('Probability Density', fontweight='bold')

            # Legend and grid
            plt.legend(framealpha=0.9, edgecolor='black', loc='upper right')
            plt.grid(True, alpha=0.2, linestyle='-')

            # Clean up spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            plt.tight_layout()
            plt.show()

        # Return absolute, unique values as per your original logic
        outliers = np.unique(np.abs(outliers))
        
        return outliers
    
    # Match outlier distances with actual array values
    def Match_dist_with_values(self, outlier_distances, data_arr, seq_dist) -> list:
        # Clean and prepare data
        # Safely convert to numeric; non-numbers and None become NaN
        data_arr = pd.to_numeric(data_arr, errors='coerce')
        
        # Now np.isnan is guaranteed to work because the array is numeric
        data_arr = data_arr[~np.isnan(data_arr)]
        data_arr_unique_sorted = np.unique(data_arr)
        
        # Handle edge cases
        if len(outlier_distances) == 0 or len(data_arr_unique_sorted) <= 1:
            return [None] * len(outlier_distances) if len(outlier_distances) > 0 else [None]
        
        matched_values = []
        mu = np.median(data_arr_unique_sorted)
        
        # Use 90% of internal data for robust median (avoid edge effects)
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
    
    def Candidate_magic_info(self, data_arr, matched_values, col_name) -> Dict:
        # Clean the array like before matching
        data_arr = pd.to_numeric(data_arr, errors='coerce')
        data_arr = data_arr[~np.isnan(data_arr)]
        
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
    
    def print_candidate_info(self, results) -> None:
        for value, info in results.items():
            print(f"\nValue: {value}")
            print(f"Column: {info['col_name']}")
            print(f"Frequency: {info['frequency']}")
            print(f"Position relative to mean: {info['position']}")
            print(f"Is first value in sorted unique array: {'Yes' if info['is_first'] else 'No'}")
            print(f"Is last value in sorted unique array: {'Yes' if info['is_last'] else 'No'}")
            print("-" * 40)

    def Overlap_Cluster_Normal_distr(self, results, data_arr, overlap_threshold, plot_graphs=False) -> tuple:
        # Standardize the array cleaning process
        data_arr_clean = pd.to_numeric(data_arr, errors='coerce')
        data_arr_clean = data_arr_clean[~np.isnan(data_arr_clean)]
        
        if data_arr_clean.size == 0:
            return [], []
        
        data_arr_sorted = np.sort(data_arr_clean)
        value_counts = Counter(data_arr_sorted)
        
        magic_non_extreme = []
        magic_extreme = []
        
        for val, info in results.items():
            if val is None:
                continue

            try:
                numeric_val = float(val)
            except (ValueError, TypeError):
                continue

            # Extract info safely
            is_extreme = info.get('is_first', False) or info.get('is_last', False)
            freq = info.get('frequency', 0)

            # Handle Extreme Values
            if is_extreme:
                if freq > 1 and numeric_val not in magic_extreme:
                    magic_extreme.append(numeric_val)
                continue # Skip to next dict item since it's extreme

           
            indices = np.argwhere(np.isclose(data_arr_sorted, numeric_val)).flatten()
            if len(indices) == 0:
                continue

            position = info.get('position')
            if position not in ('left', 'right'):
                continue

            if position == 'left':
                candidate_arr = data_arr_sorted[:indices[-1]+1]
                normal_arr = data_arr_sorted[indices[-1]+1:]
            else:  # right
                candidate_arr = data_arr_sorted[indices[0]:]
                normal_arr = data_arr_sorted[:indices[0]]

            counts = [value_counts[x] for x in candidate_arr]
            
            # Only proceed if ALL values in the candidate array appear more than once.
            if not all(c > 1 for c in counts):
                continue
            else:
                overlap_percentage = self.KDE_Overlap(
                    small_array=candidate_arr,
                    big_array=normal_arr,
                    plot_graphs=plot_graphs
                )
                
                # If the overlap returned is valid and below threshold
                if overlap_percentage is not None and overlap_percentage <= overlap_threshold:
                    # Use .extend() to unpack the array into a flat list of numbers
                    magic_non_extreme.extend(np.unique(candidate_arr).tolist())


        return magic_non_extreme, magic_extreme 


    def KDE_Overlap(self, small_array, big_array, plot_graphs=False) -> float:
        small_array = np.asarray(small_array)
        big_array = np.asarray(big_array)
        
        if len(small_array) < 2 or len(big_array) < 2:
            return None 
            
        if np.std(small_array) == 0 or np.std(big_array) == 0:
            return None # Cannot build a density curve for a single recurring point
        
        # Perform KDE
        kde_small = gaussian_kde(small_array)
        kde_big = gaussian_kde(big_array)
        
        # Create a range covering both distributions
        min_val = min(small_array.min(), big_array.min())
        max_val = max(small_array.max(), big_array.max())
        
        # Original: x = np.linspace(min_val - 1, max_val + 1, 1000)
        grid_points = np.linspace(min_val - 1, max_val + 1, 5000)
        x = np.unique(np.concatenate((grid_points, small_array, big_array)))
        x.sort()
        
        small_pdf = kde_small(x)
        big_pdf = kde_big(x)
        
        # Calculate overlap
        overlap = np.minimum(small_pdf, big_pdf)

        total_area = trapezoid(overlap, x)
        #print(f"Total overlap in column:{self.col_name} is {total_area}")

        #print("total_area:", total_area)
        
        overlap_percentage = total_area * 100 
        
        if plot_graphs:
            plt.figure(figsize=(10, 6))
            
            plt.plot(x, small_pdf, color='green', label='Candidate Array')
            plt.fill_between(x, small_pdf, color='green', alpha=0.2)
            
            plt.plot(x, big_pdf, color='blue', label='Normal Array')
            plt.fill_between(x, big_pdf, color='blue', alpha=0.2)
            
            plt.fill_between(x, overlap, color='red', alpha=0.5, label='Overlap')
            
            plt.title(f'KDE Overlap: {overlap_percentage:.2f}%')
            plt.legend()
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plt.show()   
            plt.close()
            
        return overlap_percentage