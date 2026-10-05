import numpy as np
from scipy.stats import gaussian_kde

class SmallTablesSafeguard:
    
    def __init__(self, data_arr, min_unique_values, per_inner_distr, num_imput):
        self.data_arr = data_arr
        self.min_unique_values = min_unique_values
        self.per_inner_distr = per_inner_distr
        self.num_imput = num_imput
        
        self.sampled_values = []

        self.enable_process = self.decide_if_safeguard_needed()
        
        if self.enable_process:
            self.sampled_values = self.kde_sampling()
            combined_list = self.data_arr.tolist() 
            combined_list.extend(self.sampled_values)
            self.combined_list = combined_list
            
        else: 
            self.combined_list = self.data_arr.copy()
            
        self.Output_stabilisation = {
            'enabled': self.enable_process,
            'original_data': self.data_arr,
            'sampled_values': self.sampled_values,
            'final_combined_data': self.combined_list
        }

    def decide_if_safeguard_needed(self):
        unique_values = set(self.data_arr)
        
        if len(unique_values) <= self.min_unique_values:
            return True
        else:
            return False

    def kde_sampling(self):
        data_arr_sorted = sorted(self.data_arr)
        unique_data_set = set(data_arr_sorted) # Set O(1) 

        # Percentile calculations
        excluded_distr = 100 - self.per_inner_distr
        left_percentage = excluded_distr / 2
        right_percentage = 100 - (excluded_distr / 2)

        lower_bound = np.percentile(data_arr_sorted, left_percentage)
        upper_bound = np.percentile(data_arr_sorted, right_percentage)

        filtered_list = [x for x in data_arr_sorted if lower_bound <= x <= upper_bound]
        
        # variance check (accounts for microscopic floating-point noise)
        if len(filtered_list) < 2 or np.std(filtered_list) < 1e-8:
            print("Warning: Filtered list has effectively zero variance. Skipping KDE sampling.")
            return [] 

        # safeguard: Try/Except block
        try:
            kde = gaussian_kde(filtered_list) # Fit a KDE
        except Exception as e:
            # If scipy throws a LinAlgError (or any other error), we catch it and move on safely
            print(f"Warning: KDE failed due to singular matrix (too little variance). Skipping. Details: {e}")
            return []
            
        sampled_values = []
        n_samples = 0

        while n_samples < self.num_imput:
            sampled_value = kde.resample(1)[0][0] 
            
            if sampled_value not in unique_data_set:
                sampled_values.append(sampled_value)
                n_samples += 1

        return sampled_values