import numpy as np
import pandas as pd

class SyntheticData:
    def __init__(self, data_type, variables):
        self.data_type = data_type
        self.variables = variables
        self.data = None  
        self.base_data = None

        self.std = self.variables.get("std", 1)
        self.num_samples = self.variables.get("num_samples", 1000)
        self.rnd = self.variables.get("Random", None)
        self.rng = np.random.RandomState(self.rnd) if self.rnd is not None else np.random

    def generate_data(self):
        if self.data_type == "log_normal":
            # Mu is locked to 1.0; x0_shift moves the data
            self.data = self.rng.lognormal(mean=1.0, sigma=self.std, size=self.num_samples).astype(np.float64)

        elif self.data_type == "normal":
            self.data = self.rng.normal(loc=self.mean, scale=self.std, size=self.num_samples).astype(np.float64)
            
        elif self.data_type == "poisson":
            self.data = self.rng.poisson(lam=self.mean, size=self.num_samples).astype(np.float64)
            
        elif self.data_type == "negative_binomial":
            n = self.variables.get("n", 10)
            p = self.variables.get("p", 0.5)
            self.data = self.rng.negative_binomial(n=n, p=p, size=self.num_samples).astype(np.float64)
            
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def clipping_data(self, SIGMA_LIMIT = 3.0, PERCENTILE_LIMIT = 99.9):
        if self.data is None:
            raise ValueError("No data found. Please run generate_data() first.")

        if self.data_type == "normal":
            # Standard linear boundaries
            lower_bound = self.x0_shift - SIGMA_LIMIT * self.std
            upper_bound = self.x0_shift + SIGMA_LIMIT * self.std
            self.data = np.clip(self.data, a_min=lower_bound, a_max=upper_bound)

        elif self.data_type == "log_normal":
            upper_bound = np.exp(1.0 + SIGMA_LIMIT * self.std)
            # Strict minimum clipping floor at exactly zero
            self.data = np.clip(self.data, a_min=0.0, a_max=upper_bound)
            
        elif self.data_type in ["poisson", "negative_binomial"]:
            upper_bound = np.percentile(self.data, PERCENTILE_LIMIT)
            self.data = self.data[self.data <= upper_bound]

        # Save the clean, clipped baseline so we can reset it during grid iteration
        self.base_data = self.data.copy()

    def Magic_Number_Ingestion(self, magic_numbers, quantities):
        if self.base_data is None:
            raise ValueError("No base data found. Run clipping_data() first.")

        if len(magic_numbers) != len(quantities):
            raise ValueError("Length of magic_numbers must match length of quantities.")

        # Reset data to the clean baseline before injecting new combinations
        self.data = self.base_data.copy()

        # Container list to hold new corrupted blocks alongside baseline arrays
        segments = [self.data]

        for magic_number, quantity in zip(magic_numbers, quantities):
            if quantity > 0:
                # Generate a single-value constant block matching requested quantity
                magic_samples = np.full(quantity, magic_number, dtype=np.float64)
                segments.append(magic_samples)

        self.data = np.concatenate(segments)
        self.rng.shuffle(self.data)

    def create_dataframe(self):
        if self.data is None:
            raise ValueError("No data found. Please run generate_data() first.")

        df = pd.DataFrame(self.data, columns=["Synthetic_Data"])
        return df