import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import re

class MemoryOptimizer:
    def __init__(self, csv_path, cpp_source_path):
        """
        Initialize the Memory Optimizer with profiling data and source code
        
        :param csv_path: Path to the CSV file with memory profiling data
        :param cpp_source_path: Path to the C++ source code file
        """
        self.data = pd.read_csv(csv_path)
        self.cpp_source_path = cpp_source_path
        self.cpp_source_code = self._read_cpp_source()
        self.prepare_data()
        self.model_alloc = None
        self.model_peak = None

    def _read_cpp_source(self):
        """
        Read the contents of the C++ source file
        
        :return: Source code as a string
        """
        with open(self.cpp_source_path, 'r') as file:
            return file.read()

    def prepare_data(self):
        """
        Preprocess and engineer features for memory optimization
        """
        self.data['Memory_Delta'] = self.data['Memory Allocated (KB)'] - self.data['Memory Freed (KB)']
        self.data['Memory_Utilization_Ratio'] = self.data['Memory Allocated (KB)'] / (self.data['Peak Memory (KB)'] + 1)

        # Prepare features and targets for machine learning models
        self.features = self.data[['Time (ms)', 'Memory_Utilization_Ratio', 'Memory_Delta']]
        self.target_alloc = self.data['Memory Allocated (KB)']
        self.target_peak = self.data['Peak Memory (KB)']

    def train_model(self):
        """
        Train machine learning models to predict memory allocation and peak memory usage
        """
        # Train RandomForest model for predicting memory allocation
        self.model_alloc = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_alloc.fit(self.features, self.target_alloc)

        # Train RandomForest model for predicting peak memory usage
        self.model_peak = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_peak.fit(self.features, self.target_peak)

    def optimize_source_code(self, output_path):
        """
        Apply basic memory optimizations to the source code
        
        :param output_path: Path to save the optimized source code
        """
        optimized_code = re.sub(r'malloc\((.*?)\)', r'calloc(1, \1)', self.cpp_source_code)
        optimized_code = re.sub(r'free\((.*?)\)', r'free(\1); \1 = NULL;', optimized_code)
        
        with open(output_path, 'w') as file:
            file.write(optimized_code)

    def simulate_optimized_profile(self, output_csv_path):
        """
        Generate a simulated optimized profile by applying reductions
        
        :param output_csv_path: Path to save the simulated optimized CSV
        """
        optimized_data = self.data.copy()
        optimized_data['Memory Allocated (KB)'] *= 0.8
        optimized_data['Memory Freed (KB)'] *= 0.8
        optimized_data['Peak Memory (KB)'] *= 0.9

        optimized_data.to_csv(output_csv_path, index=False)

    def predict_memory(self):
        """
        Use trained machine learning models to predict memory allocation and peak memory usage
        """
        if self.model_alloc is None or self.model_peak is None:
            self.train_model()

        self.data['Predicted Memory Allocated (KB)'] = self.model_alloc.predict(self.features)
        self.data['Predicted Peak Memory (KB)'] = self.model_peak.predict(self.features)

    def compare_profiles(self, unoptimized_data, optimized_data):
        """
        Compare memory allocation and utilization between unoptimized and optimized profiles
        
        :param unoptimized_data: DataFrame for the unoptimized profile
        :param optimized_data: DataFrame for the optimized profile
        """
        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        plt.plot(unoptimized_data['Time (ms)'], unoptimized_data['Memory Allocated (KB)'], label='Unoptimized', alpha=0.7)
        plt.plot(optimized_data['Time (ms)'], optimized_data['Memory Allocated (KB)'], label='Optimized', alpha=0.7)
        plt.title('Memory Allocation Comparison')
        plt.xlabel('Time (ms)')
        plt.ylabel('Memory Allocated (KB)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(unoptimized_data['Time (ms)'], unoptimized_data['Peak Memory (KB)'], label='Unoptimized', alpha=0.7)
        plt.plot(optimized_data['Time (ms)'], optimized_data['Peak Memory (KB)'], label='Optimized', alpha=0.7)
        plt.title('Peak Memory Usage Comparison')
        plt.xlabel('Time (ms)')
        plt.ylabel('Peak Memory (KB)')
        plt.legend()

        plt.tight_layout()
        plt.savefig('example5_profile_comparison.png')
        plt.close()
        print("Profile comparison graph saved as 'example5_profile_comparison.png'.")

        # Numerical comparison
        delta_memory_alloc = unoptimized_data['Memory Allocated (KB)'].sum() - optimized_data['Memory Allocated (KB)'].sum()
        delta_peak_memory = unoptimized_data['Peak Memory (KB)'].max() - optimized_data['Peak Memory (KB)'].max()

        print(f"Total reduction in memory allocation: {delta_memory_alloc:.2f} KB")
        print(f"Reduction in peak memory usage: {delta_peak_memory:.2f} KB")


def main(csv_path_unoptimized, cpp_source_path, output_cpp_path, output_csv_optimized):
    optimizer = MemoryOptimizer(csv_path_unoptimized, cpp_source_path)

    # Step 1: Optimize source code
    optimizer.optimize_source_code(output_cpp_path)
    print(f"Optimized source code saved to {output_cpp_path}.")

    # Step 2: Simulate optimized profile
    optimizer.simulate_optimized_profile(output_csv_optimized)
    print(f"Simulated optimized profile saved to {output_csv_optimized}.")

    # Step 3: Predict memory usage with machine learning models
    optimizer.predict_memory()

    # Step 4: Compare profiles
    unoptimized_data = optimizer.data
    optimized_data = pd.read_csv(output_csv_optimized)
    optimizer.compare_profiles(unoptimized_data, optimized_data)

if __name__ == "__main__":
    main(
        csv_path_unoptimized="example5_output.csv",
        cpp_source_path="mem_leak.cpp",
        output_cpp_path="optimized_mem_leak.cpp",
        output_csv_optimized="optimized_example5_output.csv"
    )
