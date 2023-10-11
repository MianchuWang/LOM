import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def retrieve_csv_files(directory):
    csv_files = []
    if os.path.exists(directory):
        for current_path, directories, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(current_path, file)
                    csv_files.append(full_path)
    else:
        print(f"Directory not found: {directory}")
    return csv_files
        
def plot_results(file_paths):
    plt.figure()

    # Dictionary to store all data points for each algorithm
    algo_data = defaultdict(list)

    # Read data
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        algo = file_path.split('/')[3].split('_')[0]  # Extract algo name from file path

        if 'total steps' in data.columns and 'normalized score' in data.columns:
            algo_data[algo].append(data)
        else:
            print(f"Required columns not found in the file: {file_path}")

    # For each algorithm, interpolate to common steps, then calculate mean and std dev, and plot
    for algo, data_list in algo_data.items():
        # Find the maximum 'total steps' across all dataframes for the current algorithm
        max_steps = max(df['total steps'].max() for df in data_list)

        # Create a common 'total steps' scale for the current algorithm
        common_steps = np.linspace(0, max_steps, num=1000)  # or choose another appropriate number of points

        # Interpolate 'normalized score' for the common 'total steps'
        interpolated_scores = []
        for data in data_list:
            interpolated = np.interp(common_steps, data['total steps'], data['normalized score'])
            interpolated_scores.append(interpolated)

        # Convert list of np.arrays to 2D np.array
        interpolated_scores = np.vstack(interpolated_scores)

        # Calculate mean and std dev
        y_mean = np.mean(interpolated_scores, axis=0)
        y_std = np.std(interpolated_scores, axis=0)

        plt.plot(common_steps, y_mean, label=algo)
        plt.fill_between(common_steps, y_mean - y_std, y_mean + y_std, alpha=0.2)
    
    title = file_path.split('/')[3].split('_')[1]
    plt.xlabel('Total Steps')
    plt.ylabel('Normalized Score')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()


def plot_main_experiments():
    csv_files = retrieve_csv_files('experiments/')
    plot_results(file_paths=csv_files)

if __name__=='__main__':
    plot_main_experiments()