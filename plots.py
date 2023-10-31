import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def retrieve_csv_files(directory, env_name):
    csv_files = []
    if os.path.exists(directory):
        for current_path, directories, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(current_path, file)
                    if full_path.split('/')[3].split('_')[1] == env_name:
                        csv_files.append(full_path)
    else:
        raise Exception(f"Directory not found: {directory}")
    if len(csv_files) == 0:
        raise Exception(f"Experiments on {env_name} not found")
    return csv_files

def moving_average(data, window_size):
    # Create a simple moving average for the data
    weights = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, weights, mode='valid')
    return smoothed_data
        
def plot_results(file_paths):
    plt.figure()
    
    algo_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                   '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff5733', '#4caf50',
                   '#ff3f80', '#5bc0de', '#ffd700', '#9932CC', '#00CED1', '#FF1493',
                   '#32CD32', '#8A2BE2']

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
    for i, (algo, data_list) in enumerate(algo_data.items()):
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
        
        # Smooth the data with a moving average (adjust window size as needed)
        window_size = 20
        smoothed_y_mean = moving_average(y_mean, window_size=window_size)
        
        if algo == 'BC': 
            color=algo_colors[0]
            continue
        elif algo == 'TD3BC': 
            color=algo_colors[1]
            #continue
        elif algo == 'AWR': 
            color=algo_colors[2]
            #continue
        elif algo == 'STR': 
            color=algo_colors[3]
            #continue
        elif algo == 'EXPLO':
            color=algo_colors[4]
        else: 
            color = algo_colors[5+i]
        
        plt.plot(common_steps[:-window_size+1], smoothed_y_mean, label=algo, color=color, linewidth=1)
        plt.fill_between(common_steps[:-window_size+1], smoothed_y_mean - y_std[:-window_size+1], smoothed_y_mean + y_std[:-window_size+1], color=color, alpha=0.2)
    
    title = file_path.split('/')[3].split('_')[1]
    plt.xlabel('Total Steps')
    plt.ylabel('Normalized Score')
    plt.grid(linestyle=':')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_labels, sorted_handles = zip(*sorted(zip(labels, handles)))
    plt.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.show()


def plot_main_experiments():
    envs = ['halfcheetah-medium-replay-v2', 'hopper-medium-replay-v2',
            'walker2d-medium-replay-v2']
    for env in envs:
        csv_files = retrieve_csv_files(directory='experiments/', 
                                       env_name=env)
        plot_results(file_paths=csv_files)

if __name__=='__main__':
    plot_main_experiments()