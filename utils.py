import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot(root_directory):

    # Traverse the directory to get all 'progress.csv' files
    csv_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file == 'progress.csv':
                csv_files.append(os.path.join(root, file))
    
    # Read CSVs and store in a list
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Extract experiment info from the file path
        path_parts = csv_file.split(os.sep)
        df['experiment'] = path_parts[-3] # e.g., 'five_value', 'one_value', 'two_value'
        df['seed'] = path_parts[-2] # Seed folder name
        dataframes.append(df)
    
    # Concatenate all dataframes
    all_data = pd.concat(dataframes)
    
    # Compute mean and variance for each experiment and training step
    mean_data = all_data.groupby(['experiment', 'training_steps'])['success'].mean().reset_index()
    var_data = all_data.groupby(['experiment', 'training_steps'])['success'].var().reset_index()
    
    # Plot mean and variance
    plt.figure(figsize=(10, 6))
    for experiment in mean_data['experiment'].unique():
        exp_mean_data = mean_data[mean_data['experiment'] == experiment]
        exp_var_data = var_data[var_data['experiment'] == experiment]
        plt.errorbar(exp_mean_data['training_steps'], exp_mean_data['success'], yerr=exp_var_data['success'], label=experiment)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Success')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #plot(root_directory='experiments')
    
    
    # Traverse the directory to get all 'progress.csv' files
    root_directory = 'experiments'
    
    csv_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file == 'progress.csv':
                csv_files.append(os.path.join(root, file))
    
    # Read CSVs and store in a list
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Extract experiment info from the file path
        path_parts = csv_file.split(os.sep)
        df['experiment'] = path_parts[2] # e.g., 'five_value', 'one_value', 'two_value'
        df['seed'] = path_parts[4] # Seed folder name
        dataframes.append(df)
    
    # Concatenate all dataframes
    all_data = pd.concat(dataframes)
    
    # Compute mean and variance for each experiment and training step
    mean_data = all_data.groupby(['experiment', 'total steps'])['success (policy)'].mean().reset_index()
    std_data = all_data.groupby(['experiment', 'total steps'])['success (policy)'].std().reset_index()
    
    # Plot mean and variance
    plt.figure(figsize=(10, 6))
    for experiment in mean_data['experiment'].unique():
        exp_mean_data = mean_data[mean_data['experiment'] == experiment]
        exp_std_data = std_data[std_data['experiment'] == experiment]
        plt.plot(exp_mean_data['total steps'], exp_mean_data['success (policy)'], label=experiment)
        plt.fill_between(exp_mean_data['total steps'], exp_mean_data['success (policy)'] - exp_std_data['success (policy)'],
                         exp_mean_data['success (policy)'] + exp_std_data['success (policy)'], alpha=0.2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Success')
    plt.legend()
    plt.show()