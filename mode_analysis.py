import argparse
import numpy as np
import d4rl
import matplotlib.pyplot as plt

from scipy.stats import norm
import scipy.stats as stats
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from replay_buffer import ReplayBuffer
from envs import return_environment


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='halfcheetah-medium-replay-v2')
parser.add_argument('--buffer_capacity', type=int, default=2000000)
parser.add_argument('--discount', type=float, default=0.99)
args = parser.parse_args()

env, env_info = return_environment(args.env_name)
buffer = ReplayBuffer(buffer_size=args.buffer_capacity, 
                      state_dim=env_info['state_dim'], 
                      ac_dim=env_info['ac_dim'], discount=args.discount)
buffer.load_dataset(d4rl.qlearning_dataset(env), compute_return=False)

bound = buffer.get_bounds()
bound = int(bound / 1)
obs = buffer.obs[:bound]
acs = buffer.actions[:bound]

scaler = StandardScaler()
obs_scaled = scaler.fit_transform(obs)
actions_scaled = scaler.fit_transform(acs)
combined_data = np.hstack((obs_scaled, actions_scaled))


def dataset_conditional_distribution(state, num_neighbors=200):
    # Generate samples from the joint distribution
    joint_samples = combined_data

    # Use nearest neighbors to find closest states
    nn = NearestNeighbors(n_neighbors=num_neighbors)
    nn.fit(joint_samples[:, :17])  # Fit to state part of the joint samples

    # Find nearest neighbors to the given state
    distances, indices = nn.kneighbors([state])

    # Extract the actions corresponding to these neighbors
    conditional_actions = joint_samples[indices[0], 17:]

    # Now, conditional_actions contains actions for states close to the given state
    return conditional_actions


def test_normality(conditional_actions):
    for i in range(conditional_actions.shape[1]):
        action_dimension = conditional_actions[:, i]
        stat, p = stats.shapiro(action_dimension)
        print(f'Dimension {i+1}: Statistics={stat}, p={p}')
        if p > 0.05:
            print('  Gaussian (fail to reject H0)')
        else:
            print('  Not Gaussian (reject H0)')


def plot_kde_results():
    index = [10, 100, 1000]
    
    num_action_dimensions = dataset_conditional_distribution(obs_scaled[index[0]]).shape[1]

    # Define figure size and layout
    plt.figure(figsize=(18, 30))  # Adjusted the figure size to accommodate 6 rows

    lines = []  # To store the line objects for the legend
    labels = []  # To store the labels for the legend

    for j, idx in enumerate(index):
        conditional_actions = dataset_conditional_distribution(obs_scaled[idx])
        
        for i in range(conditional_actions.shape[1]):
            # Adjusted subplot indexing to accommodate 6 rows and 3 columns
            subplot_index = j * num_action_dimensions + i + 1
            plt.subplot(6, 3, subplot_index)
            
            # Histogram
            data = conditional_actions[:, i]
            plt.hist(data, bins=30, alpha=0.6, color='gray', density=True)
            
            # Mean and Standard Deviation
            mean_value = np.mean(data)
            std_dev = np.std(data)
    
            # Gaussian Normal Distribution Curve
            x_values = np.linspace(mean_value - 3*std_dev, mean_value + 3*std_dev, 1000)
            normal_line, = plt.plot(x_values, norm.pdf(x_values, mean_value, std_dev), 
                                    color='blue', linestyle='--', linewidth=2)
            
            # KDE
            kde = KernelDensity(kernel='gaussian', bandwidth=0.08)
            kde.fit(data[:, None])
            x_d = np.linspace(mean_value - 3*std_dev, mean_value + 3*std_dev, 1000)[:, None]
            log_density = kde.score_samples(x_d)
            kde_line, = plt.plot(x_d[:, 0], np.exp(log_density), color='green', linewidth=2)
    
            # Store lines and labels for the legend
            if j == 0 and i == 0:  # Only need to add these once
                lines.extend([normal_line, kde_line])
                labels.extend(['Normal Distribution Estimation', 'KDE'])

            # Aesthetics
            plt.xlabel('Action Value', fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title(f'State {j}, Action Dimension {i + 1}', fontsize=14)
            plt.grid(True)

    # Add legend at the top of the figure
    plt.figlegend(lines, labels, loc='upper center', ncol=2, frameon=True, fontsize=18)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust layout to make room for the legend
    plt.show()

























