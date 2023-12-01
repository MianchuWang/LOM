import argparse
import numpy as np
import d4rl
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from scipy.stats import norm
import scipy.stats as stats
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from replay_buffer import ReplayBuffer
from envs import return_environment
from networks.networks import Qnetwork


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='halfcheetah-medium-v2')
parser.add_argument('--buffer_capacity', type=int, default=2000000)
parser.add_argument('--discount', type=float, default=0.99)
args = parser.parse_args()

env, env_info = return_environment(args.env_name)
buffer = ReplayBuffer(buffer_size=args.buffer_capacity, 
                      state_dim=env_info['state_dim'], 
                      ac_dim=env_info['ac_dim'], discount=args.discount)
buffer.load_dataset(d4rl.qlearning_dataset(env), compute_return=False)


q_net = Qnetwork(env_info['state_dim'], env_info['ac_dim']).to(device='cuda')
q_target_net = Qnetwork(env_info['state_dim'], env_info['ac_dim']).to(device='cuda')
q_target_net.load_state_dict(q_net.state_dict())
q_net_opt = torch.optim.Adam(q_net.parameters(), lr=3e-4) 


for it in tqdm(range(100000), mininterval=1):
    states, actions, rewards, next_states, next_actions, terminals  = buffer.sample_with_next_action(256)
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device='cuda')
    actions_tensor = torch.tensor(actions, dtype=torch.int64).to(device='cuda')
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device='cuda')
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device='cuda')
    next_actions_tensor = torch.tensor(next_actions, dtype=torch.int64).to(device='cuda')
    terminals_tensor = torch.tensor(terminals, dtype=torch.int64).to(device='cuda')
    with torch.no_grad():
        q_next_value = q_target_net(next_states_tensor, next_actions_tensor)
        target_q_value = rewards_tensor + (1 - terminals_tensor) * args.discount * q_next_value
    pred_q_value = q_net(states_tensor, actions_tensor)
    q_loss = ((target_q_value - pred_q_value)**2).mean()
    q_net_opt.zero_grad()
    q_loss.backward()
    q_net_opt.step()
    
    if it % 2 == 0:
        for param, target_param in zip(q_net.parameters(), q_target_net.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)


bound = buffer.get_bounds()
bound = int(bound / 1)
obs = buffer.obs[:bound]
acs = buffer.actions[:bound]

obs_scaler = StandardScaler()
ac_scaler = StandardScaler()
obs_scaled = obs_scaler.fit_transform(obs)
actions_scaled = ac_scaler.fit_transform(acs)
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
    indices = [10, 100, 1000, 10000]  # Multiple states
    num_action_dimensions = 2  # Plot only the first two dimensions

    # Define figure size and layout for 4x4 subplots
    plt.figure(figsize=(24, 24))

    # Lines and labels for the legend
    lines = []
    labels = []

    for row, idx in enumerate(indices):
        # Get the conditional actions for the current index
        conditional_actions = dataset_conditional_distribution(obs_scaled[idx])
        action_values = actions_scaled[idx]  # Assuming this is defined in your context

        for i in range(num_action_dimensions):
            # Unweighted Actions
            plt.subplot(4, 4, row * 4 + i * 2 + 1)
            data_unweighted = conditional_actions[:, i]
            normal_line, kde_line, ac_line = plot_data(data_unweighted, i, 'Unweighted', action_values[i], row)
            if row == 0 and i == 0:
                lines.extend([normal_line, kde_line, ac_line])
                labels.extend(['Normal Distribution Estimation', 'KDE',
                               'Original Action'])

            # Weighted Actions
            plt.subplot(4, 4, row * 4 + i * 2 + 2)
            data_weighted = get_weighted_actions(idx, conditional_actions, i)
            plot_data(data_weighted, i, 'Weighted', action_values[i], row)

    # Add legend at the top of the figure
    plt.figlegend(lines, labels, loc='upper center', ncol=3, frameon=True, fontsize=18)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust layout to make room for the legend
    plt.show()



def get_weighted_actions(idx, conditional_actions, dimension):
    # Calculate weights using your existing method
    states = torch.tensor(obs_scaler.inverse_transform(obs_scaled[idx][np.newaxis].repeat(200, axis=0))).to(device='cuda')
    actions = torch.tensor(ac_scaler.inverse_transform(conditional_actions)).to(device='cuda')
    kde_weights = q_net(states, actions).detach().cpu().numpy().squeeze()
    kde_weights = kde_weights - kde_weights.mean()
    kde_weights = np.clip(np.exp(2 * kde_weights), None, 100)
    kde_weights /= np.sum(kde_weights)

    # Resampling based on weights
    chosen_indices = np.random.choice(conditional_actions.shape[0], 
                                      size=10 * conditional_actions.shape[0], 
                                      p=kde_weights)
    resampled_actions = conditional_actions[chosen_indices]
    return resampled_actions[:, dimension]

def plot_data(data, dimension, title_prefix, action_value, row):
    # Histogram
    plt.hist(data, bins=30, alpha=0.6, color='gray', density=True)

    # Mean and Standard Deviation
    mean_value = np.mean(data)
    std_dev = np.std(data)

    # Gaussian Normal Distribution Curve
    x_values = np.linspace(mean_value - 3*std_dev, mean_value + 3*std_dev, 1000)
    normal_line, = plt.plot(x_values, norm.pdf(x_values, mean_value, std_dev), color='blue', linestyle='--', linewidth=2)

    # KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.08)
    kde.fit(data[:, None])
    x_d = np.linspace(mean_value - 3*std_dev, mean_value + 3*std_dev, 1000)[:, None]
    log_density = kde.score_samples(x_d)
    kde_line, = plt.plot(x_d[:, 0], np.exp(log_density), color='green', linewidth=2)

    # Vertical line for action_value
    ac_line = plt.axvline(x=action_value, color='red', linestyle='--', linewidth=2)

    # Aesthetics
    plt.xlabel('Action Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'State {row}, {title_prefix} Actions, Dim {dimension + 1}', fontsize=14)
    plt.grid(True)
    
    return normal_line, kde_line, ac_line

# Call the function to plot
plot_kde_results()
























