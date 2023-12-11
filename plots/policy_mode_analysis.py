import argparse
import random
import torch
import numpy as np
import wandb
import os
import time
import d4rl
from tqdm import tqdm
import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer
from envs import return_environment
from agents import return_agent
from agents.bc import BC
from networks.networks import MixedGaussianPolicy
import logger

from scipy.stats import norm
import scipy.stats as stats
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='halfcheetah-medium-v2')
parser.add_argument('--agent', type=str, default='mixedGaussian')
parser.add_argument('--buffer_capacity', type=int, default=2000000)
parser.add_argument('--discount', type=float, default=0.99)
parser.add_argument('--normalise', type=int, choices=[0, 1], default=1)
parser.add_argument('--seed', type=int, default=-1)

parser.add_argument('--enable_wandb', type=int, choices=[0, 1], default=0)
parser.add_argument('--project', type=str, default='mujoco_locomotion')
parser.add_argument('--group', type=str, default='EXPLO-baselines')
parser.add_argument('--training_steps', type=int, default=50000)  
parser.add_argument('--eval_episodes', type=int, default=25) 
parser.add_argument('--eval_every', type=int, default=10000)
parser.add_argument('--log_path', type=str, default='./experiments/')

args = parser.parse_args()
args.seed = np.random.randint(1e3) if args.seed == -1 else args.seed
args.group = args.env_name + '-' + args.group

# EXPLORATION Parameters
explo_params = {}

if args.enable_wandb:
    wandb.init(project=args.project, config=args, group=args.group, name='{}_{}_seed{}'.format(args.agent, args.env_name, args.seed))
experiments_dir = args.log_path + args.project + '/' + args.group + '/' + '{}_{}_seed{}'.format(args.agent, args.env_name, args.seed) + '/'
logger.configure(experiments_dir)
logger.log('This running starts with parameters:')
logger.log('----------------------------------------')
for k, v in args._get_kwargs():
    logger.log('- ' + str(k) + ': ' + str(v))
logger.log('----------------------------------------')
os.makedirs(experiments_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
env, env_info = return_environment(args.env_name)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

buffer = ReplayBuffer(buffer_size=args.buffer_capacity, 
                      state_dim=env_info['state_dim'], 
                      ac_dim=env_info['ac_dim'], discount=args.discount)
buffer.load_dataset(d4rl.qlearning_dataset(env))
agent = return_agent(agent=args.agent, replay_buffer=buffer, 
                     state_dim=env_info['state_dim'], ac_dim=env_info['ac_dim'], 
                     device=device, discount=args.discount, normalise=args.normalise,
                     **explo_params)
mg_agent = return_agent(agent='mixedGaussian', replay_buffer=buffer, 
                        state_dim=env_info['state_dim'], ac_dim=env_info['ac_dim'], 
                        device=device, discount=args.discount, normalise=args.normalise,
                        **explo_params)


def eval_policy(env, agent, render=False):
    avg_reward = 0 
    for i in range(args.eval_episodes):
        agent.reset()
        obs, done = env.reset(), False
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            if render:
                env.render()
                
    avg_reward /= args.eval_episodes
    normalized_score = 100 * env.get_normalized_score(avg_reward)
    return {'eval/return': avg_reward,
            'eval/normalized score': normalized_score}


epoch = 0
for steps in tqdm(range(0, args.training_steps), mininterval=1):
    t_start = time.time()
    
    policy_eval_info = {}
    training_info = mg_agent.train_models()
    
    if (steps + 1) % args.eval_every == 0:
        policy_eval_info = eval_policy(env, mg_agent)
    
        log_info = {**training_info, **policy_eval_info}
        epoch += 1
        logger.record_tabular('total steps', steps)
        logger.record_tabular('training epoch', epoch)
        logger.record_tabular('epoch time (min)', (time.time() - t_start)/60)
        for key, val in log_info.items():
            if type(val) == torch.Tensor:
                logger.record_tabular(key, val.mean().item())
            else:
                logger.record_tabular(key, val)
        logger.dump_tabular()
    
    if args.enable_wandb:
        wandb.log({**training_info, **policy_eval_info})


bound = buffer.get_bounds()
bound = int(bound / 1)
states = buffer.obs[:bound]
actions = buffer.actions[:bound]

states_prep, actions_prep, _, _, _ = agent.preprocess(states=states, actions=actions)
states_prep = states_prep.cpu().detach().numpy()
actions_prep = actions_prep.cpu().detach().numpy()
combined_data = np.hstack((states_prep, actions_prep))


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
        conditional_actions = dataset_conditional_distribution(states_prep[idx])
        action_values = actions_prep[idx]  # Assuming this is defined in your context

        for i in range(num_action_dimensions):
            # Unweighted Actions
            plt.subplot(4, 4, row * 4 + i * 2 + 1)
            data_unweighted = conditional_actions[:, i]
            normal_line, kde_line, ac_line = plot_data(data_unweighted, i, 'Unweighted', action_values[i], row, states_prep[idx])
            if row == 0 and i == 0:
                lines.extend([normal_line, kde_line, ac_line])
                labels.extend(['Normal Distribution Estimation', 'KDE',
                               'Original Action'])
                
    # Add legend at the top of the figure
    plt.figlegend(lines, labels, loc='upper center', ncol=3, frameon=True, fontsize=18)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust layout to make room for the legend
    plt.show()


def plot_data(data, dimension, title_prefix, action_value, row, state):
    # Histogram
    plt.hist(data, bins=30, alpha=0.6, color='gray', density=True)
    
    # Learned policy
    state_tensor = torch.Tensor(state).to(device='cuda')
    ac_dist, _ = agent.policy(state_tensor)
    actions = ac_dist.sample((1000, )).cpu().detach().numpy().squeeze()
    actions = np.clip(actions, -1, 1)
    plt.hist(actions[:, dimension], bins=30, alpha=0.6, color='orange', density=True)
    
    # Mean and Standard Deviation
    mean_value = np.mean(data)
    std_dev = np.std(data)

    # Gaussian Normal Distribution Curve
    #x_values = np.linspace(mean_value - 3*std_dev, mean_value + 3*std_dev, 1000)
    #normal_line, = plt.plot(x_values, norm.pdf(x_values, mean_value, std_dev), color='blue', linestyle='--', linewidth=2)
    normal_line = None
    
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







