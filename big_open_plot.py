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
from envs import return_environment, mujoco_locomotion, maze_envs
from agents import return_agent
import logger


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='maze2d-big-open')
parser.add_argument('--agent', type=str, default='BC')
parser.add_argument('--buffer_capacity', type=int, default=2000000)
parser.add_argument('--discount', type=float, default=0.99)
parser.add_argument('--normalise', type=int, choices=[0, 1], default=1)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--render', type=int, default=1)

parser.add_argument('--enable_wandb', type=int, choices=[0, 1], default=0)
parser.add_argument('--project', type=str, default='benchmark')
parser.add_argument('--group', type=str, default='seqGMM')
parser.add_argument('--training_steps', type=int, default=5000)
parser.add_argument('--eval_episodes', type=int, default=10)
parser.add_argument('--eval_every', type=int, default=10000)
parser.add_argument('--log_path', type=str, default='./experiments/')

args = parser.parse_args()
args.seed = np.random.randint(1e3) if args.seed == -1 else args.seed
args.group = args.env_name + '_' + args.agent

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
if args.env_name in mujoco_locomotion:
    buffer.load_dataset(d4rl.qlearning_dataset(env))
elif args.env_name in maze_envs:
    dataset = env.get_dataset(os.path.join('datasets', args.env_name + '.hdf5'))
    buffer.load_dataset(dataset)
bc_agent = return_agent(agent='BC', replay_buffer=buffer, 
                     state_dim=env_info['state_dim'], ac_dim=env_info['ac_dim'], 
                     device=device, discount=args.discount, normalise=args.normalise,
                     **explo_params)

buffer.actions[:, 2:] = 0

def eval_policy(env, agent, render=False):
    avg_reward = 0 
    for i in range(args.eval_episodes):
        agent.reset()
        obs, done = env.reset(), False
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            if args.env_name in maze_envs:
                if reward == 1: done=True
            avg_reward += reward
            if render:
                env.render()
                
    avg_reward /= args.eval_episodes
    return {'eval/return': avg_reward}


epoch = 0
for steps in tqdm(range(0, args.training_steps), mininterval=1):
    t_start = time.time()
    
    policy_eval_info = {}
    training_info = bc_agent.train_models()
    
    #if (steps + 1) % args.eval_every == 0:
        #policy_eval_info = eval_policy(env, bc_agent, args.render)
        


# Define your observations and threshold
target_obs_list = [np.array([3.8, 9.2]), np.array([3, 5.75]), np.array([4, 6])]
threshold = 0.2

# Initialize lists to collect matching observations and actions
matching_actions_list = []
bc_actions_list = []

# Loop over each target observation
for target_obs in target_obs_list:
    # Create the mask for observations around the target_obs
    mask = np.all(np.abs(buffer.obs[:, :2] - target_obs) < threshold, axis=1)
    
    matching_obs = buffer.obs[mask]
    
    # Use the mask to get the corresponding actions
    matching_actions = buffer.actions[mask]
    matching_actions_list.append(matching_actions)
    
    # Get actions from the behavioral cloning agent
    bc_actions = bc_agent.get_action(matching_obs)
    bc_actions_list.append(bc_actions)

# Plot the actions in different subplots
fig, axes = plt.subplots(2, len(target_obs_list), figsize=(18, 12))

# Plot original actions
for i, (target_obs, matching_actions) in enumerate(zip(target_obs_list, matching_actions_list)):
    axes[0, i].scatter(matching_actions[:, 0], matching_actions[:, 1], c='blue', label='Actions')
    axes[0, i].set_xlabel('Action Dimension 1')
    axes[0, i].set_ylabel('Action Dimension 2')
    axes[0, i].set_title(f'Actions around {target_obs}')
    axes[0, i].legend()
    axes[0, i].grid(True)
    axes[0, i].set_xlim([-1.2, 1.2])
    axes[0, i].set_ylim([-1.2, 1.2])
    
    # Calculate the mean of the points
    mean_action = np.mean(matching_actions, axis=0)
    
    # Plot an arrow from (0, 0) to the mean
    axes[0, i].arrow(0, 0, mean_action[0], mean_action[1], head_width=0.05, head_length=0.1, fc='red', ec='red')
    axes[0, i].annotate(f'Mean: ({mean_action[0]:.2f}, {mean_action[1]:.2f})', xy=(mean_action[0], mean_action[1]), xytext=(mean_action[0] + 0.1, mean_action[1] + 0.1))

# Plot bc_agent actions
for i, (target_obs, bc_actions) in enumerate(zip(target_obs_list, bc_actions_list)):
    axes[1, i].scatter(bc_actions[:, 0], bc_actions[:, 1], c='green', label='BC Actions')
    axes[1, i].set_xlabel('Action Dimension 1')
    axes[1, i].set_ylabel('Action Dimension 2')
    axes[1, i].set_title(f'BC Actions around {target_obs}')
    axes[1, i].legend()
    axes[1, i].grid(True)
    axes[1, i].set_xlim([-1.2, 1.2])
    axes[1, i].set_ylim([-1.2, 1.2])
    
    # Calculate the mean of the points
    mean_bc_action = np.mean(bc_actions, axis=0)
    
    # Plot an arrow from (0, 0) to the mean
    axes[1, i].arrow(0, 0, mean_bc_action[0], mean_bc_action[1], head_width=0.05, head_length=0.1, fc='red', ec='red')
    axes[1, i].annotate(f'Mean: ({mean_bc_action[0]:.2f}, {mean_bc_action[1]:.2f})', xy=(mean_bc_action[0], mean_bc_action[1]), xytext=(mean_bc_action[0] + 0.1, mean_bc_action[1] + 0.1))

plt.tight_layout()
plt.show()

