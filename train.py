import argparse
import random
import torch
import numpy as np
import wandb
import os
import time

from replay_buffer import ReplayBuffer
from envs import return_environment
from agents import return_agent
from controller import Controller
import logger

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='antmaze-medium-play-v2')
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--agent', type=str, default='dawog')
parser.add_argument('--buffer_capacity', type=int, default=4000000)
parser.add_argument('--discount', type=float, default=0.98)
parser.add_argument('--normalise', type=int, choices=[0, 1], default=0)
parser.add_argument('--render_mode', type=str, default=None)
parser.add_argument('--seed', type=int, default=-1)

parser.add_argument('--enable_wandb', type=int, choices=[0, 1], default=0)
parser.add_argument('--project', type=str, default='antmaze')
parser.add_argument('--group', type=str, default='two_value')
parser.add_argument('--pretrain_steps', type=int, default=100000)  
parser.add_argument('--eval_episodes', type=int, default=50) 
parser.add_argument('--eval_every', type=int, default=5000)
parser.add_argument('--log_path', type=str, default='./experiments/')
args = parser.parse_args()

if args.enable_wandb:
    wandb.init(project=args.project, config=args, group=args.group, name='{}_{}_seed{}'.format(args.agent,args.env_name, args.seed))
curr_time = time.gmtime()

experiments_dir = args.log_path + args.project + '/' + args.group + '/' + '{}_{}_seed{}'.format(args.agent,args.env_name, args.seed) + '/'
logger.configure(experiments_dir)
logger.log('This running starts with parameters:')
logger.log('----------------------------------------')
for k, v in parser.parse_args()._get_kwargs():
    logger.log('- ' + str(k) + ': ' + str(v))
logger.log('----------------------------------------')
os.makedirs(experiments_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
env, env_info = return_environment(args.env_name, render_mode=args.render_mode)
seed = np.random.randint(1e6) if args.seed == -1 else args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

buffer = ReplayBuffer(buffer_size=args.buffer_capacity, state_dim=env_info['state_dim'],
                      ac_dim=env_info['ac_dim'], goal_dim=env_info['goal_dim'],
                      max_steps=env_info['max_steps'], get_goal_from_state=env_info['get_goal_from_state'],
                      compute_reward=env_info['compute_reward'])
if args.env_name.startswith('antmaze') or env_info['goal_dim'] == 0:
    buffer.load_d4rl(env)
else:
    buffer.load_dataset(args.dataset)
    
agent = return_agent(agent=args.agent, replay_buffer=buffer, state_dim=env_info['state_dim'],
                     ac_dim=env_info['ac_dim'], goal_dim=env_info['goal_dim'], env_name=env_info['env_name'],
                     device=device, discount=args.discount, max_steps=env_info['max_steps'], 
                     normalise=args.normalise, get_goal_from_state=env_info['get_goal_from_state'],
                     compute_reward=env_info['compute_reward'])

controller = Controller(pretrain_steps=args.pretrain_steps, eval_episodes=args.eval_episodes, eval_every=args.eval_every,
                        enable_wandb=args.enable_wandb, experiments_dir=experiments_dir, env=env, env_info=env_info,
                        agent=agent, buffer=buffer)

controller.train()
