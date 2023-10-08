import argparse
import random
import torch
import numpy as np
import wandb
import os
import time
import d4rl
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from envs import return_environment
from agents import return_agent
import logger


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='walker2d-medium-v2')
parser.add_argument('--agent', type=str, default='td3bc')
parser.add_argument('--buffer_capacity', type=int, default=1000000)
parser.add_argument('--discount', type=float, default=0.98)
parser.add_argument('--normalise', type=int, choices=[0, 1], default=0)
parser.add_argument('--seed', type=int, default=-1)

parser.add_argument('--enable_wandb', type=int, choices=[0, 1], default=0)
parser.add_argument('--project', type=str, default='mujoco_locomotion')
parser.add_argument('--group', type=str, default='test')
parser.add_argument('--training_steps', type=int, default=100000)  
parser.add_argument('--eval_episodes', type=int, default=50) 
parser.add_argument('--eval_every', type=int, default=5000)
parser.add_argument('--log_path', type=str, default='./experiments/')
args = parser.parse_args()

seed = np.random.randint(1e3) if args.seed == -1 else args.seed

wandb.init(project=args.project, config=args, group=args.group, name='{}_{}_seed{}'.format(args.agent, args.env_name, seed))
experiments_dir = args.log_path + args.project + '/' + args.group + '/' + '{}_{}_seed{}'.format(args.agent, args.env_name, seed) + '/'
logger.configure(experiments_dir)
logger.log('This running starts with parameters:')
logger.log('----------------------------------------')
for k, v in parser.parse_args()._get_kwargs():
    logger.log('- ' + str(k) + ': ' + str(v))
logger.log('----------------------------------------')
os.makedirs(experiments_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
env, env_info = return_environment(args.env_name)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

buffer = ReplayBuffer(buffer_size=args.buffer_capacity, state_dim=env_info['state_dim'], ac_dim=env_info['ac_dim'])
buffer.load_dataset(d4rl.qlearning_dataset(env))
    
agent = return_agent(agent=args.agent, replay_buffer=buffer, state_dim=env_info['state_dim'],
                     ac_dim=env_info['ac_dim'], device=device, discount=args.discount, normalise=args.normalise)


def eval(env, agent, render=False):
    returns = []
    for i in range(args.eval_episodes):
        agent.reset()
        obs = env.reset()
        returns.append(0)
        dict_obs = {}
        dict_obs['desired_goal'] = np.zeros((0))
        dict_obs['observation'] = obs
        obs = dict_obs
        done = False
        while not done:
            action = agent.get_action(obs['observation'])
            obs, reward, done, info = env.step(action)
            returns[-1] += reward.item()
            dict_obs = {}
            dict_obs['observation'] = obs
            obs = dict_obs 
                  
            if render:
                env.render()
                
    mean_return = np.array(returns).mean()
    normalized_score = 100 * env.get_normalized_score(mean_return)
    return {'return': mean_return,
            'normalized score': normalized_score}


t_start = time.time()
epoch = 0
total_step = 0
for i in tqdm(range(0, args.training_steps)):
    policy_eval_info = {}
    plan_eval_info = {}
    
    training_info = agent.train_models()
    total_step += 1
    
    wandb.log(training_info)
    
    if (i + 1) % args.eval_every == 0:
        policy_eval_info = eval(env, agent)
    
        log_info = {**training_info, **policy_eval_info, **plan_eval_info}
        epoch += 1
        logger.record_tabular('total steps', total_step)
        logger.record_tabular('training epoch', epoch)
        logger.record_tabular('epoch time (min)', (time.time() - t_start)/60)
        for key, val in log_info.items():
            if type(val) == torch.Tensor:
                logger.record_tabular(key, val.mean().item())
            else:
                logger.record_tabular(key, val)
        logger.dump_tabular()
  
        t_start = time.time()