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
parser.add_argument('--agent', type=str, default='GMM-mean')
parser.add_argument('--buffer_capacity', type=int, default=2000000)
parser.add_argument('--discount', type=float, default=0.99)
parser.add_argument('--normalise', type=int, choices=[0, 1], default=1)
parser.add_argument('--seed', type=int, default=-1)

parser.add_argument('--enable_wandb', type=int, choices=[0, 1], default=1)
parser.add_argument('--project', type=str, default='benchmark')
parser.add_argument('--group', type=str, default='seqGMM')
parser.add_argument('--training_steps', type=int, default=1000000)
parser.add_argument('--eval_episodes', type=int, default=5)
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
buffer.load_dataset(d4rl.qlearning_dataset(env))
agent = return_agent(agent=args.agent, replay_buffer=buffer, 
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
            'eval/normalized_score': normalized_score}


epoch = 0
for steps in tqdm(range(0, args.training_steps), mininterval=1):
    t_start = time.time()
    
    policy_eval_info = {}
    training_info = agent.train_models()
    
    if (steps + 1) % args.eval_every == 0:
        policy_eval_info = eval_policy(env, agent)
    
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
