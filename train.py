import argparse
import random
import torch
import numpy as np
import os
import time
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from envs import return_environment, mujoco_locomotion
from agents import return_agent
import logger


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='walker2d-expert-v2')
parser.add_argument('--agent', type=str, default="LOM")
parser.add_argument('--buffer_capacity', type=int, default=2000000)
parser.add_argument('--discount', type=float, default=0.99)
parser.add_argument('--normalise', type=int, choices=[0, 1], default=1)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--render', type=int, default=0)

parser.add_argument('--enable_wandb', type=int, choices=[0, 1], default=0)
parser.add_argument('--project', type=str, default='LOM-ICLR')
parser.add_argument('--group', type=str, default='')
parser.add_argument('--training_steps', type=int, default=1000000)
parser.add_argument('--eval_episodes', type=int, default=10)
parser.add_argument('--eval_every', type=int, default=10000)
parser.add_argument('--log_path', type=str, default='./experiments/')
parser.add_argument('--save_gmm', type=int, default=1)
parser.add_argument('--gmm_pretrain_steps', type=int, default=200000)
parser.add_argument('--gmm_batch_size', type=int, default=512)
parser.add_argument('--gmm_checkpoint', type=str, default='')

parser.add_argument('--mode_selection', type=str,
                    choices=['top1', 'topk_q', 'topk_gnn', 'topk_gnn_merge'], default='top1')
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--mode_action_samples', type=int, default=8)
parser.add_argument('--gnn_hidden_dim', type=int, default=128)
parser.add_argument('--gnn_lr', type=float, default=3e-4)
parser.add_argument('--gnn_ranking_weight', type=float, default=0.25)
parser.add_argument('--gnn_temperature', type=float, default=1.0)
parser.add_argument('--gnn_prior_weight', type=float, default=1.0)


args = parser.parse_args()
args.seed = np.random.randint(1e3) if args.seed == -1 else args.seed
args.group = args.group or args.env_name + '_' + args.agent + '_' + args.mode_selection

args_dict = vars(args)

if args.enable_wandb:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Weights & Biases logging is enabled, but wandb is not installed. '
            'Install it with "python -m pip install wandb" or rerun with '
            '--enable_wandb 0.'
        ) from exc
    wandb.init(project=args.project, config=args, group=args.group,
               name='{}_{}_{}_seed{}'.format(
                   args.agent, args.env_name, args.mode_selection, args.seed))
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
    try:
        import d4rl
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'D4RL is required for MuJoCo locomotion environments. '
            'Install d4rl, gym, and mujoco-py in the Python 3.9 environment '
            'specified by the README.'
        ) from exc
    buffer.load_dataset(d4rl.qlearning_dataset(env))
agent = return_agent(replay_buffer=buffer, state_dim=env_info['state_dim'], 
                     ac_dim=env_info['ac_dim'], device=device,  **args_dict)


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
interval_start = time.time()
for steps in tqdm(range(0, args.training_steps), mininterval=5):
    step_start = time.time()
    
    policy_eval_info = {}
    training_info = agent.train_models()
    training_info['time/train_step_seconds'] = time.time() - step_start
    
    if (steps + 1) % args.eval_every == 0:
        policy_eval_info = eval_policy(env, agent, args.render)
    
        log_info = {**training_info, **policy_eval_info}
        epoch += 1
        logger.record_tabular('total steps', steps)
        logger.record_tabular('training epoch', epoch)
        logger.record_tabular('epoch time (min)',
                              (time.time() - interval_start) / 60)
        for key, val in log_info.items():
            if type(val) == torch.Tensor:
                logger.record_tabular(key, val.mean().item())
            else:
                logger.record_tabular(key, val)
        logger.dump_tabular()
        interval_start = time.time()
    
    if args.enable_wandb:
        wandb.log({**training_info, **policy_eval_info})
