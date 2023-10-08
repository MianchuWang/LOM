import numpy as np
import wandb
import logger
import torch
from tqdm import tqdm
import time

class Controller:
    def __init__(self, pretrain_steps, eval_episodes, eval_every, 
                 experiments_dir, env, env_info, agent, buffer, enable_wandb):
        self.pretrain_steps = pretrain_steps
        self.eval_episodes = eval_episodes
        self.eval_every = eval_every
        self.experiments_dir = experiments_dir
        self.env = env
        self.env_info = env_info
        self.agent = agent
        self.buffer = buffer
        self.finetune_episode_steps = 500
        self.enable_wandb = enable_wandb
        

    def train(self):
        logger.log('Pretraining ...')
        t_start = time.time()
        epoch = 0
        total_step = 0
        for i in tqdm(range(0, self.pretrain_steps)):
            policy_eval_info = {}
            plan_eval_info = {}
            
            training_info = self.agent.train_models()
            total_step += 1
            
            if (i + 1) % self.eval_every == 0:
                policy_eval_info = self.eval('policy')
            
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

                if self.enable_wandb:
                    wandb.log(log_info)
                
                t_start = time.time()
            

    def eval(self, mode='plan', render=False):
        returns = []
        successes = []
        for i in range(self.eval_episodes):
            self.agent.reset()
            obs = self.env.reset()
            returns.append(0)
            if self.env_info['goal_dim'] == 0:
                dict_obs = {}
                dict_obs['desired_goal'] = np.zeros((0))
                dict_obs['observation'] = obs
                obs = dict_obs
            
            for _ in range(self.env_info['max_steps']):
                if mode == 'plan':
                    action = self.agent.plan(obs['observation'], obs['desired_goal'])
                else:
                    action = self.agent.get_action(obs['observation'], obs['desired_goal'])
                #action = self.env.action_space.sample()
                if self.env_info['env_name'].startswith('antmaze'):
                    obs, reward, _, info = self.env.step(action)
                    if reward == 1: break
                
                elif self.env_info['goal_dim'] == 0:
                    obs, reward, done, info = self.env.step(action)
                    returns[-1] += reward.item()
                    dict_obs = {}
                    dict_obs['desired_goal'] = np.zeros((0))
                    dict_obs['observation'] = obs
                    obs = dict_obs 
                
                else:
                    obs, reward, _, info = self.env.step(action)
                    returns[-1] += reward.item()
                if render:
                    self.env.render()
                    
            successes.append(reward)
        
        mean_return = np.array(returns).mean()
        mean_success = np.array(successes).mean()
        normalized_score = 100 * self.env.get_normalized_score(mean_return)
        return {'return (' + mode + ')': mean_return,
                'success (' + mode + ')': mean_success,
                'normalized score (' + mode + ')': normalized_score}
    
