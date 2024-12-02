from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from networks.networks import Qnetwork, Policy

hyperparams = {'halfcheetah-medium-replay-v2':  {'K': 1, 'num_mixtures': 5, 'beta': 5, 'ac_rand': 0.4, 'smooth_noise': 0.2, 'C': 100},
               'hopper-medium-replay-v2':       {'K': 1, 'num_mixtures': 5, 'beta': 5, 'ac_rand': 0.0, 'smooth_noise': 0.2, 'C': 50},
               'walker2d-medium-replay-v2':     {'K': 1, 'num_mixtures': 10, 'beta': 5, 'ac_rand': 0.0, 'smooth_noise': 0.2, 'C': 50},
               
               'halfcheetah-full-replay-v2':    {'K': 1, 'num_mixtures': 10, 'beta': 5, 'ac_rand': 0.4, 'smooth_noise': 0.2, 'C': 100},
               'hopper-full-replay-v2':         {'K': 2, 'num_mixtures': 10, 'beta': 5, 'ac_rand': 0.0, 'smooth_noise': 0.2, 'C': 50},
               'walker2d-full-replay-v2':       {'K': 1, 'num_mixtures': 10, 'beta': 5, 'ac_rand': 0.0, 'smooth_noise': 0.2, 'C': 50},
               
               'halfcheetah-medium-v2':         {'K': 1, 'num_mixtures': 2, 'beta': 5, 'ac_rand': 0.4, 'smooth_noise': 0.2, 'C': 100},
               'hopper-medium-v2':              {'K': 3, 'num_mixtures': 5, 'beta': 5, 'ac_rand': 0, 'smooth_noise': 0.2, 'C': 50},
               'walker2d-medium-v2':            {'K': 1, 'num_mixtures': 2, 'beta': 5, 'ac_rand': 0.0, 'smooth_noise': 0.2, 'C': 50},
               
               'halfcheetah-medium-expert-v2':  {'K': 2, 'num_mixtures': 1, 'beta': 0.1, 'ac_rand': 0, 'smooth_noise': 0.0, 'C': 50},
               'hopper-medium-expert-v2':       {'K': 6, 'num_mixtures': 2, 'beta': 0.1, 'ac_rand': 0.0, 'smooth_noise': 0.0, 'C': 50},
               'walker2d-medium-expert-v2':     {'K': 3, 'num_mixtures': 5, 'beta': 0.1, 'ac_rand': 0.2, 'smooth_noise': 0.0, 'C': 50},
               
               'halfcheetah-expert-v2':         {'K': 1, 'num_mixtures': 2, 'beta': 0.1, 'ac_rand': 0.0, 'smooth_noise': 0.0, 'C': 50},
               'hopper-expert-v2':              {'K': 1, 'num_mixtures': 2, 'beta': 0.1, 'ac_rand': 0.0, 'smooth_noise': 0.2, 'C': 50},  
               'walker2d-expert-v2':            {'K': 1, 'num_mixtures': 5, 'beta': 0.1, 'ac_rand': 0.2, 'smooth_noise': 0.0, 'C': 50},
               }


class MixtureGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, num_mixtures):
        super(MixtureGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_mixtures = num_mixtures

        self.model = nn.Sequential(nn.Linear(state_dim, 512), 
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, action_dim * num_mixtures * 2 + num_mixtures))
    
    # Return the means, log_stds, weights for each Gaussian components.
    def forward(self, state):
        x = self.model(state)
        an = self.action_dim * self.num_mixtures
        means = torch.tanh(x[:, : an].view(-1, self.num_mixtures, self.action_dim))
        means = torch.clamp(means, -0.9999, 0.9999)
        log_stds = x[:, an: 2*an].view(-1, self.num_mixtures, self.action_dim)
        weights = F.softmax(x[:, 2*an: ], dim=-1)
        return means, log_stds, weights
    
    '''
        This function samples actions from all Gaussian components.
        Input: Tensor state
        Output: Samples with shape (batch_size, num_mixtures, ac_dim)
    '''
    def sample_actions_for_components(self, state):
       means, log_stds, _ = self.forward(state)
       stds = (log_stds.clamp(-15, 0)).exp()
       dist = torch.distributions.Normal(means, stds)
       samples = dist.sample()
       return samples



class LOM(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.params = hyperparams[self.env_name]
        print(self.params)
        self.training_steps = 0
        self.policy_delay = 2
        self.gmm = MixtureGaussianPolicy(self.state_dim, self.ac_dim, self.params['num_mixtures']).to(device=self.device)
        self.gmm_opt = torch.optim.Adam(self.gmm.parameters(), lr=1e-3)   
        #self.gmm.load_state_dict(torch.load('gmm_' + agent_params['env_name'] + '.pth'))
        
        self.policy = Policy(self.state_dim, self.ac_dim).to(device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
                
        self.mode_net = Qnetwork(self.state_dim, self.params['num_mixtures']).to(device=self.device)
        self.mode_net_opt = torch.optim.Adam(self.mode_net.parameters(), lr=1e-3)

        self.q_nets, self.q_target_nets, self.q_net_opts = [], [], []
        for k in range(self.params['K']):
            self.q_nets.append(Qnetwork(self.state_dim, self.ac_dim).to(device=self.device))
            self.q_target_nets.append(Qnetwork(self.state_dim, self.ac_dim).to(device=self.device))
            self.q_target_nets[k].load_state_dict(self.q_nets[k].state_dict())
            self.q_net_opts.append(torch.optim.Adam(self.q_nets[k].parameters(), lr=3e-4))  
        
        print('Learning the GMM model ...')
        for _ in tqdm(range(200000)):
            self.train_GMM(batch_size=512)

        torch.save(self.gmm.state_dict(), 'gmm_' + self.env_name + '.pth')
        

    def train_models(self, batch_size=512):
        self.value_info = self.train_value_function(batch_size=256)
        self.mode_info = self.train_mode_function(batch_size=1024)
        self.policy_info = self.train_policy(batch_size=256)
        if self.training_steps % self.policy_delay == 0:
            self.update_target_nets(self.q_nets, self.q_target_nets)
        self.training_steps += 1
        return {**self.value_info, **self.policy_info, **self.mode_info}
    
    '''
        Follow the TD3-BC manner to learn a Q-function conditioned on the behaviour policy
    '''
    def train_value_function(self, batch_size):
        states, actions, rewards, next_states, next_actions, terminals  = self.replay_buffer.sample_with_next_action(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals)
        _, next_actions_prep, _, _, _ = self.preprocess(actions=next_actions)
        
        with torch.no_grad():
            target_actions = next_actions_prep
            smooth_noise = torch.clamp(self.params['smooth_noise'] * torch.randn_like(target_actions), -0.5, 0.5)
            target_actions = torch.clamp(target_actions + smooth_noise, -1, 1)

            q_next_values = torch.zeros(self.params['K'], batch_size, 1).to(device=self.device)
            for i in range(self.params['K']):
                q_next_values[i] = self.q_target_nets[i](next_states_prep, target_actions)
            q_next_value = torch.min(q_next_values, dim=0)[0]
            target_q_value = rewards_prep + (1 - terminals_prep) * self.discount * q_next_value
        
        for k in range(self.params['K']):
            pred_q_value = self.q_nets[k](states_prep, actions_prep)
            q_loss = ((target_q_value - pred_q_value)**2).mean()
            self.q_net_opts[k].zero_grad()
            q_loss.backward()
            self.q_net_opts[k].step()
        
        return {'Q/loss': q_loss.item(), 
                'Q/pred_value': pred_q_value.mean().item(), 
                'Q/target_value': target_q_value.mean().item()}


    def train_mode_function(self, batch_size):
        states, _, _, _, _, _ = self.replay_buffer.sample_with_next_action(batch_size)
        states_prep, _, _, _, _ = self.preprocess(states=states)

        with torch.no_grad():
            # Create a one-hot encoding for each mixture component for every state in the batch.
            one_hot_components = torch.eye(self.params['num_mixtures']).to(self.device)  # [num_mixtures, num_mixtures]
            one_hot_components = one_hot_components.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, num_mixtures, num_mixtures]
            one_hot_components = one_hot_components.reshape(-1, self.params['num_mixtures'])  # [batch_size * num_mixtures, num_mixtures]
            
            # Create multiple copies of each state, one for each mixture component.
            extended_states = states_prep.unsqueeze(1).repeat(1, self.params['num_mixtures'], 1)
            
            # Sample an action from each mixture component for each state.
            extended_actions = self.gmm.sample_actions_for_components(states_prep)
            
            flat_states = extended_states.reshape(-1, extended_states.shape[-1])
            flat_actions = extended_actions.reshape(-1, extended_actions.shape[-1])
            q_values = self.q_nets[0](flat_states, flat_actions)
        
        mode_predictions = self.mode_net(flat_states, one_hot_components)
        mode_loss = F.mse_loss(mode_predictions, q_values, reduction='mean')

        self.mode_net_opt.zero_grad()
        mode_loss.backward()
        self.mode_net_opt.step()
    
        return {'mode_function/mode_loss': mode_loss.item()}
    
    def get_mode_actions(self, states_prep):
        batch_size = states_prep.shape[0]
        # Create a one-hot encoding for each mixture component for every state in the batch.
        one_hot_components = torch.eye(self.params['num_mixtures']).to(self.device)  # [num_mixtures, num_mixtures]
        one_hot_components = one_hot_components.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, num_mixtures, num_mixtures]
        one_hot_components = one_hot_components.reshape(-1, self.params['num_mixtures'])  # [batch_size * num_mixtures, num_mixtures]
    
        # Create multiple copies of each state, one for each mixture component.
        extended_states = states_prep.unsqueeze(1).repeat(1, self.params['num_mixtures'], 1)
        extended_states = extended_states.reshape(-1, extended_states.shape[-1])  # [batch_size * num_mixtures, state_dim]
    
        # Evaluate the mode value for each Gaussian
        mode_values = self.mode_net(extended_states, one_hot_components).reshape(batch_size, self.params['num_mixtures'])
        
        # Select the Gaussian with the highest expected return
        max_indices = torch.argmax(mode_values, dim=1)
        means, log_stds, weights = self.gmm(states_prep)
        stds = log_stds.exp()
        m = torch.distributions.Normal(means[torch.arange(batch_size), max_indices], 
                                       stds[torch.arange(batch_size), max_indices])
        actions = m.sample()
        return torch.clip(actions, -1, 1)
    
    def train_policy(self, batch_size):
        states, _, _, _, _ = self.replay_buffer.sample(batch_size)
        states_prep, _, _, _, _ = self.preprocess(states=states)
        chosen_actions = self.get_mode_actions(states_prep)        
        chosen_actions = chosen_actions + self.params['ac_rand'] * torch.randn_like(chosen_actions)
       
        gen_dist, gen_actions = self.policy(states_prep)
        with torch.no_grad():
            q_values = self.q_nets[0](states_prep, chosen_actions)
            curr_q_values = self.q_nets[0](states_prep, gen_actions)
            advs = q_values - curr_q_values
            weights = torch.clip(torch.exp(self.params['beta'] * advs), 0, self.params['C']).squeeze()
              
        policy_loss = ((gen_actions - chosen_actions).pow(2).mean(dim=1) * weights).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        return {'policy/loss': policy_loss.item(),
                'policy/weights': weights.mean().item()}

    def train_GMM(self, batch_size):
        states, actions, _, _, _ = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
        
        means, log_stds, weights = self.gmm(states_prep)
        stds = (log_stds.clamp(-15, 0)).exp()
        m = torch.distributions.Normal(means, stds)
        log_probs = m.log_prob(actions_prep.unsqueeze(1).expand_as(means))
        log_probs = log_probs.sum(-1)

        weighted_log_probs = torch.logsumexp(log_probs + torch.log(weights + 1e-8), dim=-1)
        gmm_loss = - weighted_log_probs.mean()

        self.gmm_opt.zero_grad()
        gmm_loss.backward()
        self.gmm_opt.step()

        return {
                'gmm_policy/loss': gmm_loss.item(),
                'gmm_policy/log_stds': log_stds.mean().item(),
                'gmm_policy/weights_mean': weights.mean().item(),
                'gmm_policy/weights_std': weights.std().item(),
                'gmm_policy/log_probs_mean': log_probs.mean().item(),
                'gmm_policy/log_probs_std': log_probs.std().item(),
            }
    
    @torch.no_grad()
    def get_action(self, state):
        state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
        action = self.policy(state_prep)[1].cpu().numpy().squeeze()
        clipped_action = np.clip(action, -1, 1)
        return clipped_action
     
    def update_target_nets(self, net, target_net):
        for k in range(len(net)):
            for param, target_param in zip(net[k].parameters(), target_net[k].parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
