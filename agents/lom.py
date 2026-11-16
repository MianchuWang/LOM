from tqdm import tqdm
import time

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from networks.networks import Qnetwork, Policy
from networks.gnn import ModeGraphNetwork, mode_ranking_loss

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
        self.mode_selection = agent_params.get('mode_selection', 'top1')
        self.top_k = agent_params.get('top_k', 3)
        self.mode_action_samples = agent_params.get('mode_action_samples', 8)
        if self.top_k < 1:
            raise ValueError('top_k must be at least 1')
        if (self.mode_selection != 'top1' and
                self.top_k > self.params['num_mixtures']):
            raise ValueError(
                'top_k ({}) cannot exceed num_mixtures ({})'.format(
                    self.top_k, self.params['num_mixtures']))
        if self.mode_action_samples < 1:
            raise ValueError('mode_action_samples must be at least 1')
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

        node_dim = 2 * self.ac_dim + 2
        self.mode_gnn = ModeGraphNetwork(
            node_dim=node_dim,
            hidden_dim=agent_params.get('gnn_hidden_dim', 128),
            num_layers=2,
        ).to(device=self.device)
        self.mode_gnn_opt = torch.optim.Adam(
            self.mode_gnn.parameters(), lr=agent_params.get('gnn_lr', 3e-4))
        self.gnn_ranking_weight = agent_params.get('gnn_ranking_weight', 0.25)

        self.q_nets, self.q_target_nets, self.q_net_opts = [], [], []
        for k in range(self.params['K']):
            self.q_nets.append(Qnetwork(self.state_dim, self.ac_dim).to(device=self.device))
            self.q_target_nets.append(Qnetwork(self.state_dim, self.ac_dim).to(device=self.device))
            self.q_target_nets[k].load_state_dict(self.q_nets[k].state_dict())
            self.q_net_opts.append(torch.optim.Adam(self.q_nets[k].parameters(), lr=3e-4))  
        
        gmm_checkpoint = agent_params.get('gmm_checkpoint', '')
        if gmm_checkpoint:
            self.gmm.load_state_dict(torch.load(gmm_checkpoint, map_location=self.device))
            print('Loaded GMM model from {}.'.format(gmm_checkpoint))
        else:
            gmm_steps = agent_params.get('gmm_pretrain_steps', 200000)
            print('Learning the GMM model for {} steps ...'.format(gmm_steps))
            for _ in tqdm(range(gmm_steps)):
                self.train_GMM(batch_size=agent_params.get('gmm_batch_size', 512))

            if agent_params.get('save_gmm', 1):
                torch.save(self.gmm.state_dict(), 'gmm_' + self.env_name + '.pth')
        

    def train_models(self, batch_size=512):
        value_start = time.perf_counter()
        self.value_info = self.train_value_function(batch_size=256)
        value_seconds = time.perf_counter() - value_start
        mode_start = time.perf_counter()
        self.mode_info = self.train_mode_function(batch_size=1024)
        mode_seconds = time.perf_counter() - mode_start
        if self.mode_selection == 'topk_gnn':
            gnn_start = time.perf_counter()
            self.gnn_info = self.train_mode_gnn(batch_size=256)
            gnn_seconds = time.perf_counter() - gnn_start
        else:
            self.gnn_info = {}
            gnn_seconds = 0.0
        policy_start = time.perf_counter()
        self.policy_info = self.train_policy(batch_size=256)
        policy_seconds = time.perf_counter() - policy_start
        if self.training_steps % self.policy_delay == 0:
            self.update_target_nets(self.q_nets, self.q_target_nets)
        self.training_steps += 1
        timing_info = {
            'time/value_step_seconds': value_seconds,
            'time/mode_step_seconds': mode_seconds,
            'time/gnn_step_seconds': gnn_seconds,
            'time/policy_step_seconds': policy_seconds,
        }
        return {**self.value_info, **self.policy_info, **self.mode_info,
                **self.gnn_info, **timing_info}
    
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
    
    def get_mode_scores(self, states_prep):
        batch_size = states_prep.shape[0]
        # Create a one-hot encoding for each mixture component for every state in the batch.
        one_hot_components = torch.eye(self.params['num_mixtures']).to(self.device)  # [num_mixtures, num_mixtures]
        one_hot_components = one_hot_components.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, num_mixtures, num_mixtures]
        one_hot_components = one_hot_components.reshape(-1, self.params['num_mixtures'])  # [batch_size * num_mixtures, num_mixtures]
    
        # Create multiple copies of each state, one for each mixture component.
        extended_states = states_prep.unsqueeze(1).repeat(1, self.params['num_mixtures'], 1)
        extended_states = extended_states.reshape(-1, extended_states.shape[-1])  # [batch_size * num_mixtures, state_dim]
    
        # Evaluate the mode value for each Gaussian
        return self.mode_net(extended_states, one_hot_components).reshape(
            batch_size, self.params['num_mixtures'])

    @staticmethod
    def _gather_modes(values, indices):
        gather_index = indices
        while gather_index.ndim < values.ndim:
            gather_index = gather_index.unsqueeze(-1)
        return torch.gather(values, 1, gather_index.expand(*indices.shape, *values.shape[2:]))

    def get_topk_modes(self, states_prep):
        """Return top-k component parameters ranked by the original mode net."""
        mode_scores = self.get_mode_scores(states_prep)
        top_scores, top_indices = torch.topk(mode_scores, self.top_k, dim=1)
        means, log_stds, mixture_weights = self.gmm(states_prep)
        stds = log_stds.clamp(-15, 0).exp()
        return {
            'indices': top_indices,
            'scores': top_scores,
            'means': self._gather_modes(means, top_indices),
            'stds': self._gather_modes(stds, top_indices),
            'weights': self._gather_modes(mixture_weights, top_indices),
        }

    def _sampled_q_targets(self, states_prep, modes):
        """Monte Carlo estimates of Q(s, a) for every selected component."""
        batch_size, num_modes, action_dim = modes['means'].shape
        dist = torch.distributions.Normal(modes['means'], modes['stds'])
        actions = dist.sample((self.mode_action_samples,)).clamp(-1, 1)
        flat_actions = actions.permute(1, 2, 0, 3).reshape(-1, action_dim)
        flat_states = states_prep[:, None, None, :].expand(
            -1, num_modes, self.mode_action_samples, -1).reshape(-1, self.state_dim)
        q_values = self.q_nets[0](flat_states, flat_actions)
        return q_values.reshape(
            batch_size, num_modes, self.mode_action_samples).mean(dim=-1)

    @staticmethod
    def _normalise_mode_targets(targets):
        centered = targets - targets.mean(dim=1, keepdim=True)
        if targets.shape[1] == 1:
            return centered
        return centered / targets.std(dim=1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _mode_node_features(modes):
        return torch.cat([
            modes['means'], modes['stds'], modes['weights'].unsqueeze(-1),
            modes['scores'].unsqueeze(-1)
        ], dim=-1)

    def train_mode_gnn(self, batch_size):
        states, _, _, _, _ = self.replay_buffer.sample(batch_size)
        states_prep, _, _, _, _ = self.preprocess(states=states)
        with torch.no_grad():
            modes = self.get_topk_modes(states_prep)
            q_targets = self._sampled_q_targets(states_prep, modes)
            targets = self._normalise_mode_targets(q_targets)
            node_features = self._mode_node_features(modes)

        predictions = self.mode_gnn(node_features, modes['means'])
        total_loss, regression_loss, ranking_loss = mode_ranking_loss(
            predictions, targets, self.gnn_ranking_weight)
        self.mode_gnn_opt.zero_grad()
        total_loss.backward()
        self.mode_gnn_opt.step()
        return {
            'gnn/loss': total_loss.item(),
            'gnn/regression_loss': regression_loss.item(),
            'gnn/ranking_loss': ranking_loss.item(),
        }

    def get_mode_actions(self, states_prep):
        batch_size = states_prep.shape[0]
        if self.mode_selection == 'top1':
            mode_values = self.get_mode_scores(states_prep)
            selected_indices = torch.argmax(mode_values, dim=1)
            means, log_stds, _ = self.gmm(states_prep)
            # Keep the original LOM top-1 sampling behavior unchanged.
            stds = log_stds.exp()
            selected_means = means[torch.arange(batch_size, device=self.device), selected_indices]
            selected_stds = stds[torch.arange(batch_size, device=self.device), selected_indices]
        else:
            modes = self.get_topk_modes(states_prep)
            if self.mode_selection == 'topk_q':
                with torch.no_grad():
                    refined_scores = self._sampled_q_targets(states_prep, modes)
            elif self.mode_selection == 'topk_gnn':
                with torch.no_grad():
                    refined_scores = self.mode_gnn(
                        self._mode_node_features(modes), modes['means'])
            else:
                raise ValueError('Unknown mode_selection: {}'.format(self.mode_selection))
            selected = torch.argmax(refined_scores, dim=1)
            batch_indices = torch.arange(batch_size, device=self.device)
            selected_means = modes['means'][batch_indices, selected]
            selected_stds = modes['stds'][batch_indices, selected]

        m = torch.distributions.Normal(selected_means, selected_stds)
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
