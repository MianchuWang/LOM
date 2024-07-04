import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from agents.base_agent import BaseAgent
from networks.networks import Qnetwork, Policy


class MixtureGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, num_mixtures, sample_quantile):
        super(MixtureGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_mixtures = num_mixtures
        self.sample_quantile = sample_quantile

        self.model = nn.Sequential(nn.Linear(state_dim, 512), 
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, action_dim * num_mixtures * 2 + num_mixtures))
        
    def forward(self, state):
        x = self.model(state)
        an = self.action_dim * self.num_mixtures
        means = torch.tanh(x[:, : an].view(-1, self.num_mixtures, self.action_dim))
        means = torch.clamp(means, -0.9999, 0.9999)
        log_stds = x[:, an: 2*an].view(-1, self.num_mixtures, self.action_dim)
        weights = F.softmax(x[:, 2*an: ], dim=-1)
        return means, log_stds, weights

    def sample_action(self, state):
        means, log_stds, weights = self.forward(state)
        max_weight_indices = torch.argmax(weights, dim=-1)
        chosen_means = means[torch.arange(means.size(0)), max_weight_indices]
        return chosen_means


class seqGMM(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.num_mixtures = agent_params['num_mixtures']
        self.training_steps = 0
        self.policy_delay = 2
        self.sample_quantile = agent_params['sample_quantile'] # 0 for no filtering

        self.policy = MixtureGaussianPolicy(self.state_dim, self.ac_dim, self.num_mixtures,
                                            self.sample_quantile).to(device=self.device)
        #self.policy.load_state_dict(torch.load('gmm_models/gmm/' + agent_params['env_name'] + '.pth'))
        
        self.target_policy = MixtureGaussianPolicy(self.state_dim, self.ac_dim, self.num_mixtures,
                                                   self.sample_quantile).to(device=self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

        self.delta_policy = Policy(self.state_dim, self.ac_dim, deterministic=True).to(device=self.device)
        self.target_delta_policy = Policy(self.state_dim, self.ac_dim, deterministic=True).to(device=self.device)
        self.target_delta_policy.load_state_dict(self.delta_policy.state_dict())
        self.delta_policy_opt = torch.optim.Adam(self.delta_policy.parameters(), lr=1e-3)


        self.mode_K = 1
        self.q_mode_nets, self.q_mode_target_nets, self.q_mode_net_opts = [], [], []
        for k in range(self.mode_K):
            self.q_mode_nets.append(Qnetwork(self.state_dim, 2 * self.ac_dim).to(device=self.device))
            self.q_mode_target_nets.append(Qnetwork(self.state_dim, 2 * self.ac_dim).to(device=self.device))
            self.q_mode_target_nets[k].load_state_dict(self.q_mode_nets[k].state_dict())
            self.q_mode_net_opts.append(torch.optim.Adam(self.q_mode_nets[k].parameters(), lr=3e-4))
        '''
        params = torch.load('gmm_models/value_functions/' + agent_params['env_name'] + '.pth')
        for k in range(self.mode_K):
            self.q_mode_nets[k].load_state_dict(params[k])
        '''
        self.K = 4
        self.q_nets, self.q_target_nets, self.q_net_opts = [], [], []
        for k in range(self.K):
            self.q_nets.append(Qnetwork(self.state_dim, self.ac_dim).to(device=self.device))
            self.q_target_nets.append(Qnetwork(self.state_dim, self.ac_dim).to(device=self.device))
            self.q_target_nets[k].load_state_dict(self.q_nets[k].state_dict())
            self.q_net_opts.append(torch.optim.Adam(self.q_nets[k].parameters(), lr=3e-4))


    def train_models(self, batch_size=512):
        GMM_info = {}
        GMM_info = self.train_GMM(batch_size=batch_size)
        delta_policy_info = {}#self.train_delta_policy(batch_size=batch_size)
        mode_value_info = {}#self.train_mode_value_function(batch_size=batch_size)
        value_info = {}#self.train_value_function(batch_size=batch_size)
        if self.training_steps % self.policy_delay == 0:
            self.update_target_nets([self.policy], [self.target_policy])
            #self.update_target_nets([self.delta_policy], [self.target_delta_policy])
            self.update_target_nets(self.q_mode_nets, self.q_mode_target_nets)
            #self.update_target_nets(self.q_nets, self.q_target_nets)
        self.training_steps += 1
        return {**GMM_info, **mode_value_info, **value_info, **delta_policy_info}


    def train_GMM(self, batch_size):
        states, actions, _, _, _ = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
        means, log_stds, weights = self.policy(states_prep)
        stds = (log_stds.clamp(-15, 0)).exp() # stds = log_stds.exp()
        
        m = torch.distributions.Normal(means, stds)
        log_probs = m.log_prob(actions_prep.unsqueeze(1).expand_as(means))
        log_probs = log_probs.sum(-1)

        weighted_log_probs = log_probs + torch.log(weights)
        
        gmm_loss = - weighted_log_probs.mean()

        self.policy_opt.zero_grad()
        gmm_loss.backward()
        self.policy_opt.step()

        return {
                'gmm_policy/loss': gmm_loss.item(),
                'gmm_policy/log_stds': log_stds.mean().item(),
                'gmm_policy/weights_mean': weights.mean().item(),
                'gmm_policy/weights_std': weights.std().item(),
                'gmm_policy/log_probs_mean': log_probs.mean().item(),
                'gmm_policy/log_probs_std': log_probs.std().item(),
            }
        

    def train_delta_policy(self, batch_size):
        states, actions, _, _, _ = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)

        gen_actions = self.delta_policy(states_prep)
        with torch.no_grad():
            loc, std_log = self.mode_policy(states_prep)
            std = torch.clamp(std_log.exp(), 0, 10)
            sampled_actions = torch.normal(loc, std)
            sampled_actions = torch.clip(sampled_actions, -1, 1)

            values_2 = torch.zeros(self.K, batch_size, 1).to(device=self.device)
            for k in range(self.K):
                values_2[k] = self.q_nets[k](states_prep, sampled_actions)
            values_2 = torch.min(values_2, dim=0)[0]

            curr_values_2 = torch.zeros(self.K, batch_size, 1).to(device=self.device)
            for k in range(self.K):
                curr_values_2[k] = self.q_nets[k](states_prep, gen_actions)
            curr_values_2 = torch.min(curr_values_2, dim=0)[0]

            advs_2 = values_2 - curr_values_2
            weights_exp_2 = torch.clip(torch.exp(2 * advs_2), None, 100).squeeze()

        # weights_exp_2 = torch.ones_like(weights_exp_2)
        delta_policy_loss = ((gen_actions - sampled_actions).pow(2).mean(dim=1) * weights_exp_2).mean()

        self.delta_policy_opt.zero_grad()
        delta_policy_loss.backward()
        self.delta_policy_opt.step()

        return {'delta_policy/loss': delta_policy_loss.item()}


    def train_value_function(self, batch_size):
        states, actions, rewards, next_states, terminals  = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals)

        with torch.no_grad():
            target_actions = self.target_delta_policy(next_states_prep)
            smooth_noise = torch.clamp(0.2 * torch.randn_like(target_actions), -0.5, 0.5)
            target_actions = torch.clamp(target_actions + smooth_noise, -1, 1)

            q_next_values = torch.zeros(self.K, batch_size, 1).to(device=self.device)
            for i in range(self.K):
                q_next_values[i] = self.q_target_nets[i](next_states_prep, target_actions)
            q_next_value = torch.min(q_next_values, dim=0)[0]
            target_q_value = rewards_prep + (1 - terminals_prep) * self.discount * q_next_value

        for k in range(self.K):
            pred_q_value = self.q_nets[k](states_prep, actions_prep)
            q_loss = ((target_q_value - pred_q_value)**2).mean()
            self.q_net_opts[k].zero_grad()
            q_loss.backward()
            self.q_net_opts[k].step()

        return {'Q/loss': q_loss.item(),
                'Q/pred_value': pred_q_value.mean().item(),
                'Q/target_value': target_q_value.mean().item()}


    def train_mode_value_function(self, batch_size):
        states, actions, rewards, next_states, next_actions, terminals = self.replay_buffer.sample_with_next_action(
            batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals)
        _, next_actions_prep, _, _, _ = self.preprocess(actions=next_actions)

        with torch.no_grad():
            gmm_next_means, gmm_next_log_std = self.identify_gmm_components(next_states_prep, next_actions_prep)
            next_dist = torch.cat([gmm_next_means, gmm_next_log_std], dim=-1)
            q_next_values = torch.zeros(self.mode_K, batch_size, 1).to(device=self.device)
            for i in range(self.mode_K):
                q_next_values[i] = self.q_mode_target_nets[i](next_states_prep, next_dist)
            q_next_value = torch.min(q_next_values, dim=0)[0]
            target_q_value = rewards_prep + (1 - terminals_prep) * self.discount * q_next_value

        for k in range(self.mode_K):
            gmm_means, gmm_log_std = self.identify_gmm_components(states_prep, actions_prep)
            curr_dist = torch.cat([gmm_means, gmm_log_std], dim=-1)
            pred_q_value = self.q_mode_nets[k](states_prep, curr_dist)
            q_loss = ((target_q_value - pred_q_value) ** 2).mean()
            self.q_mode_net_opts[k].zero_grad()
            q_loss.backward()
            self.q_mode_net_opts[k].step()

        return {'Q_mode/loss': q_loss.item(),
                'Q_mode/pred_value': pred_q_value.mean().item(),
                'Q_mode/target_value': target_q_value.mean().item()}


    def update_target_nets(self, net, target_net):
        for k in range(len(net)):
            for param, target_param in zip(net[k].parameters(), target_net[k].parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)


    def compute_values(self, states, means, log_stds):
        values = torch.zeros(states.shape[0], self.mode_K, 1).to(device=self.device)
        for i in range(self.mode_K):
            dist = torch.cat([means, log_stds], dim=-1)
            values[:, i] = self.q_mode_nets[i](states, dist)
        means = values.mean(dim=1)
        return means


    def reset(self):
        return

    def mode_policy(self, state):
        means, log_stds, weights = self.policy(state)
        extended_states = state.unsqueeze(1).repeat(1, self.num_mixtures, 1)
        scores = self.compute_values(extended_states.reshape(-1, self.state_dim),
                                     means.reshape(-1, self.ac_dim),
                                     log_stds.reshape(-1, self.ac_dim))
        scores = scores.reshape(-1, self.num_mixtures)
        class_label = torch.argmax(scores, dim=1)
        loc = means[0, class_label]
        std = log_stds[0, class_label].exp()
        return loc, std

    @torch.no_grad()
    def get_action(self, state):
        state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
        #action = self.delta_policy(state_prep).cpu().numpy().squeeze()
        #action = self.policy.sample_action(state_prep).cpu().numpy().squeeze()
        action = self.mode_policy(state_prep)[0].cpu().numpy().squeeze()
        clipped_action = np.clip(action, -1, 1)
        return clipped_action


    def log_prob_gaussian(self, x, mean, std):
        return -0.5 * ((x - mean) / std).pow(2) - torch.log(std) - 0.5 * torch.log(2 * torch.tensor(torch.pi))


    @torch.no_grad()
    def identify_gmm_components(self, states_prep, actions_prep):
        means, log_stds, weights = self.policy(states_prep)
        actions_expanded = actions_prep.unsqueeze(1).expand_as(means)
        stds = log_stds.exp()
        log_probs = self.log_prob_gaussian(actions_expanded, means, stds)
        log_probs = log_probs.sum(-1)  # shape: [batch_size, num_mixtures]
        most_likely_components = log_probs.argmax(dim=1)
        most_likely_means = means[torch.arange(means.size(0)), most_likely_components]
        most_likely_log_stds = log_stds[torch.arange(log_stds.size(0)), most_likely_components]
        return most_likely_means, most_likely_log_stds

    
    def plot_gmm_weights_distribution(self, num_samples=1):
            # Randomly sample states from the replay buffer
            states, _, _, _, _ = self.replay_buffer.sample(num_samples)
            
            # Preprocess the states
            states_prep, _, _, _, _ = self.preprocess(states=states)
            
            # Get GMM weights
            _, _, weights = self.policy(states_prep)
            
            # Convert weights to numpy array
            weights_np = weights.cpu().detach().numpy()
            
            # Sort weights for each state and calculate the average weights
            sorted_weights = np.sort(weights_np, axis=1)[:, ::-1]
            average_sorted_weights = np.mean(sorted_weights, axis=0)
            
            # Plot the bar chart for average sorted weights
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, self.num_mixtures + 1), average_sorted_weights)
            plt.xlabel('Gaussian Index')
            plt.ylabel('Average Weight')
            plt.title('Average Distribution of GMM Weights for Randomly Selected States')
            plt.xticks(range(1, self.num_mixtures + 1))
            plt.grid(True)
            plt.show()
    

