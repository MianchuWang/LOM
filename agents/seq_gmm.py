import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from networks.networks import CVAE_network, Qnetwork


class temp_MixtureGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, num_mixtures=2, sample_quantile=0.6):
        super(temp_MixtureGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_mixtures = num_mixtures
        self.sample_quantile = sample_quantile

        # Layers for mean, log standard deviation, and mixture weights
        self.first_layer = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())
        self.second_layer = nn.Sequential(nn.Linear(256, 256), nn.ReLU())

        self.mean_layer = nn.Linear(256, action_dim * num_mixtures)
        self.log_std_layer = nn.Linear(256, action_dim * num_mixtures)
        self.weights_layer = nn.Linear(256, num_mixtures)

    def forward(self, state):
        x = self.first_layer(state)
        x = self.second_layer(x)
        means = self.mean_layer(x).view(-1, self.num_mixtures, self.action_dim)
        log_stds = self.log_std_layer(x).view(-1, self.num_mixtures, self.action_dim)
        weights = F.softmax(self.weights_layer(x), dim=-1)
        return means, log_stds, weights

    def sample_action(self, state):
        means, log_stds, weights = self.forward(state)
        stds = log_stds.exp()

        # Choose a component from the mixture
        weights[weights < torch.quantile(weights, self.sample_quantile)] = 0
        mixture_indices = torch.multinomial(weights, 1).squeeze(-1)
        chosen_means = means[torch.arange(means.size(0)), mixture_indices]
        chosen_stds = stds[torch.arange(stds.size(0)), mixture_indices]

        # Sample from the chosen Gaussian
        normal = torch.distributions.Normal(chosen_means, chosen_stds)
        action = normal.sample()

        return action

class seqGMM(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.num_mixtures = 20
        self.alpha = 2.5
        self.training_steps = 0
        self.policy_delay = 2
        self.sample_quantile = 0.6

        self.policy = temp_MixtureGaussianPolicy(self.state_dim, self.ac_dim, self.num_mixtures, self.sample_quantile).to(device=self.device)
        self.target_policy = temp_MixtureGaussianPolicy(self.state_dim, self.ac_dim, self.num_mixtures, self.sample_quantile).to(device=self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

        self.K = 2
        self.q_nets, self.q_target_nets, self.q_net_opts = [], [], []
        for k in range(self.K):
            self.q_nets.append(Qnetwork(self.state_dim, 2*self.ac_dim).to(device=self.device))
            self.q_target_nets.append(Qnetwork(self.state_dim, 2*self.ac_dim).to(device=self.device))
            self.q_target_nets[k].load_state_dict(self.q_nets[k].state_dict())
            self.q_net_opts.append(torch.optim.Adam(self.q_nets[k].parameters(), lr=3e-4))


    def train_models(self, batch_size=512):
        GMM_info = self.train_GMM(batch_size=batch_size)
        value_info = self.train_value_function(batch_size=batch_size)
        if self.training_steps % self.policy_delay == 0:
            self.update_target_nets([self.policy], [self.target_policy])
            self.update_target_nets(self.q_nets, self.q_target_nets)
        self.training_steps += 1
        return {**GMM_info, **value_info}


    def train_GMM(self, batch_size=512):
        states, actions, _, _, _ = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
        means, log_stds, weights = self.policy(states_prep)
        gmm_loss = self.compute_loss(means, log_stds, weights, actions_prep)
        self.policy_opt.zero_grad()
        gmm_loss.backward()
        self.policy_opt.step()
        return {'training/gmm_loss': gmm_loss.item()}

    def compute_values(self, states, means, log_stds):
        values = torch.zeros(states.shape[0], self.K, 1).to(device=self.device)
        for i in range(self.K):
            dist = torch.cat([means, log_stds], dim=-1)
            values[:, i] = self.q_nets[i](states, dist)
        uncertainties = values.std(dim=1)
        means = values.mean(dim=1)
        return means, uncertainties

    def train_value_function(self, batch_size):
        states, actions, rewards, next_states, next_actions, terminals = self.replay_buffer.sample_with_next_action(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals)
        _, next_actions_prep, _, _, _ = self.preprocess(actions=next_actions)

        with torch.no_grad():
            gmm_next_means, gmm_next_log_std = self.identify_gmm_components(next_states_prep, next_actions_prep)
            next_dist = torch.cat([gmm_next_means, gmm_next_log_std], dim=-1)
            q_next_values = torch.zeros(self.K, batch_size, 1).to(device=self.device)
            for i in range(self.K):
                q_next_values[i] = self.q_target_nets[i](next_states_prep, next_dist)
            q_next_value = torch.min(q_next_values, dim=0)[0]
            target_q_value = rewards_prep + (1 - terminals_prep) * self.discount * q_next_value

        for k in range(self.K):
            gmm_means, gmm_log_std = self.identify_gmm_components(states_prep, actions_prep)
            curr_dist = torch.cat([gmm_means, gmm_log_std], dim=-1)
            pred_q_value = self.q_nets[k](states_prep, curr_dist)
            q_loss = ((target_q_value - pred_q_value) ** 2).mean()
            self.q_net_opts[k].zero_grad()
            q_loss.backward()
            self.q_net_opts[k].step()

        return {'Q/loss': q_loss.item(),
                'Q/pred_value': pred_q_value.mean().item(),
                'Q/target_value': target_q_value.mean().item()}

    def update_target_nets(self, net, target_net):
        for k in range(len(net)):
            for param, target_param in zip(net[k].parameters(), target_net[k].parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

    def reset(self):
        return

    @torch.no_grad()
    def get_action(self, state):
        state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
        means, log_stds, weights = self.policy(state_prep)

        extended_states = state_prep.unsqueeze(1).repeat(1, self.num_mixtures, 1)
        value_means, uncertainties = self.compute_values(extended_states.reshape(-1, self.state_dim),
                                                         means.reshape(-1, self.ac_dim),
                                                         log_stds.reshape(-1, self.ac_dim))

        scores = (value_means - value_means.mean())
        scores = scores.reshape(1, self.num_mixtures)
        scores[weights < torch.quantile(weights, self.sample_quantile)] = -1000
        class_label = torch.argmax(scores, dim=1)
        return means[0, class_label].cpu().numpy().squeeze()

    def log_prob_gaussian(self, x, mean, std):
        return -0.5 * ((x - mean) / std).pow(2) - torch.log(std) - 0.5 * torch.log(2 * torch.tensor(torch.pi))

    def compute_loss(self, means, log_stds, weights, actions):
        batch_size, num_mixtures, _ = means.shape

        actions_expanded = actions.unsqueeze(1).expand_as(means)

        # Compute log probabilities for each Gaussian in the mixture
        stds = log_stds.exp()
        log_probs = self.log_prob_gaussian(actions_expanded, means, stds)

        # Sum log probabilities over action dimensions
        log_probs = log_probs.sum(-1)  # shape: [batch_size, num_mixtures]

        # Weighted log sum exponent trick for numerical stability
        max_log_probs = log_probs.max(dim=1, keepdim=True)[0]
        weighted_log_probs = weights * (log_probs - max_log_probs).exp()
        log_sum_exp = max_log_probs + torch.log(weighted_log_probs.sum(dim=1, keepdim=True))

        # Negative log likelihood
        loss = -log_sum_exp.mean()
        return loss

    @torch.no_grad()
    def identify_gmm_components(self, states_prep, actions_prep):
        means, log_stds, weights = self.policy(states_prep)

        # Expand actions to match the shape of means and stds
        actions_expanded = actions_prep.unsqueeze(1).expand_as(means)

        # Compute log probabilities for each Gaussian in the mixture
        stds = log_stds.exp()
        log_probs = self.log_prob_gaussian(actions_expanded, means, stds)

        # Sum log probabilities over action dimensions
        log_probs = log_probs.sum(-1)  # shape: [batch_size, num_mixtures]

        log_probs[weights<torch.quantile(weights, self.sample_quantile)] = -10000

        # Find the index of the maximum log probability for each batch item
        most_likely_components = log_probs.argmax(dim=1)

        # Gather the means and log_stds of the most likely components
        most_likely_means = means[torch.arange(means.size(0)), most_likely_components]
        most_likely_log_stds = log_stds[torch.arange(log_stds.size(0)), most_likely_components]

        return most_likely_means, most_likely_log_stds




