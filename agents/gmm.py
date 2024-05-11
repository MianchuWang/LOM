import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from networks.networks import CVAE_network, Qnetwork, Policy


class MixtureGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, num_mixtures=2, sample_quantile=0.6):
        super(MixtureGaussianPolicy, self).__init__()
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


class GMM(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.num_mixtures = 2
        self.training_steps = 0
        self.sample_quantile = 0.6

        self.gmm_policy = MixtureGaussianPolicy(self.state_dim, self.ac_dim, self.num_mixtures,
                                                 self.sample_quantile).to(device=self.device)
        self.gmm_policy_opt = torch.optim.Adam(self.gmm_policy.parameters(), lr=1e-3)

        
    def train_models(self, batch_size=512):
        GMM_info = self.train_GMM(batch_size=batch_size)
        self.training_steps += 1
        return {**GMM_info}

    def train_GMM(self, batch_size=512):
        states, actions, _, _, _ = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
        means, log_stds, weights = self.gmm_policy(states_prep)
        gmm_loss = self.compute_loss(means, log_stds, weights, actions_prep)
        self.gmm_policy_opt.zero_grad()
        gmm_loss.backward()
        self.gmm_policy_opt.step()
        return {'gmm_policy/loss': gmm_loss.item(),
                'gmm_policy/log_stds': log_stds.mean().item()}


    def reset(self):
        return


    @torch.no_grad()
    def get_action(self, state):
        state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
        action = self.gmm_policy.sample_action(state_prep).cpu().numpy().squeeze()
        clipped_action = np.clip(action, -1, 1)
        return clipped_action


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


