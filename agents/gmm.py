import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import BaseAgent


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

        logits = torch.clamp(self.weights_layer(x), -5, 5)
        weights = F.softmax(logits, dim=-1)
        return means, log_stds, weights

    def sample_action(self, state):
        means, log_stds, weights = self.forward(state)
        stds = log_stds.exp()
        
        # Choose a component from the mixture
        weights[weights < torch.quantile(weights, self.sample_quantile)] = 0
        mixture_indices = torch.multinomial(weights, 1).squeeze(-1)
        chosen_means = means[torch.arange(means.size(0)), mixture_indices]
        chosen_stds = stds[torch.arange(stds.size(0)), mixture_indices]

        return chosen_means

        # Sample from the chosen Gaussian
        normal = torch.distributions.Normal(chosen_means, chosen_stds)
        action = normal.sample()

        return action


class GMM(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.num_mixtures = 5
        self.training_steps = 0
        self.sample_quantile = 0.6
        self.policy = MixtureGaussianPolicy(self.state_dim, self.ac_dim, self.num_mixtures,
                                                 self.sample_quantile).to(device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-4)


    def train_models(self, batch_size=512):
        GMM_info = self.train_GMM(batch_size=512)
        self.training_steps += 1
        return {**GMM_info}


    def train_GMM(self, batch_size):
        states, actions, _, _, _ = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
        means, log_stds, weights = self.policy(states_prep)
        gmm_loss = self.compute_loss(means, log_stds, weights, actions_prep)
        self.policy_opt.zero_grad()
        gmm_loss.backward()
        self.policy_opt.step()
        return {'gmm_policy/loss': gmm_loss.item(),
                'gmm_policy/log_stds': log_stds.mean().item()}


    def reset(self):
        return
    
    @torch.no_grad()
    def get_action(self, state):
        state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
        action = self.policy.sample_action(state_prep).cpu().numpy().squeeze()
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
        weighted_log_probs = weights * log_probs.exp()
        weighted_log_probs = torch.clamp(weighted_log_probs, 0.002, None)
        log_sum_exp = torch.log(weighted_log_probs.sum(dim=1, keepdim=True))

        # Negative log likelihood
        loss = - log_sum_exp.mean()
        return loss

