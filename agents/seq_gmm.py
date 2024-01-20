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
    def __init__(self, state_dim, action_dim, latent_dim, num_mixtures=2):
        super(temp_MixtureGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_mixtures = num_mixtures
        self.latent_dim = latent_dim

        # Layers for mean, log standard deviation, and mixture weights
        self.first_layer = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())
        self.second_layer = nn.Sequential(nn.Linear(256+self.latent_dim, 256), nn.ReLU())
        self.latent_layer = nn.Linear(256, self.latent_dim)

        self.mean_layer = nn.Linear(256, action_dim * num_mixtures)
        self.log_std_layer = nn.Linear(256, action_dim * num_mixtures)
        self.weights_layer = nn.Linear(256, num_mixtures)

    def forward(self, state, latent):
        x = self.first_layer(state)
        x = torch.cat([x, latent], dim=-1)
        x = self.second_layer(x)
        latent = self.latent_layer(x)
        means = self.mean_layer(x).view(-1, self.num_mixtures, self.action_dim)
        log_stds = self.log_std_layer(x).view(-1, self.num_mixtures, self.action_dim)
        weights = F.softmax(self.weights_layer(x), dim=-1)
        return means, log_stds, weights, latent

    def sample_action(self, state, latent):
        if latent == None:
            latent = torch.randn(state.shape[0], self.latent_dim).to(state.device)
        means, log_stds, weights, latent = self.forward(state, latent)
        stds = log_stds.exp()

        # Choose a component from the mixture
        weights[weights < torch.quantile(weights, 0.6)] = 0
        mixture_indices = torch.multinomial(weights, 1).squeeze(-1)
        chosen_means = means[torch.arange(means.size(0)), mixture_indices]
        chosen_stds = stds[torch.arange(stds.size(0)), mixture_indices]

        # Sample from the chosen Gaussian
        normal = torch.distributions.Normal(chosen_means, chosen_stds)
        action = normal.sample()

        return action, latent

class seqGMM(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.latent_dim = 128
        self.num_mixtures = 20
        self.alpha = 2.5
        self.training_steps = 0
        self.policy_delay = 2
        self.latent = torch.randn(1, self.latent_dim).to(device=self.device)

        self.policy = temp_MixtureGaussianPolicy(self.state_dim, self.ac_dim, self.latent_dim, self.num_mixtures).to(device=self.device)
        self.target_policy = temp_MixtureGaussianPolicy(self.state_dim, self.ac_dim, self.latent_dim, self.num_mixtures).to(device=self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

        self.attention = nn.Sequential(nn.Linear(self.latent_dim+self.state_dim+self.num_mixtures*self.ac_dim*2, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, self.num_mixtures),
                                       nn.Softmax(dim=1)).to(device=self.device)
        self.attention_opt = torch.optim.Adam(self.attention.parameters(), lr=1e-3)


        self.K = 4
        self.q_nets, self.q_target_nets, self.q_net_opts = [], [], []
        for k in range(self.K):
            self.q_nets.append(Qnetwork(self.state_dim, self.ac_dim).to(device=self.device))
            self.q_target_nets.append(Qnetwork(self.state_dim, self.ac_dim).to(device=self.device))
            self.q_target_nets[k].load_state_dict(self.q_nets[k].state_dict())
            self.q_net_opts.append(torch.optim.Adam(self.q_nets[k].parameters(), lr=3e-4))


    def train_models(self, batch_size=512):
        GMM_info = self.train_GMM(batch_size=batch_size)
        attention_info = {}#self.train_attention(batch_size=batch_size)
        value_info = self.train_value_function(batch_size=batch_size)
        if self.training_steps % self.policy_delay == 0:
            self.update_target_nets([self.policy], [self.target_policy])
            self.update_target_nets(self.q_nets, self.q_target_nets)
        self.training_steps += 1
        return {**GMM_info, **value_info, **attention_info}


    def train_GMM(self, batch_size=512):
        states, actions = self.replay_buffer.sample_sequences(batch_size, sequence_length=5)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
        latent = torch.randn(batch_size, self.latent_dim).to(device=self.device)
        gmm_loss = 0
        for t in range(5):
            means, log_stds, weights, latent = self.policy(states_prep[:, t], latent)
            gmm_loss += self.compute_loss(means, log_stds, weights, actions_prep[:, t])
        self.policy_opt.zero_grad()
        gmm_loss.backward()
        self.policy_opt.step()
        return {'training/gmm_loss': gmm_loss.item()}

    def train_attention(self, batch_size=512):
        states, actions = self.replay_buffer.sample_sequences(batch_size, sequence_length=5)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
        latent = torch.randn(batch_size, self.latent_dim).to(device=self.device)
        attention_loss = 0
        for t in range(5):
            means, log_stds, weights, next_latent = self.policy(states_prep[:, t], latent)

            extended_states = states_prep[:, t].unsqueeze(1).repeat(1, self.num_mixtures, 1)
            value_means, uncertainties = self.compute_values(extended_states.reshape(-1, self.state_dim), means.reshape(-1, self.ac_dim))
            weights = (value_means - value_means.mean()) * (1 / uncertainties)
            weights = weights.reshape(batch_size, self.num_mixtures, 1)
            class_labels = torch.argmax(weights, dim=1)

            dist_mean_input = torch.cat([means[:, i] for i in range(self.num_mixtures)], axis=-1).to(device=self.device)
            dis_log_std_input = torch.cat([log_stds[:, i] for i in range(self.num_mixtures)], axis=-1).to(device=self.device)
            attention_input = torch.cat([states_prep[:, t], latent, dist_mean_input, dis_log_std_input], dim=-1).to(device=self.device)
            estimated_weights = self.attention(attention_input)
            attention_loss = torch.nn.functional.cross_entropy(estimated_weights, class_labels.squeeze())

            latent = next_latent

        self.attention_opt.zero_grad()
        attention_loss.backward()
        self.attention_opt.step()

        return {'training/attention_loss': attention_loss.item()}

    def compute_values(self, states, actions):
        values = torch.zeros(states.shape[0], self.K, 1).to(device=self.device)
        for i in range(self.K):
            values[:, i] = self.q_nets[i](states, actions)
        uncertainties = values.std(dim=1)
        means = values.mean(dim=1)
        return means, uncertainties

    def train_value_function(self, batch_size):
        states, actions, rewards, next_states, terminals = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states,
                            terminals=terminals)

        with torch.no_grad():
            target_actions, _ = self.target_policy.sample_action(next_states_prep, None)
            smooth_noise = torch.clamp(0.2 * torch.randn_like(target_actions), -0.5, 0.5)
            target_actions = torch.clamp(target_actions + smooth_noise, -1, 1)

            q_next_values = torch.zeros(self.K, batch_size, 1).to(device=self.device)
            for i in range(self.K):
                q_next_values[i] = self.q_target_nets[i](next_states_prep, target_actions)
            q_next_value = torch.min(q_next_values, dim=0)[0]
            target_q_value = rewards_prep + (1 - terminals_prep) * self.discount * q_next_value

        for k in range(self.K):
            pred_q_value = self.q_nets[k](states_prep, actions_prep)
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
        self.latent = torch.randn(1, self.latent_dim).to(device=self.device)

    @torch.no_grad()
    def get_action(self, state):
        '''
        with torch.no_grad():
            state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
            action, self.latent = self.policy.sample_action(state_prep, self.latent)
        return action.cpu().numpy().squeeze()
        '''
        state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
        means, log_stds, weights, next_latent = self.policy(state_prep, self.latent)

        extended_states = state_prep.unsqueeze(1).repeat(1, self.num_mixtures, 1)
        value_means, uncertainties = self.compute_values(extended_states.reshape(-1, self.state_dim),
                                                         means.reshape(-1, self.ac_dim))
        scores = (value_means - value_means.mean())
        scores = scores.reshape(1, self.num_mixtures)
        scores[weights<0.1] = -1000
        class_label = torch.argmax(scores, dim=1)
        return means[0, class_label].cpu().numpy().squeeze()
        '''
        dist_mean_input = torch.cat([means[:, i] for i in range(self.num_mixtures)], axis=-1).to(device=self.device)
        dis_log_std_input = torch.cat([log_stds[:, i] for i in range(self.num_mixtures)], axis=-1).to(device=self.device)
        attention_input = torch.cat([state_prep, self.latent, dist_mean_input, dis_log_std_input], dim=-1).to(device=self.device)
        estimated_weights = self.attention(attention_input)
        class_label = torch.argmax(weights, dim=1)
        self.latent = next_latent
        return means[0, class_label].cpu().numpy().squeeze()
        '''
    def log_prob_gaussian(self, x, mean, std):
        return -0.5 * ((x - mean) / std).pow(2) - torch.log(std) - 0.5 * torch.log(2 * torch.tensor(torch.pi))

    def compute_loss(self, means, log_stds, weights, actions):
        """
        Compute the negative log-likelihood loss for the Mixture Gaussian Policy.

        :param means: Tensor of shape [batch_size, num_mixtures, action_dim], the means of the Gaussians.
        :param log_stds: Tensor of shape [batch_size, num_mixtures, action_dim], the log stds of the Gaussians.
        :param weights: Tensor of shape [batch_size, num_mixtures], the mixture weights.
        :param actions: Tensor of shape [batch_size, action_dim], the actions taken.
        :return: Tensor, the computed loss.
        """
        batch_size, num_mixtures, _ = means.shape

        # Expand actions to match the shape of means and stds
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


