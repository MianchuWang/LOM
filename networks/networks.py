import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal

class Generator(nn.Module):
    def __init__(self, state_dim, ac_dim, goal_dim, noise_dim):
        super().__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.goal_dim = goal_dim
        self.noise_dim = noise_dim
        self.input_dim = self.state_dim + self.goal_dim + noise_dim
        self.model = nn.Sequential(nn.Linear(self.input_dim, 256),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(),
                                   nn.Linear(256, self.ac_dim))

    def forward(self, state, goal, noise):
        input = torch.cat([state, goal, noise], dim=-1)
        output = self.model(input)
        return torch.tanh(output)


class Discriminator(nn.Module):
    def __init__(self, state_dim, ac_dim, goal_dim):
        super().__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.goal_dim = goal_dim
        self.input_dim = self.state_dim + self.ac_dim + self.goal_dim
        self.model = nn.Sequential(nn.Linear(self.input_dim, 256),
                                   nn.LeakyReLU(),
                                   nn.Linear(256, 256),
                                   nn.LeakyReLU(),
                                   nn.Linear(256, 1))
    def forward(self, states, actions, goals):
        input = torch.cat([states, actions, goals], dim=-1)
        score = torch.sigmoid(self.model(input))
        # clip prevents NaN in the loss function.
        return torch.clip(score, 0.0001, 0.9999)


class Dynamics(nn.Module):
    def __init__(self, state_dim, ac_dim):
        super(Dynamics, self).__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.model = nn.Sequential(nn.Linear(state_dim + ac_dim, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, state_dim))

    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        output = self.model(input)
        return state + output

class Policy(nn.Module):
    def __init__(self, state_dim, ac_dim):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.model = nn.Sequential(nn.Linear(state_dim, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, ac_dim),
                                   )
        self.log_std = nn.Parameter(torch.zeros(ac_dim), requires_grad=True)

    def forward(self, s):
        loc = torch.tanh(self.model(s))
        scale = torch.exp(self.log_std)
        normal_dist = Normal(loc, scale)
        return normal_dist, loc


class MixedGaussianPolicy(nn.Module):
    def __init__(self, state_dim, ac_dim, num_components):
        super(MixedGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.num_components = num_components

        # Common layers for means, standard deviations, and mixing coefficients
        self.common_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Output layers for means, standard deviations, and mixing coefficients
        self.means_layer = nn.Linear(256, ac_dim * num_components)
        self.log_std_layer = nn.Linear(256, ac_dim * num_components)
        self.mixing_coeffs_layer = nn.Linear(256, num_components)

    def forward(self, state):
        common_out = self.common_layers(state)

        # Reshape for means and standard deviations
        means = self.means_layer(common_out).view(-1, self.num_components, self.ac_dim)
        log_std = self.log_std_layer(common_out).view(-1, self.num_components, self.ac_dim)
        std = torch.exp(log_std)

        # Create a batch of Multivariate Normal Distributions
        cov_matrix = torch.diag_embed(std**2)  # Assuming independence among action dimensions
        normal_dists = MultivariateNormal(means, covariance_matrix=cov_matrix)

        mixing_coeffs = torch.nn.functional.softmax(self.mixing_coeffs_layer(common_out), dim=-1)

        return normal_dists, mixing_coeffs


class Vnetwork(nn.Module):
    def __init__(self, state_dim):
        super(Vnetwork, self).__init__()
        self.state_dim = state_dim
        self.model = nn.Sequential(nn.Linear(state_dim, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))
    def forward(self, state):
        output = self.model(state)
        return output

class Qnetwork(nn.Module):
    def __init__(self, state_dim, ac_dim):
        super(Qnetwork, self).__init__()
        self.model = nn.Sequential(nn.Linear(state_dim + ac_dim, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))
    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        output = self.model(input)
        return output

class CVAE_network(nn.Module):
    def __init__(self, state_dim, ac_dim, latent_dim=64):
        super(CVAE_network, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(state_dim + ac_dim, 256)
        self.fc21 = nn.Linear(256, latent_dim)  # mean
        self.fc22 = nn.Linear(256, latent_dim)  # log variance

        # Decoder
        self.fc3 = nn.Linear(state_dim + latent_dim, 256)
        self.fc4 = nn.Linear(256, ac_dim)
        
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.ac_dim = ac_dim

    def encode(self, states, actions):
        combined = torch.cat([states, actions], 1)
        h1 = F.relu(self.fc1(combined))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, states):
        combined = torch.cat([z, states], 1)
        h3 = F.relu(self.fc3(combined))
        return self.fc4(h3)

    def forward(self, states, actions):
        mu, logvar = self.encode(states, actions)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, states), mu, logvar

class MixtureGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, num_mixtures=2):
        super(MixtureGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_mixtures = num_mixtures

        # Layers for mean, log standard deviation, and mixture weights
        self.mean_layer = nn.Linear(state_dim, action_dim * num_mixtures)
        self.log_std_layer = nn.Linear(state_dim, action_dim * num_mixtures)
        self.weights_layer = nn.Linear(state_dim, num_mixtures)

    def forward(self, state):
        means = self.mean_layer(state).view(-1, self.num_mixtures, self.action_dim)
        log_stds = self.log_std_layer(state).view(-1, self.num_mixtures, self.action_dim)
        weights = F.softmax(self.weights_layer(state), dim=-1)

        return means, log_stds, weights

    def sample_action(self, state):
        means, log_stds, weights = self.forward(state)
        stds = log_stds.exp()

        # Choose a component from the mixture
        mixture_indices = torch.multinomial(weights, 1).squeeze(-1)
        chosen_means = means[torch.arange(means.size(0)), mixture_indices]
        chosen_stds = stds[torch.arange(stds.size(0)), mixture_indices]

        # Sample from the chosen Gaussian
        normal = torch.distributions.Normal(chosen_means, chosen_stds)
        action = normal.sample()

        return action

    def sample_action_from_all(self, state):
        mean = self.mean_layer(state).view(-1, self.num_mixtures, self.action_dim)
        log_std = self.log_std_layer(state).view(-1, self.num_mixtures, self.action_dim)
        std = log_std.exp()  # Convert log standard deviations to standard deviations
        normal_dist = torch.distributions.Normal(mean, std)  # Define the normal distribution
        action = normal_dist.rsample()  # Sample an action
        return action
