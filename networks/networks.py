import torch
import torch.nn as nn
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

class Binary_Classifier(nn.Module):
    def __init__(self, state_dim, ac_dim):
        super(Binary_Classifier, self).__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.model = nn.Sequential(nn.Linear(state_dim+ac_dim, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))
    
    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        output = torch.sigmoid(self.model(input))
        return output
