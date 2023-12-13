import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from networks.networks import CVAE_network


class temporal_CVAE(nn.Module):
    def __init__(self, state_dim, ac_dim, latent_dim=64, embedding_dim=128):
        super(temporal_CVAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(state_dim + ac_dim, 256)
        self.fc21 = nn.Linear(256, latent_dim)  # mean
        self.fc22 = nn.Linear(256, latent_dim)  # log variance

        # Decoder
        self.fc3 = nn.Linear(state_dim + latent_dim + embedding_dim, 256)
        self.fc4 = nn.Linear(256, ac_dim)
        self.fc5 = nn.Linear(256, embedding_dim)
        
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

    def decode(self, z, states, embedding):
        combined = torch.cat([z, states, embedding], 1)
        h3 = F.relu(self.fc3(combined))
        return self.fc4(h3), self.fc5(h3)

    def forward(self, states, actions, embedding):
        mu, logvar = self.encode(states, actions)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, states, embedding), mu, logvar


class seqCVAE(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.embedding_dim = 128
        self.embedding = torch.randn(1, self.embedding_dim).to(device=self.device)
        self.policy = temporal_CVAE(self.state_dim, self.ac_dim, embedding_dim=self.embedding_dim).to(device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def train_models(self, batch_size=512):
        
        states, actions = self.replay_buffer.sample_sequences(batch_size, sequence_length=10)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
    
        embedding = torch.randn(batch_size, self.embedding_dim).to(device=self.device)
        loss = 0
        for t in range(5):
            (recon_actions, embedding), mu, logvar = self.policy(states_prep[:, t], actions_prep[:, t], embedding)
            MSE = torch.nn.functional.mse_loss(recon_actions, actions_prep[:, t]).mean()
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += MSE + KLD
        
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()
    
        return {'policy/loss': loss.item(),
                'policy/MSE': MSE.mean().item(),
                'policy/KLD': KLD.mean().item()}

    def reset(self):
        self.embedding = torch.randn(1, self.embedding_dim).to(device=self.device)

    def get_action(self, state):
        with torch.no_grad():
            state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
            z = torch.randn(1, self.policy.latent_dim).to(self.device)
            
            generated_action, self.embedding = self.policy.decode(z, state_prep, self.embedding)
        return generated_action.cpu().numpy().squeeze()

    
    def plot_actions(self, state=None):
        if state is None:
            state, _, _, _, _ = self.replay_buffer.sample(1)
            state = state.squeeze()
            
        actions = [self.get_action(state) for _ in range(10000)]
        actions = np.array(actions)

            # Set the seaborn style for better aesthetics
        sns.set(style="whitegrid")
    
        # Determine the number of subplots needed
        num_dims = actions.shape[1]
        num_rows = num_dims // 3 + (num_dims % 3 > 0)
        plt.figure(figsize=(15, 4 * num_rows))
    
        # Plotting each action dimension
        for i in range(num_dims):
            plt.subplot(num_rows, min(num_dims, 3), i + 1)
            sns.histplot(actions[:, i], bins=50, kde=False, color='skyblue')
            plt.xlabel(f'Dimension {i+1}')
            plt.ylabel('Frequency')
            plt.title(f'Action Distribution - Dimension {i+1}')
            plt.grid(True)
    
        plt.tight_layout()
        plt.show()
    
    
