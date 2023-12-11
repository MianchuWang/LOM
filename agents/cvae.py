import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from agents.base_agent import BaseAgent
from networks.networks import CVAE_network

class CVAE(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.policy = CVAE_network(self.state_dim, self.ac_dim).to(device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def train_models(self, batch_size=512):
        states, actions, _, _, _ = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
    
        recon_actions, mu, logvar = self.policy(states_prep, actions_prep)
        
        MSE = torch.nn.functional.mse_loss(recon_actions, actions_prep).mean()
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = MSE + KLD
        
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()
    
        return {'policy/loss': loss.item(),
                'policy/MSE': MSE.mean().item(),
                'policy/KLD': KLD.mean().item()}


    def get_action(self, state):
        with torch.no_grad():
            state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
            z = torch.randn(1, self.policy.latent_dim).to(self.device)
            generated_action = self.policy.decode(z, state_prep)
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
    
    
