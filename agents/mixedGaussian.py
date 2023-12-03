import torch
import numpy as np

from agents.base_agent import BaseAgent
from networks.networks import MixedGaussianPolicy

class MixedGaussianBC(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.num_components = 20
        self.policy = MixedGaussianPolicy(self.state_dim, self.ac_dim, self.num_components).to(device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
    
    def train_models(self, batch_size=512):
        states, actions, _, _, _ = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
    
        normal_dists, mixing_coeffs = self.policy(states_prep)
    
        # Expand actions_prep to match the shape expected by log_prob
        expanded_actions = actions_prep.unsqueeze(1).expand(-1, self.num_components, -1)
    
        # Compute log probabilities for each component
        log_probs_all_components = normal_dists.log_prob(expanded_actions)
    
        # Expand and align mixing coefficients for addition
        log_mixing_coeffs = torch.log(mixing_coeffs)
    
        # Add log mixing coefficients
        log_probs_weighted = log_probs_all_components + log_mixing_coeffs
    
        # Aggregate log probabilities using log-sum-exp
        weighted_log_probs = torch.logsumexp(log_probs_weighted, dim=1)
    
        policy_loss = -weighted_log_probs.mean()
    
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
    
        return {'policy_loss': policy_loss.item()}



    
    def get_action(self, state):
        with torch.no_grad():
            state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
            normal_dists, mixing_coeffs = self.policy(state_prep)
    
            # Sample from the mixed Gaussian distribution
            # Select a component based on the mixing coefficients
            component_index = torch.multinomial(mixing_coeffs, num_samples=1, replacement=True).squeeze(-1)
    
            # Sample an action from the selected component
            # It's important to use the component index for each sample in the batch
            chosen_actions = normal_dists.sample()[torch.arange(state_prep.shape[0]), component_index]
    
            return chosen_actions.cpu().numpy().squeeze()


