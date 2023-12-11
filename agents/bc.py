import torch
import numpy as np

from agents.base_agent import BaseAgent
from networks.networks import Policy

class BC(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.policy = Policy(self.state_dim, self.ac_dim).to(device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def train_models(self, batch_size=512):
        states, actions, _, _, _ = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
        
        ac_dist, ac_mean = self.policy(states_prep)
        log_prob = ac_dist.log_prob(actions_prep)
        policy_loss = - log_prob.mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        return {'policy_loss': policy_loss.item()}

    def get_action(self, state):
        with torch.no_grad():
            state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
            ac_dist, action = self.policy(state_prep)
            action = ac_dist.sample()
        return action.cpu().numpy().squeeze()
    
    def plot_actions(self, state=None):
        if state is None:
            state, _, _, _, _ = self.replay_buffer.sample(1)
        state, _, _, _, _ = self.preprocess(states=state[np.newaxis])
        state = state.squeeze()
         
        actions = [self.policy(state)[0].sample().cpu().numpy().squeeze() for _ in range(10000)]
        actions = np.array(actions)
        
        import seaborn as sns
        import matplotlib.pyplot as plt
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