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
            #action = ac_dist.sample()
        return action.cpu().numpy().squeeze()