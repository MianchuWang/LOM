import torch
import numpy as np

from agents.base_agent import BaseAgent
from networks.networks import Policy

class BC(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.policy = Policy(self.state_dim, self.ac_dim, self.goal_dim).to(device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.sample_func = self.replay_buffer.sample
        self.her_prob = 0

    def train_models(self, batch_size=512):
        states, actions, _, goals, _ = self.sample_func(batch_size, self.her_prob)
        states_prep, actions_prep, _, goals_prep = self.preprocess(states=states, actions=actions, goals=goals)
        
        ac_dist, ac_mean = self.policy(states_prep, goals_prep)
        log_prob = ac_dist.log_prob(actions_prep)
        policy_loss = - log_prob.mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        return {'policy_loss': policy_loss.item()}

    def get_action(self, state, goal):
        with torch.no_grad():
            state_prep, _, _, goal_prep = \
                self.preprocess(states=state[np.newaxis], goals=goal[np.newaxis])
            ac_dist, action = self.policy(state_prep, goal_prep)
        return action.cpu().numpy().squeeze()
        #return ac_dist.sample().cpu().numpy().squeeze()

    def plan(self, state, goal):
        raise NotImplementedError()

    def save(self, path):
        models = (self.policy)
        import joblib
        joblib.dump(models, path)