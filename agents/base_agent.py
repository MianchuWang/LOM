import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


class BaseAgent(object):
    def __init__(self, *args, **kwargs):
        self.replay_buffer = kwargs['replay_buffer']
        self.state_dim = kwargs['state_dim']
        self.ac_dim = kwargs['ac_dim']
        self.device = kwargs['device']
        self.discount = kwargs['discount']
        self.normalise = kwargs['normalise']

        self.goal_scaler = StandardScaler()
        self.state_scaler = StandardScaler()
        self.diff_scaler = StandardScaler()

        self.fit_scalars()

    def reset(self):
        return

    def get_action(self, state):
        raise NotImplementedError()

    def plan(self, state):
        raise NotImplementedError()

    def train_models(self):
        raise NotImplementedError()

    def fit_scalars(self, state_noise=0.05):
        high = self.replay_buffer.get_bounds()
        states = self.replay_buffer.obs[:high].reshape(-1, self.state_dim)
        self.state_scaler.fit(states + state_noise * np.random.randn(*states.shape))

    def preprocess(self, states=None, actions=None, rewards=None, next_states=None, terminals=None):
        if states is not None:
            if self.normalise:
                states_shape = states.shape
                states = self.state_scaler.transform(states.reshape(-1, self.state_dim)).reshape(states_shape)
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        if actions is not None:
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        if rewards is not None:
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if next_states is not None:
            if self.normalise:
                next_states_shape = next_states.shape
                next_states = self.state_scaler.transform(next_states.reshape(-1, self.state_dim)).reshape(next_states_shape)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        if terminals is not None:
            terminals = torch.tensor(terminals, dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, terminals

    def postprocess(self, states=None, actions=None, next_states=None):
        if states is not None:
            states = states.detach().cpu().numpy()
            if self.normalise:
                states = self.state_scaler.inverse_transform(states)
        if actions is not None:
            actions = actions.detach().cpu().numpy()
        if next_states is not None:
            next_states = next_states.detach().cpu().numpy()
            if self.normalise:
                next_states = self.state_scaler.inverse_transform(next_states)
        return states, actions, next_states

    def load(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()
