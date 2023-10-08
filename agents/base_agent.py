import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


class BaseAgent(object):
    def __init__(self, *args, **kwargs):
        self.replay_buffer = kwargs['replay_buffer']
        self.state_dim = kwargs['state_dim']
        self.ac_dim = kwargs['ac_dim']
        self.goal_dim = kwargs['goal_dim']
        self.device = kwargs['device']
        self.discount = kwargs['discount']
        self.max_steps = kwargs['max_steps']
        self.normalise = kwargs['normalise']
        self.env_name = kwargs['env_name']

        self.goal_scaler = StandardScaler()
        self.state_scaler = StandardScaler()
        self.diff_scaler = StandardScaler()

        self.get_goal_from_state = kwargs['get_goal_from_state']
        self.compute_reward = kwargs['compute_reward']

        self.fit_scalars()

    def reset(self):
        return

    def get_action(self, state, goal):
        raise NotImplementedError()

    def plan(self, state, goal):
        raise self.get_action(state, goal)

    def train_models(self):
        raise NotImplementedError()

    def fit_scalars(self, state_noise=0.05, diff_noise=0.01, goal_noise=0.05):
        _, high = self.replay_buffer.get_bounds()
        states = self.replay_buffer.obs[:high].reshape(-1, self.state_dim)
        self.state_scaler.fit(states + state_noise * np.random.randn(*states.shape))
        if not self.goal_dim == 0:
            goals = self.replay_buffer.goals[:high].reshape(-1, self.goal_dim)
            self.goal_scaler.fit(goals + goal_noise * np.random.randn(*goals.shape))

    def preprocess(self, states=None, actions=None, next_states=None, goals=None):
        if states is not None:
            if self.normalise:
                states_shape = states.shape
                states = self.state_scaler.transform(states.reshape(-1, self.state_dim)).reshape(states_shape)
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        if actions is not None:
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        if next_states is not None:
            if self.normalise:
                next_states_shape = next_states.shape
                next_states = self.state_scaler.transform(next_states.reshape(-1, self.state_dim)).reshape(next_states_shape)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        if goals is not None:
            if self.normalise and not self.goal_dim == 0:
                goals_shape = goals.shape
                goals = self.goal_scaler.transform(goals.reshape(-1, self.goal_dim)).reshape(goals_shape)
            goals = torch.tensor(goals, dtype=torch.float32, device=self.device)
        return states, actions, next_states, goals

    def postprocess(self, states=None, actions=None, next_states=None, goals=None):
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
        if goals is not None:
            goals = goals.detach().cpu().numpy()
            if self.normalise:
                goals = self.goal_scaler.inverse_transform(goals)
        return states, actions, next_states, goals

    def load(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()
